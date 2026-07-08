"""Tiny Recursive Model (TRM) — faithful, minimal implementation.

Reference: Alexia Jolicoeur-Martineau, "Less is More: Recursive Reasoning
with Tiny Networks" (arXiv:2510.04871).

Core idea (paper Fig. 3): a SINGLE tiny network `net` is reused to
  (1) refine a latent reasoning feature z given (x, y, z)   -> latent reasoning
  (2) refine the current answer y given (y, z)              -> answer update
The task the net performs is signalled purely by whether the question x is
added to the input. Deep supervision carries (y, z) across steps; only the
last full recursion is back-propagated.

This module implements the attention-free (MLP-Mixer style) variant, which the
paper finds best on Sudoku (fixed, small context length L=81).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: w3( silu(w1 x) * w2 x )."""
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class MixerBlock(nn.Module):
    """One attention-free block (MLP-Mixer style).

    - token mixing: a SwiGLU MLP applied ACROSS the sequence dim L
      (this replaces self-attention; cheap when L is small & fixed)
    - channel mixing: a SwiGLU MLP applied across the feature dim D
    Both are pre-norm residual sub-layers.
    """
    def __init__(self, seq_len: int, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.token_mix = SwiGLU(seq_len, int(seq_len * 2))
        self.norm2 = RMSNorm(dim)
        self.chan_mix = SwiGLU(dim, int(dim * mlp_ratio))

    def forward(self, x):  # x: [B, L, D]
        # token mixing operates on transposed [B, D, L]
        h = self.norm1(x).transpose(1, 2)      # [B, D, L]
        h = self.token_mix(h).transpose(1, 2)  # [B, L, D]
        x = x + h
        x = x + self.chan_mix(self.norm2(x))
        return x


class TinyNet(nn.Module):
    """The single shared 2-layer network reused for both z and y updates.

    Input is a single [B, L, D] tensor (the SUM of the relevant embeddings),
    exactly as in HRM/TRM (e.g. z <- net(x + y + z), y <- net(y + z)).
    """
    def __init__(self, seq_len: int, dim: int, n_layers: int = 2, mlp_ratio: float = 4.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [MixerBlock(seq_len, dim, mlp_ratio) for _ in range(n_layers)]
        )

    def forward(self, h):  # h: [B, L, D]
        for blk in self.blocks:
            h = blk(h)
        return h


# ---------------------------------------------------------------------------
# Stablemax cross-entropy (Prieto et al. 2025) — used by TRM for stability
# ---------------------------------------------------------------------------
def _stablemax_s(x):
    # s(x) = x + 1 for x >= 0, else 1 / (1 - x)  -> positive, monotonic.
    # Both operands of torch.where must be finite EVERYWHERE, otherwise the
    # unused branch (e.g. 1/(1-x) at x=1) produces inf/nan that poisons the
    # backward pass. We clamp so each branch is finite on the whole domain.
    pos = torch.clamp(x, min=0.0) + 1.0            # x>=0 -> x+1 ; x<0 -> 1
    neg = 1.0 / (1.0 - torch.clamp(x, max=0.0))    # x<=0 -> 1/(1-x) ; x>0 -> 1
    return torch.where(x >= 0, pos, neg)


def stablemax_cross_entropy(logits, target, ignore_index: int = -100):
    """Cross-entropy where softmax is replaced by the stablemax transform.

    logits: [..., V], target: [...] (int64). Returns mean over valid entries.
    """
    s = _stablemax_s(logits.float())
    p = s / s.sum(dim=-1, keepdim=True)
    logp = torch.log(p + 1e-20)
    logp = logp.reshape(-1, logp.shape[-1])
    tgt = target.reshape(-1)
    valid = tgt != ignore_index
    tgt_c = tgt.clamp(min=0)
    nll = -logp.gather(1, tgt_c[:, None]).squeeze(1)
    nll = nll[valid]
    return nll.mean() if nll.numel() > 0 else logits.sum() * 0.0


# ---------------------------------------------------------------------------
# Tiny Recursive Model
# ---------------------------------------------------------------------------
class TRM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        dim: int = 256,
        n_layers: int = 2,
        n: int = 6,          # latent-reasoning steps per recursion
        T: int = 3,          # full recursions per supervision step (T-1 no-grad + 1 grad)
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.dim = dim
        self.n = n
        self.T = T

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, dim))
        # learned initial answer / latent features (broadcast over batch & L)
        self.y_init = nn.Parameter(torch.zeros(1, 1, dim))
        self.z_init = nn.Parameter(torch.zeros(1, 1, dim))

        self.net = TinyNet(seq_len, dim, n_layers, mlp_ratio)
        self.out_head = nn.Linear(dim, vocab_size, bias=False)
        self.q_head = nn.Linear(dim, 1, bias=True)  # halting logit

        self.apply(self._init)
        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.y_init, std=0.02)
        nn.init.normal_(self.z_init, std=0.02)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)

    # --- embeddings -------------------------------------------------------
    def embed_input(self, x_tokens):
        return self.tok_emb(x_tokens) + self.pos_emb  # [B, L, D]

    def init_yz(self, batch):
        y = self.y_init.expand(batch, self.seq_len, self.dim).contiguous()
        z = self.z_init.expand(batch, self.seq_len, self.dim).contiguous()
        return y, z

    # --- core recursion (paper Fig. 3) -----------------------------------
    def latent_recursion(self, x, y, z):
        for _ in range(self.n):
            z = self.net(x + y + z)   # latent reasoning (x present)
        y = self.net(y + z)           # answer update    (x absent)
        return y, z

    def deep_recursion(self, x, y, z):
        # T-1 improvement passes without gradient ...
        with torch.no_grad():
            for _ in range(self.T - 1):
                y, z = self.latent_recursion(x, y, z)
        # ... then one pass WITH gradient
        y, z = self.latent_recursion(x, y, z)
        y_logits = self.out_head(y)                 # [B, L, V]
        # answers are digits 1..9; token 0 (blank) must never be emitted
        y_logits = y_logits.clone()
        y_logits[..., 0] = -1e9
        q_logit = self.q_head(y.mean(dim=1))        # [B, 1] halting logit
        return y, z, y_logits, q_logit

    def forward(self, x_tokens, y, z):
        """One supervision step. Returns updated (detached) y, z and heads."""
        x = self.embed_input(x_tokens)
        y, z, y_logits, q_logit = self.deep_recursion(x, y, z)
        return y.detach(), z.detach(), y_logits, q_logit

    @torch.no_grad()
    def solve(self, x_tokens, n_sup: int = 16, record: bool = False, early_stop: bool = False):
        """Run inference for n_sup supervision steps.

        Per the paper, test-time runs the FULL n_sup steps (no early stop) to
        maximize accuracy, so `early_stop` defaults to False. If record=True,
        returns the predicted-token grid after each supervision step (for the
        interactive 'watch it solve' demo)."""
        self.eval()
        B = x_tokens.shape[0]
        y, z = self.init_yz(B)
        history = []
        for step in range(n_sup):
            x = self.embed_input(x_tokens)
            y, z, y_logits, q_logit = self.deep_recursion(x, y, z)
            pred = y_logits.argmax(-1)
            if record:
                history.append(pred.clone())
            if early_stop and torch.sigmoid(q_logit).mean() > 0.5 and step >= 1:
                break
        return (pred, history) if record else pred


def count_params(model):
    return sum(p.numel() for p in model.parameters())
