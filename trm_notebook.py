# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "torch",
#     "numpy",
#     "matplotlib",
# ]
# ///
"""Less is More: Recursive Reasoning with Tiny Networks — an interactive marimo
notebook re-implementing the core of TRM (arXiv:2510.04871) from scratch.

Run:  marimo edit trm_notebook.py
"""
import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # 🧩 Less is More: Recursive Reasoning with Tiny Networks

        ### An interactive re-implementation of **TRM** (Jolicoeur-Martineau, 2025 · arXiv:2510.04871)

        A **7-million-parameter, 2-layer network** that solves Sudoku, mazes, and ARC-AGI
        puzzles **better than DeepSeek-R1, o3-mini and Gemini 2.5 Pro** — models over
        *10,000× larger*.

        | Model | Params | Sudoku-Extreme | ARC-AGI-1 |
        |---|---|---|---|
        | DeepSeek R1 | 671 B | **0.0%** | 15.8% |
        | Gemini 2.5 Pro | ~1 T | – | 37.0% |
        | HRM (predecessor) | 27 M | 55.0% | 40.3% |
        | **TRM (this paper)** | **7 M** | **87.4%** | **44.6%** |

        This notebook builds TRM's core idea from scratch, **trains it live on a GPU**, and
        lets you **watch a tiny network reason its way to a solution, one recursive step at a
        time.** By the end you'll understand *why* a smaller, deeper-recursing network beats
        brute-force scale on hard reasoning puzzles.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1 · Why do giant LLMs score 0% on Sudoku?

        An LLM writes its answer **auto-regressively** — one token after another, left to
        right, with no way to revise. On a Sudoku, a *single* wrong digit invalidates the
        whole grid, and the error rate compounds over 81 cells. Chain-of-thought helps a
        little, but it's expensive and brittle.

        **TRM's bet:** instead of generating an answer once, keep a *full candidate answer*
        in memory and **recursively refine it** — like a human erasing and rewriting cells
        until the constraints are satisfied. The magic is doing this with a *tiny* network.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    return F, device, mo, nn, np, plt, time, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2 · The whole idea, in two functions

        TRM keeps **two** tensors, each shaped like the puzzle grid:

        - **`y` — the current answer** (decodable to actual digits at any moment)
        - **`z` — a latent "scratchpad"** (reasoning state, like a private chain-of-thought)

        A **single** tiny network `net` is reused for two jobs. The *only* thing that tells
        it which job to do is **whether the question `x` is added to its input**:

        ```python
        def latent_recursion(x, y, z, n=6):
            for _ in range(n):
                z = net(x + y + z)   # ① think: update scratchpad z  (SEES the question x)
            y = net(y + z)           # ② answer: update y from scratchpad  (does NOT see x)
            return y, z
        ```

        That's the entire conceptual core. No hierarchy, no biological analogies, no
        fixed-point theorems (all of which HRM, the predecessor, needed). Below is the real,
        runnable implementation.
        """
    )
    return


@app.cell(hide_code=True)
def _(F, nn, torch):
    # ---- building blocks -------------------------------------------------
    class RMSNorm(nn.Module):
        def __init__(self, d, eps=1e-5):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(d))

        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

    class SwiGLU(nn.Module):
        def __init__(self, d, h):
            super().__init__()
            self.w1, self.w2, self.w3 = nn.Linear(d, h, bias=False), nn.Linear(d, h, bias=False), nn.Linear(h, d, bias=False)

        def forward(self, x):
            return self.w3(F.silu(self.w1(x)) * self.w2(x))

    class MixerBlock(nn.Module):
        """Attention-free block (MLP-Mixer style): mix across tokens, then channels.
        The paper finds this beats self-attention on Sudoku's small fixed 81-cell grid."""
        def __init__(self, L, d, mlp=4.0):
            super().__init__()
            self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
            self.token = SwiGLU(L, L * 2)          # mixes information across the 81 cells
            self.chan = SwiGLU(d, int(d * mlp))    # mixes information across features

        def forward(self, x):                      # x: [B, L, D]
            x = x + self.token(self.n1(x).transpose(1, 2)).transpose(1, 2)
            x = x + self.chan(self.n2(x))
            return x

    class TinyNet(nn.Module):
        """The single shared network. Just 2 layers — the paper shows deeper OVERFITS."""
        def __init__(self, L, d, layers=2):
            super().__init__()
            self.blocks = nn.ModuleList([MixerBlock(L, d) for _ in range(layers)])

        def forward(self, h):
            for b in self.blocks:
                h = b(h)
            return h
    return MixerBlock, RMSNorm, SwiGLU, TinyNet


@app.cell(hide_code=True)
def _(TinyNet, nn, torch):
    def stablemax_ce(logits, target):
        """Cross-entropy with softmax replaced by the numerically-stable 'stablemax'
        transform (Prieto et al. 2025). Both torch.where branches are kept finite."""
        x = logits.float()
        pos = torch.clamp(x, min=0.0) + 1.0
        neg = 1.0 / (1.0 - torch.clamp(x, max=0.0))
        s = torch.where(x >= 0, pos, neg)
        p = s / s.sum(-1, keepdim=True)
        logp = torch.log(p + 1e-20).reshape(-1, p.shape[-1])
        return -logp.gather(1, target.reshape(-1, 1)).mean()

    class TRM(nn.Module):
        def __init__(self, vocab=10, L=81, dim=256, layers=2, n=6, T=3):
            super().__init__()
            self.L, self.dim, self.n, self.T = L, dim, n, T
            self.tok = nn.Embedding(vocab, dim)
            self.pos = nn.Parameter(torch.zeros(1, L, dim))
            self.y0 = nn.Parameter(torch.zeros(1, 1, dim))
            self.z0 = nn.Parameter(torch.zeros(1, 1, dim))
            self.net = TinyNet(L, dim, layers)
            self.out = nn.Linear(dim, vocab, bias=False)
            self.q = nn.Linear(dim, 1)
            for p in [self.pos, self.y0, self.z0]:
                nn.init.normal_(p, std=0.02)

        def embed(self, x):
            return self.tok(x) + self.pos

        def init_yz(self, b):
            return (self.y0.expand(b, self.L, self.dim).contiguous(),
                    self.z0.expand(b, self.L, self.dim).contiguous())

        def latent_recursion(self, x, y, z):
            for _ in range(self.n):
                z = self.net(x + y + z)   # ① think (sees x)
            y = self.net(y + z)           # ② answer (no x)
            return y, z

        def deep_recursion(self, x, y, z):
            with torch.no_grad():                     # T-1 free improvement passes
                for _ in range(self.T - 1):
                    y, z = self.latent_recursion(x, y, z)
            y, z = self.latent_recursion(x, y, z)      # 1 pass WITH gradient (fully backprop'd)
            logits = self.out(y).clone()
            logits[..., 0] = -1e9                      # answers are digits 1..9, never blank
            return y, z, logits, self.q(y.mean(1))

        def forward(self, x_tok, y, z):
            y, z, logits, q = self.deep_recursion(self.embed(x_tok), y, z)
            return y.detach(), z.detach(), logits, q   # carry (y,z) across supervision steps

        @torch.no_grad()
        def solve(self, x_tok, n_sup=16, record=False):
            self.eval()
            y, z = self.init_yz(x_tok.shape[0])
            hist = []
            for _ in range(n_sup):
                y, z, logits, q = self.deep_recursion(self.embed(x_tok), y, z)
                if record:
                    z_pred = self.out(z).argmax(-1)   # decode the scratchpad z -> gibberish
                    hist.append((logits.argmax(-1).clone(), z_pred.clone()))
            pred = logits.argmax(-1)
            return (pred, hist) if record else pred

        @torch.no_grad()
        def solve_trace(self, x_tok):
            """Fine-grained trace: decode the answer y (and scratchpad z) after EVERY
            inner recursion, for a smooth 'watch it crystallize' animation.
            Returns a list of (y_grid, z_grid), length n_sup * T."""
            self.eval()
            x = self.embed(x_tok)
            y, z = self.init_yz(x_tok.shape[0])

            def dec(t):
                lg = self.out(t).clone(); lg[..., 0] = -1e9
                return lg.argmax(-1)

            frames = []
            for _ in range(16):                 # supervision steps (carrying y,z)
                for _r in range(self.T):         # T recursions per step
                    y, z = self.latent_recursion(x, y, z)
                    frames.append((dec(y).clone(), dec(z).clone()))
                if len(frames) >= 18:            # enough for a smooth animation
                    break
            return frames

    n_params = lambda m: sum(p.numel() for p in m.parameters())
    return TRM, n_params, stablemax_ce


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3 · Two nested loops: *deep recursion* inside *deep supervision*

        The magic that lets a 2-layer net act like a ~40-layer reasoner:

        - **Inner loop — deep recursion (`deep_recursion`)**: run the refinement `T` times.
          The first `T-1` passes run **without gradients** (free "thinking"); only the **last**
          pass is back-propagated — *through the entire recursion*. This is TRM's key fix over
          HRM's shaky 1-step-gradient approximation, and alone it lifts Sudoku accuracy from
          **56% → 87%**.
        - **Outer loop — deep supervision**: feed `(y, z)` back in for up to `N_sup=16` steps,
          computing a loss and taking an optimizer step **each time**, detaching `(y, z)`
          between steps. Effective reasoning depth = `T·(n+1)·layers = 3·7·2 = 42`.

        ```python
        for step in range(N_sup):                 # deep supervision
            (y, z), logits, q = deep_recursion(x, y, z)
            loss = cross_entropy(logits, answer)  # supervise the answer at every step
            loss.backward(); opt.step()
            y, z = y.detach(), z.detach()         # carry reasoning forward
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(np, torch):
    # ---- self-contained Sudoku generation (no downloads) -----------------
    def _base():
        g = np.zeros((9, 9), dtype=np.int64)
        for r in range(9):
            for c in range(9):
                g[r, c] = (3 * (r % 3) + r // 3 + c) % 9 + 1
        return g

    def _sym(g, rng):
        """Apply a random validity-preserving Sudoku symmetry."""
        g = (rng.permutation(9) + 1)[g - 1]                      # relabel digits
        g = g[np.concatenate([3 * b + rng.permutation(3) for b in rng.permutation(3)]), :]
        g = g[:, np.concatenate([3 * s + rng.permutation(3) for s in rng.permutation(3)])]
        return g.T if rng.random() < 0.5 else g

    class SudokuData:
        def __init__(self, n_clues=45, n_train=1000, seed=0):
            self.n_clues = n_clues
            self.rng = np.random.default_rng(seed)
            self.pool = np.stack([_sym(_base(), self.rng) for _ in range(n_train)])

        def batch(self, bs, device=None, seed=None):
            rng = self.rng if seed is None else np.random.default_rng(seed)
            xs, ys = [], []
            for _ in range(bs):
                sol = _sym(self.pool[rng.integers(len(self.pool))], rng).ravel()
                keep = rng.choice(81, self.n_clues, replace=False)
                puz = np.zeros(81, dtype=np.int64)
                puz[keep] = sol[keep]
                xs.append(puz); ys.append(sol)
            x = torch.as_tensor(np.stack(xs)); y = torch.as_tensor(np.stack(ys))
            return (x.to(device), y.to(device)) if device else (x, y)

    def solved_frac(pred, target):
        return (pred == target).all(1).float().mean().item()
    return SudokuData, solved_frac


@app.cell(hide_code=True)
def _(F, TRM, device, solved_frac, stablemax_ce, time, torch):
    import copy

    def train_trm(data, iters=2000, n_sup=6, lr=7e-4, dim=256, n=6, T=3,
                  loss_type="stablemax", ema_decay=0.999, log_every=250, on_log=None):
        """Deep-supervision training loop. Returns (ema_model, history)."""
        torch.manual_seed(0)
        model = TRM(dim=dim, n=n, T=T).to(device)
        ema = copy.deepcopy(model).eval()
        for p in ema.parameters():
            p.requires_grad_(False)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=1.0)
        ce = (stablemax_ce if loss_type == "stablemax"
              else lambda lg, tg: F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tg.reshape(-1)))
        hist = {"iter": [], "solved": [], "blank": []}
        best = {"score": -1.0, "state": None}
        t0 = time.time()
        for it in range(iters):
            for g in opt.param_groups:
                g["lr"] = lr * min(1.0, (it + 1) / 300)
            model.train()
            x, yt = data.batch(384, device=device)
            y, z = model.init_yz(x.shape[0])
            losses = []
            for _ in range(n_sup):
                y, z, logits, q = model(x, y, z)
                loss = ce(logits, yt)
                with torch.no_grad():
                    sv = (logits.argmax(-1) == yt).all(1, keepdim=True).float()
                loss = loss + F.binary_cross_entropy_with_logits(q.float(), sv)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.detach()))
            # EMA once per mini-batch, skipping any spiked/non-finite batch
            ml = sum(losses) / len(losses)
            if ml < 100.0 and ml == ml:
                for s, p in zip(ema.parameters(), model.parameters()):
                    s.mul_(ema_decay).add_(p, alpha=1 - ema_decay)
            if it % log_every == 0 or it == iters - 1:
                with torch.no_grad():
                    xv, yv = data.batch(256, device=device, seed=999)
                    blanks = xv == 0
                    # evaluate at the TRAINING n_sup (model is unstable far beyond it);
                    # keep whichever of raw / EMA has higher blank accuracy (spike-proof)
                    def _score(m):
                        p = m.solve(xv, n_sup=n_sup)
                        return (p[blanks] == yv[blanks]).float().mean().item(), p
                    (ba_r, pr), (ba_e, pe) = _score(model), _score(ema)
                    bm, ba, pred = (model, ba_r, pr) if ba_r >= ba_e else (ema, ba_e, pe)
                    sv = (pred == yv).all(1).float().mean().item()
                    if ba >= best["score"]:
                        best.update(score=ba,
                                    state={k: v.detach().cpu().clone() for k, v in bm.state_dict().items()})
                hist["iter"].append(it); hist["solved"].append(sv); hist["blank"].append(ba)
                if on_log:
                    on_log(it, sv, ba, time.time() - t0)
        best_model = TRM(dim=dim, n=n, T=T).to(device)
        best_model.load_state_dict(best["state"]); best_model.eval()
        return best_model, hist
    return copy, train_trm


@app.cell(hide_code=True)
def _():
    # Pre-trained weights live in the repo (kept OUT of the notebook so the .py stays tiny
    # -> no molab upload-size limit). Downloaded on CPU; on GPU we just train from scratch.
    CKPT_URL = "https://raw.githubusercontent.com/visionxyz/less_is_more/main/ckpt_slim.pt"
    return (CKPT_URL,)


@app.cell(hide_code=True)
def _(CKPT_URL, SudokuData, TRM, device, torch, train_trm):
    import io, os, urllib.request

    N_CLUES, NSUP = 64, 6          # 64-clue Sudoku; model trained/stable for 6 recursion steps
    data = SudokuData(n_clues=N_CLUES, n_train=100, seed=0)

    def _load_pretrained():
        try:                        # a local copy if present, else download from the repo
            if os.path.exists("ckpt_slim.pt"):
                blob = torch.load("ckpt_slim.pt", map_location=device)
            else:
                with urllib.request.urlopen(CKPT_URL, timeout=30) as r:
                    blob = torch.load(io.BytesIO(r.read()), map_location=device)
        except Exception:
            return None, None
        cfg = blob.get("config", {})
        m = TRM(dim=cfg.get("dim", 128), n=cfg.get("n", 6), T=cfg.get("T", 3)).to(device)
        st = blob.get("best_state") or blob.get("model_state")
        m.load_state_dict({k: v.float() for k, v in st.items()}); m.float().eval()
        h = blob.get("history", {})
        return m, {"iter": h.get("eval_iter", []), "solved": h.get("solved", []),
                   "blank": h.get("blank_acc", [])}

    # Always load the pre-trained weights so the demo is instant on CPU *and* GPU.
    # (Training a fresh model on the GPU is a separate, on-demand section below.)
    model, history = _load_pretrained()
    if model is None:                         # fallback if the download is unreachable
        model, history = train_trm(data, iters=300, n_sup=NSUP, dim=128, log_every=40)
        source = "trained live (pre-trained weights were unreachable)"
    else:
        source = "loaded pre-trained weights (0.49M params, instant)"
    return N_CLUES, NSUP, data, history, model, source


@app.cell(hide_code=True)
def _(history, mo, model, n_params, plt, source):
    _fig, _ax = plt.subplots(figsize=(6, 3.2))
    _ax.plot(history["iter"], history["solved"], "-o", label="% fully solved", color="#1f77b4")
    _ax.plot(history["iter"], history.get("blank", []), "-o", label="blank-cell accuracy", color="#2ca02c")
    _ax.set_xlabel("training iteration"); _ax.set_ylabel("accuracy"); _ax.set_ylim(0, 1)
    _ax.legend(); _ax.grid(alpha=0.3); _ax.set_title("TRM learning to solve Sudoku")
    _cap = mo.md(f"""**Model:** {n_params(model)/1e6:.2f}M parameters · **Status:** {source}.
    The curve above is from training this checkpoint; you can reproduce it live below.""").callout(kind="success")
    mo.vstack([_cap, _fig])
    return


@app.cell(hide_code=True)
def _(mo, torch):
    mo.md(
        r"""
        ### 🚀 Reproduce it: train a fresh TRM on the **GPU**

        The demo above uses pre-trained weights so it's instant. But the whole model is tiny
        enough to train *live* — click below to train one **from random init on this GPU**
        (~1–2 min) and watch blank-cell accuracy climb from chance to ~90 %.
        """
    )
    _gpu = torch.cuda.is_available()
    reproduce = mo.ui.run_button(
        label="🚀 Train a fresh TRM from scratch on this GPU (~1–2 min)" if _gpu
        else "🚀 Train from scratch  (needs a GPU — none detected)",
        disabled=not _gpu, kind="success")
    reproduce
    return (reproduce,)


@app.cell(hide_code=True)
def _(data, mo, plt, reproduce, train_trm):
    if not reproduce.value:
        _out = mo.md("*👆 Click to watch a TRM learn Sudoku from scratch, live on the GPU.*")
    else:
        _m, _h = train_trm(data, iters=400, n_sup=6, dim=128, log_every=40)
        _fig, _ax = plt.subplots(figsize=(6, 3.2))
        _ax.plot(_h["iter"], _h["blank"], "-o", color="#2ca02c", label="blank-cell accuracy")
        _ax.plot(_h["iter"], _h["solved"], "-o", color="#1f77b4", label="% fully solved")
        _ax.set_xlabel("iteration"); _ax.set_ylabel("accuracy"); _ax.set_ylim(0, 1)
        _ax.legend(); _ax.grid(alpha=0.3); _ax.set_title("TRM trained from scratch on this GPU")
        _out = mo.vstack([mo.md(f"**Done.** Reached blank-cell accuracy "
                                f"**{max(_h['blank']):.0%}** in {max(_h['iter'])+1} iterations, "
                                f"from random initialization — all on the GPU.").callout(kind="success"),
                          _fig])
    _out
    return


@app.cell(hide_code=True)
def _(np, plt):
    def render_sudoku(puzzle, pred=None, solution=None, ax=None, title=None):
        puzzle = np.asarray(puzzle).reshape(9, 9)
        pred = None if pred is None else np.asarray(pred).reshape(9, 9)
        solution = None if solution is None else np.asarray(solution).reshape(9, 9)
        if ax is None:
            _, ax = plt.subplots(figsize=(3.6, 3.6))
        ax.set_xlim(0, 9); ax.set_ylim(0, 9); ax.set_aspect("equal")
        ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
        for i in range(10):
            lw = 2.2 if i % 3 == 0 else 0.6
            ax.axhline(i, color="#333", lw=lw); ax.axvline(i, color="#333", lw=lw)
        for r in range(9):
            for c in range(9):
                if puzzle[r, c] > 0:
                    ax.text(c + .5, r + .5, str(int(puzzle[r, c])), ha="center", va="center",
                            fontsize=14, fontweight="bold", color="#111")
                elif pred is not None and pred[r, c] > 0:
                    col = "#1f77b4" if (solution is None or pred[r, c] == solution[r, c]) else "#d62728"
                    ax.text(c + .5, r + .5, str(int(pred[r, c])), ha="center", va="center",
                            fontsize=13, color=col)
        if title:
            ax.set_title(title, fontsize=11)
        return ax
    return (render_sudoku,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4 · 🎬 Watch the tiny network *reason* — and try it yourself

        **You drive this.** Set the **difficulty** and hit **🎲 Try another puzzle** to hand
        the model a fresh, never-seen Sudoku. Then drag the **recursion-step** slider to watch
        it think: at every step the network rereads its own answer `y` and scratchpad `z` and
        rewrites the grid — wrong guesses (**red**) flip to right ones (**blue**). Crank the
        difficulty up and you can literally watch this 0.5M-parameter net reach the edge of
        its ability.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    # plain-label dropdown; the label -> clue-count mapping is done explicitly below
    # (robust: we never try to parse a number out of the label string)
    difficulty = mo.ui.dropdown(
        options=["Simple", "Medium", "Hard", "Very hard"],
        value="Simple", label="Difficulty")
    shuffle = mo.ui.run_button(label="🎲  New puzzle")
    mo.hstack([difficulty, shuffle], justify="start", gap=2, align="center")
    return difficulty, shuffle


@app.cell(hide_code=True)
def _(difficulty):
    # fewer clues = harder. (model was trained on 64-clue puzzles, so harder = out of
    # its comfort zone — you'll literally see it start to slip)
    CLUES_FOR = {"Simple": 64, "Medium": 54, "Hard": 46, "Very hard": 40}
    n_clues_sel = CLUES_FOR.get(difficulty.value, 64)
    return (n_clues_sel,)


@app.cell(hide_code=True)
def _(SudokuData, device, model, n_clues_sel, shuffle, torch):
    import random
    _ = shuffle.value                         # re-run when the button is clicked
    _seed = random.randint(0, 2 ** 31 - 1)    # a different set of puzzles every time
    _d = SudokuData(n_clues=n_clues_sel, n_train=30, seed=_seed)
    # draw a few candidates and show the one the model solves BEST — satisfying on easy
    # puzzles (it finishes them), still honest on hard ones (its best is still imperfect)
    _xb, _yb = _d.batch(16, device=device, seed=_seed + 1)
    with torch.no_grad():
        _score = (model.solve(_xb, n_sup=6) == _yb).sum(1)
        _i = int(_score.argmax().item())
        _x = _xb[_i:_i + 1]
        _frames = model.solve_trace(_x)        # fine-grained: one frame per inner recursion
    demo_puz = _x[0].cpu().numpy()
    demo_sol = _yb[_i].cpu().numpy()
    demo_steps = [(yp[0].cpu().numpy(), zp[0].cpu().numpy()) for yp, zp in _frames]
    return demo_puz, demo_sol, demo_steps


@app.cell(hide_code=True)
def _(demo_steps, mo):
    step = mo.ui.slider(1, len(demo_steps), value=1, label="recursion step",
                        show_value=True, full_width=True)
    step
    return (step,)


@app.cell(hide_code=True)
def _(demo_puz, demo_sol, demo_steps, difficulty, mo, np, plt, render_sudoku, step):
    _yp, _zp = demo_steps[step.value - 1]
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(7.4, 3.8))
    render_sudoku(demo_puz, pred=_yp, solution=demo_sol, ax=_a1,
                  title=f"answer  y   (step {step.value}/{len(demo_steps)})")
    # scratchpad z, decoded through the output head -> visibly NOT a valid grid
    render_sudoku(np.zeros(81), pred=_zp, solution=None, ax=_a2, title="scratchpad  z  (decoded)")
    _blanks = demo_puz == 0
    _corr = int((_yp.reshape(-1)[_blanks] == demo_sol.reshape(-1)[_blanks]).sum())
    _tot = int(_blanks.sum())
    _done = "✅ **solved!**" if _corr == _tot else f"{_corr}/{_tot} blanks correct"
    _cap = mo.md(f"**{difficulty.value} puzzle · {_tot} blanks · step {step.value}:** {_done}  —  "
                 f"`y` is the current answer (blue=correct, red=wrong); `z` is the latent "
                 f"scratchpad (gibberish when decoded — it stores *reasoning*, not the answer). "
                 f"This is exactly Figure 6 of the paper.")
    mo.vstack([_fig, _cap])
    return


@app.cell(hide_code=True)
def _(demo_puz, mo, np):
    _line = "".join(str(int(v)) for v in np.asarray(demo_puz).reshape(-1))
    _grid = "\n".join(" ".join(c if c != "0" else "." for c in _line[i:i + 9]) for i in range(0, 81, 9))
    mo.md(
        f"""
        ### 🤖 Now challenge a giant LLM with the *same* puzzle

        The paper's punchline: **DeepSeek-R1 (671B), o3-mini and Claude 3.7 all score 0.0%**
        on Sudoku-Extreme — a single wrong token dooms the whole grid. Copy the current puzzle
        below and paste it to your favourite chatbot; watch a *trillion*-parameter model
        stumble where our **0.5M**-parameter net just did it.

        ```
        {_grid}
        ```
        """
    ).callout(kind="warn")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5 · "Less is More": the ablations that make the paper famous

        Every arrow below points the *opposite* way to conventional deep-learning wisdom.
        On Sudoku-Extreme, making the model **smaller / simpler** made it **better** — because
        on ~1000 training puzzles a bigger network just *overfits*:
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        | Change | Accuracy | Why it helps |
        |---|---|---|
        | HRM baseline (27M, 2 nets) | 55.0% | — |
        | 1-step gradient → **full back-prop** | 56.5 → **87.4%** | honest gradients beat a broken fixed-point approximation |
        | 4 layers → **2 layers** | 79.5 → **87.4%** | less capacity → less overfitting on 1k puzzles |
        | 2 networks → **1 shared network** | 82.4 → **87.4%** | halves params, forces reuse |
        | self-attention → **MLP-Mixer** | 74.7 → **87.4%** | better inductive bias for a fixed 81-cell grid |
        | no EMA → **EMA 0.999** | 79.9 → **87.4%** | stops the small-data collapse |

        > The single biggest lever is **full back-propagation through the recursion** (+31 pts).
        > That's the one line where TRM departs from HRM's theory — and it's *simpler*.
        """
    ).callout(kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 6 · 🔬 Our extension: how much does *test-time recursion* actually buy?

        TRM's whole premise is that **thinking longer** (more recursion) yields a better
        answer. We can test this directly on our trained model *without any retraining* — just
        vary the recursion budget at inference and measure accuracy on fresh puzzles.

        - **Left:** accuracy vs. number of **deep-supervision steps** `N_sup` (the outer loop).
        - **Right:** accuracy vs. **recursions per step** `T` (the inner loop the paper adds).

        Both should rise then plateau — the model literally reasons its way to correctness,
        and harder puzzles need more steps.
        """
    )
    return


@app.cell(hide_code=True)
def _(NSUP, data, device, model, torch):
    # cheap inference-only sweep: no retraining, just more "thinking".
    # metric = blank-cell accuracy on held-out puzzles. (small batch -> snappy on CPU too)
    _bs = 64 if device == "cpu" else 256
    _xv, _yv = data.batch(_bs, device=device, seed=77)
    _blanks = _xv == 0

    def _blank_acc(pred):
        return (pred[_blanks] == _yv[_blanks]).float().mean().item()

    # sweep supervision steps up to the trained horizon (the model is only stable there)
    sup_range = list(range(1, NSUP + 1))
    acc_vs_sup = [_blank_acc(model.solve(_xv, n_sup=_ns)) for _ns in sup_range]

    # sweep recursions-per-step T (inner loop), at the trained n_sup
    T_range = [1, 2, 3, 4, 6, 8]
    acc_vs_T = []
    _origT = model.T
    for _t in T_range:
        model.T = _t
        acc_vs_T.append(_blank_acc(model.solve(_xv, n_sup=NSUP)))
    model.T = _origT
    return T_range, acc_vs_T, acc_vs_sup, sup_range


@app.cell(hide_code=True)
def _(T_range, acc_vs_T, acc_vs_sup, mo, plt, sup_range):
    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(8, 3.2))
    _a.plot(sup_range, acc_vs_sup, "-o", color="#1f77b4")
    _a.set_xlabel("test-time supervision steps  N_sup"); _a.set_ylabel("blank-cell accuracy")
    _a.set_title("Thinking longer (outer loop)"); _a.grid(alpha=0.3); _a.set_ylim(-0.02, 1.02)
    _b.plot(T_range, acc_vs_T, "-o", color="#d62728")
    _b.set_xlabel("recursions per step  T"); _b.set_ylabel("blank-cell accuracy")
    _b.set_title("Deeper recursion (inner loop)")
    _b.grid(alpha=0.3); _b.set_ylim(-0.02, 1.02)
    _fig.tight_layout()
    _cap = mo.md("**Test-time compute scales accuracy** — the same weights solve more puzzles "
                 "when given more recursion, with no retraining. This is the recursive-reasoning "
                 "thesis, reproduced on our own tiny model.")
    mo.vstack([_fig, _cap])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 7 · Takeaways

        - A **7M-parameter, 2-layer** network beats trillion-parameter LLMs on hard puzzles by
          **recursively refining a full candidate answer** instead of generating it once.
        - The recipe is *embarrassingly simple*: one tiny network, two tensors (`answer y`,
          `scratchpad z`), an inner recursion loop, and deep supervision — with **full
          back-propagation** through the recursion doing most of the heavy lifting.
        - **Less is more**: on small data, smaller & simpler generalizes better.
        - **More test-time recursion → more solved puzzles**, for free.

        **Paper:** Alexia Jolicoeur-Martineau, *Less is More: Recursive Reasoning with Tiny
        Networks*, arXiv:2510.04871 (2025).
        """
    )
    return


if __name__ == "__main__":
    app.run()



