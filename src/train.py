"""Training loop for TRM on synthetic Sudoku.

Implements deep supervision exactly as in the paper: for each mini-batch we run
up to N_sup supervision steps, and EACH step is its own optimizer update whose
gradient flows only through the final (with-grad) recursion of that step. (y, z)
are detached and carried to the next step. EMA weights are used for evaluation.
"""
from __future__ import annotations
import argparse, copy, json, math, os, time
import torch
import torch.nn.functional as F

from model import TRM, stablemax_cross_entropy, count_params
from sudoku import SudokuData, accuracy


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p, alpha=1 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)


def lr_at(it, base_lr, warmup):
    if it < warmup:
        return base_lr * (it + 1) / warmup
    return base_lr


def train(
    iters=3000, batch_size=256, n_sup=6, n=6, T=3, dim=256, n_layers=2,
    n_clues=35, n_train=1000, mode="fixed", lr=7e-4, warmup=300, wd=1.0,
    ema_decay=0.999, loss_type="stablemax", device="cuda", seed=0,
    eval_every=200, amp=True, log=print,
):
    torch.manual_seed(seed)
    import contextlib
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    amp_ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
               if (amp and device == "cuda") else contextlib.nullcontext())
    data = SudokuData(n_clues=n_clues, n_train=n_train, mode=mode, seed=seed)
    model = TRM(vocab_size=10, seq_len=81, dim=dim, n_layers=n_layers, n=n, T=T).to(device)
    ema = EMA(model, ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=wd)
    log(f"TRM params: {count_params(model)/1e6:.2f}M | device={device} "
        f"| n={n} T={T} n_sup={n_sup} dim={dim} clues={n_clues} mode={mode}")

    ce = (stablemax_cross_entropy if loss_type == "stablemax"
          else lambda lg, tg: F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tg.reshape(-1)))

    hist = {"iter": [], "loss": [], "eval_iter": [], "cell_acc": [], "solved": []}
    best = {"score": -1.0, "state": None, "which": None}
    t0 = time.time()
    for it in range(iters):
        for g in opt.param_groups:
            g["lr"] = lr_at(it, lr, warmup)
        model.train()
        x, y_true = data.batch(batch_size, device=device)
        yb, zb = model.init_yz(x.shape[0])
        step_losses = []
        for step in range(n_sup):
            with amp_ctx:
                yb, zb, y_logits, q_logit = model(x, yb, zb)
            loss_ce = ce(y_logits, y_true)  # stablemax casts to fp32 internally
            with torch.no_grad():
                solved = (y_logits.argmax(-1) == y_true).all(dim=1, keepdim=True).float()
            loss_q = F.binary_cross_entropy_with_logits(q_logit.float(), solved)
            loss = loss_ce + loss_q
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step_losses.append(loss_ce.item())
        # EMA once per mini-batch (not per supervision step), and never absorb a
        # spiked/non-finite batch — this is what corrupts the EMA otherwise.
        mean_loss = sum(step_losses) / len(step_losses)
        if mean_loss < 100.0 and all(map(lambda v: v == v, step_losses)):
            ema.update(model)
        hist["iter"].append(it)
        hist["loss"].append(mean_loss)

        if it % eval_every == 0 or it == iters - 1:
            # probe RAW model across n_sup to see the effect of extra recursion
            _ns_lo, _ns_hi = 2, 2 * n_sup
            probe = {ns: evaluate(model, data, device, n_sup=ns, batches=1, bs=256)[0]
                     for ns in (_ns_lo, n_sup, _ns_hi)}
            log(f"  probe cell@nsup: {_ns_lo}={probe[_ns_lo]:.3f} {n_sup}={probe[n_sup]:.3f} "
                f"{_ns_hi}={probe[_ns_hi]:.3f}")
            # best-tracking + main metrics at the TRAINING n_sup (matches how it was trained)
            ca, sv, ba = evaluate(ema.shadow, data, device, n_sup=n_sup, batches=4, bs=256)
            rca, rsv, rba = evaluate(model, data, device, n_sup=n_sup, batches=2, bs=256)
            hist["eval_iter"].append(it)
            hist["cell_acc"].append(ca)
            hist["solved"].append(sv)
            hist.setdefault("blank_acc", []).append(ba)
            # keep the best-performing weights (raw OR ema) by BLANK accuracy — the
            # metric that actually reflects solving; 'solved' (whole grid) stays ~0.
            src = ("raw", rba, model) if rba >= ba else ("ema", ba, ema.shadow)
            if src[1] >= best["score"]:
                best.update(score=src[1], which=src[0],
                            state={k: v.detach().cpu().clone() for k, v in src[2].state_dict().items()})
            log(f"it {it:5d} | loss {hist['loss'][-1]:.4f} | EMA[cell {ca:.3f} blank {ba:.3f} "
                f"solved {sv:.3f}] | RAW[cell {rca:.3f} blank {rba:.3f} solved {rsv:.3f}] | "
                f"best_blank {best['score']:.3f}({best['which']}) | {time.time()-t0:.0f}s")
    return model, ema, data, hist, best


@torch.no_grad()
def evaluate(model, data, device, n_sup=16, batches=4, bs=256):
    model.eval()
    cas, svs, bas = [], [], []
    for b in range(batches):
        x, y_true = data.batch(bs, device=device, seed=100000 + b)
        pred = model.solve(x, n_sup=n_sup)
        ca, sv = accuracy(pred, y_true)
        blanks = x == 0
        ba = (pred[blanks] == y_true[blanks]).float().mean().item()
        cas.append(ca); svs.append(sv); bas.append(ba)
    return sum(cas) / len(cas), sum(svs) / len(svs), sum(bas) / len(bas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--n_sup", type=int, default=6)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--T", type=int, default=3)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--n_clues", type=int, default=35)
    ap.add_argument("--n_train", type=int, default=1000)
    ap.add_argument("--mode", type=str, default="fixed")
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--loss_type", type=str, default="stablemax", choices=["stablemax", "ce"])
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--out", type=str, default="ckpt.pt")
    args = ap.parse_args()

    import sys
    sys.stdout.reconfigure(line_buffering=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ema, data, hist, best = train(
        iters=args.iters, batch_size=args.batch_size, n_sup=args.n_sup,
        n=args.n, T=args.T, dim=args.dim, n_clues=args.n_clues,
        n_train=args.n_train, mode=args.mode, lr=args.lr, device=dev,
        warmup=args.warmup, loss_type=args.loss_type,
        eval_every=args.eval_every, amp=bool(args.amp),
    )
    torch.save({
        "ema_state": ema.shadow.state_dict(),
        "model_state": model.state_dict(),
        "best_state": best["state"],          # best eval weights (spike-proof)
        "best_score": best["score"],
        "best_which": best["which"],
        "config": vars(args),
        "history": hist,
    }, args.out)
    print(f"best checkpoint: blank_acc={best['score']:.3f} from {best['which']} model")
    with open(os.path.splitext(args.out)[0] + "_hist.json", "w") as f:
        json.dump(hist, f)
    print("saved", args.out)


if __name__ == "__main__":
    main()
