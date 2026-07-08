# Less is More — Recursive Reasoning with Tiny Networks (interactive marimo notebook)

An interactive [marimo](https://marimo.io) notebook that re-implements the core of
**"Less is More: Recursive Reasoning with Tiny Networks"** (Alexia Jolicoeur-Martineau,
[arXiv:2510.04871](https://arxiv.org/abs/2510.04871)) **from scratch**, trains a
**0.49 M-parameter, 2-layer** network on the GPU, and lets you **watch it reason its way
through a Sudoku, one recursive step at a time**.

Built for the **alphaXiv × molab Notebook Competition**.

![TRM](https://img.shields.io/badge/params-0.49M-blue) ![acc](https://img.shields.io/badge/blank--cell%20acc-~90%25-green)

## What's inside

- **The whole idea in two functions** — a single tiny network reused for latent reasoning
  and answer refinement; the only switch is whether the question `x` is added to the input.
- **Live GPU training** — trains a TRM from scratch in ~1–2 min (falls back to embedded
  pre-trained weights on CPU, so it always runs).
- **🎬 Interactive "watch it solve"** — pick a difficulty, hit *Try another puzzle*, and drag
  the recursion-step slider to see wrong guesses (red) flip to right ones (blue).
- **"Less is More" ablations** — the table of results where *smaller / simpler = better*.
- **Our own extension** — how much test-time recursion actually buys, plus the finding that
  the model is only stable **within its trained recursion horizon**.
- **🤖 Challenge a giant LLM** — copy the shown puzzle to any chatbot and watch a
  trillion-parameter model score 0 % where our 0.49 M net succeeds.

## Run it

**On molab (GPU, recommended):** open [molab.marimo.io](https://molab.marimo.io), import
`trm_notebook.py`, pick the GPU runtime, and *Run all*.

**Locally:**
```bash
uv venv && source .venv/bin/activate
uv pip install marimo torch numpy matplotlib
marimo run trm_notebook.py     # or `marimo edit trm_notebook.py`
```
The notebook is self-contained — the trained weights are embedded, so it runs on CPU or GPU
with no external files or downloads.

## Repo layout

| Path | What |
|---|---|
| `trm_notebook.py` | the self-contained interactive notebook (embedded weights) |
| `src/model.py` | TRM implementation (single 2-layer MLP-Mixer net, deep recursion, stablemax) |
| `src/sudoku.py` | self-contained Sudoku generation (no downloads) |
| `src/train.py` | deep-supervision training loop (EMA, best-checkpoint tracking) |

## Paper

Alexia Jolicoeur-Martineau, *Less is More: Recursive Reasoning with Tiny Networks*,
arXiv:2510.04871 (2025). This notebook is an educational re-implementation of its core idea
on synthetic Sudoku; it is not affiliated with the authors.
