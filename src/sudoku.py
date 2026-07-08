"""Self-contained Sudoku generation for TRM.

No downloads: we generate valid full solutions from a base pattern and apply
validity-preserving symmetries (digit relabeling, band/stack and within-band
row/col permutations, transpose). Puzzles are made by masking cells.

Tokens: 0 = blank (input only), 1..9 = digits.  vocab_size = 10, L = 81.
"""
from __future__ import annotations
import numpy as np
import torch

N = 9
L = 81


def _base_solution() -> np.ndarray:
    """A valid 9x9 solution from the standard pattern."""
    g = np.zeros((N, N), dtype=np.int64)
    for r in range(N):
        for c in range(N):
            g[r, c] = (3 * (r % 3) + r // 3 + c) % N + 1  # 1..9
    return g


def _shuffle_solution(rng: np.random.Generator) -> np.ndarray:
    g = _base_solution()
    # relabel digits
    perm = rng.permutation(N) + 1
    g = perm[g - 1]
    # permute rows within each band, and the bands themselves
    bands = rng.permutation(3)
    rows = np.concatenate([3 * b + rng.permutation(3) for b in bands])
    g = g[rows, :]
    # same for columns
    stacks = rng.permutation(3)
    cols = np.concatenate([3 * s + rng.permutation(3) for s in stacks])
    g = g[:, cols]
    if rng.random() < 0.5:
        g = g.T
    return g


def _valid(grid: np.ndarray) -> bool:
    """Check a full grid is a valid Sudoku solution (used in tests)."""
    ok = np.arange(1, N + 1)
    for i in range(N):
        if set(grid[i, :]) != set(ok) or set(grid[:, i]) != set(ok):
            return False
    for br in range(0, N, 3):
        for bc in range(0, N, 3):
            if set(grid[br:br + 3, bc:bc + 3].ravel()) != set(ok):
                return False
    return True


def make_puzzle(solution: np.ndarray, n_clues: int, rng: np.random.Generator):
    """Mask cells to leave `n_clues` given; return (puzzle, solution) flat."""
    flat = solution.ravel().copy()
    mask = np.zeros(L, dtype=bool)
    keep = rng.choice(L, size=n_clues, replace=False)
    mask[keep] = True
    puzzle = np.where(mask, flat, 0)
    return puzzle.astype(np.int64), flat.astype(np.int64)


class SudokuData:
    """Generates batches of (input_tokens, target_tokens).

    modes:
      - fixed:   a fixed pool of `n_train` base solutions (paper's small-data
                 regime); each draw picks one and applies a random symmetry
                 augmentation + random clue mask.
      - infinite: a fresh random solution every draw.
    """
    def __init__(self, n_clues: int = 35, n_train: int = 1000, mode: str = "fixed",
                 seed: int = 0):
        self.n_clues = n_clues
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.pool = None
        if mode == "fixed":
            self.pool = np.stack([_shuffle_solution(self.rng) for _ in range(n_train)])

    def _one(self, rng):
        if self.mode == "fixed":
            sol = self.pool[rng.integers(len(self.pool))]
            sol = _augment(sol, rng)  # symmetry augmentation
        else:
            sol = _shuffle_solution(rng)
        return make_puzzle(sol, self.n_clues, rng)

    def batch(self, batch_size: int, device=None, seed: int | None = None):
        rng = self.rng if seed is None else np.random.default_rng(seed)
        xs, ys = [], []
        for _ in range(batch_size):
            p, s = self._one(rng)
            xs.append(p)
            ys.append(s)
        x = torch.as_tensor(np.stack(xs))
        y = torch.as_tensor(np.stack(ys))
        if device is not None:
            x, y = x.to(device), y.to(device)
        return x, y


def _augment(solution: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a random validity-preserving symmetry to a full solution."""
    g = solution.copy()
    perm = rng.permutation(N) + 1
    g = perm[g - 1]
    bands = rng.permutation(3)
    rows = np.concatenate([3 * b + rng.permutation(3) for b in bands])
    g = g[rows, :]
    stacks = rng.permutation(3)
    cols = np.concatenate([3 * s + rng.permutation(3) for s in stacks])
    g = g[:, cols]
    if rng.random() < 0.5:
        g = g.T
    return g


def accuracy(pred: torch.Tensor, target: torch.Tensor):
    """Return (cell_acc, solved_frac) where solved = whole grid correct."""
    correct = pred == target
    cell_acc = correct.float().mean().item()
    solved = correct.all(dim=1).float().mean().item()
    return cell_acc, solved


def to_grid(tokens) -> np.ndarray:
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.detach().cpu().numpy()
    return np.asarray(tokens).reshape(N, N)
