"""Binary (and M-PSK) sequence tools for Direction A'': incremental-update local search and PSL utilities.

Conventions: b in {+1,-1}^N, aperiodic autocorrelation r_k = sum_{n} b_n b_{n+k}, k = 1..N-1.
PSL = max_k |r_k| (integer), ISL = sum_k r_k^2. Flipping bit j changes r_k by  -2 b_j (b_{j+k} + b_{j-k})
(terms with out-of-range indices dropped), so all N single-flip candidates can be scored in O(N^2).

Search (`shotgun_psl`): steepest-descent single-flip local search on the lexicographic objective
(PSL, number of lags attaining PSL, ISL), with sideways moves and a short tabu list, restarted from random
sequences. This is the flavour of stochastic search used for record PSL values (Dimitrov, Baicheva, Nikolov,
IEEE SPL 2020; Mow, Du, Wu, IEEE TAES 2015). It is the classical baseline *and* the training-data generator.
"""
from __future__ import annotations
import numpy as np


def acorr_int(b: np.ndarray) -> np.ndarray:
    """r_k for k=1..N-1 as integers (b in {+1,-1})."""
    N = b.shape[-1]
    r = np.correlate(b, b, mode="full")[N:]
    return np.rint(r).astype(int)


def psl(b: np.ndarray) -> int:
    return int(np.abs(acorr_int(b)).max())


_IDX_CACHE = {}


def _idx(N: int):
    if N not in _IDX_CACHE:
        J = np.arange(N)[:, None]
        K = np.arange(1, N)[None, :]
        _IDX_CACHE[N] = (N + J + K, N + J - K)   # indices into a zero-padded copy [0]*N + b + [0]*N
    return _IDX_CACHE[N]


def flip_deltas(b: np.ndarray) -> np.ndarray:
    """D[j, k-1] = change of r_k if bit j is flipped = -2 b_j (b_{j+k} + b_{j-k}). Shape [N, N-1]. Vectorised."""
    N = b.shape[0]
    ip, im = _idx(N)
    bp = np.concatenate([np.zeros(N), b, np.zeros(N)])
    return -2.0 * b[:, None] * (bp[ip] + bp[im])


def lex_key(r: np.ndarray):
    a = np.abs(r)
    p = a.max()
    return (int(p), int((a == p).sum()), int((r ** 2).sum()))


def shotgun_psl(N: int, rng: np.random.Generator, max_iters: int = 2000, tabu: int = 8, sideways: int = 50,
                target: int | None = None):
    """One restart of steepest-descent single-flip search. Returns (b, psl, iters)."""
    b = rng.choice([-1.0, 1.0], size=N)
    r = acorr_int(b).astype(float)
    key = lex_key(r)
    recent = []
    no_improve = 0
    best_b, best_key = b.copy(), key
    for it in range(max_iters):
        D = flip_deltas(b)
        R = r[None, :] + D                      # candidate correlations after each flip [N, N-1]
        A = np.abs(R)
        P = A.max(axis=1)
        C = (A == P[:, None]).sum(axis=1)
        I = (R ** 2).sum(axis=1)
        score = P * 1e9 + C * 1e5 + I           # lexicographic (PSL, count, ISL) as one float
        for j in recent:
            score[j] = np.inf
        j = int(np.argmin(score))
        new_key = (int(P[j]), int(C[j]), int(I[j]))
        b[j] = -b[j]; r = R[j]; key = new_key
        recent.append(j); recent = recent[-tabu:]
        if key < best_key:
            best_key, best_b, no_improve = key, b.copy(), 0
            if target is not None and best_key[0] <= target:
                break
        else:
            no_improve += 1
            if no_improve > sideways:
                break
    return best_b, best_key[0], it + 1


def canonical(b: np.ndarray) -> tuple:
    """Canonical representative under the PSL-preserving symmetry group: negation, reversal, alternation."""
    N = b.shape[0]
    alt = (-1.0) ** np.arange(N)
    cands = []
    for s in (b, -b):
        for t in (s, s[::-1]):
            for u in (t, t * alt):
                cands.append(tuple((u > 0).astype(int)))
    return min(cands)
