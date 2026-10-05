"""Closed-form unimodular code families and random sampling on the torus."""
import numpy as np


def random_unimodular(n_samples: int, N: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform (Haar) samples on the N-torus: i.i.d. phases in [0, 2pi)."""
    return np.exp(1j * rng.uniform(0, 2 * np.pi, size=(n_samples, N)))


def frank(M: int) -> np.ndarray:
    """Frank code of length N = M^2."""
    p, q = np.meshgrid(np.arange(M), np.arange(M), indexing="ij")
    return np.exp(2j * np.pi * p * q / M).reshape(-1)


def p4(N: int) -> np.ndarray:
    n = np.arange(N)
    return np.exp(1j * (np.pi * n ** 2 / N - np.pi * n))


def zadoff_chu(N: int, u: int = 1) -> np.ndarray:
    n = np.arange(N)
    if N % 2 == 0:
        return np.exp(-1j * np.pi * u * n ** 2 / N)
    return np.exp(-1j * np.pi * u * n * (n + 1) / N)


# ---- binary sequence families used by classical code-shift-keying (CSK) DFRC baselines ----

def _lfsr_mseq(taps: list[int], n: int, seed: int = 1) -> np.ndarray:
    """m-sequence of length 2^n-1 from a primitive polynomial given as exponent list (incl. n and 0)."""
    state = [(seed >> i) & 1 for i in range(n)]
    out = []
    for _ in range(2 ** n - 1):
        out.append(state[-1])
        fb = 0
        for t in taps:
            if 0 < t <= n:
                fb ^= state[t - 1]
        state = [fb] + state[:-1]
    return np.array(out, dtype=int)


def gold_set(n: int = 6) -> np.ndarray:
    """Gold sequences of length 2^n-1 from a preferred pair (n=6: x^6+x+1, x^6+x^5+x^2+x+1). Returns [2^n+1, N] in {+1,-1}."""
    if n == 6:
        a = _lfsr_mseq([6, 1], 6); b = _lfsr_mseq([6, 5, 2, 1], 6)
    elif n == 5:
        a = _lfsr_mseq([5, 2], 5); b = _lfsr_mseq([5, 4, 3, 2], 5)
    else:
        raise ValueError("only n in {5,6} wired up")
    N = len(a)
    seqs = [a, b] + [a ^ np.roll(b, s) for s in range(N)]
    return 1.0 - 2.0 * np.array(seqs, dtype=float)


def kasami_small_set(n: int = 6) -> np.ndarray:
    """Small Kasami set, length 2^n-1 (n even): 2^(n/2) sequences in {+1,-1}."""
    assert n % 2 == 0
    a = _lfsr_mseq([6, 1], 6) if n == 6 else _lfsr_mseq([4, 1], 4)
    N = len(a)
    d = 2 ** (n // 2) + 1
    bdec = a[::d][: (N // d)]
    b = np.tile(bdec, d)[:N]
    seqs = [a] + [a ^ np.roll(b, s) for s in range(2 ** (n // 2) - 1)]
    return 1.0 - 2.0 * np.array(seqs, dtype=float)
