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
