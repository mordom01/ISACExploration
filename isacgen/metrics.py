"""Waveform quality metrics for unimodular (phase-coded) sequences.

Conventions
-----------
* A waveform is a complex vector x of length N (batch: [..., N]).
* Aperiodic autocorrelation r[k] = sum_n x[n] conj(x[n-k]), k = -(N-1)..(N-1).
* PSL  = max_{k != 0} |r[k]| / |r[0]|          (reported in dB: 20 log10)
* ISL  = sum_{k != 0} |r[k]|^2                  (two-sided, unnormalised; also /|r[0]|^2 in dB)
* Cross-correlation between x and y: c[k] = sum_n x[n] conj(y[n-k]); we report
  peak  = max_k |c[k]| / N   and   mean = mean_k |c[k]| / N   over all 2N-1 lags,
  and the zero-lag value |c[0]|/N separately (the quantity a synchronised comm receiver sees).
* PAPR = max |x|^2 / mean |x|^2.
All routines are numpy and batch-friendly; they are the reference implementations the
learned models will be scored with, so keep them boring and exact.
"""
from __future__ import annotations
import numpy as np


def _acorr_fft(x: np.ndarray) -> np.ndarray:
    """Aperiodic autocorrelation for batch x[..., N] -> r[..., 2N-1] (lags -(N-1)..N-1)."""
    N = x.shape[-1]
    L = 2 * N
    X = np.fft.fft(x, n=L, axis=-1)
    r = np.fft.ifft(np.abs(X) ** 2, axis=-1)
    # r[0..N-1] are lags 0..N-1; r[L-(N-1)..L-1] are lags -(N-1)..-1
    return np.concatenate([r[..., L - (N - 1):], r[..., :N]], axis=-1)


def _xcorr_fft(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Aperiodic cross-correlation c[k] = sum_n x[n] conj(y[n-k]) for lags -(N-1)..N-1."""
    N = x.shape[-1]
    L = 2 * N
    X = np.fft.fft(x, n=L, axis=-1)
    Y = np.fft.fft(y, n=L, axis=-1)
    c = np.fft.ifft(X * np.conj(Y), axis=-1)
    return np.concatenate([c[..., L - (N - 1):], c[..., :N]], axis=-1)


def psl_isl(x: np.ndarray):
    """Return (psl_linear, isl_linear_normalised) for batch x[..., N].

    psl_linear = max_{k!=0}|r[k]|/|r[0]|;  isl_norm = sum_{k!=0}|r[k]|^2 / |r[0]|^2.
    """
    r = _acorr_fft(x)
    N = x.shape[-1]
    r0 = np.abs(r[..., N - 1])
    side = np.abs(np.delete(r, N - 1, axis=-1))
    psl = side.max(axis=-1) / r0
    isl = (side ** 2).sum(axis=-1) / r0 ** 2
    return psl, isl


def psl_db(x):
    return 20 * np.log10(psl_isl(x)[0])


def isl_db(x):
    return 10 * np.log10(psl_isl(x)[1])


def isl_raw(x: np.ndarray) -> np.ndarray:
    """Unnormalised two-sided ISL sum_{k!=0} |r[k]|^2 (what CAN minimises)."""
    r = _acorr_fft(x)
    N = x.shape[-1]
    side = np.abs(np.delete(r, N - 1, axis=-1))
    return (side ** 2).sum(axis=-1)


def xcorr_stats(X: np.ndarray, lag_window: int | None = None):
    """Pairwise cross-correlation statistics for a set X[K, N].

    Returns dict with per-pair arrays over the K(K-1)/2 unordered pairs:
      peak      : max_k |c[k]|/N over all lags (or |k| <= lag_window if given)
      mean      : mean_k |c[k]|/N over the same lags
      zero_lag  : |c[0]|/N
      cross_isl : sum_k |c[k]|^2 (all lags, unnormalised)
    """
    K, N = X.shape
    peaks, means, zl, cisl = [], [], [], []
    for i in range(K):
        for j in range(i + 1, K):
            c = np.abs(_xcorr_fft(X[i], X[j]))
            if lag_window is not None:
                c_w = c[N - 1 - lag_window: N + lag_window]
            else:
                c_w = c
            peaks.append(c_w.max() / N)
            means.append(c_w.mean() / N)
            zl.append(c[N - 1] / N)
            cisl.append((c ** 2).sum())
    return dict(peak=np.array(peaks), mean=np.array(means), zero_lag=np.array(zl),
                cross_isl=np.array(cisl))


def set_total_isl(X: np.ndarray) -> float:
    """Total ISL of a set: sum of auto sidelobe energies + sum over ordered pairs of cross energies.
    Equals sum_k ||R_k - N I delta_k||_F^2 with R_k the KxK correlation matrix at lag k.
    Lower bound (aperiodic sequence-set bound) is N^2 (K-1)."""
    K, N = X.shape
    total = isl_raw(X).sum()
    st = xcorr_stats(X)
    total += 2 * st["cross_isl"].sum()  # ordered pairs
    return float(total)


def papr(x: np.ndarray) -> np.ndarray:
    p = np.abs(x) ** 2
    return p.max(axis=-1) / p.mean(axis=-1)


def ambiguity_function(x: np.ndarray, n_doppler: int = 64, max_norm_doppler: float = 0.5) -> np.ndarray:
    """Discrete ambiguity |A(k, nu)| for a single waveform x[N].
    nu is normalised Doppler (cycles/sample) on a grid in [-max, max]; k over all 2N-1 lags.
    Returns array [n_doppler, 2N-1], normalised so that A(0,0)=1."""
    N = x.shape[-1]
    nus = np.linspace(-max_norm_doppler, max_norm_doppler, n_doppler)
    n = np.arange(N)
    xd = x[None, :] * np.exp(2j * np.pi * nus[:, None] * n[None, :])  # Doppler-shifted copies
    c = _xcorr_fft(xd, np.broadcast_to(x, xd.shape))
    return np.abs(c) / N


def af_psl_db(x: np.ndarray, n_doppler: int = 64, max_norm_doppler: float = 0.5,
              exclude_lag: int = 1, exclude_dop: int = 1) -> float:
    """Peak sidelobe of the full AF outside a small mainlobe box around (0,0), in dB."""
    A = ambiguity_function(x, n_doppler, max_norm_doppler)
    N = x.shape[-1]
    nd = A.shape[0]
    c0 = nd // 2
    mask = np.ones_like(A, dtype=bool)
    mask[c0 - exclude_dop: c0 + exclude_dop + 1, N - 1 - exclude_lag: N + exclude_lag] = False
    return float(20 * np.log10(A[mask].max()))


def spectrum_db(x: np.ndarray, nfft: int | None = None) -> np.ndarray:
    """Power spectrum in dB (normalised to peak), fftshifted."""
    N = x.shape[-1]
    nfft = nfft or 8 * N
    P = np.abs(np.fft.fftshift(np.fft.fft(x, n=nfft, axis=-1), axes=-1)) ** 2
    P = P / P.max(axis=-1, keepdims=True)
    return 10 * np.log10(P + 1e-12)


def mask_violation_db(x: np.ndarray, mask_db: np.ndarray) -> np.ndarray:
    """Max excess of the spectrum over a mask (same length as spectrum grid), in dB; <=0 means compliant."""
    S = spectrum_db(x, nfft=mask_db.shape[-1])
    return (S - mask_db).max(axis=-1)
