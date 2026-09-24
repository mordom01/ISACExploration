"""Classical per-instance optimisers used as baselines.

* can           : CAN (Stoica, He, Li, IEEE TSP 2009) - cyclic ISL minimisation for one unimodular sequence.
* multi_can     : Multi-CAN (He, Stoica, Li, IEEE TSP 2009) - joint auto+cross ISL minimisation for a set.
                  Uses the identity  sum_k ||R_k - N I delta_k||_F^2 = (1/2N) sum_p (||z_p||^2 - NK)^2 + N^2 K (K-1),
                  i.e. the set criterion only constrains the *summed* power spectrum to be flat, and the
                  total ISL of any set of K unimodular length-N sequences is >= N^2 K (K-1).
* torus_gd      : Riemannian gradient descent on the N-torus (x = exp(j phi)) for weighted-ISL objectives,
                  with an IRLS option that approximates the l_p norm of the sidelobes (PSL surrogate,
                  cf. Song, Babu, Palomar, IEEE TSP 2016 for the MM version).
All are numpy, single-threaded, deliberately simple: they are baselines, not contributions.
"""
from __future__ import annotations
import numpy as np
from .metrics import _acorr_fft, isl_raw


def _unimod(z):
    return np.exp(1j * np.angle(z))


def can(x0: np.ndarray, n_iter: int = 2000, tol: float = 1e-6, return_hist: bool = False):
    """CAN for a single sequence. x0[N] unimodular init. Returns optimised x (and ISL history)."""
    N = x0.shape[-1]
    L = 2 * N
    x = x0.copy()
    hist = []
    prev = np.inf
    for it in range(n_iter):
        z = np.fft.fft(x, n=L) / np.sqrt(L)             # unitary DFT of zero-padded x
        v = _unimod(z) / np.sqrt(2.0)                    # |v_p|^2 = 1/2  <=> |a_p^H x|^2 = N
        g = np.fft.ifft(v * np.sqrt(L), n=L) * np.sqrt(L) / L  # unitary IDFT
        x = _unimod(g[:N])
        if return_hist or tol > 0:
            J = float(isl_raw(x))
            hist.append(J)
            if abs(prev - J) < tol * max(1.0, J):
                break
            prev = J
    return (x, np.array(hist)) if return_hist else x


def multi_can(X0: np.ndarray, n_iter: int = 2000, tol: float = 1e-7, return_hist: bool = False):
    """Multi-CAN for a set X0[K, N]. Minimises total auto+cross ISL.

    Cyclic steps: z_p = unitary-DFT rows (K-vectors per frequency p); v_p = sqrt(NK/L) z_p/||z_p||;
    X = unimodular projection of unitary-IDFT(v)[:N]. (The scale of v is irrelevant after projection.)"""
    K, N = X0.shape
    L = 2 * N
    X = X0.copy()
    hist = []
    prev = np.inf
    for it in range(n_iter):
        Z = np.fft.fft(X, n=L, axis=-1) / np.sqrt(L)     # [K, L]
        norms = np.linalg.norm(Z, axis=0, keepdims=True) + 1e-12
        V = Z / norms * np.sqrt(N * K / L)               # ||v_p||^2 = NK/L  <=> summed spectrum flat
        G = np.fft.ifft(V, n=L, axis=-1) * np.sqrt(L)    # unitary IDFT
        X = _unimod(G[:, :N])
        if return_hist or tol > 0:
            S = np.abs(np.fft.fft(X, n=L, axis=-1)) ** 2  # unnormalised |a_p^H x|^2
            J = float(((S.sum(axis=0) - N * K) ** 2).sum() / L)  # = total ISL - N^2 K (K-1)
            hist.append(J)
            if abs(prev - J) < tol * max(1.0, J):
                break
            prev = J
    return (X, np.array(hist)) if return_hist else X


def set_isl_bound(K: int, N: int) -> float:
    """Lower bound on total set ISL (auto sidelobes + all ordered cross terms): N^2 K (K-1).

    Derivation: with z_p the K-vector of 2N-point DFT values at frequency p,
    total = (1/2N) sum_p ||z_p z_p^H - N I||_F^2 = (1/2N) sum_p (||z_p||^4 - 2N||z_p||^2 + N^2 K),
    and Parseval fixes sum_p ||z_p||^2 = 2N * NK, so the minimum is at ||z_p||^2 = NK for all p.
    For i.i.d. random unimodular sets the expected total is N^2 K^2 - KN; the gap to the bound is
    exactly the expected auto-sidelobe energy K N (N-1). Cross-correlation energy cannot be reduced
    below ~N^2 per ordered pair on average: only its lag distribution and peak can be shaped."""
    return float(N ** 2 * K * (K - 1))


def wisl_grad_phase(x: np.ndarray, w: np.ndarray):
    """Gradient of J = sum_k w_k |r_k|^2 (lags -(N-1)..N-1, w symmetric, w[N-1]=0) w.r.t. phases phi.
    Returns (J, dJ/dphi)."""
    N = x.shape[-1]
    r = _acorr_fft(x)
    J = float((w * np.abs(r) ** 2).sum())
    g = 2 * np.convolve(w * r, x, mode="full")[N - 1: 2 * N - 1]   # dJ/d conj(x)
    dphi = 2 * np.real(np.conj(g) * 1j * x)
    return J, dphi


def torus_gd(x0: np.ndarray, n_iter: int = 500, lr: float = 0.05, objective: str = "isl",
             p: float = 8.0, irls_every: int = 1, tol: float = 1e-8, return_hist: bool = False):
    """Riemannian GD on the torus with Armijo backtracking.

    objective='isl'  : plain ISL (w_k = 1, k != 0)
    objective='lp'   : IRLS approximation of sum_k |r_k|^p  (w_k = |r_k|^{p-2}, normalised) - PSL surrogate.
    """
    N = x0.shape[-1]
    phi = np.angle(x0).copy()
    w = np.ones(2 * N - 1)
    w[N - 1] = 0.0
    hist = []
    step = lr
    best_x, best_psl = np.exp(1j * phi), np.inf   # best-so-far guard on the *true* PSL (IRLS is not monotone in PSL)
    for it in range(n_iter):
        x = np.exp(1j * phi)
        if objective == "lp":
            cur = float(np.abs(np.delete(_acorr_fft(x), N - 1)).max() / N)
            if cur < best_psl:
                best_psl, best_x = cur, x
        if objective == "lp" and it % irls_every == 0:
            r = np.abs(_acorr_fft(x))
            r[N - 1] = 0.0
            w = r ** (p - 2)
            w[N - 1] = 0.0
            w = w / (w.max() + 1e-12)
        J, g = wisl_grad_phase(x, w)
        hist.append(J)
        gn2 = float((g ** 2).sum())
        if gn2 < tol:
            break
        # backtracking line search on the same (fixed-w) objective
        while True:
            phi_new = phi - step * g
            J_new, _ = wisl_grad_phase(np.exp(1j * phi_new), w)
            if J_new <= J - 1e-4 * step * gn2 or step < 1e-10:
                break
            step *= 0.5
        phi = phi_new
        step = min(step * 1.5, 10.0)
    x = np.exp(1j * phi)
    if objective == "lp":
        cur = float(np.abs(np.delete(_acorr_fft(x), N - 1)).max() / N)
        x = x if cur < best_psl else best_x
    return (x, np.array(hist)) if return_hist else x


def set_wisl_grad_phase(X: np.ndarray, w_auto: np.ndarray, w_cross: np.ndarray):
    """Gradient w.r.t. phases of
        J = sum_m sum_k w_auto[k] |r_mm[k]|^2 + sum_{m != m'} sum_k w_cross[k] |r_mm'[k]|^2
    for a set X[K, N]; weights indexed by lag -(N-1)..N-1 and symmetric. Returns (J, dJ/dphi [K, N])."""
    from .metrics import _xcorr_fft
    K, N = X.shape
    J = 0.0
    G = np.zeros_like(X)
    for m in range(K):
        r = _acorr_fft(X[m])
        J += float((w_auto * np.abs(r) ** 2).sum())
        G[m] += 2 * np.convolve(w_auto * r, X[m], mode="full")[N - 1: 2 * N - 1]
        for mp in range(K):
            if mp == m:
                continue
            c = _xcorr_fft(X[m], X[mp])
            J += float((w_cross * np.abs(c) ** 2).sum())
            G[m] += 2 * np.convolve(w_cross * c, X[mp], mode="full")[N - 1: 2 * N - 1]
    dphi = 2 * np.real(np.conj(G) * 1j * X)
    return J, dphi


def torus_gd_set(X0: np.ndarray, w_auto: np.ndarray, w_cross: np.ndarray, n_iter: int = 300,
                 lr: float = 0.02, tol: float = 1e-8, return_hist: bool = False):
    """Riemannian GD on the K*N torus for a weighted auto/cross ISL set objective (Armijo backtracking)."""
    phi = np.angle(X0).copy()
    hist = []
    step = lr
    for it in range(n_iter):
        X = np.exp(1j * phi)
        J, g = set_wisl_grad_phase(X, w_auto, w_cross)
        hist.append(J)
        gn2 = float((g ** 2).sum())
        if gn2 < tol:
            break
        while True:
            phi_new = phi - step * g
            J_new, _ = set_wisl_grad_phase(np.exp(1j * phi_new), w_auto, w_cross)
            if J_new <= J - 1e-4 * step * gn2 or step < 1e-10:
                break
            step *= 0.5
        phi = phi_new
        step = min(step * 1.5, 10.0)
    X = np.exp(1j * phi)
    return (X, np.array(hist)) if return_hist else X
