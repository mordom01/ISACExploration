"""Batched torch implementations of the classical optimisers (CPU now, GPU later).

* can_batched        : CAN for a batch of single codes, phases [B, N].
* set_objective      : weighted auto/cross ISL of a batch of sets, phases [B, K, N]; differentiable.
* torus_gd_batched   : Riemannian GD (Adam on phases) for the set objective, batched over B codebooks.
These generate training data for the set-level flow model and serve as the random-restart baseline at matched
wall-clock. Numerics are checked against the numpy versions in isacgen/classical.py (see tests in experiments).
"""
from __future__ import annotations
import math
import torch


def _unimod(z: torch.Tensor) -> torch.Tensor:
    return torch.polar(torch.ones_like(z.real), torch.angle(z))


def can_batched(phi0: torch.Tensor, n_iter: int = 500) -> torch.Tensor:
    """CAN on a batch. phi0 [B, N] phases -> phases [B, N]."""
    N = phi0.shape[-1]
    L = 2 * N
    x = torch.polar(torch.ones_like(phi0), phi0)
    for _ in range(n_iter):
        z = torch.fft.fft(x, n=L, dim=-1)
        v = _unimod(z)
        g = torch.fft.ifft(v, n=L, dim=-1)[..., :N]
        x = _unimod(g)
    return torch.angle(x)


def xcorr_all(x: torch.Tensor) -> torch.Tensor:
    """All pairwise aperiodic correlations of a batch of sets. x [B, K, N] complex -> c [B, K, K, 2N-1]
    with c[b, m, m', :] = sum_n x_m[n] conj(x_m'[n-k]) for lags -(N-1)..N-1."""
    B, K, N = x.shape
    L = 2 * N
    X = torch.fft.fft(x, n=L, dim=-1)                       # [B, K, L]
    C = X[:, :, None, :] * torch.conj(X[:, None, :, :])     # [B, K, K, L]
    c = torch.fft.ifft(C, dim=-1)
    return torch.cat([c[..., L - (N - 1):], c[..., :N]], dim=-1)


def set_objective(phi: torch.Tensor, w_auto: torch.Tensor, w_cross: torch.Tensor) -> torch.Tensor:
    """J[b] = sum_m sum_k w_auto[k] |r_mm[k]|^2 + sum_{m != m'} sum_k w_cross[k] |r_mm'[k]|^2  (phi [B, K, N])."""
    x = torch.polar(torch.ones_like(phi), phi)
    c = xcorr_all(x).abs() ** 2                              # [B, K, K, 2N-1]
    K = phi.shape[1]
    eye = torch.eye(K, device=phi.device, dtype=torch.bool)
    auto = (c[:, eye] * w_auto).sum(dim=(-1, -2))
    cross = (c[:, ~eye] * w_cross).sum(dim=(-1, -2))
    return auto + cross


def window_weights(N: int, W: int, cross_weight: float = 50.0, zero_lag_target: float = 0.0, device="cpu"):
    """Auto ISL weight (all lags but 0) and windowed cross weight (|lag| <= W). zero_lag_target is reserved for
    the simplex variant (handled in set_objective_simplex)."""
    wa = torch.ones(2 * N - 1, device=device); wa[N - 1] = 0.0
    wc = torch.zeros(2 * N - 1, device=device); wc[N - 1 - W: N + W] = cross_weight
    return wa, wc


def set_objective_simplex(phi: torch.Tensor, w_auto: torch.Tensor, w_cross: torch.Tensor, rho0: float) -> torch.Tensor:
    """Variant that targets zero-lag correlation Re<x_m, x_m'> = rho0 * N (simplex: rho0 = -1/(K-1)) instead of
    0, while still penalising |c|^2 at the other lags in the window. Phases [B, K, N]."""
    B, K, N = phi.shape
    x = torch.polar(torch.ones_like(phi), phi)
    c = xcorr_all(x)
    eye = torch.eye(K, device=phi.device, dtype=torch.bool)
    auto = ((c[:, eye].abs() ** 2) * w_auto).sum(dim=(-1, -2))
    wc = w_cross.clone(); wc[N - 1] = 0.0
    cross = ((c[:, ~eye].abs() ** 2) * wc).sum(dim=(-1, -2))
    zl = c[:, ~eye][..., N - 1]
    cross = cross + w_cross[N - 1] * ((zl.real - rho0 * N) ** 2 + zl.imag ** 2).sum(-1)
    return auto + cross


def torus_gd_batched(phi0: torch.Tensor, objective, n_iter: int = 300, lr: float = 0.05) -> torch.Tensor:
    """Adam on phases for a batched objective J(phi) -> [B]. Returns best-iterate phases [B, K, N]."""
    phi = phi0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([phi], lr=lr)
    best = phi0.clone(); best_J = torch.full((phi0.shape[0],), float("inf"), device=phi0.device)
    for it in range(n_iter):
        opt.zero_grad()
        J = objective(phi)
        J.sum().backward()
        with torch.no_grad():
            better = J < best_J
            best[better] = phi[better]
            best_J[better] = J[better]
        opt.step()
    with torch.no_grad():
        J = objective(phi)
        better = J < best_J
        best[better] = phi[better]
    return torch.remainder(best + math.pi, 2 * math.pi) - math.pi
