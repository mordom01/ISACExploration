"""Flow matching on the flat N-torus for unimodular phase codes.

State: phases phi in [-pi, pi)^N (global phase canonicalised: phi[0] = 0 in the data; the model is free to
move it, we re-canonicalise after sampling). Source p0 = uniform (Haar) on the torus. Target p1 = optimiser
outputs. Conditional path: wrapped geodesic phi_t = wrap(phi_0 + t * wrap(phi_1 - phi_0)); target velocity
u_t = wrap(phi_1 - phi_0), constant along the path. This is Riemannian flow matching (Chen & Lipman, ICLR
2024) specialised to the flat torus, where the logarithmic map is the wrapped difference; it is also what
TorchCFM's OT-CFM reduces to with angle wrapping. Sampling: Euler on the torus with wrapping at each step.

Model: a small 1D transformer over the N chips. Input features per chip: (cos phi, sin phi), a sinusoidal
positional embedding, and a time embedding added to every token. Output: one real velocity per chip.
Conditioning hooks (tokens via cross-attention) are left for Phase 1 proper; the smoke test is unconditional.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn


def wrap(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x + math.pi, 2 * math.pi) - math.pi


class TimeEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(1e4) * torch.arange(half, device=t.device) / half)
        a = t[:, None] * freqs[None, :] * 1000.0
        return self.mlp(torch.cat([a.sin(), a.cos()], dim=-1))


class TorusVelocityNet(nn.Module):
    def __init__(self, N: int, d_model: int = 128, n_layers: int = 4, n_heads: int = 4, cond_dim: int = 0):
        super().__init__()
        self.N = N
        self.inp = nn.Linear(2, d_model)
        self.pos = nn.Parameter(torch.randn(1, N, d_model) * 0.02)
        self.temb = TimeEmbed(d_model)
        self.cond = nn.Linear(cond_dim, d_model) if cond_dim > 0 else None
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model, dropout=0.0,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, n_layers)
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, phi: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        h = self.inp(torch.stack([phi.cos(), phi.sin()], dim=-1)) + self.pos
        g = self.temb(t)
        if self.cond is not None and cond is not None:
            g = g + self.cond(cond)
        h = h + g[:, None, :]
        h = self.blocks(h)
        return self.out(h).squeeze(-1)


def cfm_loss(model: nn.Module, phi1: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
    """Conditional flow-matching loss on the flat torus with uniform source."""
    B = phi1.shape[0]
    phi0 = (torch.rand_like(phi1) * 2 * math.pi) - math.pi
    t = torch.rand(B, device=phi1.device)
    u = wrap(phi1 - phi0)                       # log map on the flat torus
    phit = wrap(phi0 + t[:, None] * u)
    v = model(phit, t, cond)
    return ((v - u) ** 2).mean()


@torch.no_grad()
def sample(model: nn.Module, n: int, N: int, nfe: int = 8, cond: torch.Tensor | None = None,
           device: str = "cpu", guidance=None, guidance_scale: float = 0.0) -> torch.Tensor:
    """Euler sampling on the torus. `guidance(phi_hat1) -> scalar energy` (lower is better) is optional:
    at each step we form the x1-prediction phi_hat1 = wrap(phi_t + (1-t) v) and subtract scale * grad."""
    phi = (torch.rand(n, N, device=device) * 2 * math.pi) - math.pi
    ts = torch.linspace(0, 1, nfe + 1, device=device)
    for i in range(nfe):
        t = ts[i].expand(n)
        dt = (ts[i + 1] - ts[i]).item()
        v = model(phi, t, cond)
        if guidance is not None and guidance_scale > 0:
            with torch.enable_grad():
                p = phi.detach().requires_grad_(True)
                phat = wrap(p + (1 - t[:, None]) * v.detach())
                e = guidance(phat).sum()
                g = torch.autograd.grad(e, p)[0]
            v = v - guidance_scale * g
        phi = wrap(phi + dt * v)
    return phi


def canonicalise(phi: torch.Tensor) -> torch.Tensor:
    return wrap(phi - phi[:, :1])


def psl_db_torch(phi: torch.Tensor) -> torch.Tensor:
    """Differentiable zero-Doppler PSL (dB) for phases [B, N] via 2N-point FFT."""
    N = phi.shape[-1]
    x = torch.polar(torch.ones_like(phi), phi)
    X = torch.fft.fft(x, n=2 * N, dim=-1)
    r = torch.fft.ifft(X.abs() ** 2, dim=-1)[..., :N]  # lags 0..N-1 (one-sided suffices, symmetric)
    side = r[..., 1:].abs()
    return 20 * torch.log10(side.max(dim=-1).values / N)


def isl_torch(phi: torch.Tensor) -> torch.Tensor:
    """Differentiable one-sided ISL (normalised by N^2) for phases [B, N]."""
    N = phi.shape[-1]
    x = torch.polar(torch.ones_like(phi), phi)
    X = torch.fft.fft(x, n=2 * N, dim=-1)
    r = torch.fft.ifft(X.abs() ** 2, dim=-1)[..., 1:N]
    return (r.abs() ** 2).sum(-1) / N ** 2
