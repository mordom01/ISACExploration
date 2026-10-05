"""Constructive policy for binary sequences: p(b_n | b_<n) with the running aperiodic autocorrelation as state.

State after emitting chips b_0..b_{n-1}: the partial autocorrelations r^{(n)}_k = sum_{m=0}^{n-1-k} b_m b_{m+k}
(k = 1..N_MAX-1, zero where undefined), plus the chip just emitted, the position n/N, and N/N_MAX. These are
exactly the quantities exhaustive PSL searches prune on. A GRU consumes the per-step state and outputs the logit
of b_n = +1. Training: teacher forcing on search outputs (imitation). Sampling: sequential, with an O(N)
incremental correlation update per step, batched.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn


def running_acorr(B: torch.Tensor, n_max: int) -> torch.Tensor:
    """B [batch, N] in {+1,-1} (padded with 0 beyond the true length). Returns R [batch, N, n_max-1] where
    R[:, n, k-1] = r^{(n+1)}_k = sum_{m<=n-k} b_m b_{m+k}: the partial autocorrelation after chips 0..n."""
    bs, N = B.shape
    R = torch.zeros(bs, N, n_max - 1, dtype=B.dtype, device=B.device)
    for k in range(1, min(N, n_max)):
        prod = B[:, :N - k] * B[:, k:]                     # term m contributes once chip m+k is emitted
        R[:, k:, k - 1] = torch.cumsum(prod, dim=1)        # available from step n = m+k
    return R


class SeqPolicy(nn.Module):
    def __init__(self, n_max: int = 128, hidden: int = 256, layers: int = 2):
        super().__init__()
        self.n_max = n_max
        din = (n_max - 1) + 1 + 2                            # running acorr, last chip, n/N, N/n_max
        self.inp = nn.Sequential(nn.Linear(din, hidden), nn.SiLU())
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def features(self, B: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Teacher-forcing features: step n uses state after chips 0..n-1 (step 0 uses the empty state)."""
        B = B.float()
        bs, N = B.shape
        R = running_acorr(B, self.n_max)                                      # after chip n
        R_prev = torch.cat([torch.zeros_like(R[:, :1]), R[:, :-1]], dim=1)   # before chip n
        last = torch.cat([torch.zeros_like(B[:, :1]), B[:, :-1]], dim=1)[..., None]
        pos = (torch.arange(N, device=B.device)[None, :, None].float() / lengths[:, None, None].float())
        Ln = (lengths[:, None, None].float() / self.n_max).expand(bs, N, 1)
        scale = 1.0 / torch.sqrt(lengths[:, None, None].float())
        return torch.cat([R_prev * scale, last, pos, Ln], dim=-1)

    def forward(self, B: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(self.inp(self.features(B, lengths)))
        return self.out(h).squeeze(-1)                                        # logits for b_n = +1, [bs, N]

    @torch.no_grad()
    def sample(self, n_samples: int, N: int, temperature: float = 1.0, device: str = "cpu", greedy_frac: float = 0.0):
        bs = n_samples
        B = torch.zeros(bs, N, device=device)
        r = torch.zeros(bs, self.n_max - 1, device=device)
        last = torch.zeros(bs, 1, device=device)
        hstate = None
        scale = 1.0 / math.sqrt(N)
        for n in range(N):
            f = torch.cat([r * scale, last, torch.full((bs, 1), n / N, device=device),
                           torch.full((bs, 1), N / self.n_max, device=device)], dim=-1)
            h, hstate = self.gru(self.inp(f)[:, None, :], hstate)
            logit = self.out(h[:, 0]).squeeze(-1) / temperature
            p = torch.sigmoid(logit)
            if greedy_frac > 0:
                u = torch.rand(bs, device=device)
                b = torch.where(u < greedy_frac, (p > 0.5).float(), (torch.rand(bs, device=device) < p).float()) * 2 - 1
            else:
                b = (torch.rand(bs, device=device) < p).float() * 2 - 1
            B[:, n] = b
            if n > 0:
                # update r_k += b_n * b_{n-k} for k = 1..n
                k = min(n, self.n_max - 1)
                r[:, :k] += b[:, None] * torch.flip(B[:, n - k: n], dims=[1])   # r_k += b_n b_{n-k}
            last = b[:, None]
        return B


def psl_torch_int(B: torch.Tensor) -> torch.Tensor:
    """Integer PSL for B [batch, N] in {+1,-1} via FFT."""
    N = B.shape[1]
    X = torch.fft.rfft(B, n=2 * N, dim=-1)
    r = torch.fft.irfft(X.abs() ** 2, n=2 * N, dim=-1)[:, 1:N]
    return torch.round(r.abs().max(dim=1).values).long()
