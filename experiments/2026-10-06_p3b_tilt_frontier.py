"""P3b - The covertness/sensing frontier: exponential tilt vs truncation of the background law.
Date: 2026-10-06
Theory. Let P be the background traffic law over frames (here GMSK, N=64) and c(x) a sensing cost (mainlobe-
excluded ISL). Among laws Q with KL(Q||P) <= eps, the one minimising E_Q[c] is the exponential tilt
Q_beta(x) ∝ P(x) exp(-beta c(x)) (Gibbs variational principle), with KL(Q_beta||P) = -beta E_Q[c] - log E_P[e^{-beta c}].
"Search inside the family" (keep frames with c <= t) is the truncation Q_t = P(. | c <= t), with
KL(Q_t||P) = -log P(c <= t). For equal KL the tilt has lower expected cost; moreover the truncation has a hard
edge in c that a detector can exploit. A generator trained against a detector is a way to realise Q_beta when
P is known only through samples. This script computes both frontiers by importance weighting over 100k GMSK
frames, and measures operational covertness (held-out learned-detector AUC) for both at matched KL.
Output: results/p3/p3b_frontier.csv, results/p3/p3b_frontier.png
"""
import sys, pathlib, math, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from isacgen import plotstyle; plotstyle.apply()
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(2)
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "p3"; OUT.mkdir(parents=True, exist_ok=True)
N, SPS, BT, KMIN = 64, 2, 0.3, 3

from scipy.special import erf
def gmsk_pulse(bt=BT, sps=SPS, L=4):
    t = (np.arange(-L * sps, L * sps + 1) + 0.5) / sps; s = np.sqrt(np.log(2)) / (2 * np.pi * bt)
    q = 0.5 * (erf((t + 0.5) / (s * np.sqrt(2))) - erf((t - 0.5) / (s * np.sqrt(2)))); return q / q.sum()
G_PULSE = gmsk_pulse()
def gmsk_frames(n, rng):
    nb = N // SPS + 8; bits = rng.choice([-1.0, 1.0], size=(n, nb)); up = np.zeros((n, nb * SPS)); up[:, ::SPS] = bits
    freq = np.apply_along_axis(lambda r: np.convolve(r, G_PULSE, mode="same"), 1, up)
    phase = np.cumsum(np.pi / 2 * freq, axis=1)[:, 4 * SPS: 4 * SPS + N] + rng.uniform(0, 2 * np.pi, size=(n, 1))
    return torch.tensor(np.angle(np.exp(1j * phase)), dtype=torch.float32)
def acorr(phi):
    x = torch.polar(torch.ones_like(phi), phi); X = torch.fft.fft(x, n=2 * N, dim=-1); return torch.fft.ifft(X.abs() ** 2, dim=-1)[..., :N]
def cost(phi):
    r = acorr(phi); return (r[..., KMIN:].abs() ** 2).sum(-1) / N ** 2
def psl_db(phi):
    r = acorr(phi); return 20 * torch.log10(r[..., KMIN:].abs().max(-1).values / r[..., 0].abs())
def feats(phi):
    d = torch.remainder(phi[:, 1:] - phi[:, :-1] + math.pi, 2 * math.pi) - math.pi; d = F.pad(d, (1, 0))
    return torch.stack([phi.cos(), phi.sin(), d / math.pi], 1)
class Det(nn.Module):
    def __init__(s, ch=64):
        super().__init__(); s.net = nn.Sequential(nn.Conv1d(3, ch, 5, padding=2), nn.LeakyReLU(0.2), nn.Conv1d(ch, ch, 5, padding=2, stride=2), nn.LeakyReLU(0.2), nn.Conv1d(ch, ch, 5, padding=2, stride=2), nn.LeakyReLU(0.2), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(ch, 1))
    def forward(s, phi): return s.net(feats(phi)).squeeze(-1)
def auc(Xr, Xq, Xr_test, Xq_test, steps=500):
    D = Det(); o = torch.optim.Adam(D.parameters(), 1e-3); n = min(len(Xr), len(Xq))
    for _ in range(steps):
        i = torch.randint(0, n, (128,)); j = torch.randint(0, len(Xq), (128,))
        l = F.binary_cross_entropy_with_logits(D(Xr[i]), torch.ones(128)) + F.binary_cross_entropy_with_logits(D(Xq[j]), torch.zeros(128))
        o.zero_grad(); l.backward(); o.step()
    with torch.no_grad(): sr, sq = D(Xr_test), D(Xq_test)
    s = torch.cat([sr, sq]); lab = torch.cat([torch.ones(len(sr)), torch.zeros(len(sq))])
    order = s.argsort(); ranks = torch.empty_like(order, dtype=torch.float); ranks[order] = torch.arange(1, len(s) + 1).float()
    a = (ranks[lab == 1].sum() - len(sr) * (len(sr) + 1) / 2) / (len(sr) * len(sq)); return float(max(a, 1 - a))

rng = np.random.default_rng(0)
t0 = time.time(); M = 100_000
P = gmsk_frames(M, rng); c = cost(P); p = psl_db(P)
print(f"background: {M} GMSK frames; cost mean {c.mean():.4f}, PSL(excl) median {p.median():.1f} dB, min {p.min():.1f} dB ({time.time()-t0:.0f}s)", flush=True)
Pr_test = gmsk_frames(2000, rng); Pr_train = gmsk_frames(6000, rng)
rows = []
def resample(w, n):
    idx = torch.multinomial(w / w.sum(), n, replacement=True); return idx
# exponential tilts
for beta in [0, 5, 10, 20, 40, 80, 160, 320]:
    w = torch.exp(-beta * (c - c.min()))
    q = w / w.sum(); kl = float((q * torch.log(q * M + 1e-30)).sum())          # KL(Q||P) with P uniform over the M samples
    Ec = float((q * c).sum()); Ep = float((q * p).sum()); ess = float(1 / (q ** 2).sum())
    i_tr, i_te = resample(w, 6000), resample(w, 2000)
    a = auc(Pr_train, P[i_tr], Pr_test, P[i_te])
    print(f"tilt beta={beta:4d}: KL {kl:.3f} nats, E[cost] {Ec:.4f}, E[PSL] {Ep:.1f} dB, ESS {ess:.0f}, detector AUC {a:.3f}", flush=True)
    rows.append(("tilt", beta, kl, Ec, Ep, ess, a))
# truncations at matched quantiles
for qf in [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]:
    thr = torch.quantile(c, qf) if qf < 1 else c.max() + 1
    w = (c <= thr).float(); kl = -math.log(float(w.mean())) if qf < 1 else 0.0
    q = w / w.sum(); Ec = float((q * c).sum()); Ep = float((q * p).sum())
    i_tr, i_te = resample(w, 6000), resample(w, 2000)
    a = auc(Pr_train, P[i_tr], Pr_test, P[i_te])
    print(f"trunc q={qf:5.3f}: KL {kl:.3f} nats, E[cost] {Ec:.4f}, E[PSL] {Ep:.1f} dB, kept {int(w.sum())}, detector AUC {a:.3f}", flush=True)
    rows.append(("truncation", qf, kl, Ec, Ep, float(w.sum()), a))
with open(OUT / "p3b_frontier.csv", "w") as f:
    f.write("family,param,kl_nats,E_cost,E_psl_db,ess_or_kept,detector_auc\n")
    for r in rows: f.write(",".join(str(v) for v in r) + "\n")
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
for fam, j in [("tilt", 0), ("truncation", 1)]:
    rr = [r for r in rows if r[0] == fam]
    ax[0].plot([r[2] for r in rr], [r[4] for r in rr], marker="o", label=fam, color=plotstyle.SERIES[j])
    ax[1].plot([r[2] for r in rr], [r[6] for r in rr], marker="o", label=fam, color=plotstyle.SERIES[j])
ax[0].set_xlabel("KL(Q || background) (nats)"); ax[0].set_ylabel("expected PSL, mainlobe excluded (dB)"); ax[0].legend(); ax[0].set_title("Sensing cost vs divergence budget")
ax[1].set_xlabel("KL(Q || background) (nats)"); ax[1].set_ylabel("held-out detector AUC"); ax[1].legend(); ax[1].set_title("Operational covertness vs divergence")
fig.tight_layout(); fig.savefig(OUT / "p3b_frontier.png"); print("saved")
