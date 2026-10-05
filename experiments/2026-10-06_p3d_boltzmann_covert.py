"""P3d - Boltzmann generator for covert ISAC: exact MCMC reference and an amortised autoregressive sampler.
Date: 2026-10-06
Theory (P3b): with background P uniform over GMSK symbol patterns s in {+-1}^NB and sensing cost c(s) (mainlobe-
excluded ISL of the modulated frame), the covert-optimal emission law under a KL budget is the Gibbs law
Q_beta(s) ∝ exp(-beta c(s)).  Importance reweighting of data cannot reach beta >~ 20 (ESS collapse, P3b); a GAN
cannot match the near-uniform target (P3a/P3c: detector AUC 1.0).  Two samplers that can:
 (1) Metropolis MCMC over single-symbol flips with energy beta*c  (exact, reference, ~1000 flips per frame).
 (2) Autoregressive Boltzmann generator (variational autoregressive network, Wu-Wang-Zhang PRL 2019 style):
     q_theta(s) = prod_i q(s_i | s_<i) (GRU), trained with the reverse KL  E_q[log q(s) + beta c(s)]  by the score-
     function estimator with a moving baseline.  No training data, one forward pass per frame at inference.
Free energy / KL: log Z_beta by thermodynamic integration over the MCMC grid, log Z_beta = NB log 2 - int_0^beta E_b[c] db,
so KL(Q_beta || P) = NB log 2 - beta E_beta[c] - log Z_beta.  For the learned sampler, KL(q || Q_beta) is estimated
as E_q[log q + beta c] + log Z_beta.
Reported per beta: E[c], E[PSL], KL to background, held-out detector AUC (MCMC and learned), sampling time.
Comm: CSK with seed-shared implicit codebooks (K=8 messages = 8 generator draws from a shared seed), SER vs Eb/N0
with the lag-searching correlation receiver, for the learned sampler.  Usage: <train_steps>
Outputs: results/p3/p3d_boltzmann.csv
"""
import sys, pathlib, math, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.special import erf
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "p3"; OUT.mkdir(parents=True, exist_ok=True)
train_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
N, SPS, BT, KMIN, W, K = 64, 2, 0.3, 3, 2, 8
NB = N // SPS + 8

def gmsk_pulse(bt=BT, sps=SPS, L=4):
    t = (np.arange(-L * sps, L * sps + 1) + 0.5) / sps; s = np.sqrt(np.log(2)) / (2 * np.pi * bt)
    q = 0.5 * (erf((t + 0.5) / (s * np.sqrt(2))) - erf((t - 0.5) / (s * np.sqrt(2)))); return q / q.sum()
G_PULSE_T = torch.tensor(gmsk_pulse(), dtype=torch.float32)[None, None, :]
def modulate(sym, rand_phase=True):
    B = sym.shape[0]; up = torch.zeros(B, NB * SPS); up[:, ::SPS] = sym
    freq = F.conv1d(up[:, None, :], G_PULSE_T, padding=G_PULSE_T.shape[-1] // 2)[:, 0, :NB * SPS]
    phase = torch.cumsum(math.pi / 2 * freq, dim=1)[:, 4 * SPS: 4 * SPS + N]
    if rand_phase: phase = phase + torch.rand(B, 1) * 2 * math.pi
    return torch.remainder(phase + math.pi, 2 * math.pi) - math.pi
def acorr(phi):
    x = torch.polar(torch.ones_like(phi), phi); X = torch.fft.fft(x, n=2 * N, dim=-1); return torch.fft.ifft(X.abs() ** 2, dim=-1)[..., :N]
def cost_sym(sym):
    r = acorr(modulate(sym, rand_phase=False)); return (r[..., KMIN:].abs() ** 2).sum(-1) / N ** 2
def psl_db(phi):
    r = acorr(phi); return 20 * torch.log10(r[..., KMIN:].abs().max(-1).values / r[..., 0].abs())
def feats(phi):
    d = torch.remainder(phi[:, 1:] - phi[:, :-1] + math.pi, 2 * math.pi) - math.pi; d = F.pad(d, (1, 0))
    return torch.stack([phi.cos(), phi.sin(), d / math.pi], 1)
class Det(nn.Module):
    def __init__(s, ch=64):
        super().__init__(); s.net = nn.Sequential(nn.Conv1d(3, ch, 5, padding=2), nn.LeakyReLU(0.2), nn.Conv1d(ch, ch, 5, padding=2, stride=2), nn.LeakyReLU(0.2), nn.Conv1d(ch, ch, 5, padding=2, stride=2), nn.LeakyReLU(0.2), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(ch, 1))
    def forward(s, phi): return s.net(feats(phi)).squeeze(-1)
def auc_vs_background(Xq, steps=500):
    """Fresh detector: uniform-GMSK frames vs Xq (6000 train / 2000 test each)."""
    D = Det(); o = torch.optim.Adam(D.parameters(), 1e-3)
    Xr = modulate(torch.randint(0, 2, (8000, NB)).float() * 2 - 1); Xq = Xq.detach()
    for _ in range(steps):
        i = torch.randint(0, 6000, (128,)); j = torch.randint(0, len(Xq) - 2000, (128,))
        l = F.binary_cross_entropy_with_logits(D(Xr[i]), torch.ones(128)) + F.binary_cross_entropy_with_logits(D(Xq[j]), torch.zeros(128))
        o.zero_grad(); l.backward(); o.step()
    with torch.no_grad(): sr, sq = D(Xr[6000:]), D(Xq[-2000:])
    s = torch.cat([sr, sq]); lab = torch.cat([torch.ones(2000), torch.zeros(2000)])
    order = s.argsort(); ranks = torch.empty_like(order, dtype=torch.float); ranks[order] = torch.arange(1, 4001).float()
    a = (ranks[lab == 1].sum() - 2000 * 2001 / 2) / (2000 * 2000); return float(max(a, 1 - a))

# ---------------- (1) Metropolis MCMC over symbols ----------------
@torch.no_grad()
def mcmc(beta, n_chains=8000, sweeps=25):
    s = torch.randint(0, 2, (n_chains, NB)).float() * 2 - 1; c = cost_sym(s); E = beta * c
    for _ in range(sweeps * NB):
        j = torch.randint(0, NB, (n_chains,)); s2 = s.clone(); s2[torch.arange(n_chains), j] *= -1
        c2 = cost_sym(s2); E2 = beta * c2
        acc = torch.rand(n_chains) < torch.exp(-(E2 - E)).clamp(max=1.0)
        s[acc] = s2[acc]; c[acc] = c2[acc]; E[acc] = E2[acc]
    return s, c
betas = [0, 5, 10, 20, 40, 80, 160]
rows = []; Ec = {}; mc_samples = {}
t0 = time.time()
for b in betas:
    s, c = mcmc(b); Ec[b] = float(c.mean()); mc_samples[b] = s
    print(f"MCMC beta={b:4d}: E[c] {Ec[b]:.4f}  E[PSL] {psl_db(modulate(s)).mean():.1f} dB  ({time.time()-t0:.0f}s)", flush=True)
# thermodynamic integration (trapezoid on the beta grid)
logZ = {0: NB * math.log(2)}
for i in range(1, len(betas)):
    b0, b1 = betas[i - 1], betas[i]; logZ[b1] = logZ[b0] - 0.5 * (Ec[b0] + Ec[b1]) * (b1 - b0)
for b in betas:
    kl = NB * math.log(2) - b * Ec[b] - logZ[b]
    s = mc_samples[b]; phi = modulate(s); a = auc_vs_background(phi) if b > 0 else 0.5
    print(f"MCMC beta={b:4d}: KL(Q||P) {kl:.2f} nats  E[PSL] {psl_db(phi).mean():.1f} dB  worst-of-8 PSL (median over draws) {psl_db(phi)[:8000].view(-1, 8).max(1).values.median():.1f} dB  detector AUC {a:.3f}", flush=True)
    rows.append(("MCMC", b, kl, Ec[b], float(psl_db(phi).mean()), a, float("nan"), float("nan")))

# ---------------- (2) autoregressive Boltzmann generator ----------------
class VAN(nn.Module):
    def __init__(s, h=128):
        super().__init__(); s.gru = nn.GRU(1, h, batch_first=True); s.out = nn.Linear(h, 1); s.h0 = nn.Parameter(torch.zeros(1, 1, h))
    def sample(s, n, gen=None):
        x = torch.zeros(n, 1, 1); hs = s.h0.expand(1, n, -1).contiguous(); sym = []; logq = torch.zeros(n)
        for i in range(NB):
            o, hs = s.gru(x, hs); p = torch.sigmoid(s.out(o[:, 0]).squeeze(-1))
            u = torch.rand(n, generator=gen) if gen is not None else torch.rand(n)
            b = (u < p).float(); logq = logq + torch.log(torch.where(b > 0, p, 1 - p) + 1e-9)
            x = (b * 2 - 1)[:, None, None]; sym.append(b * 2 - 1)
        return torch.stack(sym, 1), logq
    def logprob(s, sym):
        x = torch.cat([torch.zeros(sym.shape[0], 1, 1), sym[:, :-1, None]], 1); o, _ = s.gru(x, s.h0.expand(1, sym.shape[0], -1).contiguous())
        p = torch.sigmoid(s.out(o).squeeze(-1)); return torch.log(torch.where(sym > 0, p, 1 - p) + 1e-9).sum(1)

def ser_csk(sample_fn, ebn0_db, n=4000, seed=123):
    """CSK with K seed-shared draws as the codebook (transmitter and receiver regenerate it from the seed)."""
    gen = torch.Generator().manual_seed(seed); sym, _ = sample_fn(K, gen); cb = modulate(sym, rand_phase=False)
    m = torch.randint(0, K, (n,)); x = torch.polar(torch.ones(n, N), cb[m])
    off = torch.randint(-W, W + 1, (n,)); xp = F.pad(x, (W, W)); idx = torch.arange(N)[None, :] + W + off[:, None]; y = torch.gather(xp, 1, idx)
    N0 = N / (math.log2(K) * 10 ** (ebn0_db / 10)); y = y + math.sqrt(N0 / 2) * (torch.randn_like(y.real) + 1j * torch.randn_like(y.real))
    C = torch.polar(torch.ones(K, N), cb); best = torch.full((n, K), -1.0); yp = F.pad(y, (W, W))
    for k in range(-W, W + 1): best = torch.maximum(best, (yp[:, W + k: W + k + N] @ C.conj().T).abs())
    return float((best.argmax(1) != m).float().mean())

for b in [10, 40, 160]:
    van = VAN(); opt = torch.optim.Adam(van.parameters(), 1e-3); base = None; t1 = time.time()
    for it in range(train_steps):
        sym, logq = van.sample(512); c = cost_sym(sym); R = (logq + b * c).detach()      # reverse-KL objective per sample
        base = R.mean() if base is None else 0.9 * base + 0.1 * R.mean()
        loss = ((R - base) * logq).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(van.parameters(), 1.0); opt.step()
        if it % 500 == 0 or it == train_steps - 1:
            print(f"  VAN beta={b} it {it}: E_q[c] {c.mean():.4f}  KL(q||Q_beta) {float((logq + b * c).mean()) + logZ[b]:.2f} nats ({time.time()-t1:.0f}s)", flush=True)
    with torch.no_grad():
        t2 = time.time(); sym, logq = van.sample(8000); dt = (time.time() - t2) / 8000
        c = cost_sym(sym); phi = modulate(sym); kl_q_P = float(logq.mean()) + NB * math.log(2); kl_q_Qb = float((logq + b * c).mean()) + logZ[b]
    a = auc_vs_background(phi)
    ser = {e: ser_csk(van.sample, e) for e in [0, 4, 8, 12]}
    print(f"VAN  beta={b:4d}: KL(q||P) {kl_q_P:.2f} nats, KL(q||Q_beta) {kl_q_Qb:.2f}, E[c] {c.mean():.4f}, E[PSL] {psl_db(phi).mean():.1f} dB, detector AUC {a:.3f}, {dt*1e3:.3f} ms/frame; CSK SER @0/4/8/12 dB {ser[0]:.3f}/{ser[4]:.3f}/{ser[8]:.3f}/{ser[12]:.3f}", flush=True)
    rows.append(("VAN", b, kl_q_P, float(c.mean()), float(psl_db(phi).mean()), a, kl_q_Qb, dt * 1e3))
    torch.save(van.state_dict(), OUT / f"p3d_van_beta{b}.pt")
with open(OUT / "p3d_boltzmann.csv", "w") as f:
    f.write("sampler,beta,kl_to_background_nats,E_cost,E_psl_db,detector_auc,kl_to_gibbs_nats,ms_per_frame\n")
    for r in rows: f.write(",".join(str(v) for v in r) + "\n")
print("saved")
