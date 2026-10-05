"""P3c - Covert ISAC with a *structured* generator: symbols through the background modulator (GMSK) itself.
Date: 2026-10-06
Change from P3a: P3a's free-phase generator was separable from GMSK by a held-out detector (AUC 1.0 even with no
sensing penalty). Here G emits 40 binary symbols (straight-through sign of logits) that pass through the same
differentiable GMSK modulator as the background, so every output is a GMSK frame by construction and covertness
reduces to the statistics of the symbol law: G is a learned sampler of the tilted symbol distribution Q_beta
(see P3b). Everything else (D, R, channel, metrics, baselines) is unchanged. Original P3a header follows.
Rationale (why generative here and nowhere else in this project): every earlier objective (PSL, ISL, windowed
cross-correlation, SER) is a per-waveform functional, and per-instance optimisation won every matched comparison.
Covertness is a *distributional* objective: the emitted waveform's statistics must match a background traffic
distribution so that a detector cannot tell them apart. That is the one objective an optimiser cannot express
and a generator trained against a detector addresses directly.

Setup (N = 64 samples, constant envelope throughout):
  background  : GMSK frames (2 samples/symbol, 32 random bits, BT = 0.3), the legacy constant-envelope traffic.
  generator G : (message m in {0..K-1}, noise z) -> phases phi in R^N (unit modulus by construction).
  detector D  : adversary, 1-D conv on (cos, sin, phase increment), GMSK vs. generated.
  receiver R  : cooperative classifier of m from the received waveform under AWGN and a random timing offset
                of up to W = 2 samples (the autoencoder-communication formulation).
  sensing     : differentiable normalised ISL of G's output (radar range sidelobes).
Losses: G minimises  adv(D) + lam_comm * CE(R) + lam_af * ISL;  D and R trained on their own objectives.
All sidelobe metrics exclude the mainlobe (|lag| < 3 samples = 1.5 symbols) because the background is narrowband.
Evaluation (held-out): a *fresh* detector trained post hoc on 4000 GMSK vs 4000 generated frames and tested on
held-out frames (AUC); PSL/ISL of generated waveforms; SER vs Eb/N0 with timing uncertainty.
Baselines: (B1) random GMSK frames used as a CSK codebook (one random frame per message), AUC by construction
~0.5, poor PSL; (B2) PSL-searched GMSK bit patterns: best of M random GMSK frames per message (M = 2000), the
classical "search inside the covert family" approach, which is covert against a naive detector but biased;
(B3) CAN-optimised unimodular codes (not covert). Usage: <steps> <lam_af> [K]
Outputs: results/p3/p3a_<lam_af>.{log,csv,pt}
"""
import sys, pathlib, time, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from isacgen.torusfm import isl_torch, psl_db_torch
KMIN = 3   # mainlobe exclusion: GMSK at 2 samples/symbol is narrowband, lags 1..2 are mainlobe; sidelobes are |k| >= 3

def _acorr(phi):
    N_ = phi.shape[-1]; x = torch.polar(torch.ones_like(phi), phi); X = torch.fft.fft(x, n=2 * N_, dim=-1)
    return torch.fft.ifft(X.abs() ** 2, dim=-1)[..., :N_]

def psl_db_excl(phi):
    r = _acorr(phi); return 20 * torch.log10(r[..., KMIN:].abs().max(dim=-1).values / r[..., 0].abs())

def isl_excl(phi):
    r = _acorr(phi); N_ = phi.shape[-1]; return (r[..., KMIN:].abs() ** 2).sum(-1) / N_ ** 2
from isacgen.classical import can
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "p3"; OUT.mkdir(parents=True, exist_ok=True)
steps = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
lam_af = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
K = int(sys.argv[3]) if len(sys.argv) > 3 else 8
N, SPS, BT, W = 64, 2, 0.3, 2
lam_comm = 1.0
tag = f"struct_laf{lam_af:g}_K{K}"

# ---------------- GMSK background ----------------
def gmsk_pulse(bt=BT, sps=SPS, L=4):
    t = (np.arange(-L * sps, L * sps + 1) + 0.5) / sps
    s = np.sqrt(np.log(2)) / (2 * np.pi * bt)
    from scipy.special import erf
    q = 0.5 * (erf((t + 0.5) / (s * np.sqrt(2))) - erf((t - 0.5) / (s * np.sqrt(2))))   # Gaussian-filtered rect
    return q / q.sum()
G_PULSE = gmsk_pulse()

def gmsk_frames(n, rng):
    """n GMSK frames of N samples (phases). Random bits, random initial phase, pulse tails from preceding bits."""
    nb = N // SPS + 8
    bits = rng.choice([-1.0, 1.0], size=(n, nb))
    up = np.zeros((n, nb * SPS)); up[:, ::SPS] = bits
    freq = np.apply_along_axis(lambda r: np.convolve(r, G_PULSE, mode="same"), 1, up)
    phase = np.cumsum(np.pi / 2 * freq / SPS * SPS, axis=1) * (1.0 / 1.0)
    phase = phase[:, 4 * SPS: 4 * SPS + N]
    phase = phase + rng.uniform(0, 2 * np.pi, size=(n, 1))
    return torch.tensor(np.angle(np.exp(1j * phase)), dtype=torch.float32)

def feats(phi):
    d = torch.remainder(phi[:, 1:] - phi[:, :-1] + math.pi, 2 * math.pi) - math.pi
    d = F.pad(d, (1, 0))
    return torch.stack([phi.cos(), phi.sin(), d / math.pi], dim=1)          # [B, 3, N]

class Conv1dNet(nn.Module):
    def __init__(self, cout, ch=64):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(3, ch, 5, padding=2), nn.LeakyReLU(0.2), nn.Conv1d(ch, ch, 5, padding=2, stride=2), nn.LeakyReLU(0.2),
                                 nn.Conv1d(ch, ch, 5, padding=2, stride=2), nn.LeakyReLU(0.2), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(ch, cout))
    def forward(self, phi): return self.net(feats(phi))

NB = N // SPS + 8
G_PULSE_T = torch.tensor(G_PULSE, dtype=torch.float32)[None, None, :]
def gmsk_modulate(sym):
    """Differentiable GMSK modulator identical to gmsk_frames: sym [B, NB] in {-1,+1} -> phases [B, N]."""
    B = sym.shape[0]
    up = torch.zeros(B, NB * SPS); up[:, ::SPS] = sym
    freq = F.conv1d(up[:, None, :], G_PULSE_T, padding=G_PULSE_T.shape[-1] // 2)[:, 0, :NB * SPS]
    phase = torch.cumsum(math.pi / 2 * freq, dim=1)[:, 4 * SPS: 4 * SPS + N]
    phase = phase + torch.rand(B, 1) * 2 * math.pi
    return torch.remainder(phase + math.pi, 2 * math.pi) - math.pi

class Generator(nn.Module):
    def __init__(self, K, zdim=16, h=256):
        super().__init__()
        self.emb = nn.Embedding(K, 32)
        self.net = nn.Sequential(nn.Linear(zdim + 32, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, NB))
        self.zdim = zdim
    def forward(self, m, z=None):
        z = torch.randn(m.shape[0], self.zdim) if z is None else z
        logit = self.net(torch.cat([self.emb(m), z], -1))
        soft = torch.tanh(logit)
        hard = torch.sign(soft) + (soft - soft.detach())      # straight-through binarisation: exact GMSK frames forward
        return gmsk_modulate(hard)

def channel(phi, ebn0_db, bits, rng_t):
    """AWGN + random integer timing offset in [-W, W]; receiver window at zero offset."""
    B = phi.shape[0]; x = torch.polar(torch.ones_like(phi), phi)
    off = torch.randint(-W, W + 1, (B,), generator=rng_t)
    xp = F.pad(x, (W, W))
    idx = (torch.arange(N)[None, :] + W + off[:, None])
    y = torch.gather(xp, 1, idx)
    N0 = N / (bits * 10 ** (ebn0_db / 10))
    y = y + torch.sqrt(torch.tensor(N0 / 2)) * (torch.randn_like(y.real) + 1j * torch.randn_like(y.real))
    return torch.angle(y), y

# ---------------- training ----------------
rng = np.random.default_rng(0); rng_t = torch.Generator().manual_seed(0)
G, D, R = Generator(K), Conv1dNet(1), Conv1dNet(K)
oG = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.999)); oD = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.999)); oR = torch.optim.Adam(R.parameters(), 1e-3)
bs = 128; t0 = time.time(); bits = math.log2(K)
def rx_feats(y):
    """receiver input: noisy complex samples -> (cos, sin, increments) of the noisy phase plus magnitude"""
    return torch.angle(y)
for it in range(steps):
    m = torch.randint(0, K, (bs,))
    real = gmsk_frames(bs, rng)
    fake = G(m)
    # D step
    lD = F.binary_cross_entropy_with_logits(D(real).squeeze(-1), torch.ones(bs)) + F.binary_cross_entropy_with_logits(D(fake.detach()).squeeze(-1), torch.zeros(bs))
    oD.zero_grad(); lD.backward(); oD.step()
    # R step (on noisy, offset waveforms at a training SNR)
    phin, _ = channel(fake.detach(), 6.0, bits, rng_t)
    lR = F.cross_entropy(R(phin), m)
    oR.zero_grad(); lR.backward(); oR.step()
    # G step
    fake = G(m)
    phin, _ = channel(fake, 6.0, bits, rng_t)
    adv = F.binary_cross_entropy_with_logits(D(fake).squeeze(-1), torch.ones(bs))
    comm = F.cross_entropy(R(phin), m)
    af = isl_excl(fake).mean()
    lG = adv + lam_comm * comm + lam_af * af
    oG.zero_grad(); lG.backward(); oG.step()
    if it % 250 == 0 or it == steps - 1:
        with torch.no_grad():
            p = psl_db_excl(G(torch.randint(0, K, (500,))))
        print(f"it {it} D {lD.item():.3f} R-CE {lR.item():.3f} adv {adv.item():.3f} ISL {af.item():.4f} | gen PSL median {p.median():.1f} dB ({time.time()-t0:.0f}s)", flush=True)
torch.save({"G": G.state_dict(), "R": R.state_dict()}, OUT / f"p3a_{tag}.pt")

# ---------------- evaluation ----------------
G.eval(); R.eval()
def held_out_auc(gen_fn, n=4000, train_steps=600):
    """Fresh detector trained on n GMSK vs n generated, tested on fresh held-out sets. Returns AUC."""
    Dh = Conv1dNet(1); o = torch.optim.Adam(Dh.parameters(), 1e-3)
    Xr, Xf = gmsk_frames(n, rng), gen_fn(n).detach()
    with torch.enable_grad():
        for _ in range(train_steps):
            i = torch.randint(0, n, (128,))
            l = F.binary_cross_entropy_with_logits(Dh(Xr[i]).squeeze(-1), torch.ones(128)) + F.binary_cross_entropy_with_logits(Dh(Xf[i]).squeeze(-1), torch.zeros(128))
            o.zero_grad(); l.backward(); o.step()
    with torch.no_grad():
        sr, sf = Dh(gmsk_frames(2000, rng)).squeeze(-1), Dh(gen_fn(2000)).squeeze(-1)
    # AUC via rank statistic (prob that a real frame scores above a generated one)
    s = torch.cat([sr, sf]); lab = torch.cat([torch.ones(2000), torch.zeros(2000)])
    order = s.argsort(); ranks = torch.empty_like(order, dtype=torch.float); ranks[order] = torch.arange(1, 4001).float()
    auc = (ranks[lab == 1].sum() - 2000 * 2001 / 2) / (2000 * 2000)
    return float(max(auc, 1 - auc))

def ser_corr(codebook_fn, ebn0_db, n=4000):
    """CSK with a correlation receiver that knows the codebook (one waveform per message) and searches +-W lags."""
    m = torch.randint(0, K, (n,)); phi = codebook_fn(m, fixed=True)      # transmitter and receiver share the seed
    _, y = channel(phi, ebn0_db, bits, rng_t)
    allm = torch.arange(K); C = torch.polar(torch.ones(K, N), codebook_fn(allm, fixed=True))
    best = torch.full((n, K), -1.0)
    yp = F.pad(y, (W, W))
    for k in range(-W, W + 1):
        yk = yp[:, W + k: W + k + N]
        best = torch.maximum(best, (yk @ C.conj().T).abs())
    return float((best.argmax(1) != m).float().mean())

def ser_learned_rx(ebn0_db, n=4000):
    m = torch.randint(0, K, (n,)); phin, _ = channel(G(m), ebn0_db, bits, rng_t)
    with torch.no_grad(): return float((R(phin).argmax(1) != m).float().mean())

# codebooks: GAN (one fixed z per message for the correlation receiver; fresh z for covertness), random GMSK, searched GMSK, CAN
zfix = torch.randn(K, 16)
def gan_cb(m, fixed=False):
    with torch.no_grad(): return G(m, zfix[m]) if fixed else G(m)
gm_cb_tab = gmsk_frames(K, rng)
def gmsk_cb(m, fixed=False): return gm_cb_tab[m]
best_tab = []
for k in range(K):
    cands = gmsk_frames(2000, rng); p = psl_db_excl(cands); best_tab.append(cands[p.argmin()])
best_tab = torch.stack(best_tab)
def searched_cb(m, fixed=False): return best_tab[m]
can_tab = torch.tensor(np.stack([np.angle(can(np.exp(1j * rng.uniform(0, 2 * np.pi, N)), n_iter=500)) for _ in range(K)]), dtype=torch.float32)
def can_cb(m, fixed=False): return can_tab[m]

rows = []
with torch.no_grad():
    for name, fn, gen_for_auc in [("GAN", gan_cb, lambda n: G(torch.randint(0, K, (n,)))),
                                  ("random GMSK", gmsk_cb, lambda n: gmsk_frames(n, rng)),
                                  ("searched GMSK (best of 2000)", searched_cb, lambda n: best_tab[torch.randint(0, K, (n,))]),
                                  ("CAN codes", can_cb, lambda n: can_tab[torch.randint(0, K, (n,))])]:
        auc = held_out_auc(gen_for_auc)
        X = fn(torch.arange(K), fixed=True); p = psl_db_excl(X)
        ser = {e: ser_corr(fn, e) for e in [0, 4, 8, 12]}
        print(f"{name:30s} detector AUC {auc:.3f} | codebook PSL mean {p.mean():.1f} dB (worst {p.max():.1f}) | SER corr-rx @0/4/8/12 dB: {ser[0]:.3f}/{ser[4]:.3f}/{ser[8]:.3f}/{ser[12]:.3f}", flush=True)
        rows.append((name, auc, float(p.mean()), float(p.max()), ser[0], ser[4], ser[8], ser[12]))
    serR = {e: ser_learned_rx(e) for e in [0, 4, 8, 12]}
    print(f"GAN with learned receiver R (fresh z per symbol): SER @0/4/8/12 dB: {serR[0]:.3f}/{serR[4]:.3f}/{serR[8]:.3f}/{serR[12]:.3f}", flush=True)
    pg = psl_db_excl(G(torch.randint(0, K, (2000,))))
    print(f"GAN fresh-z waveforms: PSL median {pg.median():.1f} dB, 10th pct {pg.quantile(0.1):.1f}, 90th pct {pg.quantile(0.9):.1f}; GMSK random PSL median {psl_db_excl(gmsk_frames(2000, rng)).median():.1f} dB", flush=True)
with open(OUT / f"p3a_{tag}.csv", "w") as f:
    f.write("method,detector_auc,psl_mean_db,psl_worst_db,ser_0dB,ser_4dB,ser_8dB,ser_12dB\n")
    for r in rows: f.write(",".join(str(v) for v in r) + "\n")
    f.write(f"GAN learned rx,,,,{serR[0]},{serR[4]},{serR[8]},{serR[12]}\n")
print("saved")
