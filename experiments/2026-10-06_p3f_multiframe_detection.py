"""P3f - Multi-frame detectability: does per-frame covertness survive a warden that accumulates evidence?
Date: 2026-10-06
Rationale: All detector AUCs so far (P3b, P3d, P3e) are single-frame. A warden observing n frames can sum its
per-frame log-likelihood-ratio estimates; with per-frame KL D, the optimal test's error decays roughly like
exp(-n D) (Stein), so covertness over many frames needs D -> 0 (the square-root law of covert communication).
Here: train the same CNN detector per law (6000 vs 6000 frames), then score held-out *groups* of n frames by the
sum of logits and report AUC for n in {1, 4, 16}. Laws: exact tilts (MCMC) at beta in {2.5, 5, 10} and the
forward-KL sampler from P3e at beta=10. Output: results/p3/p3f_multiframe.csv
"""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F
src = open(pathlib.Path(__file__).resolve().parents[0] / "2026-10-06_p3d_boltzmann_covert.py").read()
ns = {"__file__": str(pathlib.Path(__file__).resolve().parents[0] / "2026-10-06_p3d_boltzmann_covert.py")}
exec(compile(src.split("betas = [0, 5, 10, 20, 40, 80, 160]")[0], "p3d_head", "exec"), ns)
exec(compile(src.split("# ---------------- (2) autoregressive Boltzmann generator ----------------")[1].split("for b in [10, 40, 160]:")[0], "p3d_van", "exec"), ns)
mcmc, modulate, Det, VAN, NB = ns["mcmc"], ns["modulate"], ns["Det"], ns["VAN"], ns["NB"]
torch.manual_seed(7); OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "p3"
NTR, NTE = 6000, 16 * 400

def auc(sr, sq):
    s = torch.cat([sr, sq]); lab = torch.cat([torch.ones(len(sr)), torch.zeros(len(sq))])
    o = s.argsort(); r = torch.empty_like(o, dtype=torch.float); r[o] = torch.arange(1, len(s) + 1).float()
    a = (r[lab == 1].sum() - len(sr) * (len(sr) + 1) / 2) / (len(sr) * len(sq)); return float(max(a, 1 - a))

def background(n): return modulate(torch.randint(0, 2, (n, NB)).float() * 2 - 1)

def evaluate(name, Xq):
    D = Det(); o = torch.optim.Adam(D.parameters(), 1e-3); Xr = background(NTR + NTE)
    for _ in range(500):
        i = torch.randint(0, NTR, (128,))
        l = F.binary_cross_entropy_with_logits(D(Xr[i]), torch.ones(128)) + F.binary_cross_entropy_with_logits(D(Xq[i]), torch.zeros(128))
        o.zero_grad(); l.backward(); o.step()
    with torch.no_grad(): sr, sq = D(Xr[NTR:]), D(Xq[NTR:])
    res = []
    for n in [1, 4, 16]:
        res.append(auc(sr[: (NTE // n) * n].view(-1, n).sum(1), sq[: (NTE // n) * n].view(-1, n).sum(1)))
    print(f"{name:28s} AUC over 1 / 4 / 16 frames: {res[0]:.3f} / {res[1]:.3f} / {res[2]:.3f}", flush=True)
    return (name, *res)

rows = []
for b in [2.5, 5, 10]:
    s, _ = mcmc(b, n_chains=NTR + NTE, sweeps=25); rows.append(evaluate(f"exact tilt beta={b}", modulate(s)))
van = VAN(); van.load_state_dict(torch.load(OUT / "p3e_van_fkl_beta10.pt"))
with torch.no_grad(): sym, _ = van.sample(NTR + NTE)
rows.append(evaluate("forward-KL sampler beta=10", modulate(sym)))
rows.append(evaluate("background (control)", background(NTR + NTE)))
with open(OUT / "p3f_multiframe.csv", "w") as f:
    f.write("law,auc_1frame,auc_4frames,auc_16frames\n")
    for r in rows: f.write(",".join(str(v) for v in r) + "\n")
print("saved")
