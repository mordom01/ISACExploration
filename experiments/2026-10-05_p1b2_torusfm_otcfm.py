"""P1b2 - Torus flow matching with minibatch OT coupling and in-flight evaluation (fallback / second variant).
Date: 2026-10-05
Rationale: In P1b the CFM loss sat at the irreducible floor pi^2/3 because, with a uniform source on the torus,
a chip's own phase carries no information about its velocity (per-chip marginals of good codes are near
uniform); only the joint across chips does. Two changes that reduce the conditional variance without changing
the target distribution: (1) minibatch optimal-transport coupling of source and data samples under the wrapped
L2 metric (OT-CFM, Tong et al., TMLR 2024, here on the flat torus), and (2) keeping the data in canonical global
phase (phi[0]=0) without the random-rotation augmentation used in P1b. Also prints sampled median PSL every
1000 steps so progress is visible. Usage: python <script> <steps> [ot|indep] [aug|noaug]
Outputs: results/p1/p1b2_<tag>.{log,csv,png,pt}
"""
import sys, pathlib, time, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from isacgen.torusfm import TorusVelocityNet, sample, canonicalise, wrap, psl_db_torch
from isacgen import plotstyle
plotstyle.apply()
torch.manual_seed(0); np.random.seed(0)
steps = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
coupling = sys.argv[2] if len(sys.argv) > 2 else "ot"
aug = (sys.argv[3] if len(sys.argv) > 3 else "noaug") == "aug"
tag = f"{coupling}_{'aug' if aug else 'noaug'}"
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "p1"
D = np.load(OUT / "singles_N64.npz")
P = torch.from_numpy(D["phases"]).float(); N = P.shape[1]
train, held = P[:-2000], P[-2000:]
torch.set_num_threads(4)
model = TorusVelocityNet(N, d_model=128, n_layers=4, n_heads=4)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
bs = 256

def ot_pair(phi0, phi1):
    d = wrap(phi0[:, None, :] - phi1[None, :, :]).pow(2).sum(-1).numpy()
    r, c = linear_sum_assignment(d)
    return phi0[r], phi1[c]

t0 = time.time(); losses = []
for it in range(steps):
    idx = torch.randint(0, train.shape[0], (bs,))
    phi1 = train[idx]
    if aug:
        phi1 = wrap(phi1 + (torch.rand(bs, 1) * 2 * math.pi - math.pi))
    phi0 = torch.rand(bs, N) * 2 * math.pi - math.pi
    if coupling == "ot":
        phi0, phi1 = ot_pair(phi0, phi1)
    t = torch.rand(bs)
    u = wrap(phi1 - phi0)
    phit = wrap(phi0 + t[:, None] * u)
    loss = ((model(phit, t) - u) ** 2).mean()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    losses.append(loss.item())
    if it % 500 == 0 or it == steps - 1:
        print(f"it {it} loss {np.mean(losses[-100:]):.4f} ({time.time()-t0:.0f}s)", flush=True)
    if (it % 1000 == 0 and it > 0) or it == steps - 1:
        model.eval()
        with torch.no_grad():
            S = sample(model, 500, N, nfe=8); psl = psl_db_torch(S)
        model.train()
        print(f"   [eval it {it}] sampled PSL median {psl.median():.2f} mean {psl.mean():.2f} frac<=-25dB {(psl<=-25).float().mean():.3f} (train median {np.median(D['psl_db']):.2f})", flush=True)
        torch.save(model.state_dict(), OUT / f"p1b2_{tag}.pt")

model.eval()
rows = []
def nn_dist(A, B):
    A = canonicalise(A); B = canonicalise(B); d = []
    for i in range(0, A.shape[0], 256):
        d.append(wrap(A[i:i+256, None, :] - B[None, :, :]).norm(dim=-1).min(dim=1).values)
    return torch.cat(d)
haar = torch.rand(2000, N) * 2 * math.pi - math.pi
for name, X in {"held-out": held, "Haar": haar}.items():
    psl = psl_db_torch(X); rows.append((name, 0, float(psl.median()), float(psl.mean()), float((psl <= -25).float().mean()), float(nn_dist(X, train[:20000]).mean())))
    print(f"{name:10s} PSL median {psl.median():.2f} mean {psl.mean():.2f} frac<=-25dB {(psl<=-25).float().mean():.3f} NN-dist {rows[-1][-1]:.3f}")
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.5))
for nfe in [1, 2, 4, 8, 16, 32]:
    with torch.no_grad(): S = sample(model, 2000, N, nfe=nfe)
    psl = psl_db_torch(S); nnd = float(nn_dist(S, train[:20000]).mean())
    rows.append((f"FM nfe={nfe}", nfe, float(psl.median()), float(psl.mean()), float((psl <= -25).float().mean()), nnd))
    print(f"FM nfe={nfe:2d} PSL median {psl.median():.2f} mean {psl.mean():.2f} frac<=-25dB {(psl<=-25).float().mean():.3f} NN-dist {nnd:.3f}")
    if nfe in (1, 4, 8, 32): ax[0].hist(psl.numpy(), bins=40, histtype="step", lw=1.6, label=f"FM nfe={nfe}")
ax[0].hist(psl_db_torch(held).numpy(), bins=40, histtype="step", lw=1.6, ls="--", color=plotstyle.TEXT, label="training distribution")
ax[0].hist(psl_db_torch(haar).numpy(), bins=40, histtype="step", lw=1.2, ls=":", color=plotstyle.TEXT2, label="Haar")
ax[0].set_xlabel("PSL (dB)"); ax[0].legend(fontsize=7); ax[0].set_title(f"{tag}: sampled PSL vs training")
nfes = [r[1] for r in rows if r[1] > 0]; meds = [r[2] for r in rows if r[1] > 0]
ax[1].plot(nfes, meds, marker="o", label="FM median PSL"); ax[1].axhline(rows[0][2], ls="--", color=plotstyle.TEXT, label="training median")
ax[1].set_xscale("log", base=2); ax[1].set_xlabel("NFE"); ax[1].set_ylabel("median PSL (dB)"); ax[1].legend()
fig.tight_layout(); fig.savefig(OUT / f"p1b2_{tag}.png")
with open(OUT / f"p1b2_{tag}.csv", "w") as f:
    f.write("name,nfe,psl_median_db,psl_mean_db,frac_le_m25dB,nn_dist_to_train\n")
    for r in rows: f.write(",".join(str(v) for v in r) + "\n")
print("saved")
