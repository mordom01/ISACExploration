"""P1b - Torus flow-matching smoke test (unconditional prior over optimiser outputs, N=64).
Date: 2026-10-05
Rationale: Before any conditioning or set-level work, check that a small flow-matching model on the flat torus
can reproduce the distribution of CAN+polish codes: (1) sampled PSL median within 1 dB of the training median at
8 NFE (gate from Phase 0 Section 6), (2) samples are novel (nearest-neighbour distance to the training set is
comparable to the training set's own nearest-neighbour distance, i.e. no memorisation) and (3) quality vs NFE.
Also reports the rejection-sampling reference (Haar) and the training distribution for the same metrics.
CPU only here; the model is deliberately small (4 layers, d=128). Outputs: results/p1/p1b_*.{csv,png,pt}
"""
import sys, pathlib, time, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from isacgen.torusfm import TorusVelocityNet, cfm_loss, sample, canonicalise, wrap, psl_db_torch
from isacgen import plotstyle
plotstyle.apply()
torch.manual_seed(0); np.random.seed(0)
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "p1"
D = np.load(OUT / "singles_N64.npz")
P = torch.from_numpy(D["phases"]).float(); N = P.shape[1]
train, held = P[:-2000], P[-2000:]
print(f"train {train.shape}, held {held.shape}, train PSL median {np.median(D['psl_db'][:-2000]):.2f} dB")

torch.set_num_threads(4)
model = TorusVelocityNet(N, d_model=128, n_layers=4, n_heads=4)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
steps = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
bs = 256
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
t0 = time.time(); losses = []
for it in range(steps):
    idx = torch.randint(0, train.shape[0], (bs,))
    phi1 = train[idx]
    # data augmentation: random global phase (metrics invariant) so the model does not over-fit phi[0]=0
    phi1 = wrap(phi1 + (torch.rand(bs, 1) * 2 * math.pi - math.pi))
    loss = cfm_loss(model, phi1)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    losses.append(loss.item())
    if it % 500 == 0 or it == steps - 1:
        print(f"it {it} loss {np.mean(losses[-100:]):.4f} ({time.time()-t0:.0f}s)")
torch.save(model.state_dict(), OUT / "p1b_torusfm_N64.pt")

# ---- evaluation ----
model.eval()
def nn_dist(A, B, exclude_self=False):
    """mean over rows of A of min over B of wrapped L2 phase distance (after canonicalising global phase)."""
    A = canonicalise(A); B = canonicalise(B)
    d = []
    for i in range(0, A.shape[0], 256):
        a = A[i:i+256]
        diff = wrap(a[:, None, :] - B[None, :, :]).norm(dim=-1)
        if exclude_self: diff[torch.arange(a.shape[0]), torch.arange(i, i + a.shape[0])] = float("inf")
        d.append(diff.min(dim=1).values)
    return torch.cat(d)
rows = []
haar = torch.rand(2000, N) * 2 * math.pi - math.pi
ref = {"train": train[:2000], "held-out": held, "Haar": haar}
for name, X in ref.items():
    psl = psl_db_torch(X)
    rows.append((name, 0, float(psl.median()), float(psl.mean()), float((psl <= -25).float().mean()), float(nn_dist(X, train[:20000], exclude_self=(name=="train")).mean())))
    print(f"{name:10s} PSL median {psl.median():.2f} mean {psl.mean():.2f} frac<=-25dB {(psl<=-25).float().mean():.3f} NN-dist {rows[-1][-1]:.3f}")
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.5))
for j, nfe in enumerate([1, 2, 4, 8, 16, 32]):
    t1 = time.time(); S = sample(model, 2000, N, nfe=nfe); ts = time.time() - t1
    psl = psl_db_torch(S)
    nnd = float(nn_dist(S, train[:20000]).mean())
    rows.append((f"FM nfe={nfe}", nfe, float(psl.median()), float(psl.mean()), float((psl <= -25).float().mean()), nnd))
    print(f"FM nfe={nfe:2d} PSL median {psl.median():.2f} mean {psl.mean():.2f} frac<=-25dB {(psl<=-25).float().mean():.3f} NN-dist {nnd:.3f}  ({ts*1e3/2000:.2f} ms/sample CPU)")
    if nfe in (1, 4, 8, 32):
        ax[0].hist(psl.numpy(), bins=40, histtype="step", lw=1.6, label=f"FM nfe={nfe}")
ax[0].hist(psl_db_torch(held).numpy(), bins=40, histtype="step", lw=1.6, ls="--", color=plotstyle.TEXT, label="training distribution")
ax[0].hist(psl_db_torch(haar).numpy(), bins=40, histtype="step", lw=1.2, ls=":", color=plotstyle.TEXT2, label="Haar (random phases)")
ax[0].set_xlabel("PSL (dB)"); ax[0].set_ylabel("count / 2000"); ax[0].legend(fontsize=7); ax[0].set_title("Sampled PSL vs training data")
nfes = [r[1] for r in rows if r[1] > 0]; meds = [r[2] for r in rows if r[1] > 0]
ax[1].plot(nfes, meds, marker="o", label="FM median PSL"); ax[1].axhline(rows[1][2], ls="--", color=plotstyle.TEXT, label="training median")
ax[1].set_xscale("log", base=2); ax[1].set_xlabel("NFE (Euler steps)"); ax[1].set_ylabel("median PSL (dB)"); ax[1].legend(); ax[1].set_title("Quality vs sampling cost")
fig.tight_layout(); fig.savefig(OUT / "p1b_smoke.png")
with open(OUT / "p1b_smoke.csv", "w") as f:
    f.write("name,nfe,psl_median_db,psl_mean_db,frac_le_m25dB,nn_dist_to_train\n")
    for r in rows: f.write(",".join(str(v) for v in r) + "\n")
print("saved")
