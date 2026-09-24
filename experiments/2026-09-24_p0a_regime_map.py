"""P0a - Regime map: when is a learned sampler needed at all?
Date: 2026-09-24
Rationale: A conditional generative model for unimodular waveforms only earns its keep in the regime where
naive alternatives fail. The cheapest alternative is rejection sampling from the uniform (Haar) distribution
on the torus: draw random phases, keep those whose PSL meets the target. If the acceptance rate is >= 1e-3,
a GPU can produce thousands of feasible, maximally diverse waveforms per second with no model at all.
We measure the acceptance rate vs PSL threshold for several N, and the PSL reached by a cheap local polish
(CAN from random init), which defines the *upper* end of what an amortised model must reproduce.
Outputs: results/p0/p0a_acceptance.csv, results/p0/p0a_polish.csv, results/p0/p0a_regime_map.png
Never mutate this file; copy it for a new run. (Edited once before first commit: l_p polish now uses a best-PSL-iterate guard and irls_every=5.)
"""
import sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from isacgen.metrics import psl_isl
from isacgen.codes import random_unimodular, frank, p4
from isacgen.classical import can, torus_gd
from isacgen import plotstyle
plotstyle.apply()
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "p0"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(2026)

Ns = [31, 64, 128, 256]
n_samp = 200_000
thr_db = np.arange(-30, -9.9, 0.5)
rows = []
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
for i, N in enumerate(Ns):
    psl = np.concatenate([20 * np.log10(psl_isl(random_unimodular(20_000, N, rng))[0]) for _ in range(n_samp // 20_000)])
    acc = [(psl <= t).mean() for t in thr_db]
    for t, a in zip(thr_db, acc):
        rows.append((N, t, a))
    ax[0].semilogy(thr_db, np.maximum(acc, 1e-7), label=f"N={N}")
    print(f"N={N}: random PSL median {np.median(psl):.2f} dB, 1e-3 quantile {np.quantile(psl,1e-3):.2f} dB, min {psl.min():.2f} dB")
ax[0].axhline(1e-3, color=plotstyle.TEXT2, lw=0.8, ls="--")
ax[0].set_xlabel("PSL target (dB)"); ax[0].set_ylabel("acceptance rate, uniform on torus"); ax[0].legend(); ax[0].set_ylim(1e-6, 1.1)
ax[0].set_title("Rejection sampling from Haar measure")
np.savetxt(OUT / "p0a_acceptance.csv", np.array(rows), delimiter=",", header="N,psl_thr_db,acceptance", comments="")

# polish: CAN from random init, then l_p PSL surrogate GD; 200 restarts each N
prow = []
for i, N in enumerate(Ns):
    X0 = random_unimodular(200, N, rng)
    t0 = time.time()
    Xc = np.stack([can(x, n_iter=1000) for x in X0])
    tc = (time.time() - t0) / 200
    t0 = time.time()
    Xp = np.stack([torus_gd(x, objective="lp", p=16, n_iter=150, irls_every=5) for x in Xc])
    tp = (time.time() - t0) / 200
    pc = 20 * np.log10(psl_isl(Xc)[0]); pp = 20 * np.log10(psl_isl(Xp)[0])
    ref = {}
    if N == 64: ref = {"Frank": 20 * np.log10(psl_isl(frank(8)[None])[0][0]), "P4": 20 * np.log10(psl_isl(p4(64)[None])[0][0])}
    print(f"N={N}: CAN-from-random PSL mean {pc.mean():.2f} (min {pc.min():.2f}, max {pc.max():.2f}) dB, {tc*1e3:.1f} ms/solve; "
          f"+lp-polish PSL mean {pp.mean():.2f} (min {pp.min():.2f}) dB, +{tp*1e3:.1f} ms; refs {ref}")
    prow.append((N, pc.mean(), pc.min(), pc.max(), tc, pp.mean(), pp.min(), pp.max(), tp))
    ax[1].hist(pc, bins=25, histtype="step", lw=1.6, label=f"N={N} CAN", color=plotstyle.SERIES[i])
    ax[1].hist(pp, bins=25, histtype="step", lw=1.6, ls="--", color=plotstyle.SERIES[i])
ax[1].set_xlabel("PSL after local polish from random init (dB); dashed = + l_p polish"); ax[1].set_ylabel("count / 200 restarts"); ax[1].legend()
ax[1].set_title("What random-restart optimisation reaches")
np.savetxt(OUT / "p0a_polish.csv", np.array(prow), delimiter=",",
           header="N,can_psl_mean_db,can_psl_min_db,can_psl_max_db,can_sec,lp_psl_mean_db,lp_psl_min_db,lp_psl_max_db,lp_sec", comments="")
fig.tight_layout(); fig.savefig(OUT / "p0a_regime_map.png")
print("saved", OUT)
