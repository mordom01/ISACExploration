"""P0b - Set-level diversity: what can joint design buy over i.i.d. sampling?
Date: 2026-09-24
Rationale: The prior GAN work reports mean pairwise cross-correlation of 0.09-0.15 after slot embeddings and
calls it "matching real code families". We test the hypothesis that this is simply the i.i.d.-random level,
and quantify what joint design (Multi-CAN; weighted-window Riemannian GD) can and cannot improve:
 (a) total ISL is bounded below by N^2 K (K-1), so *mean* cross-correlation energy cannot beat i.i.d.;
 (b) peak cross-correlation and windowed / zero-lag cross-correlation can be driven far below i.i.d.
This fixes which set-level objective a generative model should be judged on.
Outputs: results/p0/p0b_set_stats.csv, results/p0/p0b_set_bounds.png
"""
import sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from isacgen.metrics import psl_isl, xcorr_stats, set_total_isl
from isacgen.codes import random_unimodular
from isacgen.classical import can, multi_can, torus_gd_set, set_isl_bound
from isacgen import plotstyle
plotstyle.apply()
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "p0"; OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)
N, W, trials = 64, 3, 30
Ks = [2, 4, 8]
methods = ["iid_random", "iid_CAN", "MultiCAN", "windowGD"]
rows = []
summary = {}
for K in Ks:
    acc = {m: dict(peak=[], mean=[], zl=[], wpeak=[], psl=[], tot=[], sec=[]) for m in methods}
    wa = np.ones(2 * N - 1); wa[N - 1] = 0
    wc = np.zeros(2 * N - 1); wc[N - 1 - W: N + W] = 1.0
    for t in range(trials):
        X0 = random_unimodular(K, N, rng)
        t0 = time.time(); X1 = np.stack([can(x, n_iter=1000) for x in X0]); t1 = time.time() - t0
        t0 = time.time(); X2 = multi_can(X0, n_iter=1500); t2 = time.time() - t0
        t0 = time.time(); X3 = torus_gd_set(X0, wa, 50 * wc, n_iter=300); t3 = time.time() - t0
        for m, X, sec in zip(methods, [X0, X1, X2, X3], [0.0, t1, t2, t3]):
            st = xcorr_stats(X); stw = xcorr_stats(X, lag_window=W)
            acc[m]["peak"].append(st["peak"].max()); acc[m]["mean"].append(st["mean"].mean())
            acc[m]["zl"].append(st["zero_lag"].max()); acc[m]["wpeak"].append(stw["peak"].max())
            acc[m]["psl"].append(20 * np.log10(psl_isl(X)[0]).max()); acc[m]["tot"].append(set_total_isl(X)); acc[m]["sec"].append(sec)
    for m in methods:
        a = {k: float(np.mean(v)) for k, v in acc[m].items()}
        summary[(K, m)] = a
        rows.append((K, m, a["peak"], a["mean"], a["zl"], a["wpeak"], a["psl"], a["tot"], set_isl_bound(K, N), a["sec"]))
        print(f"K={K} {m:11s} peak-x(all lags, worst pair) {a['peak']:.3f} | mean-x {a['mean']:.3f} | zero-lag worst {a['zl']:.3f} | "
              f"peak-x |k|<={W} worst {a['wpeak']:.3f} | worst PSL {a['psl']:.1f} dB | total ISL {a['tot']:.0f} (bound {set_isl_bound(K,N):.0f}) | {a['sec']*1e3:.0f} ms")
with open(OUT / "p0b_set_stats.csv", "w") as f:
    f.write("K,method,peak_xcorr_worst,mean_xcorr,zero_lag_worst,window_peak_worst,worst_psl_db,total_isl,isl_bound,sec\n")
    for r in rows: f.write(",".join(str(v) for v in r) + "\n")

fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
x = np.arange(len(Ks)); wdt = 0.2
for j, m in enumerate(methods):
    ax[0].bar(x + (j - 1.5) * wdt, [summary[(K, m)]["peak"] for K in Ks], wdt, label=m, color=plotstyle.SERIES[j])
    ax[1].bar(x + (j - 1.5) * wdt, [summary[(K, m)]["wpeak"] for K in Ks], wdt, label=m, color=plotstyle.SERIES[j])
    ax[2].bar(x + (j - 1.5) * wdt, [10 * np.log10(summary[(K, m)]["tot"] / set_isl_bound(K, N)) for K in Ks], wdt, label=m, color=plotstyle.SERIES[j])
for a, t, yl in zip(ax, ["Worst-pair peak cross-corr (all lags)", f"Worst-pair peak cross-corr, |lag|<={W}", "Total set ISL above bound (dB)"],
                    ["|c|/N", "|c|/N", "dB"]):
    a.set_xticks(x); a.set_xticklabels([f"K={K}" for K in Ks]); a.set_title(t); a.set_ylabel(yl)
ax[1].set_yscale("log"); ax[0].legend(fontsize=7)
fig.suptitle(f"N={N}: joint design cannot beat i.i.d. in total ISL, but can zero a lag window", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "p0b_set_bounds.png"); print("saved")
