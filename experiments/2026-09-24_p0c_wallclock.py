"""P0c - Latency bar: how fast are the classical optimisers the generative model must beat?
Date: 2026-09-24
Rationale: The "few-step ODE sampling is PHY-latency compatible" argument only matters if the per-instance
optimiser is slow at the relevant scale. We time CAN (single) and Multi-CAN (set) in single-threaded numpy
vs N and K, to convergence (relative ISL change < 1e-6). A learned sampler with ~10 network evaluations of a
6-block transformer at N=64 costs a few ms on a GPU; the comparison below sets the bar.
Outputs: results/p0/p0c_wallclock.csv
"""
import sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from isacgen.codes import random_unimodular
from isacgen.classical import can, multi_can
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "p0"; OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(3)
rows = []
for N in [64, 128, 256, 512, 1024, 2048]:
    ts, its = [], []
    for r in range(10):
        x0 = random_unimodular(1, N, rng)[0]
        t0 = time.perf_counter(); x, h = can(x0, n_iter=5000, tol=1e-6, return_hist=True); ts.append(time.perf_counter() - t0); its.append(len(h))
    rows.append(("CAN", N, 1, np.mean(ts), np.mean(its))); print(f"CAN N={N}: {np.mean(ts)*1e3:.1f} ms, {np.mean(its):.0f} iters")
for N in [64, 256]:
    for K in [2, 4, 8, 16]:
        ts, its = [], []
        for r in range(5):
            X0 = random_unimodular(K, N, rng)
            t0 = time.perf_counter(); X, h = multi_can(X0, n_iter=5000, tol=1e-6, return_hist=True); ts.append(time.perf_counter() - t0); its.append(len(h))
        rows.append(("MultiCAN", N, K, np.mean(ts), np.mean(its))); print(f"MultiCAN N={N} K={K}: {np.mean(ts)*1e3:.1f} ms, {np.mean(its):.0f} iters")
with open(OUT / "p0c_wallclock.csv", "w") as f:
    f.write("method,N,K,sec,iters\n")
    for r in rows: f.write(",".join(str(v) for v in r) + "\n")
