"""P1g - Baseline for Direction A'': PSL-objective local search for binary codes, N=64.
Date: 2026-10-05
Rationale: Direction A'' (learned constructive sampler for discrete-phase codes) must beat the cheapest
classical competitor at matched wall-clock: coordinate descent (single-bit flips) on the PSL objective with ISL
tie-breaking, from random starts. We measure the PSL distribution over restarts, the fraction reaching PSL <= 5
and PSL = 4 (optimum, Coxson & Russo 2005), the number of distinct codes found, and the time per restart.
Output: results/p1/p1g_binary_psl_cd.csv
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "p1"
rng = np.random.default_rng(9)
N, R = 64, 300

def acorr_side(x):
    r = np.correlate(x, x, mode="full")[N:]  # lags 1..N-1
    return np.abs(r)

def score(x):
    s = acorr_side(x); return (int(s.max()), float((s ** 2).sum()))

def cd_psl(x, max_sweeps=200):
    cur = score(x)
    for _ in range(max_sweeps):
        improved = False
        for n in rng.permutation(N):
            x[n] = -x[n]; new = score(x)
            if new < cur: cur = new; improved = True
            else: x[n] = -x[n]
        if not improved: break
    return x, cur[0]

psls, codes, t0 = [], set(), time.time()
for r in range(R):
    x = rng.choice([-1.0, 1.0], size=N)
    x, p = cd_psl(x); psls.append(p); codes.add(tuple((x > 0).astype(int)))
dt = (time.time() - t0) / R
psls = np.array(psls)
vals, cnts = np.unique(psls, return_counts=True)
print("PSL histogram (value: count):", dict(zip(vals.tolist(), cnts.tolist())))
print(f"frac PSL<=5: {(psls<=5).mean():.3f}  frac PSL==4: {(psls==4).mean():.3f}  distinct codes {len(codes)}/{R}  {dt*1e3:.1f} ms/restart  mean PSL dB {np.mean(20*np.log10(psls/N)):.2f}")
with open(OUT / "p1g_binary_psl_cd.csv", "w") as f:
    f.write("psl,count\n"); [f.write(f"{v},{c}\n") for v, c in zip(vals, cnts)]
    f.write(f"# frac_le5,{(psls<=5).mean()}\n# frac_eq4,{(psls==4).mean()}\n# distinct,{len(codes)}\n# ms_per_restart,{dt*1e3}\n")
