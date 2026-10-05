"""P1f - Feasibility probe for a discrete-phase pivot: how good are QPSK/8PSK codes from cheap local search?
Date: 2026-10-05
Rationale: Continuous-phase design is dominated by analytic gradient descent (P1c). For M-PSK alphabets GD
does not apply directly; the classical tools are CAN + quantisation and single-chip coordinate descent (local
search), with GAs for harder cases. If cheap local search leaves a large gap to the continuous optimum, a
learned sampler distilled from heavy search has room to contribute; if not, the discrete pivot is also closed.
Measures at N=64: Haar-random M-PSK, CAN->quantise, coordinate descent (ISL objective, best-of-sweeps) from
random init and from quantised CAN, with wall-clock. Output: results/p1/p1f_discrete.csv
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from isacgen.metrics import psl_isl, isl_raw, _acorr_fft
from isacgen.classical import can
from isacgen.codes import random_unimodular
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "p1"
rng = np.random.default_rng(5)
N, R = 64, 60

def quantise(x, M):
    return np.exp(1j * (2 * np.pi / M) * np.round(np.angle(x) / (2 * np.pi / M)))

def coord_descent(x, M, max_sweeps=50):
    """Greedy single-chip updates minimising ISL; stops when a full sweep makes no improvement."""
    x = x.copy(); J = float(isl_raw(x)); alph = np.exp(2j * np.pi * np.arange(M) / M)
    for s in range(max_sweeps):
        improved = False
        for n in rng.permutation(N):
            best_v, best_J = x[n], J
            for a in alph:
                if a == x[n]: continue
                x[n] = a; Jn = float(isl_raw(x))
                if Jn < best_J - 1e-9: best_v, best_J = a, Jn
            x[n] = best_v; 
            if best_J < J: J = best_J; improved = True
        if not improved: break
    return x

rows = []
for M in [2, 4, 8]:
    res = {k: [] for k in ["random", "CAN+quant", "CD from random", "CD from CAN+quant", "continuous CAN"]}
    tm = {k: 0.0 for k in res}
    for r in range(R):
        x0 = quantise(random_unimodular(1, N, rng)[0], M)
        res["random"].append(psl_isl(x0)[0])
        t = time.time(); xc = can(random_unimodular(1, N, rng)[0], n_iter=1000); tm["continuous CAN"] += time.time() - t
        res["continuous CAN"].append(psl_isl(xc)[0])
        t = time.time(); xq = quantise(xc, M); tm["CAN+quant"] += time.time() - t; res["CAN+quant"].append(psl_isl(xq)[0])
        t = time.time(); xcd = coord_descent(x0, M); tm["CD from random"] += time.time() - t; res["CD from random"].append(psl_isl(xcd)[0])
        t = time.time(); xcq = coord_descent(xq, M); tm["CD from CAN+quant"] += time.time() - t; res["CD from CAN+quant"].append(psl_isl(xcq)[0])
    for k, v in res.items():
        v = 20 * np.log10(np.array(v))
        rows.append((M, k, v.mean(), v.min(), np.median(v), tm[k] / R * 1e3))
        print(f"M={M} {k:18s} PSL mean {v.mean():6.2f} best {v.min():6.2f} median {np.median(v):6.2f} dB   {tm[k]/R*1e3:7.1f} ms/code")
with open(OUT / "p1f_discrete.csv", "w") as f:
    f.write("M,method,psl_mean_db,psl_best_db,psl_median_db,ms_per_code\n")
    for r in rows: f.write(",".join(str(v) for v in r) + "\n")
