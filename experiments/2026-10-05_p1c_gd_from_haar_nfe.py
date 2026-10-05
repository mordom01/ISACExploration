"""P1c - "Guidance-only" baseline: how good is plain ISL gradient descent from Haar-random phases after k steps?
Date: 2026-10-05
Rationale: A learned flow sampled with k network evaluations must beat the cheapest alternative at the same k:
k steps of Riemannian gradient descent on the differentiable ISL from the same random start (no model, no
training data). This fixes the bar for the smoke test's quality-vs-NFE curve and tests the Phase 0 claim that
random-restart optimisation is the dangerous baseline. Adam on phases, lr swept; batch of 1000 codes; N=64.
Output: results/p1/p1c_gd_nfe.csv
"""
import sys, pathlib, time, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch
from isacgen.torusfm import isl_torch, psl_db_torch
torch.manual_seed(1); torch.set_num_threads(1)
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "p1"
N, B = 64, 1000
rows = []
for lr in [0.03, 0.1, 0.3]:
    phi = (torch.rand(B, N) * 2 * math.pi - math.pi).requires_grad_(True)
    opt = torch.optim.Adam([phi], lr=lr)
    marks = {1, 2, 4, 8, 16, 32, 64, 128, 256}
    t0 = time.time()
    for k in range(1, 257):
        opt.zero_grad(); J = isl_torch(phi); J.sum().backward(); opt.step()
        if k in marks:
            with torch.no_grad():
                p = psl_db_torch(phi)
            rows.append((lr, k, float(p.median()), float(p.mean()), float((p <= -25).float().mean()), (time.time() - t0) / B * 1e3))
            print(f"lr {lr} k={k:3d}: PSL median {p.median():.2f} mean {p.mean():.2f} frac<=-25dB {(p<=-25).float().mean():.3f}  ({(time.time()-t0)/B*1e3:.3f} ms/code cum.)")
with open(OUT / "p1c_gd_nfe.csv", "w") as f:
    f.write("lr,steps,psl_median_db,psl_mean_db,frac_le_m25dB,ms_per_code_cumulative\n")
    for r in rows: f.write(",".join(str(v) for v in r) + "\n")
