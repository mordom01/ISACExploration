"""P1e - How much of the flow path is informative? PSL of the wrapped-geodesic interpolant between Haar phases
and optimiser-generated codes as a function of t.
Date: 2026-10-05
Rationale: Flow matching trains on t ~ U(0,1). If the interpolant phi_t already has Haar-level sidelobes for
t below some t*, the velocity target is unpredictable from phi_t on [0, t*] and only the sliver (t*, 1]
carries signal. Per-chip, the interpolant adds wrapped noise of scale (1-t)*pi; PSL is a quartic functional,
so the structure is destroyed quickly. This quantifies t* for N=64 and motivates time-weighting / alternatives.
Output: results/p1/p1e_path.csv
"""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch
from isacgen.torusfm import wrap, psl_db_torch
torch.manual_seed(0); torch.set_num_threads(1)
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "p1"
P = torch.from_numpy(np.load(OUT / "singles_N64.npz")["phases"]).float()[-2000:]
N = P.shape[1]
phi0 = torch.rand_like(P) * 2 * math.pi - math.pi
u = wrap(P - phi0)
rows = []
for t in [1.0, 0.995, 0.99, 0.98, 0.97, 0.95, 0.93, 0.9, 0.85, 0.8, 0.7, 0.5, 0.0]:
    phit = wrap(phi0 + t * u)
    p = psl_db_torch(phit)
    rows.append((t, float(p.median()), float((p <= -20).float().mean())))
    print(f"t={t:5.3f}  per-chip noise std ~ {(1-t)*math.pi/math.sqrt(3):.3f} rad  PSL median {p.median():6.2f} dB  frac<=-20dB {(p<=-20).float().mean():.3f}")
with open(OUT / "p1e_path.csv", "w") as f:
    f.write("t,psl_median_db,frac_le_m20dB\n")
    for r in rows: f.write(",".join(str(v) for v in r) + "\n")
