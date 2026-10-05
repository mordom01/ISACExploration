"""P1a - Training set for the torus flow-matching smoke test: optimiser-generated single codes.
Date: 2026-10-05
Rationale: Phase 0 showed the feasible set at PSL <= -22 dB (N=64) is unreachable by rejection sampling, so a
learned sampler must be a distilled optimiser. We generate the distribution to be learned: CAN from Haar-random
initialisation followed by the l_p PSL polish (best-iterate guard, irls_every=5). Each sample is an independent
random restart, so the set is as diverse as the optimiser's basin structure allows. Global phase is canonicalised
(phi[0] = 0) because all correlation metrics are invariant to it.
Outputs: results/p1/singles_N64.npz (phases [n, N] float32, psl_db [n]) and a log.
"""
import sys, pathlib, time, multiprocessing as mp
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from isacgen.codes import random_unimodular
from isacgen.classical import can, torus_gd
from isacgen.metrics import psl_isl
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "p1"; OUT.mkdir(parents=True, exist_ok=True)
N, n_total = 64, 30000

def work(seed):
    rng = np.random.default_rng(seed)
    x0 = random_unimodular(1, N, rng)[0]
    x = can(x0, n_iter=1000)
    x = torus_gd(x, objective="lp", p=16, n_iter=150, irls_every=5)
    phi = np.angle(x * np.conj(x[0]))  # canonical global phase
    return phi.astype(np.float32)

if __name__ == "__main__":
    t0 = time.time()
    with mp.Pool(4) as pool:
        phis = pool.map(work, range(n_total), chunksize=50)
    P = np.stack(phis)
    psl = 20 * np.log10(psl_isl(np.exp(1j * P.astype(np.float64)))[0])
    np.savez_compressed(OUT / "singles_N64.npz", phases=P, psl_db=psl.astype(np.float32))
    print(f"n={n_total} N={N} time {time.time()-t0:.0f}s  PSL mean {psl.mean():.2f} median {np.median(psl):.2f} "
          f"min {psl.min():.2f} max {psl.max():.2f} dB; frac<=-25dB {(psl<=-25).mean():.3f}")
