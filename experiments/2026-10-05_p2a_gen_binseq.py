"""P2a - Training data for the constructive policy: distinct low-PSL binary sequences by stochastic search.
Date: 2026-10-05
Rationale: The policy is trained by imitation on heavy-search outputs. For each length N we run the
Dimitrov-style shotgun search (steepest single-flip descent on (PSL, #lags at PSL, ISL), sideways moves,
tabu, random restarts) for a fixed wall-clock budget per length on 3 worker processes, keep every sequence
whose PSL is within 1 of the best PSL found at that length, de-duplicate under the PSL-preserving symmetry
group (negation, reversal, alternation), and store canonical representatives with their PSL. The attained PSL
distribution per length is itself the classical baseline (G1/G3 in docs/phase2_plan.md).
Outputs: results/p2/binseq_N{N}.npz (seqs int8 [M, N] in {0,1}, psl int16 [M]); results/p2/p2a.log
"""
import sys, pathlib, time, multiprocessing as mp
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from isacgen.binseq import shotgun_psl, canonical
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "p2"; OUT.mkdir(parents=True, exist_ok=True)
LENGTHS = [32, 40, 48, 56, 64, 72]
BUDGET_S = float(sys.argv[1]) if len(sys.argv) > 1 else 480.0   # per length, per worker
WORKERS = 3

def worker(args):
    N, seed, budget = args
    rng = np.random.default_rng(seed)
    t0 = time.time(); found = {}
    while time.time() - t0 < budget:
        b, p, _ = shotgun_psl(N, rng, max_iters=5000, sideways=200)
        c = canonical(b)
        if c not in found or found[c] > p:
            found[c] = p
    return found

if __name__ == "__main__":
    for N in LENGTHS:
        t0 = time.time()
        with mp.Pool(WORKERS) as pool:
            parts = pool.map(worker, [(N, 1000 * N + w, BUDGET_S) for w in range(WORKERS)])
        found = {}
        for d in parts:
            for c, p in d.items():
                if c not in found or found[c] > p: found[c] = p
        psls = np.array(list(found.values()))
        best = psls.min()
        keep = [(c, p) for c, p in found.items() if p <= best + 1]
        seqs = np.array([c for c, _ in keep], dtype=np.int8); ps = np.array([p for _, p in keep], dtype=np.int16)
        np.savez_compressed(OUT / f"binseq_N{N}.npz", seqs=seqs, psl=ps)
        hist = dict(zip(*[x.tolist() for x in np.unique(psls, return_counts=True)]))
        print(f"N={N}: {len(found)} distinct canonical sequences in {time.time()-t0:.0f}s (3 workers); PSL hist {hist}; kept {len(keep)} with PSL<={best+1}", flush=True)
