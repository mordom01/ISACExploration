"""P2e - Short-length training data (N=16..28, optimal PSL=2) for the DFS-ordering test.
Date: 2026-10-05
Rationale: The policy trained on N=32..72 (P2b) was tested as a DFS branching heuristic only at N=20, outside its
training range, and the Python DFS cannot reach N>=28 with the simple bound. To get a fair, decisive test of
"learned ordering vs lexicographic/random" we train on lengths where DFS completes: optimal (PSL 2) sequences
for N=16..28 found by the shotgun search (seconds per length), then test DFS at N=24..28 (in range) and 30, 32
(transfer). Outputs: results/p2/binseq_N{N}.npz for N in 16..28 (even), log via stdout.
"""
import sys, pathlib, time, multiprocessing as mp
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from isacgen.binseq import shotgun_psl, canonical
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "p2"
def worker(args):
    N, seed, budget = args; rng = np.random.default_rng(seed); t0 = time.time(); found = {}
    while time.time() - t0 < budget:
        b, p, _ = shotgun_psl(N, rng, max_iters=3000, sideways=100); c = canonical(b)
        if c not in found or found[c] > p: found[c] = p
    return found
if __name__ == "__main__":
    for N in [16, 18, 20, 22, 24, 26, 28]:
        with mp.Pool(3) as pool: parts = pool.map(worker, [(N, 7 * N + w, 60.0) for w in range(3)])
        found = {}
        for d in parts:
            for c, p in d.items():
                if c not in found or found[c] > p: found[c] = p
        ps = np.array(list(found.values())); best = ps.min()
        keep = [(c, p) for c, p in found.items() if p <= best]
        np.savez_compressed(OUT / f"binseq_N{N}.npz", seqs=np.array([c for c, _ in keep], dtype=np.int8), psl=np.array([p for _, p in keep], dtype=np.int16))
        print(f"N={N}: distinct {len(found)}, PSL hist {dict(zip(*[x.tolist() for x in np.unique(ps, return_counts=True)]))}, kept {len(keep)} at PSL {best}", flush=True)
