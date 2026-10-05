"""P2d - The policy as a branching heuristic inside exact depth-first search with bound pruning.
Date: 2026-10-05
Rationale: Ancestral sampling from the imitation policy fails because errors compound without backtracking
(P2b, P2c). Exhaustive PSL searches are DFS with the partial-autocorrelation bound
|r_k^{(n+1)}| - (N-1-n) <= T; their cost is dominated by the branching order. We test whether the learned
p(b_n | state) is a better branching heuristic than lexicographic or random order: for each ordering we run a
bounded DFS and record nodes expanded and wall-clock to the first k solutions. Settings: N=40 and 48 with
T=3 (optimal), N=64 with T=5, and N=64 with T=4 (optimal; time-limited). The learned ordering costs one GRU
step per node, so it must cut the node count by more than the per-node overhead ratio to win in time; both
numbers are reported. Usage: <policy.pt> [budget_seconds]
Output: results/p2/p2d_dfs.csv
"""
import sys, pathlib, time, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch
from isacgen.policy import SeqPolicy
from isacgen.binseq import canonical, psl as psl_np
torch.manual_seed(0); torch.set_num_threads(1)
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "p2"
ckpt = sys.argv[1] if len(sys.argv) > 1 else str(OUT / "p2b_policy.pt")
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
model = SeqPolicy(n_max=128, hidden=256, layers=2); model.load_state_dict(torch.load(ckpt)); model.eval()
N_MAX = 128


def dfs(N, T, ordering, k_solutions=20, budget=BUDGET, rng=None):
    """Iterative DFS. Node = (n, b_prefix, r, hstate). Children ordered by `ordering`.
    Returns list of (nodes_expanded, seconds) at each solution found, plus totals."""
    t0 = time.time(); nodes = 0; sols = []; found = set()
    b = np.zeros(N); r = np.zeros(N_MAX - 1)
    # stack entries: (n, chip_to_try_list, r_before, hstate_before, feats_before)
    def children(n, r_before, hstate):
        """return ordered list of chips and the policy hidden state after consuming the step-n features."""
        f = torch.cat([torch.from_numpy(r_before[None, :]).float() / math.sqrt(N),
                       torch.tensor([[b[n - 1] if n > 0 else 0.0]]), torch.tensor([[n / N]]), torch.tensor([[N / N_MAX]])], -1)
        if ordering == "policy":
            with torch.no_grad():
                h, hs = model.gru(model.inp(f)[:, None, :], hstate)
                p = float(torch.sigmoid(model.out(h[:, 0])))
            return ([1.0, -1.0] if p >= 0.5 else [-1.0, 1.0]), hs
        if ordering == "random":
            return ([1.0, -1.0] if rng.random() < 0.5 else [-1.0, 1.0]), None
        return [1.0, -1.0], None                      # lexicographic
    # symmetry breaking: fix b_0 = +1 (negation symmetry)
    stack = []
    ch, hs = children(0, r, None)
    stack.append([0, [1.0], r.copy(), hs])
    while stack and time.time() - t0 < budget and len(sols) < k_solutions:
        n, chips, r_before, hs = stack[-1]
        if not chips:
            stack.pop(); continue
        c = chips.pop(0)
        nodes += 1
        b[n] = c
        r_new = r_before.copy()
        if n > 0:
            kk = min(n, N_MAX - 1)
            r_new[:kk] += c * b[n - 1: n - 1 - kk if n - 1 - kk >= 0 else None: -1][:kk]
        remaining = N - 1 - n
        if n > 0 and (np.abs(r_new[:min(n, N_MAX - 1)]) - remaining > T).any():
            continue                                   # pruned
        if n == N - 1:
            if psl_np(b) <= T:
                key = canonical(b)
                if key not in found:
                    found.add(key); sols.append((nodes, time.time() - t0))
            continue
        ch, hs2 = children(n + 1, r_new, hs)
        stack.append([n + 1, ch, r_new, hs2])
    return sols, nodes, time.time() - t0


rows = []
rng = np.random.default_rng(0)
for N, T, k in [(40, 3, 20), (48, 3, 20), (64, 5, 20), (64, 4, 3)]:
    for ordering in ["lexicographic", "random", "policy"]:
        sols, nodes, secs = dfs(N, T, ordering, k_solutions=k, rng=rng)
        first = sols[0] if sols else (None, None)
        print(f"N={N} T={T} {ordering:13s}: {len(sols)} solutions in {secs:.1f}s, {nodes} nodes ({nodes/secs:.0f} nodes/s); "
              f"first solution at {first[0]} nodes / {first[1] if first[1] is None else round(first[1],2)} s", flush=True)
        rows.append((N, T, ordering, len(sols), secs, nodes, first[0], first[1]))
with open(OUT / "p2d_dfs.csv", "w") as f:
    f.write("N,T,ordering,solutions,seconds,nodes,first_nodes,first_seconds\n")
    for r in rows: f.write(",".join(str(v) for v in r) + "\n")
print("saved")
