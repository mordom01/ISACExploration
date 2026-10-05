# Phase 2 interim report: constructive policies for low-PSL binary sequences (Direction A″)

Date: 2026-10-05 (updated as runs complete). Scripts `experiments/2026-10-05_p2*.py`, outputs `results/p2/`.
CPU-only container, 4 cores.

## 1. Status in one paragraph

The classical search engine works and sets a hard bar: steepest single-flip descent on (PSL, lag count, ISL)
with sideways moves, tabu and restarts yields 4 to 5 distinct PSL-5 sequences per second per core at N=64 and
found two of the optimal PSL-4 sequences in 24 core-minutes. A chip-by-chip imitation policy (GRU on the running
autocorrelation, 0.82M parameters, trained on 133k search outputs across N=32..72 with 8-fold symmetry
augmentation) is perfectly diverse (10⁴ distinct, zero copies of training data) and fast (0.6 ms per
sequence), but its samples have median PSL 10 at N=64 against 5 to 6 for the search at the same CPU time.
Adding the exhaustive-search pruning bound to the sampler (no backtracking) raises the PSL-5 yield only to
0.4 per second versus 21 for the search; an uninformed policy with the same pruning yields zero. The per-chip
NLL (0.50 nats) is near the entropy floor of the feasible set itself (≈0.45 nats per chip from its estimated
size), so imitation is close to the best a factorised left-to-right policy can do: the failure is compounding
error, not under-training. Gates G2 and G3 fail. The last form of the idea, the policy as the branching
heuristic inside exact DFS, is being tested (Section 4).

## 2. Measurements

| ID | Experiment | Result |
|---|---|---|
| P1g | PSL-objective coordinate descent, N=64, 300 restarts | best PSL 6; 0% ≤ 5; 2.6 ms/restart |
| binseq | shotgun search (sideways 200, tabu 8), N=64 | 28 ms/restart; 13% at PSL 5, 87% at PSL 6; all distinct; PSL 4 never in 900 restarts; iterated local search 7890 kicks in 4 min: 0 PSL-4 |
| P2a | training data, 3 workers × 8 min per length | N=32: 3261 (415 at optimal 3); N=40: 25k; N=48: 7.4k at 4 (2 at optimal 3); N=56: 51k at 5 (154 at optimal 4); N=64: 7.5k at 5 (2 at optimal 4); N=72: 39k at 6 (238 at 5) |
| P2b | imitation, 4000 steps, bs 128, NLL/chip 0.693 → 0.496 | N=64: median PSL 10 (T=1.0 and 0.8), 0.14 to 0.24% at PSL 6, 10⁴/10⁴ distinct, 0 train overlap, 0.6 ms/seq. N=80: median 13 (12 at T=0.8). N=96: median 15. Matched CPU-time shotgun: N=64 100% at PSL ≤ 6 (16% at 5); N=80 100% ≤ 7 (22% at 6); N=96 100% ≤ 8 (44% at 7) |
| P2c | policy + pruning bound (no backtracking) vs uninformed + pruning vs shotgun, same CPU-seconds | N=64 T=5: policy 1 distinct (0.4/s), uninformed 0, shotgun 56 (21/s). T=6: policy 18 (7.6/s), uninformed 0, shotgun 333 (141/s). N=80, 96: policy 0 at every T tested; shotgun 63 at T=6 (N=80), 164 at T=7 (N=96) |
| P2d sanity | DFS with the bound, orderings lexicographic / random / policy, N=13 and 20 | correct (Barker-13 found). N=20 T=2 complete tree = 182k nodes; first solution at 4.2k (lex), 2.3k (random), 16.3k (policy) nodes; policy costs 7× per node (2.8k vs 20k nodes/s) |

## 3. Interpretation

- Low-PSL binary sequences are a high-entropy set with weak chip-wise regularities, exactly as the continuous
  codes were (Phase 1). A left-to-right factorisation p(b_n | b_<n) can carry at most the set's entropy per
  chip; what matters for PSL is the global quartic constraint, and small per-chip deviations compound into
  PSL 10 at N=64.
- The classical search does not have this problem because it operates on complete sequences with exact
  objective evaluation and local moves; its cost per candidate (28 ms) buys a guaranteed local optimum.
- The pruning bound |r_k| − (N−1−n) ≤ T is too loose early in the sequence to steer a no-backtracking sampler,
  and the learned policy adds too little to it.

## 4. Pending: the policy as a DFS branching heuristic (P2e/P2f)

Fair test in progress: policy trained on exhaustively known optimal sequences at N=16..28, DFS with the bound
at N=24, 26, 28 (in range) and 30, 32 (transfer), comparing nodes and time to the first solutions under
lexicographic, random and policy orderings. If the policy ordering reduces nodes-to-first-solution by much more
than its 7× per-node overhead, the "learned branch-and-bound for sequence design" framing survives; otherwise
Direction A″ is closed and the paper is the benchmark paper (B″) with the learned-sampler negative results as
a section.
