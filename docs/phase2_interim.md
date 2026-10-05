# Phase 2 interim report: constructive policies for low-PSL binary sequences (Direction A″)

Date: 2026-10-05 (final). Scripts `experiments/2026-10-05_p2*.py`, outputs `results/p2/`.
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
heuristic inside exact DFS, gives a real but modest node-count advantage (1.4 to 4.5×, including at a transfer
length) that is outweighed by the per-node cost of the network in this implementation (Section 4). One
positive observation: at short lengths, inside its training range, the policy emits *new* optimal sequences
absent from its training set at roughly 10× the rate of continued search (Section 5). Direction A″ is closed
as a method for the lengths of interest; the paper is the benchmark paper with these results as a section.

## 2. Measurements

| ID | Experiment | Result |
|---|---|---|
| P1g | PSL-objective coordinate descent, N=64, 300 restarts | best PSL 6; 0% ≤ 5; 2.6 ms/restart |
| binseq | shotgun search (sideways 200, tabu 8), N=64 | 28 ms/restart; 13% at PSL 5, 87% at PSL 6; all distinct; PSL 4 never in 900 restarts; iterated local search 7890 kicks in 4 min: 0 PSL-4 |
| P2a | training data, 3 workers × 8 min per length | N=32: 3261 (415 at optimal 3); N=40: 25k; N=48: 7.4k at 4 (2 at optimal 3); N=56: 51k at 5 (154 at optimal 4); N=64: 7.5k at 5 (2 at optimal 4); N=72: 39k at 6 (238 at 5) |
| P2b | imitation, 4000 steps, bs 128, NLL/chip 0.693 → 0.496 | N=64: median PSL 10 (T=1.0 and 0.8), 0.14 to 0.24% at PSL 6, 10⁴/10⁴ distinct, 0 train overlap, 0.6 ms/seq. N=80: median 13 (12 at T=0.8). N=96: median 15. Matched CPU-time shotgun: N=64 100% at PSL ≤ 6 (16% at 5); N=80 100% ≤ 7 (22% at 6); N=96 100% ≤ 8 (44% at 7) |
| P2c | policy + pruning bound (no backtracking) vs uninformed + pruning vs shotgun, same CPU-seconds | N=64 T=5: policy 1 distinct (0.4/s), uninformed 0, shotgun 56 (21/s). T=6: policy 18 (7.6/s), uninformed 0, shotgun 333 (141/s). N=80, 96: policy 0 at every T tested; shotgun 63 at T=6 (N=80), 164 at T=7 (N=96) |
| P2g | policy samples vs random sequences as initialisations for the same shotgun descent, 300 starts each | N=64: policy init median PSL 10 → final {5: 43, 6: 257}, 363 iterations; random init median 16 → final {5: 41, 6: 259}, 372 iterations. N=80: policy {6: 50, 7: 250}; random {5: 1, 6: 52, 7: 247}. No benefit: the search forgets the initialisation |
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
- As an initialiser for local search the policy is worthless (P2g): a few hundred steepest-descent moves erase
  any advantage of starting at PSL 10 instead of 16.

## 4. The policy as a DFS branching heuristic (P2e/P2f)

Setup: a second policy trained for 1500 steps on the optimal sequences at N=16..28 found by the search
(913 sequences in total; small sets are intrinsic at these lengths, e.g. only 2 canonical PSL-2 sequences at
N=28), then DFS with the bound |r_k| − (N−1−n) ≤ T, b_0 fixed, children ordered lexicographically, randomly,
or by the policy's probability. Budget 120 s per ordering; the policy step costs 7× per node (3.2k vs 22k
nodes/s). `results/p2/p2f_dfs_short.csv`.

| N, T | lexicographic | random | policy | node advantage of policy |
|---|---|---|---|---|
| 20, 2 (complete tree 182k nodes) | first at 4.2k | first at 15.0k | first at 11.6k | none |
| 24, 3 (20 solutions) | 26.3k nodes | 19.9k | 11.9k | 1.7 to 2.2× |
| 28, 3 (20 solutions) | 253k; first 78k | 410k; first 208 | 185k; first 28 | 1.4 to 2.2× |
| 32, 3 (transfer; per 120 s) | 13 in 2.76M | 9 in 2.62M | 6 in 381k | 3.4 to 4.7× per node; first solution at 56k vs 731k / 298k |
| 36, 3 (per 120 s) | 0 in 2.72M | 2 in 2.56M (first 403k) | 0 in 383k | inconclusive (policy budget below random's first hit) |

Reading: the learned ordering is consistently the most solution-dense per node from N=24 up, and the effect
survives one step beyond the training range (N=32), but it is a factor of 2 to 5, not the order of magnitude
needed to pay for a network evaluation per node. A compiled implementation would shrink the per-node overhead
(the GRU step is ~0.3 ms in PyTorch at batch 1; ~0.03 ms is realistic in C with batched node expansion), at
which point the policy ordering would win wall-clock by about the node factor. This is a legitimate but small
contribution ("learned branching for exact PSL search"), and it does not reach the regime the user cares
about (N ≥ 64, where exact search is infeasible in any ordering).

## 5. Within-length novelty at short lengths (P2h)

`results/p2/p2h_novelty.csv`, 20k samples per length vs. the shotgun search for the same wall-clock, counting
distinct canonical optimal sequences *not in the training set*:

| N, T | policy: optimal samples / distinct / new | new per second | shotgun: restarts / distinct / new | new per second |
|---|---|---|---|---|
| 22, 3 | 1679 / 242 / 58 | 4.6 | 2149 / 146 / 2 | 0.16 |
| 24, 3 | 2281 / 280 / 92 | 6.6 | 1876 / 234 / 7 | 0.51 |
| 26, 3 | 1063 / 95 / 0 | 0 | 1923 / 202 / 0 | 0 |
| 30, 3 (transfer) | 5 / 4 / 4 | 0.23 | 2322 / 78 / 78 | 4.5 |
| 32, 3 (transfer) | 5 / 4 / 0 | 0 | 2178 / 287 / 0 | 0 |

Inside the training range the policy generalises: it produces optimal sequences the 3-core-minute search had
not found, 10 to 30× faster than continued search finds new ones (which saturates as its found set grows).
One length beyond the range the search wins 20×, and by N=64 (P2b/P2c) the policy no longer reaches the
optimal class at all. Caveat: at N ≤ 74 the optimal sets are exhaustively enumerable, so this has no practical
value for binary codes; it would matter only if it scaled, and it does not.

## 6. Verdict on Direction A″

Closed as a method for N ≥ 64. Three facts decide it: the per-chip entropy floor of the feasible set (0.45
nats) means a factorised policy cannot be sharp; compounding errors without backtracking put samples at
PSL 10 where search gets 5 to 6 in the same time; and exact search with the learned ordering gains only
2 to 5× in nodes. What survives for the paper: the search-engine yield curves as the classical reference, the
novelty-and-transfer protocol, and the two small positive effects above, reported as such.
