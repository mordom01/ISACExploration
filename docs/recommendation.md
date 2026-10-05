# Recommendation after Phases 0 to 2

Date: 2026-10-05. Status: all Phase 2 runs complete. The DFS-ordering test showed a 2 to 5× node-count
advantage for the learned ordering (including one transfer length), not enough to change the recommendation;
details in `docs/phase2_interim.md` Sections 4 to 6.

## 1. What the evidence says

Across two families of learned generators and two problem classes, the per-instance classical method won every
matched comparison, and the learned models were diverse but not good:

| Problem | Learned model | Best classical at matched cost | Outcome |
|---|---|---|---|
| Continuous unimodular codes, N=64, analytic PSL/ISL | torus flow matching (6 variants incl. OT coupling, late-time weighting, Euclidean embedding, correlation features) | Adam on ISL from random phases: −19.7 dB at 8 steps (0.12 ms), −25.4 dB at 128 steps (1.7 ms) | model samples indistinguishable from random (−13.5 dB) at 1 to 32 NFE; even a 2-parameter family is not learned |
| Binary codes, N=64..96, PSL | chip-wise imitation policy (GRU on running autocorrelation), plus pruning, plus as search initialiser | shotgun local search: 100% at PSL ≤ 6 (16% at 5) in the same CPU-seconds | policy median PSL 10; with pruning 0.4 vs 21 distinct PSL-5 per second; as initialiser, no effect |

Mechanisms, each with a measurement behind it: feasible sets are high-entropy and spread like random points
(nearest-neighbour test; per-chip NLL at the entropy floor); the structure is a global quartic constraint that
per-element networks do not see and that gradient descent or exact local moves exploit for free; flow paths
from a uniform source are uninformative for 80% of the time axis; constructive sampling without backtracking
compounds small per-chip errors.

What did pass: diversity (every learned sampler produced 100% distinct, novel outputs), sampling cost (0.6 to
1 ms per sequence), all of the Phase 0/0b analytic results (set-ISL bound, regime map, CSK parity and the
simplex-optimality observation), and two small positive effects at short binary lengths: within-range
generation of new optimal sequences at ~10× the rate of continued search (N=22, 24), and a 2 to 5× node-count
reduction when the policy orders an exact DFS (N=24 to 32, including one transfer length).

## 2. Recommended paper

**Framing.** "When does learned generation beat per-instance optimisation for constant-modulus and binary
waveform design? Bounds, regime maps and controlled negative results." Target: IEEE Transactions on Signal
Processing if the bound and regime-map sections are developed to full rigour; otherwise IEEE Transactions on
Radar Systems or IEEE TAES, with an ICASSP/RadarConf short version. This is the "benchmark paper" (B″) with the
learned-sampler results as a section, which is what the user asked to keep.

**Headline contribution.** A quantitative map of the regime in which a learned waveform generator can add value,
built from (i) the aperiodic set-ISL floor N²K(K−1) and the fact that i.i.d. random sets sit 0.5 to 3 dB above
it (so set-level "diversity" claims must be made on windowed or peak quantities), (ii) the rejection-sampling
frontier and gradient-descent cost curves for continuous codes, (iii) search-yield curves for binary codes, and
(iv) the comm-side parity result for CSK codebooks including the simplex-optimal zero-lag target.

**Supporting contributions.** Controlled negative results with mechanisms (Sections 3 of the Phase 1 and 2
reports), a reusable evaluation protocol (matched wall-clock, distinct-canonical counts, train overlap,
transfer lengths), and the classical baseline library.

**Minimum experiment set a TSP reviewer will demand.** Everything already in `results/` plus: GPU timings for
the batched optimisers and the samplers on the same device; N up to 1024 for the continuous regime map; the
bits-per-pulse vs sensing-loss figure from the Phase 0b audit; a conditional WGAN re-implementation on the same
data so the GAN line is covered by the same protocol; and one positive-control problem where learning should
win (an expensive non-analytic objective such as CA-CFAR detection probability in clutter), so the paper is not
only negative.

**Likely objections.** "Negative results on small models and CPU compute" (answer: the mechanisms are
compute-independent and measured; report the GPU re-run), "the right generative model was not tried"
(answer: six flow variants, two policy variants, pruning, initialisation; the entropy-floor argument applies
to any factorised sampler; invite the reader to beat the published baselines with the released code),
"index-modulation rate is low" (answer: parity is the claim, see Phase 0b).

## 3. If the group prefers to keep building a learned method

The one setting the evidence does not close is objectives that are not analytic functions of the waveform:
detection probability through a CFAR detector in clutter, hardware or power-amplifier effects, or
set-level metrics that require simulation. There the per-instance optimiser loses its free gradient and its
cheap local moves, and a learned surrogate or critic (used to guide gradient descent or local search, not a
flow) is the natural tool. That is Direction C of Phase 0 and the positive-control experiment above; it should
be built on the classical search engines in `isacgen/` and evaluated with the same protocol.

## 4. Deliverables in the repository

- `docs/phase0_landscape_and_directions.md`, `docs/phase0b_comm_side_audit.md`, `docs/phase1_interim.md`,
  `docs/phase2_plan.md`, `docs/phase2_interim.md`, `docs/references.md` (every citation with verification status).
- `isacgen/`: metrics, classical optimisers (numpy and batched torch), binary search engine, torus flow
  matching, constructive policy.
- `experiments/`: 20 dated, never-mutated run scripts; `results/`: their tables, figures, logs and datasets.
