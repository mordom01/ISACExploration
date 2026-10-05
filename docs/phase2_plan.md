# Phase 2 plan: learned constructive sampler for low-PSL discrete-phase sequences (Direction A″)

Date: 2026-10-05. Decision: pursue A″; the continuous-phase results of Phase 0/1 become the benchmark section.

## 1. Problem

Binary (later QPSK) sequences b ∈ {±1}^N with small aperiodic peak sidelobe level PSL(b) = max_{k≥1} |Σ_n b_n b_{n+k}|,
and codebooks of K such sequences with windowed cross-correlation constraints (the CSK / index-modulation ISAC
setting audited in Phase 0b). Facts that define the regime:

- Optimal PSL is known by exhaustive search up to N = 74 (Leukhin & Potekhin 2013): PSL = 4 for 64 ≤ N ≤ 82
  region reported in the literature; for N = 64 there are 1859 balance-inequivalent optimal codes (Coxson &
  Russo 2005). Beyond N ≈ 74 only best-known values exist (Nunn & Coxson 2008; Dimitrov et al. 2020/2021).
- Cheap local search is far from optimal: PSL-objective coordinate descent from random starts never reached
  PSL ≤ 5 in 300 restarts at N = 64 (P1g, best 6, mean −18.9 dB vs optimum −24.1 dB), at 2.6 ms per restart.
  Stochastic search with sideways moves and tabu (Dimitrov-style) reaches lower PSL but at a cost that grows
  with N; its yield of distinct optimal codes per second is the quantity to beat.
- No gradient: the continuous-phase shortcut that dominated Phase 1 (Adam on the analytic ISL) does not apply.

## 2. Hypothesis and mechanism

**H1 (constructive policy).** A stochastic policy π_θ(b_n | state_n) that emits chips one at a time, with
state_n = (running partial autocorrelations r^{(n)}_k = Σ_{m<n} b_m b_{m+k} restricted to m+k < n, position n,
length N), trained by maximum likelihood on sequences found by heavy search (imitation), samples sequences whose
PSL distribution is close to the search outputs at a per-sample cost of N policy evaluations, and yields many
distinct sequences because it is a sampler, not an optimiser.

Why the state is sufficient: the final r_k is the running r_k plus future contributions; exhaustive searches
prune on exactly these partial sums (Coxson & Russo 2005 use partial-autocorrelation bounds). The policy learns
a stochastic version of the branching heuristic. Unlike the Phase 1 flows, no inference over a noisy
interpolant is needed: the state is exact and the target (next chip) is a one-bit decision.

**H2 (transfer).** Trained on lengths 32 ≤ N ≤ 72 with N as an input, the policy produces at N ∈ {80, 96, 128}
sequences with PSL within 1 of the best-known values more often per unit time than restarted local search.
Mechanism: the branching heuristic is local in the partial-correlation state and does not depend on N except
through the remaining length.

**H3 (reward fine-tuning).** Fine-tuning π_θ with REINFORCE on −PSL (plus an entropy bonus) improves the tail
(fraction at optimal PSL) without collapsing diversity, because the reward is exact and cheap.

**H4 (codebooks).** The same policy, conditioned on previously emitted sequences' cross-correlation state,
produces codebooks with windowed cross-correlation below the i.i.d. level at matched per-code PSL, which is
the set-level claim of Phase 0 transplanted to the discrete alphabet.

## 3. Falsifiers

- F1: imitation samples are no better than random restarts of the search truncated to the same wall-clock.
- F2: samples are near-copies of training sequences (canonical-form overlap > 20%) or the number of distinct
  canonical sequences per 10⁴ samples is < 10³.
- F3: no transfer: at N = 96 the policy's PSL distribution is no better than local search at matched time.
- F4: reward fine-tuning collapses diversity (distinct count drops by > 5×) for < 1 PSL unit of gain.

## 4. Gates (pass/fail, N = 64 unless stated)

| Gate | Criterion |
|---|---|
| G1 data | search engine yields ≥ 2000 distinct canonical sequences with PSL ≤ 5 at N = 64 within 1 CPU-hour. (Measured 2026-10-05: 4 to 5 PSL-5 sequences per second per core; PSL 4 was not reached in 900 restarts nor by 7890 iterated-local-search kicks in 4 minutes, so PSL 4 at N = 64 is out of reach of this search and is a stretch target for the policy, not a data requirement.) |
| G2 imitation | ≥ 50% of policy samples have PSL ≤ 6 and ≥ 10% have PSL ≤ 5 at N = 64, at ≤ 1 ms per sequence (batched), ≥ 10³ distinct canonical sequences per 10⁴ samples, ≤ 20% overlap with training set (for reference: random sequences have PSL ≤ 6 with probability ≈ 0 and coordinate descent reaches PSL 6 in 8% of restarts, P1g) |
| G3 matched-time | at equal wall-clock, policy sampling (optionally + one local-search polish) finds more distinct PSL ≤ 5 sequences than restarted search |
| G4 transfer | at N = 96 (not in training), median policy PSL ≤ median of restarted search at matched time, and ≥ 1% of samples within 1 of the best-known PSL |
| G5 codebooks | K = 8 codebooks with windowed cross-correlation (|lag| ≤ 3) peak ≤ 8/64 on ≥ 50% of codebooks at per-code PSL ≤ 6 |

## 5. Baselines (all classical, all implemented in-repo)

Random; CAN + sign quantisation; PSL coordinate descent (P1g); Dimitrov-style shotgun search with sideways
moves and tabu (this phase, `isacgen/binseq.py`); Gold / Kasami for codebooks; best-known PSL tables from the
literature as reference lines. Learned baseline: BiSCorN-style per-instance network optimiser (Rezaei 2023),
re-implemented minimally only if G2/G3 pass.

## 6. Compute and order of work

1. Search engine + yield measurement (CPU, minutes to an hour) → G1.
2. Training set: N ∈ {32, 40, 48, 56, 64, 72}, a few thousand distinct canonical sequences each at best
   reachable PSL (CPU hours, 4 cores).
3. Policy: GRU or small transformer over the running-correlation state (≈0.5M params), imitation training
   (CPU: tens of minutes; GPU: minutes) → G2, G3.
4. Transfer evaluation at N = 80, 96, 128 → G4.
5. REINFORCE fine-tuning → H3; codebook conditioning → G5.
