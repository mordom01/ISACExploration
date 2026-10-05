# Phase 1 interim report: the torus flow-matching smoke test failed, and why

Date: 2026-10-05. Scripts: `experiments/2026-10-05_p1*.py`; outputs in `results/p1/`. Everything below ran on a
4-core CPU container (no GPU); the model sizes were chosen for that and are stated.

## 1. Result in one paragraph

An unconditional flow-matching model on the flat 64-torus, trained for 6000 steps on 28k optimiser-generated
codes (median PSL −27.5 dB), produces samples with Haar-random statistics at every sampling budget from 1 to 32
Euler steps (median PSL −13.4 to −13.6 dB; training distribution −27.5 dB; Haar −13.6 dB). The gate from Phase 0
Section 6 (median within 1 dB of training at 8 NFE) is failed by 14 dB. Six follow-up probes show the failure is
not a bug and not specific to optimiser-generated data: the same model family fails on a two-parameter
analytic family (P4 under global phase and linear ramp) with a transformer, with an MLP, on the torus, in a
Euclidean cos/sin embedding with a Gaussian source, with late-time weighting, and with correlation-domain
input features. The only target learned was a point mass. Meanwhile, plain Adam gradient descent on the
analytic ISL from the same Haar start reaches median PSL −19.7 dB in 8 steps (0.12 ms per code, one CPU thread)
and −25.4 dB in 128 steps (1.7 ms), with perfect diversity by construction. For continuous-phase codes with
analytic correlation objectives, the generative-prior idea is dominated on quality, cost and diversity, and the
density-matching formulation is also the hardest possible learning problem for this data. The hypothesis in
the task statement is falsified in its naive form at N=64.

## 2. What was run

| ID | Experiment | Outcome |
|---|---|---|
| P1a | 30k codes, N=64: CAN from Haar + l_p PSL polish, canonical global phase (4 cores, 805 s) | median PSL −27.49 dB, 92.8% ≤ −25 dB |
| P1b | Torus CFM, 4-layer transformer (0.84M params), uniform source, wrapped-geodesic path, 6000 steps, bs 256, lr 3e-4 cosine, random global-phase augmentation | loss flat at 3.29 = π²/3 (zero-velocity floor) from step 0; samples Haar-level at NFE 1, 2, 4, 8, 16, 32; nearest-neighbour distance to training set identical to Haar's (10.93 vs 10.92) |
| P1c | Adam on analytic ISL from Haar, batch 1000, lr ∈ {0.03, 0.1, 0.3} | median PSL after k steps (lr 0.3): −16.2 (1), −19.7 (8), −23.4 (32), −24.8 (64), −25.4 (128), −25.6 dB (256); 0.016 ms per code per step on one thread |
| P1d | Control: same transformer on the 2-parameter P4 family, lr 1e-3, 800 steps, in-flight sampling | Haar-level through step 600 (run still finishing; no learning) |
| P1e | PSL of the interpolant φ_t along the wrapped-geodesic path | structure survives only for t > 0.85: median PSL −22.8 dB at t=0.9, −20.6 at 0.85, −18.9 at 0.8, −16.5 at 0.7, Haar (−14.2) at 0.5 |
| P1f | Discrete alphabets, N=64: Haar M-PSK, CAN+quantise, ISL coordinate descent (CD) from random and from CAN+quantise, continuous CAN | mean PSL (dB): M=2: random −11.8, CAN+quant −13.5, CD −17.8 / −18.3, continuous −24.6. M=4: −13.6, −17.7, −19.8 / −20.1, −24.6. M=8: −13.2, −21.9, −21.8 / −22.9, −24.6. CD costs 17 to 164 ms per code |
| probes (scratch, not committed as experiments) | MLP on point-mass target; MLP on P4 family; MLP with t ~ Beta(3,1); Euclidean cos/sin FM with Gaussian source; MLP with autocorrelation and spectrum input features | point mass: loss 3.3 → 0.55, samples approach the code (min distance 0.87) but median PSL only −16.4 dB; all P4-family variants: loss stays at the floor, samples Haar-level after 1500 to 3000 steps |
| P1b2 | Torus CFM with minibatch OT coupling, no augmentation, t ~ Beta(3,1), lr 1e-3, 3000 steps, in-flight evaluation | running at the time of writing; result appended in Section 6 when available |

## 3. Why it fails (mechanism, with the measurements that support each step)

1. **Per-chip marginals of good codes are uniform; all structure is in the correlation functional.** The nearest
   neighbour of a Haar sample in the training set is as close as training points are to each other (10.93 vs
   10.81, wrapped L2 over 64 chips; independent uniform would give about 14.5). Optimised codes are scattered
   over the torus like random points; the feasible set is thin but spread out, exactly as the Phase 0 regime
   map implied (acceptance < 10⁻⁶ at −22 dB).
2. **The flow path is uninformative for most of the time axis.** P1e: the interpolant has Haar-level sidelobes
   for t ≤ 0.5 and loses most structure by t = 0.8. The conditional velocity u = wrap(φ₁ − φ₀) is therefore
   unpredictable from φ_t on roughly 80% of the training time axis, and the regression loss floor π²/3 is
   irreducible there. Reweighting t toward 1 does not fix it (probe: Beta(3,1) sampling, loss 3.20 to 3.28).
3. **Even where the path is informative, the required inference is hard.** To predict u near t ≈ 1 the network
   must infer the hidden structure of φ₁ from φ_t. For the P4 family that is a joint (offset, ramp) estimate
   from 64 wrapped noisy phases: a periodogram-like frequency estimation. Neither an MLP nor a 4-layer
   transformer learns it in 3000 steps, with or without autocorrelation and spectrum features as inputs. For
   optimiser-generated codes the hidden structure is "flat 2N-point spectrum", a quartic constraint, and the
   inference is the optimisation problem itself.
4. **Density matching is the wrong objective for this data.** Flow matching tries to cover the whole
   high-entropy feasible set with a learned transport from Haar. The earlier conditional WGAN never had to:
   a GAN generator is a forward map z → x judged by a critic, i.e. amortised optimisation with a learned
   objective and no inference step; mode dropping is tolerated. That is why the GAN reached 86% pass on
   parametric families and why "native diversity through coverage" (the premise in the task statement) is a
   burden here, not an advantage.
5. **The analytic gradient makes every learned prior redundant for continuous phases.** P1c: one FFT-sized
   gradient step costs ~0.016 ms per code and 8 steps already beat −19 dB. A transformer evaluation costs
   ~1 ms per code on CPU (both will scale similarly on GPU, the ratio stays ~50 to 100×). A learned sampler at
   k NFE must beat GD at ~100k steps to break even on wall-clock. For k = 8 that means beating −25 dB median
   with full random-restart diversity. No amortised model is going to do that at N = 64 to 256.

## 4. What this does and does not falsify

- Falsified: "flow matching from the Haar measure on the torus is the right generator for continuous
  constant-modulus codes under analytic correlation objectives." Also weakened: any density-matching
  generative model (diffusion included) for this target; the Euclidean cos/sin probe failed identically.
- Not falsified: generative or learned components where the objective is not an analytic function of the
  phases, or the alphabet is discrete, or the per-instance optimiser is itself slow and weak. P1f is the
  evidence for the discrete case: binary local search sits 6.3 dB above the known optimum PSL = 4
  (−24.1 dB, 1859 optimal length-64 codes, Coxson & Russo IEEE TAES 2005) and 6.8 dB above continuous CAN,
  while costing more than CAN. QPSK: 4.6 dB gap. 8-PSK: 1.7 dB gap.
- Unchanged from Phase 0 and 0b: the set-ISL bound, the regime map, the CSK comm-side parity result and the
  classical baseline library all stand and are reusable.

## 5. Re-ranked directions (replaces Phase 0 Section 4)

**A″ (recommended): learned constructive sampler for discrete-phase sequence sets, distilled from heavy search
and fine-tuned on the exact objective.** Setting: binary / QPSK codes and codebooks (the Tedesso & Romero CSK
setting, and Eedara et al.'s GA setting). Model: an autoregressive chip-by-chip policy whose state is the
partial sequence's running autocorrelation (the physics-informed state the probes showed is needed), trained
first by imitation on optimal or near-optimal codes (exhaustive-search sets exist up to length 74 for binary;
PSL-objective tabu / evolutionary search for longer and for QPSK), then by reward fine-tuning on −PSL and
set-level windowed cross-correlation. This avoids the inference problem of density matching (the policy never
has to invert a noisy interpolant), targets a regime where per-instance search is slow and 4 to 7 dB
suboptimal, and keeps the ISAC codebook framing and comm-side audit intact. Closest prior to verify before
claiming novelty: Rezaei, Ahmadi, Naghsh, Aubry, Nayebi, De Maio, "A learning-inspired strategy to design
binary sequences with good correlation properties: SISO and MIMO radar systems," arXiv:2305.08936 (IEEE
Xplore 10138370); method not yet read. Gate: at N=64, M=2, sampled codes reach PSL ≤ 5 (−22.1 dB) on ≥ 50% of
samples at ≤ 1 ms per code, with ≥ 10³ distinct codes per 10⁴ samples; coordinate descent reaches PSL 5 on
< 10% in P1f-type runs. Compute: data from search, hours on CPU; policy training hours on the 5070 Ti.

**B″: the benchmark paper.** The regime map, the set-ISL bound and its consequence for "diversity", the
random-restart GD cost curves, the CSK parity result and the failure analysis of Section 3 form a methodology
and bounds paper on their own ("when can learned generation beat per-instance optimisation for unimodular
waveform design, and how to measure it"). Lower novelty risk, lower ceiling; fits a letters or ICASSP/SPAWC
venue, or the evaluation half of A″.

**C″: learned critic as reward for a non-analytic sensing metric**, used to guide GD or the A″ policy (not a
flow). Candidate metric: CA-CFAR detection probability in a two-target scene. Supporting only.

**Dropped:** continuous-phase flow / diffusion priors (A, B of Phase 0) unless N ≥ 1024 and the objective is
non-analytic, which is a different paper. Guidance with analytic objectives on a learned prior (Phase 0 B) is
pointless when GD on the same objective from Haar is cheaper and better.

## 6. Pending at the time of writing

P1d (control, 800 steps) and P1b2 (OT coupling, late-time weighting) are still running; their final lines will be
appended here. Expectation from the probes: both Haar-level.
