# Phase 3: Covert ISAC as constrained distribution design (the application paper)

Date: 2026-10-06. Scripts `experiments/2026-10-06_p3*.py`; outputs `results/p3/`. CPU container, 4 cores, a few hours.

## 1. Why this is the generative-AI paper and the earlier directions were not

Phases 1 and 2 established that for any per-waveform objective (PSL, ISL, windowed cross-correlation, SER) a
per-instance optimiser or search beats every learned generator at matched cost. Covertness is different in
kind: the requirement is that the *law* of the emitted frames be close to the law of the background traffic,
so that the best detector cannot tell them apart. That is a constraint on a distribution, not on a waveform.
No optimiser expresses it; a sampler does. The strengths of generative models that matter here are exactly the
ones measured in Phases 1 to 2: every learned sampler produced 100% distinct, novel outputs at ~1 ms or less,
and they are the only objects that can realise a target law that is known only through samples.

## 2. Theory (new, simple, and the paper's spine)

Let P be the background law over frames (here GMSK, N=64 samples, 2 samples/symbol, BT=0.3, i.e. uniform over
2^40 symbol patterns through a fixed modulator) and c(x) ≥ 0 a sensing cost (mainlobe-excluded normalised ISL).
Among all laws Q with KL(Q‖P) ≤ ε, the minimiser of E_Q[c] is the exponential tilt

    Q_β(x) ∝ P(x) exp(−β c(x)),   KL(Q_β‖P) = −β E_β[c] − log E_P[e^{−β c}],

by the Gibbs variational principle. Consequences:

1. **Optimal covert waveform law = Boltzmann distribution of the background at inverse temperature β.** The
   covertness budget picks β; the achievable (sidelobe, divergence) pairs form a frontier.
2. **"Search inside the covert family" is a truncation Q_t = P(·|c ≤ t)**, with KL = −log P(c ≤ t). At equal
   KL its expected cost is higher than the tilt's (measured below), and its hard edge is a signature.
3. **Information-bearing codebooks are seed-shared draws from Q_β**: K draws per frame from a PRNG seed known to
   transmitter and receiver (an implicit codebook that changes every frame, so no public family exists to
   enumerate). The comm side is CSK with a lag-searching correlation receiver, as audited in Phase 0b.
4. **Realising Q_β needs a sampler.** Importance reweighting of background samples collapses (ESS) beyond
   KL ≈ 5 nats; single-flip MCMC freezes beyond β ≈ 20 at this N; a trained generator is the amortised sampler,
   and because Q_β must retain most of the background's entropy (27.7 nats here) to be covert, the target is in
   the high-entropy regime where learned samplers are feasible (unlike the near-zero-entropy optimal-PSL sets of
   Phase 2).

## 3. Measurements

### 3.1 Frontiers (`p3b_frontier.csv`, `p3d_boltzmann.csv`)

Importance-weighted tilt and truncation over 10⁵ GMSK frames, plus exact Metropolis MCMC for the tilt
(8000 chains, 25 sweeps). Detector = fresh 1-D CNN trained post hoc (6000 vs 6000), AUC on held-out 2000 vs 2000.

| law | KL to background (nats) | expected PSL, mainlobe excluded (dB) | detector AUC |
|---|---|---|---|
| background (uniform GMSK) | 0 | −9.6 | 0.50 |
| tilt β=5 (IS / MCMC) | 0.94 / 1.36 | −11.6 / −11.6 | 0.61 / 0.60 |
| tilt β=10 (IS / MCMC) | 2.15 / 2.63 | −12.5 / −12.5 | 0.65 / 0.66 |
| tilt β=20 (IS / MCMC) | 4.50 / 4.85 | −13.6 / −13.6 | 0.73 / 0.68 |
| truncation, best 10% | 2.30 | −12.2 | 0.64 |
| truncation, best 1% | 4.61 | −13.4 | 0.69 |
| truncation, best 0.5% | 5.30 | −13.6 | 0.73 |
| **searched GMSK, best of 2000 per message** (classical covert-family search, P3a baseline) | ≈ 7.6 | −15.5 | **0.998** |
| CAN-optimised unimodular codes (not covert) | n/a | −24.3 | 1.000 |

Reading: (i) the tilt gives lower expected cost than truncation at matched KL (0.676 vs 0.721 at ≈2.2 nats,
0.512 vs 0.549 at ≈4.5 nats, in cost units), as the theory says; the learned detector finds the two about
equally detectable at equal KL, so the tilt's advantage is in sensing per unit of covertness budget.
(ii) The classical approach of picking the best-sidelobe member of the family is *not covert*: AUC 0.998.
(iii) MCMC rows with β ≥ 40 are unequilibrated (PSL stops improving, thermodynamic-integration KL becomes
inconsistent) and are excluded; the frontier beyond β ≈ 20 needs tempered chains or a trained sampler.

### 3.2 Learned samplers

| sampler | result |
|---|---|
| Free-phase GAN (P3a): MLP phase increments, adversary D, cooperative receiver R, ISL penalty λ=0 | matches background PSL (−8.7 dB median), carries bits (SER 0.003 at 8 dB with seed-shared codebook), but **held-out detector AUC 1.000**: artefacts an adversary learns |
| Structured GAN (P3c): symbols through the GMSK modulator (straight-through), so every output is a legal frame | **AUC 1.000** after 30 steps: a latent-variable generator cannot cover a uniform law over 2^40 patterns; the detector learns its low-entropy support |
| Autoregressive Boltzmann generator (P3d), reverse-KL (REINFORCE) to Q_β, 2500 CPU steps | β=10: KL(q‖P) 1.08, residual KL(q‖Q_β) 1.98 nats, PSL −11.0 dB, AUC 0.69, SER = random GMSK (0.003 at 8 dB), **0.036 ms/frame**. β=40: mode collapse (KL(q‖P) 17.7, AUC 0.99, seed-shared draws too similar: SER 0.21 at 12 dB). β=160: collapsed |

Reading: the autoregressive sampler is the right object (it reaches the covert regime with bits intact at
sub-0.1 ms per frame) but is under-trained at low β and mode-collapsed at high β, the two textbook pathologies
of reverse-KL training. The known remedies are forward-KL on MCMC samples (available from P3d), α-divergence
or entropy-annealed objectives, and more compute (GPU hours instead of CPU minutes). The GAN formulations are
closed: against a maximum-entropy background a GAN loses to a fresh detector by construction.

## 4. The paper (non-incremental version)

**Title direction.** "Covert integrated sensing and communication by constrained distribution design: Boltzmann
generators for traffic-mimicking waveforms."

**Headline.** The covert-optimal emission law is the exponentially tilted background; the (sidelobe, divergence,
detectability) frontier is computable; the classical covert-family search sits far off it (AUC 0.998); and an
amortised autoregressive Boltzmann generator samples the tilted law at < 0.1 ms per frame with seed-shared
implicit codebooks, so there is no public code family to enumerate. Generative modelling is not an alternative
optimiser here; it is the only way to realise the optimal law from traffic data.

**Supporting contributions.** (1) The Gibbs characterisation and the tilt-vs-truncation gap. (2) Detectability
frontier against learned detectors, with the square-root-law context from covert communications. (3) Seed-shared
implicit codebooks: CSK rate, SER with timing uncertainty, and an enumeration-attack argument. (4) Sensing side:
mainlobe-excluded PSL/ISL and a range-sidelobe-modulation analysis across frames (each frame is a different
draw). (5) Generality: the same construction for any background available as data, including non-constant-
envelope traffic (OFDM) under a PAPR constraint, which no CPM-based covert design can do.

**Minimum experiment set.** Frontier at N ∈ {64, 256} for GMSK and OFDM-QPSK backgrounds; tempered MCMC to
β ≫ 20; forward-KL-trained autoregressive and masked-diffusion samplers vs the MCMC frontier (KL to Q_β,
AUC, PSL); detectors of increasing strength (energy, cyclostationary, CNN, transformer) and their AUC vs KL;
CSK SER vs Eb/N0 with timing and frequency offsets; radar P_d in a two-target scene at fixed P_fa; wall-clock
per frame on GPU; baselines: random background frames, best-of-M family search (Tedesso-Romero-style CSK with
GMSK/Gold), Ziemann-Metzler GAN (free-phase; expect AUC → 1 against a fresh detector), Wang et al. 2025 covert
DFRC.

**Likely objections and answers.** "A GAN was already used for LPD radar": a fresh detector separates it; our
object is the optimal law, not a GAN. "KL is not detectability": we report both and the frontier shows their
relation. "Learned sampler under-performs MCMC": true at CPU scale; the paper needs the forward-KL-trained
version on GPU to close the 2-nat gap, which is the main remaining engineering item. "Rate is low": CSK parity,
as in Phase 0b; the contribution is covertness per unit sensing, not rate.

## 5. What is left to do (ordered)

1. Train the autoregressive sampler by forward KL on MCMC samples (data exist from P3d) and by entropy-annealed
   reverse KL; target KL(q‖Q_β) < 0.3 nats at β ∈ {10, 20}. GPU: hours.
2. Tempered MCMC for β up to 200 to extend the exact frontier; verify the detector AUC curve.
3. OFDM background with PAPR constraint (drop constant modulus; add PAPR to the energy).
4. Stronger detectors and the detectability-vs-KL study; security argument for seed-shared codebooks.
5. Sensing-side system metrics (two-target P_d, range-sidelobe modulation across frames).
