# Phase 0: Landscape, premise audit, and ranked directions for generative ISAC waveform synthesis

Date: 2026-09-24. Author: research-collaborator agent for M. Qi (USC, Pedram / Kamal). Scope: generative backend only.

Citations use the keys in `docs/references.md`; every key there was verified on the date above. Numbers quoted
below come from `experiments/2026-09-24_p0{a,b,c}_*.py` with outputs in `results/p0/`.

---

## 0. Executive summary

1. **The direct competition is thinner than the prompt assumes, and one premise is wrong.** No published work
   uses diffusion or flow matching to synthesise constant-modulus radar or ISAC waveforms under sidelobe /
   cross-correlation / spectral constraints. "CoSMIC" (Bai2026) uses rectified flow at the *receiver* for
   semantic recovery on a chirp carrier, not for waveform generation. RF-Diffusion (Chi2024) generates
   Wi-Fi / FMCW / 5G signals for downstream sensing tasks, not constraint-conditioned design. The real
   competition is the Aalto conditional-WGAN line (Saarinen2020/2021/2023/2024) and Ziemann2025. Both are GANs.
2. **Three measured facts reshape the directions** (Section 2):
   - The total ISL of any set of K unimodular length-N sequences is bounded below by N²K(K−1); i.i.d. random
     sets sit only 0.5 to 3 dB above it, and Multi-CAN reaches it exactly. **Mean cross-correlation cannot be
     driven below the i.i.d. level.** The prior GAN's "0.09-0.15 mean cross-correlation matching real code
     families" is the i.i.d. level (0.109 at N=31, 0.075 at N=64). Slot embeddings cured collapse; they did not
     produce designed orthogonality. A set-level generative contribution must therefore target what joint design
     *can* change: peak cross-correlation, and cross-correlation in a lag window or at zero lag (which joint
     design drives from 0.25 to 0.002 at N=64, K=4).
   - Rejection sampling from the uniform distribution on the torus reaches PSL ≈ −17 dB (N=64) at 10⁻³
     acceptance, while CAN plus an l_p polish from a random start reaches −27 dB (Frank-64 level) in ~50 ms
     single-threaded. A learned sampler
     is only justified in the regime between those numbers and beyond, and must be benchmarked against
     random-restart optimisation, not only against a single optimiser run.
   - At N ≤ 256 the classical optimisers are fast (tens of ms single-threaded, sub-ms if batched on GPU). The
     "latency compatibility" argument for few-step flows is weak on its own. The defensible advantages of a
     generative model are (i) diversity at matched quality, (ii) inference-time composition of constraints
     without deriving a new algorithm, (iii) objectives that are non-differentiable or expensive.
3. **Recommended headline** (Direction A): joint generation of *waveform sets* on the (KN)-torus with a
   set-level guidance potential, framed as codebook synthesis for index-modulation ISAC, benchmarked against
   Multi-CAN / Multi-WeCAN, ADMM, random-restart manifold optimisation, and the conditional WGAN. Direction B
   (a single trained torus-flow prior plus zero-shot constraint composition, with an explicit "when is a
   generative model worth it" regime map) is the supporting framework and second experiment block.
4. Flow matching is the right *training* paradigm, but not for the reasons listed in the prompt (Section 3.1).
   The proposal's cGAN-to-CFG progression and the "Unified Conditioning Space" are the wrong emphasis for
   constraints that are differentiable functionals of the waveform (Section 3.4).

---

## 1. Landscape map

### 1.1 Generative synthesis of radar / ISAC waveforms (direct competitors)

| Work | Model | Output / constraint | Diversity handling | Baselines | Gap we can exploit |
|---|---|---|---|---|---|
| Saarinen2020, Saarinen2021 (Aalto) | WGAN(-GP), later variable length | Unimodular phase codes with AF shape; constant modulus by construction | none / implicit | none quantitative | no conditioning on numeric specs, no sets |
| Saarinen2023 (Aalto) | GAN for MIMO sets | sets of codes | not reported (abstract) | Multi-CAN-type | venue not confirmed; see references |
| Saarinen2024 (Aalto, arXiv) | conditional WGAN on off-diagonal of target R (M=10, N=41) | beampattern via X^H X = R | measured by Frobenius similarity; authors state GAN covers less of the training distribution than the data; diversity flagged as future work | Multi-CAO; 0.002 s vs 0.098 s | conditioning is on R only; diversity open; no ISAC comm metric |
| Ziemann2025 (UMD, IEEE TRS) | adversarial generator vs critic on ambient RF + AF loss | LPD waveforms matching background distribution | distributional by design | classical LPD waveforms | different objective (covertness); no constraint conditioning |
| radarwgan (ours, unpublished) | conditional WGAN-GP, conformer generator, AF-image critic, slot embeddings, CcGAN vicinal weighting | Frank/Oppermann/P4 families, PSL threshold, MIMO slots | slot embeddings restore i.i.d. level | none classical yet | see Section 2.1 |
| Luo2026 | causal temporal transformer, MI loss | orthogonal MIMO sets | not generative (as far as the abstract shows) | unclear | low threat |

Assessment: nobody has (a) a distributional model whose set-level statistics beat the i.i.d. floor on the
right metric, (b) a controllable numeric conditioning axis with measured monotonicity, (c) inference-time
constraint composition, or (d) an honest comparison against random-restart optimisation at matched wall-clock.
Any two of these are a paper if executed with TSP-level rigour.

### 1.2 Generative models in wireless (adjacent)

RF-Diffusion (Chi2024) is the reference for "diffusion on complex-valued RF signals" and should be cited as the
Euclidean, unconstrained alternative: it does not enforce constant modulus, and its outputs are not scored on
AF metrics. Bai2026, Chen2025scoring, Jiang2025 and the two 2025 surveys establish that generative models in
ISAC are used for channel/semantic/sensing inference, not for transmit-waveform design. That absence is the
opening.

### 1.3 ML tools that transfer

- Flow matching on the flat torus is a solved primitive (Chen2024rfm includes flat-torus experiments; Jing2022
  did score-based diffusion on the hypertorus). On a flat torus the Riemannian machinery reduces to wrapped
  angle differences, so implementation cost is near zero (TorchCFM, Tong2024, plus a wrap).
- Hard-constraint sampling for flows: Utkarsh2025 (PCFM, zero-shot projection during sampling), Liang2025
  (chance-constrained), GCFM2025. These are the ML-side prior art for "spectral mask / PAPR at inference."
- Reward / non-differentiable guidance: Li2024svdd (derivative-free, works with exact PSL), Chung2023 (gradient
  guidance through the x1-prediction), DomingoEnrich2025 and Jensen2026 (reward fine-tuning of flows). These
  are the correct successors of the "critic as implicit AF signal" idea.
- Set-level sampling: Corso2024 (particle guidance) is the exact mechanism for joint non-i.i.d. sampling with a
  pairwise potential. Our delta is the potential (windowed cross-correlation energy on the torus) and the
  benchmark (Multi-WeCAN).
- Few-step: Geng2025 (MeanFlow) and Woo2026 (Riemannian MeanFlow) give one-step flow maps on manifolds.

### 1.4 Classical baselines (must appear in the paper)

Single sequence: CAN / WeCAN (Stoica2009); MM for WISL and l_p-PSL (Song2016a); complex-circle manifold
gradient methods (Alhujaili2019). Sets: Multi-CAN / Multi-WeCAN (He2009), MM sets (Song2016b), consensus
ADMM/PDMM (Wang2021). ISAC: Liu2018 (MIMO DFRC trade-off), Liu2021slp (symbol-level precoding, Swindlehurst),
Huang2020 (index modulation, MAJoRCom), Lee2024 (constant-modulus DFRC with space-time sidelobes, ADMM+MM),
Krish2023 (deep unfolding for constant-modulus JCAS). The random-restart versions of CAN and manifold GD are
the *diversity* baselines and are the ones most likely to embarrass a generative model.

---

## 2. Phase 0 measurements (CPU, numpy, minutes)

### 2.1 Set-level diversity is bounded (P0b, `results/p0/p0b_set_stats.csv`, N=64, 30 trials)

With z_p the K-vector of 2N-point DFT values at frequency p, the total set ISL (auto sidelobes plus all
ordered cross-correlation energies) is

    sum_k ||R_k − N I δ_k||_F² = (1/2N) sum_p (||z_p||⁴ − 2N||z_p||² + N²K),   sum_p ||z_p||² = 2N·NK,

minimised at ||z_p||² = NK for all p, giving the floor N²K(K−1). For i.i.d. random sets the expectation is
N²K² − KN, so the entire gap to the floor equals the expected auto-sidelobe energy. Consequence: joint design
can remove auto sidelobes but cannot reduce cross-correlation energy below ~N² per ordered pair on average.
Multi-CAN reaches the floor to four digits.

| K | method | worst-pair peak x-corr (all lags) | mean x-corr | worst zero-lag | worst peak, lags ≤3 | worst PSL (dB) | total ISL / floor (dB) |
|---|---|---|---|---|---|---|---|
| 4 | i.i.d. random | 0.277 | 0.075 | 0.174 | 0.247 | −12.1 | 1.22 |
| 4 | i.i.d. CAN | 0.290 | 0.074 | 0.202 | 0.254 | −22.9 | 0.11 |
| 4 | Multi-CAN | 0.246 | 0.068 | 0.145 | 0.202 | −14.8 | 0.00 |
| 4 | window-GD (ours, weighted set objective) | 0.295 | 0.072 | **0.001** | **0.002** | −19.3 | 0.26 |
| 8 | i.i.d. random | 0.319 | 0.074 | 0.251 | 0.300 | −11.5 | 0.54 |
| 8 | Multi-CAN | 0.288 | 0.071 | 0.219 | 0.257 | −12.8 | 0.00 |
| 8 | window-GD | 0.338 | 0.071 | **0.003** | **0.005** | −16.0 | 0.12 |

Reading: mean cross-correlation is 0.07 for every method (i.i.d. level, analytic 0.074). Multi-CAN buys 1 to
2 dB in all-lag peak cross-correlation and pays for it in per-sequence PSL. A windowed objective (zero
cross-correlation for |lag| ≤ 3 plus auto ISL) is where joint design gives two orders of magnitude over
i.i.d. while keeping PSL near −20 dB. **The set-level generative target should be windowed / zero-lag
orthogonality plus auto PSL, not "mutually low cross-correlation" in the mean sense.** The radarwgan
numbers (0.09-0.15 mean) equal the i.i.d. reference (0.109 at N=31, 0.075 at N=64); the open issue in that
work about "Oppermann dominating" is moot at the mean level, and the report should switch to worst-pair peak
and windowed metrics.

### 2.2 Regime map for a learned sampler (P0a, `results/p0/p0a_regime_map.png`)

| N | random-phase PSL median (dB) | PSL at 10⁻³ acceptance (dB) | best of 200k random (dB) | CAN from random: mean / best of 200 (dB) | + l_p polish: mean / best (dB) | time / solve (CAN + polish) |
|---|---|---|---|---|---|---|
| 31 | −11.4 | −16.1 | −18.3 | −21.4 / −25.1 | −23.4 / −26.8 | 20 + 19 ms |
| 64 | −13.5 | −17.3 | −18.8 | −24.8 / −27.5 | −27.4 / −29.5 | 30 + 21 ms |
| 128 | −15.7 | −18.9 | −20.0 | −27.2 / −29.7 | −30.6 / −33.0 | 44 + 29 ms |
| 256 | −18.0 | −20.7 | −21.7 | −29.8 / −32.2 | −32.8 / −35.4 | 81 + 53 ms |

(Frank-64: −27.8 dB, P4-64: −24.4 dB. Times are single-threaded numpy under CPU contention; treat as upper bounds.)

Reading: below roughly −17 dB at N=64 rejection sampling from the Haar measure is free and maximally diverse;
above −22 dB it is impossible (acceptance < 10⁻⁶) and the feasible set can only be reached by optimisation,
which from random starts lands at −25 to −29 dB with 50 ms of single-threaded CPU per waveform, i.e. every
random restart is a fresh, diverse, Frank-quality code.
A learned sampler therefore (a) must be trained on optimiser outputs, i.e. it is an amortised, distilled
optimiser, and (b) must be evaluated at PSL targets tighter than the rejection frontier, otherwise a
reviewer will rightly ask why a model is needed. The prior GAN's 86% pass rate at family-dependent thresholds
should be re-read against this frontier.

Note: the l_p PSL polish in `isacgen/classical.py` is an IRLS surrogate with a best-iterate guard, not the MM
algorithm of Song2016a; it already beats Frank-64 on average (−27.4 vs −27.8 dB best-of-one) but a faithful
MM-PSL implementation remains a Phase 1 baseline task.

### 2.3 Latency bar (P0c, `results/p0/p0c_wallclock.csv`)

CAN to 10⁻⁶ relative ISL change: 44 ms (N=64), 106 ms (N=256), 477 ms (N=1024), 1.4 s (N=2048), all
single-threaded numpy with contention. Multi-CAN N=64, K=8: 130 ms. A batched GPU CAN would run thousands of
instances in parallel at sub-millisecond amortised cost. Conclusion: at N ≤ 256 the amortisation argument does
not carry a paper. It becomes real when the objective is expensive (learned or simulation-based metrics), or
when many distinct constraint combinations must be served without re-deriving solvers.

---

## 3. Premise audit (things in the prompt or proposal I would not build on as stated)

### 3.1 "Flow matching is right because of native diversity, few-step ODE, stable training, torus compatibility"

- *Native diversity*: flow matching reproduces the diversity of its training distribution; it does not create
  it. If the training set is optimiser outputs from random restarts, the diversity ceiling is that of
  random-restart optimisation, which is free to obtain. The defensible claim is "no collapse pathology and
  faithful coverage," which is a *relative* advantage over the GAN, measurable by the coverage statistic
  Saarinen2024 themselves report as deficient.
- *Few-step ODE / PHY latency*: see 2.3. True but not decisive at these sizes.
- *Stable regression training*: true and the strongest practical reason; it removes the vicinal-weighting and
  critic-degeneracy machinery that continuous PSL labels forced on the GAN (Ding2021), because a regression
  objective with continuous conditioning has no per-label discriminator to starve.
- *Torus compatibility*: the GAN also outputs phases and is constant-modulus by construction. Constant modulus
  is not a differentiator versus GANs; it is a differentiator versus Euclidean diffusion (RF-Diffusion), which
  would need projection and would misrepresent the density.

So: adopt flow matching, but claim (i) faithful coverage, (ii) continuous conditioning without adversarial
degeneracy, (iii) inference-time composability of guidance. Do not claim diversity as intrinsic.

### 3.2 "The critic acts as an implicit learned AF-quality signal, avoiding a hand-crafted differentiable AF loss"

The AF is differentiable: A(k, ν) is a product of shifted conjugates followed by an FFT; ISL and windowed
AF energy are smooth polynomials of degree 4 in the phases; PSL is a max (subgradient or l_p smoothing, which is
exactly Song2016a). Bara2025 is a preprint about implementing this in autodiff. A reviewer at TSP will not
accept "avoids a hand-crafted loss" as a reason for a critic. A learned critic is justified only for metrics
that are non-differentiable *and* not cheaply smoothable (CFAR detection probability in clutter, discrete-phase
quantisation, hardware nonlinearity), or as a cheap surrogate of an expensive simulation. Direction C keeps the
critic in exactly that role.

### 3.3 "Slot conditioning is structurally analogous to best-of-K rejection sampling"

It is closer to "restore conditional independence." A conditional GAN that maps one latent to K outputs (or
one latent per slot but a shared conditioning path) has a collapse mechanism; a per-slot embedding breaks the
shared path. The measured result (mean cross-correlation at the i.i.d. level) is precisely what conditional
independence predicts. A flow model sampled i.i.d. per slot trivially reaches the same level. The cleanest
structural statement: **i.i.d. sampling is a floor for mean cross-correlation, not an achievement; anything
better requires joint (non-i.i.d.) sampling with a set potential, and the achievable gain is only in
peak / windowed quantities.** That is Direction A.

### 3.4 The proposal's Tier 1 design: cGAN with penalties → conditional diffusion with CFG, and a Unified Conditioning Space

For constraints that are differentiable functionals of the waveform (PSL/ISL, spectral masks, PAPR, windowed
cross-correlation, beampattern error), the right mechanism is not conditioning-plus-CFG but *guidance or
projection at inference* against an unconditional (or lightly conditioned) prior. Conditioning is needed for
categorical / structural choices (code family, length, number of codewords) and for speed. CFG on a numeric
spec produces "spec-likeness," not constraint satisfaction, and the guidance scale has no unit. The paper
should include the ablation "conditioning vs guidance vs both" precisely to make this point. The "Unified
Conditioning Space" then shrinks to a set of tokens (scalars, a sampled mask curve, a family id) with
cross-attention, which is what radarwgan already does, and the science is in the controllability measurement,
not the encoder.

---

## 4. Ranked directions

Each entry: hypothesis, why it should hold, what falsifies it, gate, novelty risk, closest prior work and
delta, compute.

### Direction A (headline): joint set generation on the torus with a windowed cross-correlation potential, framed as ISAC codebook synthesis

**Setting.** K unimodular length-N codes X ∈ T^{KN}. Sensing: per-code zero-Doppler PSL/ISL and an
AF sidelobe level in a Doppler band that excludes the chirp ridge. Comm (index-modulation ISAC in the spirit of
Huang2020 / Hassanien2016): the transmitter sends one of L = K codewords per pulse (log2 K bits); an AWGN
correlation receiver with timing uncertainty of up to W chips identifies the codeword. Its error probability is
governed by the worst pairwise cross-correlation in the lag window |k| ≤ W, so windowed orthogonality *is* the
comm metric, and the radar receiver (which knows the sent code) sees the code's own PSL. Trade-off knobs: K,
W, and the weight between auto and windowed cross terms. LPI/agility motivation: refresh the codebook every
CPI from a distribution.

**Hypothesis.** A flow model on T^{KN}, trained on jointly optimised sets (window-GD / Multi-WeCAN outputs)
and sampled with a set-level guidance potential Φ(X) = Σ_{m≠m'} Σ_{|k|≤W} |r_{mm'}[k]|² (particle guidance,
Corso2024, with a physical potential), produces codebooks whose worst-pair windowed cross-correlation is
≥ 20 dB below i.i.d. sampling at matched per-code PSL, and whose *distribution over codebooks* has coverage
(nearest-neighbour distance between independently sampled codebooks) matching random-restart window-GD.

**Why it should work.** Section 2.1 shows the target has a large feasible set at the required level: the
windowed objective is satisfiable to 10⁻³ at K=8 by local optimisation from random starts, so the target
distribution is broad and the flow has something to learn. The potential is a smooth quartic on the torus with
an analytic gradient (`set_wisl_grad_phase`), so guidance is cheap and exact.

**Falsifiers.** (i) The guided flow does no better than i.i.d. flow samples plus a few steps of window-GD
polish at the same wall-clock (the "just polish" baseline). (ii) Coverage collapses as guidance strength
rises (all codebooks converge to a few). (iii) Multi-WeCAN from random starts already gives equal diversity
and quality faster.

**Gate (Phase 1, N=64, K∈{4,8}, W=3).** Pass if: worst-pair windowed peak ≤ 0.01 on ≥ 90% of sampled
codebooks; per-code PSL ≤ −20 dB on ≥ 90% of codes; codebook coverage within 10% of random-restart
window-GD; and wall-clock per codebook ≤ that of random-restart window-GD on the same GPU. Fail on any one.

**Novelty risk: medium-low.** Closest prior: Saarinen2024 (conditional WGAN on R, M=10 codes; diversity flagged
open) and Corso2024 (particle guidance, generic). Delta: physical set potential on the torus, windowed-orthogonality
target justified by the ISL bound, ISAC codebook framing with an SER metric, full classical baseline set.
Compute: data generation 50k sets on GPU (batched torus GD) ~1 h; training a 2-5M-parameter 1D transformer
on the 5070 Ti ~2-4 h.

### Direction B (framework, second experiment block): one torus-flow prior plus zero-shot constraint composition, with the regime map as an explicit result

**Hypothesis.** An unconditional (family-conditioned only) flow prior over CAN/MM outputs on T^N, combined with
inference-time guidance for (a) a PSL target (l_p smoothed, or derivative-free via Li2024svdd on the exact PSL),
(b) a spectral mask (projection or PCFM-style correction, Utkarsh2025), (c) a PAPR bound (trivial on the
torus), reaches within 1 dB of the per-instance MM/ADMM optimiser specialised for that constraint combination,
at ≥ 10× lower wall-clock than *random-restart* optimisation at matched diversity, and only in the regime
beyond the rejection frontier of Section 2.2.

**Why.** The prior concentrates mass on the low-sidelobe region (−25 dB at N=64) that the Haar measure cannot
reach; guidance moves samples inside that region. Constraint composition without re-deriving an MM/ADMM
solver is the practical value proposition, and the regime map converts "why generative?" from a philosophical
objection into a quantitative boundary.

**Falsifiers.** Random-restart manifold GD from Haar initialisations with the same constraints matches the
guided flow at equal wall-clock (this is the most likely failure at N=64; if so, the paper moves to N ≥ 256 or
to expensive objectives). Guidance degrades PSL faster than it improves mask compliance (a Pareto curve that
is dominated by the optimiser).

**Gate.** Monotonicity: sampled PSL vs target PSL has Spearman ρ ≥ 0.9 with a controllable spread ≥ 4 dB
across the target range; mask compliance ≥ 95% at ≤ 1 dB PSL cost relative to the mask-aware WeCAN/MM
solution; wall-clock advantage ≥ 10× over random-restart GD at matched PSL and matched nearest-neighbour
diversity. Compute: 100k CAN solutions (CPU, ~1 h on 4 cores; minutes on GPU batched); training ~2 h.

**Novelty risk: medium.** Closest ML prior: Utkarsh2025 / Liang2025 (constraints in flows, generic PDE data);
closest SP prior: Saarinen2024. Delta: the regime map, the random-restart baseline, and controllability
measurements are the signal-processing contributions.

### Direction C (supporting): guidance ladder for non-differentiable sensing metrics

Analytic-gradient guidance (ISL, windowed AF energy) vs derivative-free guidance on the exact PSL (Li2024svdd)
vs a learned critic on AF images (the radarwgan critic, reused as a reward) vs reward fine-tuning
(DomingoEnrich2025 / Jensen2026). Test metric where a critic is honestly needed: detection probability of a
CA-CFAR detector at a fixed false-alarm rate in a synthetic two-target scene (non-differentiable). Gate: the
critic-guided flow improves P_d at fixed PSL by a measurable margin over analytic guidance; if not, the critic
idea is retired with evidence. Risk: low novelty as ML; value is in answering "does the GAN critic carry over."

### Direction D (supporting): conditioning and controllability methodology

Tokenised heterogeneous specs (scalars, sampled mask curve, family id, K) with cross-attention (radarwgan
design carried over). Contribution is the *measurement*: monotonicity curves, guidance-scale vs satisfaction
curves, conditioning-vs-guidance ablation (Section 3.4). Not a standalone paper; it is the evaluation
section of A/B.

### Direction E (not recommended): MIMO DFRC precoding-style ISAC (Liu2018 / Liu2021slp form) with generative models

Instance-specific feasible sets (channel H, symbols S) make a distributional prior nearly useless, diversity
has no value in precoding, and deep unfolding (Krish2023) is the natural learned competitor and will win on
this problem. Flag only: if the sponsor pushes here, it is a different paper and a different tool.

### Direction F (surprise candidate, optional): the aperiodic set bound as a design axiom for ISAC codebooks

The identity in Section 2.1 says the summed power spectrum of a good set is flat and the cross-correlation
energy budget is fixed; the only freedom is how that energy is distributed over lags. A short theoretical
section deriving the best achievable windowed-orthogonality / PSL trade-off (energy moved out of the window
must appear elsewhere) would give the generative results a reference curve and a bound that reviewers at TSP
value. Cost: a week of analysis; risk: it may already exist in the ZCZ / low-correlation-zone literature (check
before claiming).

---

## 5. Recommended paper framing (IEEE TSP primary)

**Title direction.** "Generative synthesis of unimodular waveform sets for integrated sensing and
communications: flow matching on the torus with set-level guidance."

**Headline contribution.** A flow-matching generator on T^{KN} with a physical set potential that samples
*distributions of codebooks* meeting windowed orthogonality and per-code sidelobe specifications, with
measured coverage, benchmarked against Multi-WeCAN, ADMM, random-restart manifold optimisation and a
conditional WGAN.

**Supporting contributions.** (1) The set-ISL bound and its consequence for what generative "diversity" can
mean; (2) a regime map that says when a learned sampler beats rejection and random-restart optimisation;
(3) zero-shot constraint composition (mask, PAPR, PSL target) on one prior with controllability curves;
(4) an ISAC index-modulation link between windowed cross-correlation and symbol error rate.

**Minimum experiment set a TSP reviewer will demand.**
- Per-code: PSL, ISL, AF sidelobe level in a Doppler band, PAPR (=1 by construction, report anyway), spectral
  mask compliance; per-family breakdowns.
- Set: worst-pair and mean peak cross-correlation (all lags), windowed peak, zero-lag; total ISL vs the bound.
- Diversity: nearest-neighbour distance among sampled sets (modulo global phase), coverage vs training data
  (Saarinen2024's statistic), and diversity-vs-quality curves against random-restart optimisation at matched
  wall-clock.
- Baselines: CAN/WeCAN, MM-PSL (Song2016a), Multi-WeCAN, ADMM (Wang2021), manifold GD (Alhujaili2019),
  conditional WGAN (radarwgan / Saarinen2024 re-implemented), Euclidean flow with projection (RF-Diffusion-like).
- ISAC: codeword SER vs SNR for K ∈ {4, 8, 16} with timing offset up to W, and radar P_d / range sidelobe
  masking in a two-target scene.
- Cost: NFE vs quality, and wall-clock on the same GPU for all methods.

**Most likely reviewer objections and prepared answers.**
1. "Optimisation already solves this; why a generative model?" → regime map + random-restart baseline at
   matched wall-clock + constraint composition without new solver derivations. If the numbers do not support
   it at N=64, move to N ≥ 256 and expensive objectives before submitting.
2. "Training data comes from the optimiser, so the model cannot exceed it." → agreed and stated; the claims
   are coverage, speed, composition, and set-level structure, not optimality.
3. "Mean cross-correlation improvements are impossible." → we say so first and use the bound.
4. "AF metrics at zero Doppler only." → include Doppler-band AF sidelobe metrics; note that chirp-like codes
   (Frank/P4) have a −3 dB AF ridge by design, so full-AF PSL without ridge exclusion is meaningless.
5. "ISAC comm model is toy." → it is intentionally idealised (AWGN, ideal sync up to W chips); the
   mapping from windowed cross-correlation to SER is exact under that model, and that is the point.

---

## 6. What Phase 1 would do (awaiting go)

1. Batched GPU torus optimisers (CAN, window-GD, MM-PSL) to generate 100k singles and 50k sets at N=64.
2. Torus flow-matching smoke test: unconditional prior over CAN outputs; report PSL distribution vs training
   data, coverage, NFE curve. Gate: sampled PSL median within 1 dB of training median at 8 NFE.
3. Direction A prototype: joint flow on T^{KN} + windowed potential guidance; the gate in Section 4A.
4. Direction B: PSL-target guidance and mask projection; monotonicity and mask-compliance curves; the
   random-restart baseline at matched wall-clock.
5. Re-implement the conditional WGAN baseline from radarwgan for the same data and metrics (do not retrain the
   full golden run; a small faithful version is enough for a fair table).

---

## 7. Amendment (2026-10-05): comm-side audit of Direction A

See `docs/phase0b_comm_side_audit.md`. Summary of what changed:

- Direction A's comm side is code-shift keying with a generated codebook. The direct prior is Tedesso & Romero
  (DSP 2018, Gold/Kasami CSK, SER vs the M-ary bound, PACF/periodic AF, LPI) and Eedara, Amin, Hoorfar, Chalise
  (IEEE TAES 2022, CSK on FH-MIMO). The MIMO drop-in targets are Hassanien's waveform-diversity (TSP 2016) and
  waveform-permutation (DSP 2018) schemes.
- We cannot win on rate (log2 K per antenna vs 7 to 16+ bits for MAJoRCom-type index modulation at 4 to 8
  antennas) or on SER (the Kasami small set is near-simplex and already optimal under perfect timing; windowed
  orthogonality buys at most 10 log10(1 − ρ_window) ≈ 1.5 dB under timing uncertainty, measured 0.3 to 0.5 dB
  at K=8). Comm is a parity check.
- The set potential should target zero-lag ρ = −1/(K−1) (simplex) for coherent receivers, not ρ = 0.
- The sensing claims survive: mean per-code PSL 3.0 dB (Kasami) to 4.5 dB (Gold) better at K=8, N=63; windowed
  cross-correlation 0.006 vs 0.17 to 0.29; codebook agility; constraint composition. Latency is "to be measured".
- Added to the Direction A gate: SER parity under timing uncertainty, simplex-bound parity under perfect timing,
  and a bits-per-pulse vs sensing-loss figure with Gold/Kasami/Multi-CAN/window-GD/deep-unfolded/generated
  codebooks and MAJoRCom / permutation rate reference lines. Engineering cost before training: about one week.
