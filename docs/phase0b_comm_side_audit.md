# Phase 0b: Communications-side audit of Direction A (codebook / waveform-selection ISAC)

Date: 2026-10-05. Scope: the comm side of Direction A only. Numbers from `experiments/2026-10-05_p0d_csk_codebooks.py`
(outputs in `results/p0/p0d_*`) and from `results/p0/p0b_*`. Every baseline below was verified against a primary
page on 2026-10-05; see `docs/references.md` for the exact citation and what was confirmed. Items marked
"not confirmed" are stated as such.

---

## 0. Bottom line

1. **There is a direct comm-side prior for exactly our formulation.** Tedesso & Romero (DSP 2018) do code-shift
   keying (CSK): one of K pseudo-random codes is sent per symbol, the same code is the radar waveform, SER is
   measured by Monte Carlo against the M-ary orthogonal-signalling bound, and the radar side is scored by the
   periodic autocorrelation and periodic ambiguity function, with LPI as the motivation. Eedara, Amin, Hoorfar
   and Chalise (TAES 2022) carry CSK onto frequency-hopping MIMO radar and co-design the codes with a genetic
   algorithm against range sidelobes. Direction A is, on the comm side, "CSK with a generated codebook." We should
   say so in the paper and place our SER curves next to theirs.
2. **We cannot win on rate or on SER.** Single-antenna CSK carries log2 K bits per pulse (3 bits at K=8). The
   MIMO index-modulation family (Hassanien's sidelobe-control and waveform-permutation schemes, MAJoRCom, FRaC,
   HIM) carries 8 to 16+ bits per pulse at 8 antennas. On SER, any codebook with zero-lag correlations at the
   simplex value already sits on the M-ary bound; a generated codebook can only match it. Measured: at K=8, N=63,
   the Kasami small set has the best perfect-timing SER of all codebooks tested, including our jointly designed
   one, because its zero-lag correlations are −9/63 ≈ −1/(K−1).
3. **What windowed orthogonality buys on the comm side is small and bounded.** Under timing uncertainty of ±3
   chips with a delay-searching receiver, the worst-pair windowed cross-correlation sets a pairwise-distance
   penalty of 10 log10(1−ρ): −1.5 dB for Gold / i.i.d., −0.8 dB for Kasami, −0.03 dB for the window-designed
   codebook. The measured SER gap at 10⁻³ is about 0.3 to 0.5 dB. That is a parity-plus-margin result, not a
   headline.
4. **What we can win** (Section 3): per-code sidelobes at fixed K (mean PSL −17.6 dB vs −13.1 dB Gold and −14.6 dB
   Kasami at K=8, N=63), MIMO separability (windowed cross-correlation 0.006 vs 0.17 to 0.29), codebook agility
   for LPI (a fresh codebook per CPI from a distribution, which fixed Gold/Kasami families cannot offer and which
   classical optimisers offer only at their wall-clock), and inference-time constraint composition (mask, PAPR,
   window, PSL) without a new solver per combination. Latency is unproven: window-GD at K=8, N=63 takes 4 s in
   single-thread numpy and a batched GPU version has not been timed.
5. **Minimum change to Direction A so a fair comm-side comparison exists** (Section 4): adopt CSK exactly as in
   Tedesso & Romero for the single-antenna case, add per-antenna codeword selection (Hassanien's waveform
   diversity / permutation) for the MIMO case, and report one figure that no prior paper has: bits per pulse vs
   sensing loss (worst-code PSL and worst-pair windowed cross-correlation) with the Gold/Kasami, Multi-CAN,
   window-GD, deep-unfolded, and generated codebooks on the same axes and MAJoRCom / permutation rate lines as
   references. Cost: about one week of engineering before any model training, listed per item in Section 4.

---

## 1. Comm-side baselines that use the same or an adjacent formulation, and what they report

Formulation key: CSK = one codeword out of K per symbol, same code is the radar waveform; WD = waveform diversity
(several orthogonal waveforms transmitted simultaneously, bits in which/how); IM = index modulation over
frequency / antenna / permutation indices.

| # | Baseline (verified) | Formulation | Bits per pulse | Comm metrics reported | Radar metrics reported | Channel / receiver |
|---|---|---|---|---|---|---|
| B1 | Tedesso & Romero, DSP 80:48-56, 2018, "Code shift keying based joint radar and communications for EMCON applications" | CSK with Gold or Kasami codes; BPSK/QPSK code-bit modulation; CW pseudo-random coded radar | log2 K (K = codebook size; exact K used not confirmed, abstract only) | SER vs SNR by Monte Carlo, compared to an SER upper bound for M-ary FSK | periodic autocorrelation (PACF), periodic ambiguity function; LPI argument | AWGN, correlation receiver (details not confirmed) |
| B2 | Eedara, Amin, Hoorfar, Chalise, IEEE TAES 58(3):1501-1513, 2022, "Dual-function frequency-hopping MIMO radar system with CSK signaling" | CSK pulse sequences multiply FH hops in fast time on a MIMO platform; CSK sequences and FH code jointly designed by GA | not confirmed (abstract) | abstract states embedding strategy; BER curves not confirmed | range sidelobe levels, range-sidelobe modulation (RSM) across pulses | MIMO FH radar |
| B3 | Hassanien, Amin, Zhang, Ahmad, IEEE TSP 64(8):2168-2181, 2016, "Dual-function radar-communications: information embedding using sidelobe control and waveform diversity" | WD: Q orthogonal waveforms sent simultaneously; one bit per waveform via two sidelobe levels toward the comm receiver | Q (one bit per orthogonal waveform) | BER vs SNR | transmit beampattern (mainlobe preserved, sidelobe levels), MIMO mode | comm receiver in the sidelobe region; matched filtering to the Q waveforms |
| B4 | Hassanien, Aboutanios, Amin, DSP 83:118-128, 2018, "A dual-function MIMO radar-communication system via waveform permutation" | WD: which orthogonal waveform goes to which antenna | log2(Q!) (≈15.3 bits at Q=8) | BER (per the Hassanien line; figure-level details not confirmed) | beampattern / MIMO radar unchanged | MIMO |
| B5 | Huang, Shlezinger, Xu, Liu, Eldar, IEEE TSP 68:3423-3438, 2020, MAJoRCom | IM over carrier-frequency selection and antenna-frequency permutation on CAESAR (frequency-agile phased array) | log2 C(M,K) + log2[LR!/(LK!)^K]; grows as K log2 M + LR log2 K; example 7 to 16 bits at LR = 4 to 8, M = 7 | achievable-rate upper and lower bounds vs SNR (Figs. 5-6, up to ~10 bits/pulse), BER of ML vs low-complexity decoders (Fig. 7), codebook design by minimum distance | radar unchanged from CAESAR (target recovery); effect of codebook pruning on radar | spatial-decay and Rayleigh channels; dedicated-comm-antenna baselines |
| B6 | Ma, Shlezinger, Huang, Liu, Eldar, IEEE JSTSP 15(6):1348-1364, 2021, FRaC | IM over sparse antenna subset and carrier selection plus PM symbols on FMCW | log2 J + ⌊log2 C(P,K)⌋ (+ permutation term for the MAJoRCom special case) | BER vs PM-only embedding; rate | range/velocity/angle estimation, max recoverable targets via phase transitions, ambiguity function | vehicular; sparse MIMO |
| B7 | Ma, Shlezinger, Huang, Shavit, Namer, Liu, Eldar, IEEE TVT 70(3):2283-, 2021, SpaCoR (GSM-based DFRC, hardware prototype) | generalized spatial modulation: antenna subset carries bits, dedicated comm waveform on the rest | GSM rate (antenna-selection bits + constellation bits) | BER vs fixed allocation at equal rate | beampattern, angular resolution, sidelobes of the expected beampattern; prototype measurements | MIMO |
| B8 | Xu, Wang, Aboutanios, Cui, IEEE TVT 72(3):3186-3200, 2023, hybrid index modulation (HIM) for FH-MIMO DFRC | IM over 3-tuples (frequency, phase, antenna) | high (dictionary size) | SER, data rate | ambiguity function | FH-MIMO |
| B9 | Chen, Kaushik, Masouros, IEEE GLOBECOM 2022, "Pre-scaling and codebook design for JRC based on index modulation" | MAJoRCom-type IM; codebook pruned by maximum minimum Euclidean distance; constellation randomisation | ⌊log2 of valid codewords⌋ | BER vs the unpruned baseline | not the focus | multi-carrier MIMO radar |
| B10 | Şahin & Girici, IET RSN 18(12):2608-2616, 2024, "Phase coded waveforms for ISAC systems" | intra-pulse phase-modulated Barker / Zadoff-Chu / LFM; M = 4 symbols; one matched filter per symbol | log2 M (2 bits) | SER vs SNR | range and velocity MSE, sidelobes at close range | monostatic pulse radar, automotive |
| B11 | Wang, Fu, Chen, Wu, Wang, IEEE Comm. Letters 29(2):244-248, 2025, covert DFRC with phase-reduced shift keying and code-domain IM (spherical codes) | code-domain IM on LFM | not confirmed | BER, covertness | detection performance | LFM |
| B12 | Eedara, Hassanien, Amin, IET RSN 15(4):402-418, 2021, FH-MIMO DFRC with PSK | PSK symbols multiply FH hops | per-hop PSK bits | BER analysis | RSM, range sidelobes | FH-MIMO |
| B13 | Baxter, Aboutanios, Hassanien, Asilomar 2018, FH code selection; Wu, Zhang, Huang, Guo, IEEE TWC 21(7):5392-5405, 2022, FH permutations with secure comm | IM over FH codes / permutations | log2 of permutations | BER / demodulation, secrecy | mutual interference among targets | FH-MIMO |
| S1 | Elbir, Celik, Eltawil, Amin, IEEE SPM 2024 (arXiv:2401.08186), "Index modulation for ISAC: a signal processing perspective" | survey; Table I lists IM-ISAC techniques by index medium with rate and pros/cons; names CSK as "a special case of IM where a communication symbol is mapped to a radar waveform" (their ref. [33] is B2) | | | | |

Reading of the table. Every paper in the family reports BER or SER vs SNR in AWGN (sometimes Rayleigh) with a
matched-filter or ML receiver, and a rate figure of merit in bits per pulse. The radar side is reported as
beampattern, range sidelobes or RSM, ambiguity function, or estimation error, never as a joint
"bits per pulse vs sensing loss" curve. B1 is the only one with our single-antenna CSK formulation and scores
exactly what we score (SER vs the M-ary bound, autocorrelation and ambiguity function). B3/B4 are the MIMO
versions (waveform diversity / permutation) into which a generated codebook drops in directly.

## 2. Measured comm-side numbers (K=8, N=63, W=3; `results/p0/p0d_csk_codebooks.csv`)

| codebook | worst PSL (dB) | mean PSL (dB) | worst zero-lag |ρ| | worst peak, |lag|≤3 | worst peak, all lags | SER @ 6 dB, perfect timing | SER @ 6 dB, offset ±3, delay-search rx |
|---|---|---|---|---|---|---|---|
| Gold (63) | −9.1 | −13.1 | 0.238 | 0.286 | 0.349 | 2.1e-3 | 3.7e-2 |
| Kasami small (63) | −13.1 | −14.6 | 0.143 | 0.175 | 0.222 | 0.7e-3 | 3.3e-2 |
| i.i.d. random unimodular | −12.7 | −13.7 | 0.243 | 0.292 | 0.345 | 2.4e-3 | 3.7e-2 |
| Multi-CAN | −13.3 | −14.9 | 0.190 | 0.255 | 0.318 | 2.5e-3 | 3.3e-2 |
| window-GD (ours, classical) | −14.6 | −17.6 | 0.003 | 0.006 | 0.325 | 1.8e-3 | 3.1e-2 |
| M-ary orthogonal union bound | | | 0 | | | 2.0e-3 | |

Three things to state plainly in the paper:

- **Perfect timing: orthogonality is not optimal; the simplex is.** For coherent K-ary detection the optimal
  equal-energy set has ρ = −1/(K−1) between every pair. The Kasami small set (ρ ∈ {−9/63, 7/63}) is close to it
  and beats both the orthogonal bound and our window-designed codebook. A comm-optimised codebook objective
  should therefore target zero-lag ρ = −1/(K−1) for coherent receivers and ρ = 0 for noncoherent ones. This is
  a one-line change to the set potential and costs nothing on the sensing side.
- **Timing uncertainty: the gain from windowed orthogonality is bounded by 10 log10(1−ρ_window).** That is 1.5 dB
  for Gold / i.i.d., 0.8 dB for Kasami, ≈0 for the window design. Measured SER gap at 10⁻³ is about 0.3 to 0.5 dB
  because with a delay-searching noncoherent receiver the K(2W+1) noise-only hypotheses dominate. The comm side
  is a parity check with a small margin, not a contribution.
- **Sensing is where the codebook matters.** Mean per-code PSL improves by 3.0 dB over Kasami and 4.5 dB over
  Gold at the same K and N, and windowed cross-correlation (the quantity that governs MIMO waveform separation
  and cross-code range sidelobes within the timing window) drops from 0.17 to 0.29 down to 0.006. Note that the
  joint constraints cost per-code PSL relative to a single optimised code (−25 dB from CAN at N=64): the
  "bits per pulse vs sensing loss" curve is exactly this trade.

## 3. Claims we can and cannot make, per baseline

| Baseline | Can win | Cannot win | Evidence / status |
|---|---|---|---|
| B1 Tedesso & Romero (CSK, Gold/Kasami) | per-code aperiodic PSL at fixed K (3.0 to 4.5 dB mean, Section 2); windowed cross-correlation (30 to 50×); codebook agility for LPI (fresh codebook per CPI vs a public, enumerable family); constraint composition (spectral mask / PAPR added at inference) | rate (identical, log2 K); SER at perfect timing (Kasami is near-simplex and already optimal); SER under timing uncertainty beyond ~1.5 dB | measured at K=8, N=63; Gold/Kasami implemented and checked (periodic correlation values {−17,−1,15} and {−9,7}) |
| B2 Eedara et al. (CSK on FH-MIMO, GA co-design) | wall-clock per codebook vs a genetic algorithm (plausible, unmeasured); constraint composition; no RSM term is in our objective yet, so RSM parity must be shown | sidelobe optimality: a GA tuned to their exact objective is a per-instance optimiser like Multi-WeCAN | their BER figures not confirmed from abstract; need the paper for a side-by-side |
| B3/B4 Hassanien et al. (waveform diversity, permutation) | nothing on the embedding scheme itself; our codebook substitutes for their "Q orthogonal waveforms" and can be scored on the same BER axis; sensing gain is the per-code PSL and windowed orthogonality of the Q waveforms | rate (Q or log2 Q! bits per pulse is theirs, not ours); beampattern-based embedding is orthogonal to our work | plug-in comparison possible with M antennas = Q |
| B5 MAJoRCom | nothing head-to-head: different physical layer (frequency-agile phased array); only a rate reference line | rate (7 to 16 bits at 4 to 8 antennas vs 3 bits single-antenna CSK; per-antenna CSK reaches 16 to 32 but needs the full codebook pairwise windowed-orthogonal, which we have only shown at K=8); their radar side is "unchanged from CAESAR" so sensing-loss comparisons are not like-for-like | formula verified from the paper text; example values computed in P0d |
| B6/B7 FRaC, SpaCoR | same as B5 | same as B5 | |
| B8 HIM, B9 Chen et al. codebook pruning | codebook *generation* vs codebook *pruning*: we can adopt their MED-pruning idea as a post-step | rate | |
| B10 Şahin & Girici | per-code PSL and SER at equal M (their M=4 Barker/ZC vs ours) | nothing else; low bar | |
| Deep unfolding (Krishnananthalingam, Nguyen, Juntti, arXiv 2023, constant-modulus JCAS precoding; Zhao et al. GRSL 2024 range-ISL unfolding, not confirmed) | diversity (unfolded nets are deterministic per input) and constraint composition | per-instance quality at fixed objective; latency (an unfolded net is a few layers) | a deep-unfolded window-GD is the right in-house baseline; see Section 4 |
| Random-restart window-GD / Multi-WeCAN (classical) | wall-clock only if the batched GPU optimiser is slower than the sampler at matched quality (unmeasured) | quality; diversity (random restarts are diverse by construction) | this remains the most dangerous baseline for the whole paper |

## 4. Minimum change to Direction A for a fair comm-side comparison, with cost

1. **Adopt CSK exactly as B1 for the single-antenna case.** Receiver: coherent correlation (perfect timing) and
   delay-searching noncoherent (timing uncertainty ±W). Metrics: SER vs Eb/N0 against the M-ary
   orthogonal bound and the simplex bound; report both Gold and Kasami codebooks at the same K and N.
   Status: implemented in P0d. Remaining cost: 0.5 day to add the simplex target option to the set potential.
2. **Add per-antenna codeword selection for the MIMO case** (B3/B4 style): M antennas each pick one of K
   codewords, giving M log2 K bits per pulse, with the requirement that every pair in the codebook is windowed
   orthogonal so any simultaneously transmitted subset separates at the radar receiver. This is where our
   codebook adds something B3/B4 do not have (they assume orthogonal waveforms exist). Cost: 1 day for the MIMO
   receiver simulation (per-antenna matched filtering, cross-code leakage as the sensing loss).
3. **One new figure: bits per pulse vs sensing loss.** x-axis: bits per pulse (log2 K single antenna; M log2 K
   per-antenna). y-axes (two panels, one scale each): worst-code PSL and worst-pair windowed cross-correlation.
   Series: Gold, Kasami, Multi-CAN, window-GD, deep-unfolded window-GD, generated codebooks (mean and 10th
   percentile over sampled codebooks). Reference lines: MAJoRCom rate at matched antenna count (formula), waveform
   permutation log2(M!), sidelobe control M bits. Caveat printed on the figure: reference lines are rates of
   different physical layers, not SER-equivalent. Cost: K sweep {2,4,8,16,32} with window-GD at N=63 is minutes
   per codebook on CPU at K ≤ 16 and tens of minutes at K=32 (cost grows as K²); 30 codebooks per K: a few CPU
   hours, or minutes on a batched GPU implementation (which is Phase 1 item 1 anyway).
4. **Deep-unfolded baseline.** Unfold T = 10 to 20 iterations of `set_wisl_grad_phase` with learned per-layer
   step sizes and lag weights, trained on the same objective; this is the honest "learned optimiser" competitor
   for latency and quality, and is deterministic, which is what makes the diversity comparison meaningful.
   Cost: 1 to 2 days. Re-implementing Krishnananthalingam et al. (precoding formulation) is not needed unless the
   precoding direction is adopted, which Phase 0 recommends against.
5. **Do not re-implement MAJoRCom / FRaC end to end.** A faithful CAESAR plus ML decoder is about a week and
   would still not be like-for-like. Use the rate formulas as reference lines and cite their BER figures
   descriptively.
6. **Acquire B1 and B2 full texts** before writing the related-work paragraph: B1's codebook size and receiver,
   B2's BER figures are not confirmed from the abstracts. Cost: library access, hours.

Total before model training: about one engineering week, dominated by items 2 to 4, all of which are reusable
as evaluation code for Phase 1.

## 5. Consequences for the Direction A gate

The gate in `phase0_landscape_and_directions.md` Section 4A stands, with two additions and one reweighting:

- Add: codeword SER under timing uncertainty within 0.5 dB of the window-GD codebook at the same K (parity
  check), and perfect-timing SER within 0.3 dB of the simplex bound when the simplex target is used.
- Add: bits-per-pulse vs sensing-loss curve where generated codebooks lie within 1 dB of worst-code PSL and
  within 2× of windowed cross-correlation of the classical window-GD codebooks at every K, while being
  sampled, not optimised, per codebook.
- Reweight: the headline sensing claim is per-code PSL and windowed cross-correlation at fixed bits per pulse,
  plus codebook agility. The comm claim is parity. Latency is a claim only after the batched GPU optimiser is
  timed; until then it is listed as "to be measured".
