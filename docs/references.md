# Verified references

Every entry below was checked against a primary or index page on 2026-09-24 (arXiv, IEEE Xplore, ACM DL,
OpenReview, NeurIPS/ICLR proceedings, publisher pages). "Status" notes what was confirmed. Anything not
in this list that appears in the Phase 0 document is marked "unverified" there.

## Generative models for radar / ISAC waveforms (the direct competition)

| Key | Citation | Status |
|---|---|---|
| Saarinen2020 | V. Saarinen, V. Koivunen, "Radar Waveform Synthesis Using Generative Adversarial Networks," IEEE RadarConf 2020, Florence, pp. 1-6. | verified (IEEE Xplore 9266709) |
| Saarinen2021 | V. Saarinen, V. Koivunen, "Generative Adversarial Network for Variable-Length Sensing Waveform Synthesis," IEEE SPAWC 2021, Lucca, pp. 456-460. | verified (Aalto research portal) |
| Saarinen2023 | V. Saarinen, V. Koivunen, "MIMO Radar Waveform Synthesis Using Generative Adversarial Networks," IEEE conference paper 2023 (IEEE Xplore 10285933). | exists; exact conference name not confirmed (Aalto page returned 403) |
| Saarinen2024 | V. Saarinen, R. Rajamäki, V. Koivunen, "Generative Deep Synthesis of MIMO Sensing Waveforms with Desired Transmit Beampattern," arXiv:2412.20883, Dec 2024. Conditional WGAN, M=10 codes of N=41, conditioned on the off-diagonal of the target cross-correlation matrix R; baseline Multi-CAO; 0.002 s (GPU) vs 0.098 s per sample. Conclusion states: "The diversity of outputs could also be improved to better match or exceed the full diversity of the training data." | verified arXiv; carries an IEEE copyright notice; journal/conference acceptance not confirmed. Cite as arXiv preprint. |
| Ziemann2025 | M. R. Ziemann, C. A. Metzler, "Adaptive LPD Radar Waveform Design with Generative Deep Learning," arXiv:2403.12254; IEEE Transactions on Radar Systems, 2025 (Xplore 10887245). Adversarial generator vs critic on ambient RF background + AF-based loss. | verified |
| Luo2026 | Y. Luo, G. Du, J. Dang, "Orthogonal waveform design for MIMO radars on the basis of deep learning," Signal, Image and Video Processing, 2026. Causal temporal transformer producing phase sequences, mutual-information loss. | exists (Springer); not a generative/distributional model as far as the abstract shows |
| Kang2023 | B. Kang et al., "Deep Learning for Radar Waveform Design: Retrospectives and the Road Ahead," IEEE International Radar Conference 2023. | verified (Xplore 10371126) |
| Hu2024 | Y. Hu et al., "Radar waveform generative design method," Electronics Letters, 2024. Architecture-engineering-inspired parametric design, not a deep generative model. | exists; low relevance |

## Generative models in wireless (adjacent, not direct competitors)

| Key | Citation | Status |
|---|---|---|
| Chi2024 | G. Chi, Z. Yang, C. Wu, J. Xu, Y. Gao, Y. Liu, T. X. Han, "RF-Diffusion: Radio Signal Generation via Time-Frequency Diffusion," ACM MobiCom 2024, pp. 77-92. Generates Wi-Fi / FMCW / 5G signals for downstream sensing and channel estimation; not constraint-conditioned waveform design. | verified |
| Bai2026 | Y. Bai et al., "Covert Semantic Transmission in ISAC: Dual-Functional Waveform Design and Rectified Flow-Assisted Recovery," arXiv:2607.25354, July 2026 (states IEEE TWC). Rectified flow is used at the receiver for semantic recovery; the waveform is chirp-based with Gaussian-pair rotation coding. Not a waveform generator. | verified. This is the "CoSMIC" in the prompt; the premise that it is rectified flow for ISAC waveform synthesis is incorrect. |
| Chen2025scoring | L. Chen, C. Cai, H. Yang, X. Yuan, Y.-J. A. Zhang, "Scoring ISAC: Benchmarking Integrated Sensing and Communications via Score-Based Generative Modeling," arXiv:2508.02117. Score models estimate MI/MMSE/BCRB from data; not waveform generation. | verified |
| Jiang2025 | Y. Jiang, F. Gao, S. Jin, T. J. Cui, "Electromagnetic Property Sensing Based on Diffusion Model in ISAC System," arXiv:2407.03075 (IEEE TWC 2025 per citing surveys). Sensing-side use. | arXiv verified; TWC status from secondary source |
| Surveys | "Diffusion Models for Future Networks and Communications: A Comprehensive Survey," arXiv:2508.01586; "Generative Diffusion Models for Wireless Networks," arXiv:2507.16733. | exist; use for landscape only |

## Flow matching, manifolds, constraints, guidance (ML tools)

| Key | Citation | Status |
|---|---|---|
| Lipman2023 | Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, M. Le, "Flow Matching for Generative Modeling," ICLR 2023. | verified |
| Liu2023rf | X. Liu, C. Gong, Q. Liu, "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow," ICLR 2023. | verified |
| Tong2024 | A. Tong, K. Fatras, N. Malkin, G. Huguet, Y. Zhang, J. Rector-Brooks, G. Wolf, Y. Bengio, "Improving and generalizing flow-based generative models with minibatch optimal transport," TMLR 2024. TorchCFM library. | verified |
| Chen2024rfm | R. T. Q. Chen, Y. Lipman, "Flow Matching on General Geometries," ICLR 2024 (oral). Riemannian flow matching; flat-torus experiments included; simulation-free on simple geometries. | verified |
| Jing2022 | B. Jing, G. Corso, J. Chang, R. Barzilay, T. Jaakkola, "Torsional Diffusion for Molecular Conformer Generation," NeurIPS 2022. Score-based diffusion on the hypertorus. | verified |
| Collas2025 | A. Collas, C. Ju, N. Salvy, B. Thirion, "Riemannian Flow Matching for Brain Connectivity Matrices via Pullback Geometry," NeurIPS 2025 (arXiv:2505.18193). | verified |
| Utkarsh2025 | U. Utkarsh, P. Cai, A. Edelman, R. Gomez-Bombarelli, C. V. Rackauckas, "Physics-Constrained Flow Matching: Sampling Generative Models with Hard Constraints," NeurIPS 2025 (arXiv:2506.04171). Zero-shot hard-constraint enforcement during sampling. | verified |
| Liang2025 | J. Liang, Y. Sun, A. Samaddar, S. Madireddy, F. Fioretto, "Chance-constrained Flow Matching for High-Fidelity Constraint-aware Generation," arXiv:2509.25157. | verified arXiv; venue not confirmed |
| GCFM2025 | "Generalized Constrained Flow Matching for Constraint-Aware Generative Modeling," NeurIPS 2025 (neurips.cc/virtual/2025/123323). | listing exists; authors not confirmed |
| Woo2026 | D. Woo, M. Skreta, S. Park, K. Neklyudov, S. Ahn, "Riemannian MeanFlow," arXiv:2602.07744, Feb 2026. One/few-step flow maps on manifolds. | verified arXiv |
| Geng2025 | Z. Geng, M. Deng, X. Bai, J. Z. Kolter, K. He, "Mean Flows for One-step Generative Modeling," NeurIPS 2025 (arXiv:2505.13447). | verified |
| Corso2024 | G. Corso, Y. Xu, V. De Bortoli, R. Barzilay, T. Jaakkola, "Particle Guidance: non-I.I.D. Diverse Sampling with Diffusion Models," ICLR 2024. Joint-particle potential for diverse sets. | verified |
| Ho2022 | J. Ho, T. Salimans, "Classifier-Free Diffusion Guidance," arXiv:2207.12598, 2022. | verified |
| Chung2023 | H. Chung et al., "Diffusion Posterior Sampling for General Noisy Inverse Problems," ICLR 2023. | verified |
| Li2024svdd | X. Li, ..., M. Uehara, "Derivative-Free Guidance in Continuous and Discrete Diffusion Models with Soft Value-Based Decoding," arXiv:2408.08252, 2024. | verified arXiv |
| DomingoEnrich2025 | C. Domingo-Enrich, M. Drozdzal, B. Karrer, R. T. Q. Chen, "Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control," ICLR 2025. | verified |
| Jensen2026 | C. Perez Jensen, L. Schaufelberger, R. De Santi, K. Jorner, A. Krause, "Value Matching: Scalable and Gradient-Free Reward-Guided Flow Adaptation," ICLR 2026. | verified |
| Ding2021 | X. Ding, Y. Wang, Z. Xu, W. J. Welch, Z. J. Wang, "CcGAN: Continuous Conditional Generative Adversarial Networks for Image Generation," ICLR 2021. | verified |

## Classical sequence / waveform design (baselines)

| Key | Citation | Status |
|---|---|---|
| Stoica2009 | P. Stoica, H. He, J. Li, "New Algorithms for Designing Unimodular Sequences With Good Correlation Properties," IEEE TSP 57(4):1415-1425, 2009. CAN, WeCAN. | verified |
| He2009 | H. He, P. Stoica, J. Li, "Designing Unimodular Sequence Sets With Good Correlations - Including an Application to MIMO Radar," IEEE TSP 57(11):4391-4405, 2009. Multi-CAN, Multi-WeCAN. | verified |
| Song2016a | J. Song, P. Babu, D. P. Palomar, "Sequence Design to Minimize the Weighted Integrated and Peak Sidelobe Levels," IEEE TSP 64(8):2051-2064, 2016. MM for WISL and l_p-norm PSL. | verified |
| Song2016b | J. Song, P. Babu, D. P. Palomar, "Sequence Set Design With Good Correlation Properties via Majorization-Minimization," IEEE TSP 64(11):2866-2879, 2016. | verified |
| Wang2021 | J. Wang, Y. Wang, "Designing Unimodular Sequences With Optimized Auto/Cross-Correlation Properties via Consensus-ADMM/PDMM Approaches," IEEE TSP 69:2987-2999, 2021. | verified |
| Alhujaili2019 | K. Alhujaili, V. Monga, M. Rangaswamy, "Transmit MIMO Radar Beampattern Design via Optimization on the Complex Circle Manifold," IEEE TSP 67(13):3561-3575, 2019. | verified |
| Alhujaili2019b | K. Alhujaili, V. Monga, M. Rangaswamy, "Quartic Gradient Descent for Tractable Radar Slow-Time Ambiguity Function Shaping," IEEE TAES, 2019. | verified (TAES, not TSP) |
| Bara2025 | M. Bara Iniesta, "Differentiable Radar Ambiguity Functions: Mathematical Formulation and Computational Implementation," arXiv:2506.22935, 2025. | verified arXiv (single-author preprint) |

## ISAC / DFRC formulations relevant to the comm side

| Key | Citation | Status |
|---|---|---|
| Liu2018 | F. Liu, L. Zhou, C. Masouros, A. Li, W. Luo, A. Petropulu, "Toward Dual-functional Radar-Communication Systems: Optimal Waveform Design," IEEE TSP 66(16):4264-4279, 2018. | verified |
| Liu2021slp | R. Liu, M. Li, Q. Liu, A. L. Swindlehurst, "Dual-Functional Radar-Communication Waveform Design: A Symbol-Level Precoding Approach," IEEE JSTSP 15(6):1316-1331, 2021. | verified |
| Huang2020 | T. Huang, N. Shlezinger, X. Xu, Y. Liu, Y. C. Eldar, "MAJoRCom: A Dual-Function Radar Communication System Using Index Modulation," IEEE TSP 68:3423-3438, 2020. | verified |
| Lu2024 | S. Lu, F. Liu, F. Dong, Y. Xiong, J. Xu, Y.-F. Liu, S. Jin, "Random ISAC Signals Deserve Dedicated Precoding," IEEE TSP 72:3453-3469, 2024. | verified |
| Chen2024papr | Y. Chen, C. Wen, Y. Huang, L. Liang, J. Li, H. Zhang, W. Hong, "Joint Design of ISAC Waveform under PAPR Constraints," arXiv:2311.11594; China Communications 2024. | verified |
| Lee2024 | B. Lee, A. B. Das, D. J. Love, C. G. Brinton, J. V. Krogmeier, "Constant Modulus Waveform Design with Space-Time Sidelobe Reduction for DFRC Systems," arXiv:2406.18951, 2024. ADMM and MM solvers. | verified arXiv |
| Krish2023 | P. Krishnananthalingam, N. T. Nguyen, M. Juntti, "Deep Unfolding Enabled Constant Modulus Waveform Design for Joint Communications and Sensing," arXiv:2306.14702, 2023. | verified arXiv |
| Hassanien2016 | A. Hassanien et al., "Phase-modulation based dual-function radar-communications," IET Radar, Sonar & Navigation, 2016. | exists (Wiley page) |
