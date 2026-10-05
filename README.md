# ISACExploration

Research scaffold and run log for generative models of radar / ISAC waveforms (constant-modulus phase codes and
binary sequences), with the classical optimisers they must beat. Phase-gated; every experiment is a dated script
that is never edited after its run, so the scripts double as the lab notebook.

Read in order:
1. `docs/phase0_landscape_and_directions.md` : literature map, premise audit, measured facts, ranked directions
2. `docs/phase0b_comm_side_audit.md`        : comm-side baselines (CSK / index modulation), what can and cannot be won
3. `docs/phase1_interim.md`                 : torus flow matching fails; mechanism; gradient descent dominates
4. `docs/phase2_plan.md`, `docs/phase2_interim.md` : discrete-phase constructive policy; classical search; results
5. `docs/recommendation.md`                 : paper framing and next steps
6. `docs/references.md`                     : every citation used, with what was verified

Code: `isacgen/` (metrics, classical optimisers, binary search engine, torus flow matching, constructive policy),
`experiments/` (run scripts), `results/` (tables, figures, logs, datasets).

Environment: Python 3.11, numpy, scipy, matplotlib, torch (CPU wheels used here; GPU works unchanged).
