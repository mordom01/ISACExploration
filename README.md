# ISACExploration

Research scaffold for generative models of constant-modulus radar / ISAC waveforms (phase codes on the torus).
Phase-gated: read `docs/phase0_landscape_and_directions.md` first.

Layout
- `docs/phase0_landscape_and_directions.md` : literature map, premise audit, measured facts, ranked directions, paper framing
- `docs/references.md`                      : every citation used, with what was verified
- `isacgen/`                                : metrics (AF, PSL/ISL, cross-correlation, PAPR, masks), closed-form codes, classical baselines (CAN, Multi-CAN, torus GD)
- `experiments/YYYY-MM-DD_<id>_<name>.py`   : one script per experiment with a dated rationale header; never mutated after its run (copy for a new run)
- `results/<phase>/`                        : csv + png outputs and logs

Environment: numpy, scipy, matplotlib (Phase 0). PyTorch + TorchCFM from Phase 1.
