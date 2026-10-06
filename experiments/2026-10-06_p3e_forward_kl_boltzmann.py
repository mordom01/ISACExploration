"""P3e - Closing the sampler gap: forward-KL (maximum-likelihood on MCMC samples) training of the autoregressive
Boltzmann generator, with a short reverse-KL refinement.
Date: 2026-10-06
Rationale: P3d's reverse-KL (REINFORCE) training was under-converged at beta=10 (KL(q||Q_beta) = 1.98 nats) and
mode-collapsed at beta>=40. Forward KL on exact samples is mode-covering and has a stable, low-variance gradient.
Protocol per beta in {10, 20}: (1) 4 x 8000 Metropolis chains (25 sweeps) -> 32k approximately-Gibbs samples;
(2) train the GRU sampler by maximum likelihood (3000 steps, batch 512, 8-fold symmetry augmentation is NOT used
because the Gibbs law is not symmetric under sign of individual symbols); (3) 300 steps of reverse-KL refinement
from that initialisation; (4) evaluate KL(q||Q_beta) via thermodynamic-integration log Z on an equilibrated grid
(beta <= 20), KL(q||P), E[PSL], fraction of distinct frames in 10k draws, held-out detector AUC, CSK SER with a
seed-shared K=8 codebook, ms per frame. Reference rows: MCMC at the same beta; P3d reverse-KL numbers quoted.
Output: results/p3/p3e_forward_kl.csv
"""
import sys, pathlib, math, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch
src = open(pathlib.Path(__file__).resolve().parents[0] / "2026-10-06_p3d_boltzmann_covert.py").read()
# reuse P3d's definitions (modulator, cost, detector, MCMC, VAN, CSK) without running its experiment body
head = src.split("betas = [0, 5, 10, 20, 40, 80, 160]")[0]
ns = {"__file__": str(pathlib.Path(__file__).resolve().parents[0] / "2026-10-06_p3d_boltzmann_covert.py")}
exec(compile(head, "p3d_head", "exec"), ns)
van_src = src.split("# ---------------- (2) autoregressive Boltzmann generator ----------------")[1].split("for b in [10, 40, 160]:")[0]
exec(compile(van_src, "p3d_van", "exec"), ns)
mcmc, cost_sym, modulate, psl_db, auc_vs_background, VAN, ser_csk, NB, N = (ns[k] for k in ["mcmc", "cost_sym", "modulate", "psl_db", "auc_vs_background", "VAN", "ser_csk", "NB", "N"])
torch.manual_seed(1); OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "p3"
steps_ml = int(sys.argv[1]) if len(sys.argv) > 1 else 3000

# equilibrated grid for thermodynamic integration
grid = [0, 2.5, 5, 10, 15, 20]; Ec = {}; samples = {}
t0 = time.time()
for b in grid:
    s, c = mcmc(b, n_chains=8000, sweeps=25); Ec[b] = float(c.mean()); samples[b] = s
    print(f"MCMC beta={b:5.1f}: E[c] {Ec[b]:.4f} ({time.time()-t0:.0f}s)", flush=True)
logZ = {0: NB * math.log(2)}
for i in range(1, len(grid)):
    b0, b1 = grid[i - 1], grid[i]; logZ[b1] = logZ[b0] - 0.5 * (Ec[b0] + Ec[b1]) * (b1 - b0)
rows = []
for b in [10, 20]:
    # more exact samples for training: 3 extra MCMC runs
    S = [samples[b]] + [mcmc(b, n_chains=8000, sweeps=25)[0] for _ in range(3)]
    S = torch.cat(S); print(f"beta={b}: {S.shape[0]} MCMC samples, E[c] {cost_sym(S).mean():.4f} ({time.time()-t0:.0f}s)", flush=True)
    kl_mcmc = NB * math.log(2) - b * Ec[b] - logZ[b]
    phi_ref = modulate(samples[b]); a_ref = auc_vs_background(phi_ref)
    rows.append(("MCMC", b, kl_mcmc, 0.0, Ec[b], float(psl_db(phi_ref).mean()), a_ref, float("nan"), float("nan"), float("nan")))
    print(f"MCMC beta={b}: KL(Q||P) {kl_mcmc:.2f}, E[PSL] {psl_db(phi_ref).mean():.1f} dB, AUC {a_ref:.3f}", flush=True)
    van = VAN(); opt = torch.optim.Adam(van.parameters(), 2e-3); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps_ml)
    for it in range(steps_ml):
        idx = torch.randint(0, S.shape[0], (512,))
        loss = -van.logprob(S[idx]).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(van.parameters(), 1.0); opt.step(); sched.step()
        if it % 500 == 0 or it == steps_ml - 1:
            with torch.no_grad():
                sym, logq = van.sample(4000); c = cost_sym(sym)
            print(f"  ML beta={b} it {it}: nll/frame {loss.item():.2f} nats | KL(q||Q_beta) {float((logq + b * c).mean()) + logZ[b]:.2f} | E_q[c] {c.mean():.4f} ({time.time()-t0:.0f}s)", flush=True)
    def evaluate(tag):
        with torch.no_grad():
            t2 = time.time(); sym, logq = van.sample(10000); dt = (time.time() - t2) / 10000
            c = cost_sym(sym); phi = modulate(sym)
            kl_qP = float(logq.mean()) + NB * math.log(2); kl_qQ = float((logq + b * c).mean()) + logZ[b]
            distinct = len(set(map(tuple, (sym > 0).to(torch.int8).tolist()))) / 10000
        a = auc_vs_background(phi); ser = {e: ser_csk(van.sample, e) for e in [0, 4, 8, 12]}
        print(f"{tag} beta={b}: KL(q||P) {kl_qP:.2f}, KL(q||Q_beta) {kl_qQ:.2f}, E[c] {c.mean():.4f}, E[PSL] {psl_db(phi).mean():.1f} dB, distinct {distinct:.3f}, AUC {a:.3f}, {dt*1e3:.3f} ms/frame; SER @0/4/8/12 {ser[0]:.3f}/{ser[4]:.3f}/{ser[8]:.3f}/{ser[12]:.3f}", flush=True)
        rows.append((tag, b, kl_qP, kl_qQ, float(c.mean()), float(psl_db(phi).mean()), a, distinct, dt * 1e3, ser[8]))
    evaluate("VAN-forwardKL")
    torch.save(van.state_dict(), OUT / f"p3e_van_fkl_beta{b}.pt")
    # short reverse-KL refinement from the ML initialisation (low lr, baseline)
    opt = torch.optim.Adam(van.parameters(), 2e-4); base = None
    for it in range(300):
        sym, logq = van.sample(512); c = cost_sym(sym); R = (logq + b * c).detach()
        base = R.mean() if base is None else 0.9 * base + 0.1 * R.mean()
        loss = ((R - base) * logq).mean(); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(van.parameters(), 1.0); opt.step()
    evaluate("VAN-forwardKL+revKL")
    torch.save(van.state_dict(), OUT / f"p3e_van_hybrid_beta{b}.pt")
with open(OUT / "p3e_forward_kl.csv", "w") as f:
    f.write("sampler,beta,kl_to_background,kl_to_gibbs,E_cost,E_psl_db,detector_auc,distinct_frac,ms_per_frame,ser_8dB\n")
    for r in rows: f.write(",".join(str(v) for v in r) + "\n")
print("saved")
