"""P2g - Hybrid: imitation-policy samples as initialisations for the shotgun search.
Date: 2026-10-05
Rationale: The policy alone is weak (median PSL 10 at N=64) but cheap (0.6 ms) and diverse; the search is strong
but starts from random sequences (median PSL ~13.6). If policy initialisations reach PSL <= 5 more often or in
fewer iterations than random initialisations, the learned model has a modest but real role as an initialiser
(a "learned restart distribution"). Same search settings for both; 300 starts each; N=64 and N=80.
Output: results/p2/p2g_init.csv
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch
from isacgen.policy import SeqPolicy, psl_torch_int
from isacgen.binseq import acorr_int, flip_deltas, lex_key, canonical
torch.manual_seed(0); torch.set_num_threads(1)
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "p2"
model = SeqPolicy(n_max=128, hidden=256, layers=2); model.load_state_dict(torch.load(OUT / "p2b_policy.pt")); model.eval()
rng = np.random.default_rng(0)

def descend(b, sideways=200, tabu=8, max_iters=5000):
    N = len(b); r = acorr_int(b).astype(float); key = lex_key(r); best_b, best_key = b.copy(), key; recent = []; ni = 0
    for it in range(max_iters):
        D = flip_deltas(b); R = r[None, :] + D; A = np.abs(R); P = A.max(1); C = (A == P[:, None]).sum(1); I = (R ** 2).sum(1)
        score = P * 1e9 + C * 1e5 + I
        for j in recent: score[j] = np.inf
        j = int(np.argmin(score)); b[j] = -b[j]; r = R[j]; key = (int(P[j]), int(C[j]), int(I[j])); recent.append(j); recent = recent[-tabu:]
        if key < best_key: best_key, best_b, ni = key, b.copy(), 0
        else:
            ni += 1
            if ni > sideways: break
    return best_b, best_key[0], it + 1

rows = []
for N in [64, 80]:
    S = model.sample(300, N).numpy().astype(float)
    inits = {"policy": S, "random": rng.choice([-1.0, 1.0], size=(300, N))}
    for name, X in inits.items():
        t0 = time.time(); ps = []; its = []; init_psl = [int(np.abs(acorr_int(x)).max()) for x in X]
        for x in X:
            b, p, it = descend(x.copy()); ps.append(p); its.append(it)
        ps = np.array(ps); its = np.array(its); dt = (time.time() - t0) / 300
        hist = dict(zip(*[v.tolist() for v in np.unique(ps, return_counts=True)]))
        print(f"N={N} init={name:6s}: init PSL median {np.median(init_psl):.0f}; final PSL hist {hist}; frac<=5 {(ps<=5).mean():.3f}; "
              f"mean iters {its.mean():.0f}; {dt*1e3:.1f} ms/start", flush=True)
        rows.append((N, name, np.median(init_psl), float((ps <= 5).mean()), float((ps <= 6).mean()), its.mean(), dt * 1e3, str(hist).replace(",", ";")))
with open(OUT / "p2g_init.csv", "w") as f:
    f.write("N,init,init_psl_median,frac_le5,frac_le6,mean_iters,ms_per_start,hist\n")
    for r in rows: f.write(",".join(str(v) for v in r) + "\n")
