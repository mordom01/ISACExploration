"""P2h - Are the optimal-PSL samples of the short-length policy new sequences or memorised ones?
Date: 2026-10-05
Rationale: At N=24 the P2e policy emits optimal (PSL 3) sequences in ~12% of samples. If those are mostly new
canonical sequences not in the 416-sequence training set, the policy generalises within length and the yield
of *new* optimal sequences per second can be compared with the shotgun search's. Per PSL class: samples,
distinct canonical, in-train, new; for N = 22, 24, 26 (train) and 30, 32 (transfer, T = 3 best known/optimal).
Then shotgun for the same wall-clock, counting new optimal sequences not in its own earlier finds (fair: both
methods are credited for sequences outside the training set).
Output: results/p2/p2h_novelty.csv
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch
from isacgen.policy import SeqPolicy, psl_torch_int
from isacgen.binseq import canonical, shotgun_psl
torch.manual_seed(1); torch.set_num_threads(1)
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "p2"
model = SeqPolicy(n_max=128, hidden=256, layers=2); model.load_state_dict(torch.load(OUT / "p2e_policy.pt")); model.eval()
rng = np.random.default_rng(2)
rows = []
for N, T in [(22, 3), (24, 3), (26, 3), (30, 3), (32, 3)]:
    f = OUT / f"binseq_N{N}.npz"
    train = set(map(tuple, np.load(f)["seqs"].tolist())) if f.exists() else set()
    t0 = time.time(); S = model.sample(20000, N); dt = time.time() - t0
    p = psl_torch_int(S).numpy(); ok = p <= T
    canon = [canonical(s) for s in S.numpy()[ok]]
    distinct = set(canon); new = distinct - train
    print(f"policy  N={N} T={T}: {ok.sum()} samples at PSL<={T} of 20000 ({dt:.1f}s, 1 thread); distinct {len(distinct)}; in-train {len(distinct & train)}; NEW {len(new)}; new/s {len(new)/dt:.2f}", flush=True)
    rows.append((N, T, "policy", int(ok.sum()), len(distinct), len(distinct & train), len(new), dt, len(new) / dt))
    t0 = time.time(); found = set(); cnt = 0
    while time.time() - t0 < dt:
        b, ps, _ = shotgun_psl(N, rng, max_iters=3000, sideways=100); cnt += 1
        if ps <= T: found.add(canonical(b))
    newf = found - train
    print(f"shotgun N={N} T={T}: {cnt} restarts in {dt:.1f}s; distinct {len(found)}; in-train {len(found & train)}; NEW {len(newf)}; new/s {len(newf)/dt:.2f}", flush=True)
    rows.append((N, T, "shotgun", cnt, len(found), len(found & train), len(newf), dt, len(newf) / dt))
with open(OUT / "p2h_novelty.csv", "w") as fh:
    fh.write("N,T,method,n_or_restarts,distinct,in_train,new,seconds,new_per_sec\n")
    for r in rows: fh.write(",".join(str(v) for v in r) + "\n")
