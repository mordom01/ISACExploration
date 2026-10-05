"""P2b - Imitation training of the constructive policy on search outputs, and gates G2-G4.
Date: 2026-10-05
Rationale: Train p(b_n | running autocorrelation, position, N) by teacher forcing on the P2a sequences
(lengths 32..72, PSL within 1 of the best found), with the 8-fold symmetry augmentation (negation, reversal,
alternation: all preserve PSL). Evaluate at N=64: PSL distribution of 10^4 samples, distinct canonical
sequences, overlap with the training set, time per sequence; and transfer at N=80 and 96 (not in training).
Classical reference at matched wall-clock: shotgun search yield from P2a logs / isacgen.binseq.
Usage: python <script> <steps> [hidden] [layers]
Outputs: results/p2/p2b_policy.pt, results/p2/p2b_eval.csv, results/p2/p2b.log (stdout)
"""
import sys, pathlib, time, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch, torch.nn.functional as F
from isacgen.policy import SeqPolicy, psl_torch_int
from isacgen.binseq import canonical, shotgun_psl
torch.manual_seed(0); np.random.seed(0)
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "p2"
steps = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
hidden = int(sys.argv[2]) if len(sys.argv) > 2 else 256
layers = int(sys.argv[3]) if len(sys.argv) > 3 else 2
N_MAX = 128
TRAIN_N = [32, 40, 48, 56, 64, 72]

# ---- data ----
data = {}
train_canon = {}
for N in TRAIN_N:
    f = OUT / f"binseq_N{N}.npz"
    if not f.exists():
        print(f"missing {f}; skipping length {N}"); continue
    d = np.load(f); S = d["seqs"].astype(np.float32) * 2 - 1; P = d["psl"]
    # sharpen the imitation target: keep only the best PSL class when it is populous enough, else best+1
    best = P.min(); keep = P <= (best if (P == best).sum() >= 2000 else best + 1)
    S, P = S[keep], P[keep]
    data[N] = (torch.from_numpy(S), P)
    train_canon[N] = set(map(tuple, d["seqs"].tolist()))
    print(f"N={N}: {S.shape[0]} sequences, PSL hist {dict(zip(*[x.tolist() for x in np.unique(P, return_counts=True)]))}")
assert data, "no data"

def augment(B):
    """Random element of the PSL-preserving group applied per sequence."""
    bs, N = B.shape
    sgn = (torch.rand(bs, 1) < 0.5).float() * 2 - 1
    B = B * sgn
    rev = torch.rand(bs) < 0.5
    B[rev] = torch.flip(B[rev], dims=[1])
    alt = (torch.rand(bs) < 0.5)
    alt_sign = ((-1.0) ** torch.arange(N))[None, :]
    B[alt] = B[alt] * alt_sign
    return B

def batch(bs):
    N = TRAIN_N[np.random.randint(len(data))] if len(data) == len(TRAIN_N) else list(data.keys())[np.random.randint(len(data))]
    S, _ = data[N]
    idx = torch.randint(0, S.shape[0], (bs,))
    return augment(S[idx].clone()), N

torch.set_num_threads(4)
model = SeqPolicy(n_max=N_MAX, hidden=hidden, layers=layers)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
t0 = time.time(); losses = []
for it in range(steps):
    B, N = batch(128)
    L = torch.full((B.shape[0],), N)
    logits = model(B, L)
    loss = F.binary_cross_entropy_with_logits(logits, (B > 0).float())
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    losses.append(loss.item())
    if it % 250 == 0 or it == steps - 1:
        print(f"it {it} nll/chip {np.mean(losses[-50:]):.4f} ({time.time()-t0:.0f}s)", flush=True)
    if (it % 1000 == 0 and it > 0) or it == steps - 1:
        model.eval()
        S = model.sample(2000, 64); p = psl_torch_int(S)
        model.train()
        print(f"   [eval it {it}] N=64 sampled PSL hist {dict(zip(*[x.tolist() for x in torch.unique(p, return_counts=True)]))}", flush=True)
torch.save(model.state_dict(), OUT / "p2b_policy.pt")

# ---- evaluation ----
model.eval()
rows = []
def evaluate(N, n_samp=10000, temperature=1.0):
    t1 = time.time(); S = model.sample(n_samp, N, temperature=temperature); dt = (time.time() - t1) / n_samp
    p = psl_torch_int(S).numpy()
    canon = set(canonical(s) for s in S.numpy())
    overlap = len(canon & train_canon.get(N, set())) / max(1, len(canon))
    hist = dict(zip(*[x.tolist() for x in np.unique(p, return_counts=True)]))
    print(f"N={N} T={temperature}: PSL hist {hist}; median {np.median(p):.0f}; distinct canonical {len(canon)}/{n_samp}; "
          f"overlap with train {overlap:.3f}; {dt*1e3:.3f} ms/seq (4 threads)", flush=True)
    rows.append((N, temperature, np.median(p), float((p <= 5).mean()), float((p <= 6).mean()), len(canon), overlap, dt * 1e3, str(hist).replace(",", ";")))
    return p
for N in [64, 80, 96]:
    for T in [1.0, 0.8]:
        evaluate(N, temperature=T)
# matched-time classical reference at N=64, 80, 96: shotgun restarts for the same wall-clock as 10^4 samples
rng = np.random.default_rng(1)
for N in [64, 80, 96]:
    budget = rows[[r[0] for r in rows].index(N)][7] * 1e-3 * 10000 * 4   # same CPU-seconds (4 threads) as the policy
    t1 = time.time(); ps = []
    while time.time() - t1 < budget:
        _, p, _ = shotgun_psl(N, rng, max_iters=5000, sideways=200); ps.append(p)
    ps = np.array(ps)
    print(f"shotgun N={N} in {budget:.1f}s (1 thread): {len(ps)} restarts, PSL hist {dict(zip(*[x.tolist() for x in np.unique(ps, return_counts=True)]))}", flush=True)
    rows.append((N, "shotgun", np.median(ps), float((ps <= 5).mean()), float((ps <= 6).mean()), len(ps), 0.0, budget * 1e3 / max(1, len(ps)), ""))
with open(OUT / "p2b_eval.csv", "w") as f:
    f.write("N,temperature_or_method,psl_median,frac_le5,frac_le6,distinct_or_restarts,train_overlap,ms_per_seq,hist\n")
    for r in rows: f.write(",".join(str(v) for v in r) + "\n")
print("saved")
