"""P2c - Policy-guided constrained sampling: imitation policy + partial-autocorrelation pruning.
Date: 2026-10-05
Rationale: Exhaustive PSL searches prune a branch as soon as any partial autocorrelation exceeds the target T,
because |r_k| only accumulates at lags k >= n (new terms) and the already-closed lags are final. We combine the
trained policy p(b_n | state) with the same rule: at each step, a chip that makes any closed lag exceed T is
forbidden (if both chips are forbidden the policy's choice stands and the sample will fail). This turns the
sampler into a stochastic depth-first search guided by the learned branching probabilities, without backtracking.
We measure yield (distinct sequences with PSL <= T per second) vs. the shotgun search at matched wall-clock, for
T in {5, 6} at N=64 and T in {5, 6} at N=80, 96 (transfer). Usage: <policy.pt> [hidden] [layers]
Outputs: results/p2/p2c_pruned.csv
"""
import sys, pathlib, time, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch
from isacgen.policy import SeqPolicy, psl_torch_int
from isacgen.binseq import canonical, shotgun_psl
torch.manual_seed(0)
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "p2"
ckpt = sys.argv[1] if len(sys.argv) > 1 else str(OUT / "p2b_policy.pt")
hidden = int(sys.argv[2]) if len(sys.argv) > 2 else 256
layers = int(sys.argv[3]) if len(sys.argv) > 3 else 2
torch.set_num_threads(4)
model = SeqPolicy(n_max=128, hidden=hidden, layers=layers); model.load_state_dict(torch.load(ckpt)); model.eval()


@torch.no_grad()
def sample_pruned(model, bs, N, T, temperature=1.0):
    """Ancestral sampling with hard pruning: a chip is forbidden if it makes a *closed* partial lag exceed T.
    Lag k is closed after chip n if n - k ... (all terms present) never holds before the end for aperiodic
    correlation, so we use the exhaustive-search bound: after emitting n+1 chips, the final r_k satisfies
    |r_k| >= |r^{(n+1)}_k| - (N - 1 - n) for lags k <= n (each remaining chip adds at most 1 to each lag);
    forbid chips for which |r^{(n+1)}_k| - (N-1-n) > T for some k."""
    n_max = 128
    B = torch.zeros(bs, N)
    r = torch.zeros(bs, n_max - 1)
    last = torch.zeros(bs, 1); hstate = None; scale = 1 / math.sqrt(N)
    for n in range(N):
        f = torch.cat([r * scale, last, torch.full((bs, 1), n / N), torch.full((bs, 1), N / n_max)], -1)
        if model is None:                                               # control: uninformed policy
            p = torch.full((bs,), 0.5)
        else:
            h, hstate = model.gru(model.inp(f)[:, None, :], hstate)
            p = torch.sigmoid(model.out(h[:, 0]).squeeze(-1) / temperature)
        if n > 0:
            k = min(n, n_max - 1)
            prev = torch.flip(B[:, n - k: n], dims=[1])                    # b_{n-1} .. b_{n-k}
            remaining = N - 1 - n
            ok_plus = ((r[:, :k] + prev).abs() - remaining <= T).all(dim=1)
            ok_minus = ((r[:, :k] - prev).abs() - remaining <= T).all(dim=1)
            p = torch.where(ok_plus & ~ok_minus, torch.ones_like(p), p)
            p = torch.where(ok_minus & ~ok_plus, torch.zeros_like(p), p)
        b = (torch.rand(bs) < p).float() * 2 - 1
        B[:, n] = b
        if n > 0:
            r[:, :k] += b[:, None] * torch.flip(B[:, n - k: n], dims=[1])
        last = b[:, None]
    return B

rows = []
rng = np.random.default_rng(3)
for N, Ts in [(64, [5, 6]), (80, [5, 6]), (96, [6, 7])]:
    for T in Ts:
        n_samp = 4000
        t0 = time.time(); S = sample_pruned(model, n_samp, N, T); dt = time.time() - t0
        p = psl_torch_int(S).numpy(); ok = p <= T
        canon = set(canonical(s) for s in S.numpy()[ok])
        hist = dict(zip(*[x.tolist() for x in np.unique(p, return_counts=True)]))
        print(f"policy+prune N={N} T={T}: {dt:.1f}s for {n_samp} samples; frac PSL<={T}: {ok.mean():.3f}; distinct {len(canon)}; "
              f"yield {len(canon)/dt:.2f} distinct/s (4 threads); hist {hist}", flush=True)
        rows.append((N, T, "policy+prune", n_samp, dt, float(ok.mean()), len(canon), len(canon) / dt))
        t0 = time.time(); S0 = sample_pruned(None, n_samp, N, T); dt0 = time.time() - t0
        p0 = psl_torch_int(S0).numpy(); ok0 = p0 <= T; canon0 = set(canonical(s) for s in S0.numpy()[ok0])
        print(f"random+prune N={N} T={T}: {dt0:.1f}s; frac PSL<={T}: {ok0.mean():.4f}; distinct {len(canon0)}; yield {len(canon0)/dt0:.2f} distinct/s", flush=True)
        rows.append((N, T, "random+prune", n_samp, dt0, float(ok0.mean()), len(canon0), len(canon0) / dt0))
        # matched wall-clock shotgun (1 thread; multiply yield by 4 for a fair 4-thread comparison)
        t0 = time.time(); found = set(); cnt = 0
        while time.time() - t0 < dt * 4:
            b, ps, _ = shotgun_psl(N, rng, max_iters=5000, sideways=200); cnt += 1
            if ps <= T: found.add(canonical(b))
        print(f"shotgun     N={N} T={T}: {cnt} restarts in {dt*4:.1f}s (1 thread = same CPU-seconds); distinct PSL<={T}: {len(found)}; yield {len(found)/(dt*4):.2f} distinct/s/thread -> x4 = {len(found)/dt:.2f}", flush=True)
        rows.append((N, T, "shotgun", cnt, dt * 4, len(found) / max(1, cnt), len(found), len(found) / dt))
with open(OUT / "p2c_pruned.csv", "w") as f:
    f.write("N,T,method,n,seconds,frac_success,distinct,distinct_per_sec_4thread_equiv\n")
    for r in rows: f.write(",".join(str(v) for v in r) + "\n")
print("saved")
