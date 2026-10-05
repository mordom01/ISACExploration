"""P1d - Control: is the torus flow-matching code path able to learn a LOW-entropy target?
Date: 2026-10-05
Rationale: P1b (optimiser-generated codes, high entropy, Haar-like spread) produced Haar-level samples. To
separate "the target is practically unlearnable at this compute" from "the code is broken", train the same model
on a 2-parameter family: P4(64) under a random global phase and a random linear phase ramp (both leave |r_k|
invariant), which is the kind of parametric family the earlier conditional WGAN was trained on (Frank / P4 /
Oppermann). Success = sampled PSL near P4's -24.4 dB within a few hundred steps. Usage: <steps>
Output: results/p1/p1d_control.log (via stdout)
"""
import sys, pathlib, time, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch
from isacgen.torusfm import TorusVelocityNet, cfm_loss, sample, wrap, psl_db_torch
from isacgen.codes import p4
torch.manual_seed(0); torch.set_num_threads(4)
steps = int(sys.argv[1]) if len(sys.argv) > 1 else 800
N = 64
base = torch.tensor(np.angle(p4(N)), dtype=torch.float32)
n = torch.arange(N, dtype=torch.float32)
def batch(bs):
    c = torch.rand(bs, 1) * 2 * math.pi - math.pi
    w = torch.rand(bs, 1) * 2 * math.pi - math.pi
    return wrap(base[None, :] + c + w * n[None, :])
print("target family PSL", float(psl_db_torch(batch(4)).mean()))
model = TorusVelocityNet(N, d_model=128, n_layers=4, n_heads=4)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
t0 = time.time()
for it in range(steps):
    loss = cfm_loss(model, batch(256))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if it % 100 == 0 or it == steps - 1:
        model.eval()
        with torch.no_grad():
            S = sample(model, 500, N, nfe=8); p = psl_db_torch(S)
        model.train()
        print(f"it {it} loss {loss.item():.3f} sampled PSL median {p.median():.2f} frac<=-22dB {(p<=-22).float().mean():.3f} ({time.time()-t0:.0f}s)", flush=True)
