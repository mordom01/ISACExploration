"""P0d - Comm-side audit numbers: classical CSK codebooks vs jointly designed unimodular codebooks.
Date: 2026-10-05
Rationale: The closest comm-side prior for Direction A is code-shift keying (CSK) DFRC with Gold/Kasami
codebooks (Tedesso & Romero, DSP 2018; Eedara et al., TAES 2022 on FH-MIMO). To state plainly what a
generated codebook can and cannot win, we compare at N=63, K=8: (i) Gold and Kasami codebooks, (ii) i.i.d.
random unimodular, (iii) Multi-CAN, (iv) window-GD (zero cross-correlation for |lag|<=W plus auto ISL),
on per-code PSL, zero-lag and windowed cross-correlation, and the Monte-Carlo SER of a coherent ML
correlation receiver in AWGN with and without a timing offset of up to W chips (receiver assumes zero offset).
The M-ary orthogonal-signalling SER is the reference bound used by Tedesso & Romero.
Outputs: results/p0/p0d_csk_codebooks.csv, results/p0/p0d_csk_ser.png
(Edited once before first commit: the timing-uncertain receiver now searches lags within +-W; a zero-lag receiver
 makes every codebook fail identically because the sent code's own autocorrelation sidelobe is what it sees.)
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from isacgen.metrics import psl_isl, xcorr_stats
from isacgen.codes import gold_set, kasami_small_set, random_unimodular
from isacgen.classical import multi_can, torus_gd_set
from isacgen import plotstyle
plotstyle.apply()
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "p0"; OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(11)
N, K, W = 63, 8, 3

books = {}
books["Gold(63)"] = gold_set(6)[:K].astype(complex)
books["Kasami(63)"] = kasami_small_set(6)[:K].astype(complex)
X0 = random_unimodular(K, N, rng)
books["iid random"] = X0
books["Multi-CAN"] = multi_can(X0, n_iter=1500)
wa = np.ones(2 * N - 1); wa[N - 1] = 0
wc = np.zeros(2 * N - 1); wc[N - 1 - W: N + W] = 1.0
t0 = time.time(); books["window-GD"] = torus_gd_set(X0, wa, 50 * wc, n_iter=300); tgd = time.time() - t0

def ser_mc(X, ebn0_db, n_trials=20000, max_offset=0, rng=rng):
    """Correlation receiver in AWGN. Perfect timing: coherent ML, argmax_m Re<y, x_m>.
    Timing uncertainty: the codeword arrives with an unknown offset uniform in [-max_offset, max_offset] chips
    and the receiver decides argmax_m max_{|k|<=W} |<y_k, x_m>| (noncoherent in delay and phase), so the error
    events are governed by the *windowed* cross-correlation peaks of the codebook. Eb = N / log2 K per bit."""
    K, N = X.shape
    bits = np.log2(K)
    errs = 0
    Es = N
    Xc = np.conj(X)
    for _ in range(n_trials // 1000):
        m = rng.integers(0, K, size=1000)
        off = rng.integers(-max_offset, max_offset + 1, size=1000) if max_offset > 0 else np.zeros(1000, int)
        rx = np.zeros((1000, N + 2 * W), complex)
        for i in range(1000):
            rx[i, W + off[i]: W + off[i] + N] = X[m[i]]
        N0 = Es / (bits * 10 ** (ebn0_db / 10))
        rx += np.sqrt(N0 / 2) * (rng.standard_normal(rx.shape) + 1j * rng.standard_normal(rx.shape))
        if max_offset == 0:
            corr = np.real(rx[:, W: W + N] @ Xc.T)
            errs += (corr.argmax(axis=1) != m).sum()
        else:
            best = np.full((1000, K), -np.inf)
            for k in range(-W, W + 1):
                corr = np.abs(rx[:, W + k: W + k + N] @ Xc.T)
                best = np.maximum(best, corr)
            errs += (best.argmax(axis=1) != m).sum()
    return errs / (n_trials // 1000 * 1000)

ebn0 = np.arange(0, 13, 2)
rows = []
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
for j, (name, X) in enumerate(books.items()):
    X = X / np.abs(X)  # all unimodular
    psl = 20 * np.log10(psl_isl(X)[0])
    st = xcorr_stats(X); stw = xcorr_stats(X, lag_window=W)
    ser0 = [ser_mc(X, e, max_offset=0) for e in ebn0]
    serW = [ser_mc(X, e, max_offset=W) for e in ebn0]
    rows.append((name, psl.max(), psl.mean(), st["zero_lag"].max(), stw["peak"].max(), st["peak"].max(), ser0[3], serW[3]))
    print(f"{name:12s} worst PSL {psl.max():6.1f} dB | mean PSL {psl.mean():6.1f} | worst zero-lag {st['zero_lag'].max():.3f} | "
          f"worst peak |k|<={W}: {stw['peak'].max():.3f} | worst peak all lags {st['peak'].max():.3f} | SER@6dB sync {ser0[3]:.4f} / offset<= {W}: {serW[3]:.4f}")
    ax[0].semilogy(ebn0, np.maximum(ser0, 1e-5), marker="o", label=name, color=plotstyle.SERIES[j])
    ax[1].semilogy(ebn0, np.maximum(serW, 1e-5), marker="o", label=name, color=plotstyle.SERIES[j])
# M-ary orthogonal signalling SER (union bound) for reference
from scipy.stats import norm
bits = np.log2(K)
ub = [(K - 1) * norm.sf(np.sqrt(bits * 10 ** (e / 10))) for e in ebn0]
for a in ax:
    a.semilogy(ebn0, np.minimum(ub, 1), ls="--", color=plotstyle.TEXT2, label="M-ary orthogonal, union bound")
    a.set_xlabel("Eb/N0 (dB)"); a.set_ylabel("codeword SER"); a.set_ylim(1e-5, 1)
ax[0].set_title(f"CSK, K={K}, N={N}, perfect timing"); ax[1].set_title(f"CSK, timing offset uniform in ±{W} chips, delay-searching receiver")
ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "p0d_csk_ser.png")
with open(OUT / "p0d_csk_codebooks.csv", "w") as f:
    f.write("codebook,worst_psl_db,mean_psl_db,worst_zero_lag,worst_window_peak,worst_peak_all,ser_6dB_sync,ser_6dB_offset\n")
    for r in rows: f.write(",".join(str(v) for v in r) + "\n")
print("window-GD time %.1f s" % tgd, "saved")
