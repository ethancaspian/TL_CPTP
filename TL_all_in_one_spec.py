#!/usr/bin/env python3
# TL: minimal, single-file simulation + plots
# Reproduces: PSD (Sx), cross-spectra demo, and sensitivity contour.
# Purely synthetic; OU surrogate matches target Lorentzian Sx.

import argparse, numpy as np, matplotlib.pyplot as plt
from scipy.signal import welch, csd, coherence

# ---------- Model ----------
def Sx_lorentzian(f, tau0, sig_xi):
    # one-sided PSD for timing deviation x(t)
    return (sig_xi**2 * tau0**3) / (1.0 + (2.0*np.pi*f*tau0)**2)

def gen_common_x(N, fs, tau0, sig_xi, seed=0):
    """Stable OU step; then global scale so low-f PSD matches Sx(0)."""
    rng = np.random.default_rng(seed)
    dt = 1.0/fs
    a = np.exp(-dt/tau0)
    x = np.zeros(N)
    for n in range(1, N):
        x[n] = a*x[n-1] + np.sqrt(1 - a*a)*rng.standard_normal()
    # Calibrate amplitude to match analytic Sx near f≈0:
    f, Pxx = welch(x, fs=fs, nperseg=min(1<<14, N//2), scaling='density')
    idx = 1 if len(f) > 1 else 0
    target = sig_xi**2 * tau0**3
    x *= np.sqrt(target / (Pxx[idx] + 1e-30))
    return x

def add_local(xc, alpha, rms_frac=0.6, seed=0):
    rng = np.random.default_rng(seed)
    loc = rng.standard_normal(len(xc))
    loc = loc/np.std(loc) * (np.std(xc)*rms_frac)
    return alpha*xc + loc

# ---------- Figures ----------
def fig_psd(fs, tau0, sig_xi, seed, out="figure2_jitter_spectrum.pdf"):
    N = 1<<18
    x = gen_common_x(N, fs, tau0, sig_xi, seed)
    f, Pxx = welch(x, fs=fs, nperseg=1<<14, noverlap=1<<13, scaling='density')
    plt.figure(figsize=(3.5,2.6), dpi=200)
    plt.loglog(f, Pxx, label="Simulated $S_x$")
    plt.loglog(f, Sx_lorentzian(f, tau0, sig_xi), "--", label="Analytic $S_x$")
    fc = 1/(2*np.pi*tau0)
    plt.axvline(fc, ls=":", lw=1.0)
    plt.xlabel("Frequency $f$ [Hz]"); plt.ylabel(r"$S_x(f)\,[\mathrm{s}^3]$")
    plt.title("Timing-deviation PSD (one-sided)"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out); plt.close(); return out

def fig_cross(M, fs, tau0, sig_xi, seed, out="fig2-cross-spectra.pdf", out2="fig2-cross-spectra_coherence.pdf"):
    N = 1<<17; rng = np.random.default_rng(seed)
    xc = gen_common_x(N, fs, tau0, sig_xi, seed)
    alphas = 1.0 + 0.2*rng.standard_normal(M)
    X = np.stack([add_local(xc, alphas[m], seed=seed+m+1) for m in range(M)], 0)
    a,b = 0,1
    f, Saa = welch(X[a], fs=fs, nperseg=1<<12, noverlap=1<<11)
    _, Sbb = welch(X[b], fs=fs, nperseg=1<<12, noverlap=1<<11)
    _, Sab = csd(X[a], X[b], fs=fs, nperseg=1<<12, noverlap=1<<11)
    _, Coh = coherence(X[a], X[b], fs=fs, nperseg=1<<12, noverlap=1<<11)

    plt.figure(figsize=(3.5,2.6), dpi=200)
    plt.loglog(f, np.abs(Sab), label=r"$|S_{ab}|$")
    plt.loglog(f, Saa, "--", label=r"$S_{aa}$")
    plt.loglog(f, Sbb, "--", label=r"$S_{bb}$")
    plt.xlabel("Frequency $f$ [Hz]"); plt.ylabel("PSD / cross-PSD")
    plt.title("Cross-spectral signature"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out); plt.close()

    plt.figure(figsize=(3.5,2.0), dpi=200)
    plt.semilogx(f, Coh); plt.ylim(0,1.05)
    plt.xlabel("Frequency $f$ [Hz]"); plt.ylabel("Coherence")
    plt.title("Pairwise coherence"); plt.tight_layout(); plt.savefig(out2); plt.close()
    return out, out2

def fig_sensitivity(M, Nseg, Sloc_fc, C=10.0, tau0_grid=None, sig_grid=None, out="fig3-sensitivity.pdf"):
    if tau0_grid is None: tau0_grid = np.logspace(-14.5, -12.0, 120)
    if sig_grid  is None: sig_grid  = np.logspace(-6.0,  -3.0, 120)
    Tau, Sig = np.meshgrid(tau0_grid, sig_grid, indexing='xy')
    lhs = (Sig**2)*Tau
    rhs = C*Sloc_fc/np.sqrt(M*(M-1)/2 * Nseg)
    det = lhs >= rhs
    plt.figure(figsize=(3.5,2.6), dpi=200)
    plt.contourf(Tau, Sig, det, levels=[-0.5,0.5,1.5], alpha=0.85)
    plt.contour(Tau, Sig, lhs, levels=[rhs], colors='k', linewidths=1.0)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel(r"$\tau_0$ [s]"); plt.ylabel(r"$\sigma_\xi$")
    plt.title("Detectable region (shaded)")
    plt.tight_layout(); plt.savefig(out); plt.close(); return out

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Temporal Locations: single-file simulation (PSD, cross-PSD, sensitivity).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fs", type=float, default=1e4)
    ap.add_argument("--tau0", type=float, default=1e-13)
    ap.add_argument("--sigma_xi", type=float, default=1e-4)
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--Nseg", type=int, default=512)
    ap.add_argument("--Sloc_fc", type=float, default=1e-27)
    ap.add_argument("--no_coherence", action="store_true", help="skip coherence panel")
    args = ap.parse_args()

    print(f"[info] fs={args.fs:g} Hz, tau0={args.tau0:g} s, sigma_xi={args.sigma_xi:g}, seed={args.seed}")
    f1 = fig_psd(args.fs, args.tau0, args.sigma_xi, args.seed)
    f2, f2c = fig_cross(args.M, args.fs, args.tau0, args.sigma_xi, args.seed)
    if args.no_coherence:
        print("[note] skipping coherence panel"); f2c = "(skipped)"
    f3 = fig_sensitivity(args.M, args.Nseg, args.Sloc_fc)
    print("[ok] wrote:", f1, f2, f2c, f3)

if __name__ == "__main__":
    main()
