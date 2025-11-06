
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, csd, coherence

def lorentzian_psd_x(f, tau0, sigma_xi):
    return (sigma_xi**2 * tau0**3) / (1.0 + (2.0*np.pi*f*tau0)**2)

def generate_common_mode_x(N, fs, tau0, sigma_xi, seed=0):
    rng = np.random.default_rng(seed)
    dt = 1.0/fs
    x = np.zeros(N, dtype=float)
    eta_std = 1.0
    for n in range(1, N):
        x[n] = x[n-1] + dt*(-x[n-1]/tau0) + np.sqrt(dt)*eta_std*rng.standard_normal()
    f_tmp, Pxx = welch(x, fs=fs, nperseg=min(1<<14, N//2), noverlap=None, detrend='constant', return_onesided=True, scaling='density')
    target0 = sigma_xi**2 * tau0**3
    idx0 = 1 if len(f_tmp) > 1 else 0
    scale = np.sqrt(target0 / (Pxx[idx0] + 1e-30))
    x *= scale
    return x

def add_local_noise(x_common, alpha, fs, local_rms_frac=0.5, seed=0):
    rng = np.random.default_rng(seed)
    N = len(x_common)
    local = rng.standard_normal(N)
    local = local / np.std(local) * (np.std(x_common) * local_rms_frac)
    return alpha * x_common + local

def figure2_jitter_spectrum(fs, tau0, sigma_xi, seed=0, out="figure2_jitter_spectrum.pdf"):
    N = 1<<18
    x = generate_common_mode_x(N, fs, tau0, sigma_xi, seed=seed)
    f, Pxx = welch(x, fs=fs, nperseg=1<<14, noverlap=1<<13, detrend='constant', return_onesided=True, scaling='density')
    Sx = lorentzian_psd_x(f, tau0, sigma_xi)
    fc = 1.0/(2.0*np.pi*tau0)

    plt.figure(figsize=(3.5, 2.6), dpi=200)
    plt.loglog(f, Pxx, label="Simulated $S_x(f)$")
    plt.loglog(f, Sx, linestyle="--", label="Analytic $S_x(f)$")
    plt.axvline(fc, linestyle=":", linewidth=1.0)
    plt.text(fc*1.05, Sx[np.argmin(np.abs(f-fc))], r"$f_c=(2\pi\tau_0)^{-1}$", fontsize=8, rotation=90, va='bottom')
    plt.xlabel("Frequency $f$ [Hz]")
    plt.ylabel(r"$S_x(f)\ [\mathrm{s}^3]$")
    plt.title("Timing-deviation PSD (one-sided)")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out

def fig2_cross_spectra(M, fs, tau0, sigma_xi, seed=0, out="fig2-cross-spectra.pdf"):
    N = 1<<17
    rng = np.random.default_rng(seed)
    x_common = generate_common_mode_x(N, fs, tau0, sigma_xi, seed=seed)
    alphas = 1.0 + 0.2*rng.standard_normal(M)
    X = np.stack([add_local_noise(x_common, alphas[m], fs, local_rms_frac=0.6, seed=seed+m+1) for m in range(M)], axis=0)
    a, b = 0, 1
    f, Paa = welch(X[a], fs=fs, nperseg=1<<12, noverlap=1<<11, scaling='density')
    f, Pbb = welch(X[b], fs=fs, nperseg=1<<12, noverlap=1<<11, scaling='density')
    f, Pab = csd(X[a], X[b], fs=fs, nperseg=1<<12, noverlap=1<<11, scaling='density')
    f, Coh = coherence(X[a], X[b], fs=fs, nperseg=1<<12, noverlap=1<<11)

    plt.figure(figsize=(3.5, 2.6), dpi=200)
    plt.loglog(f, np.abs(Pab), label=r"$|S_{ab}(f)|$")
    plt.loglog(f, Paa, linestyle="--", label=r"$S_{aa}(f)$")
    plt.loglog(f, Pbb, linestyle="--", label=r"$S_{bb}(f)$")
    plt.xlabel("Frequency $f$ [Hz]")
    plt.ylabel(r"PSD / cross-PSD")
    plt.title("Cross-spectral signature")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()

    out2 = out.replace(".pdf", "_coherence.pdf")
    plt.figure(figsize=(3.5, 2.0), dpi=200)
    plt.semilogx(f, Coh)
    plt.ylim(0, 1.05)
    plt.xlabel("Frequency $f$ [Hz]")
    plt.ylabel("Coherence")
    plt.title("Pairwise coherence")
    plt.tight_layout()
    plt.savefig(out2, bbox_inches="tight")
    plt.close()
    return out, out2

def fig3_sensitivity(fs, M, Nseg, Sloc_fc, C=10.0, tau0_grid=None, sigxi_grid=None, out="fig3-sensitivity.pdf"):
    import numpy as np
    if tau0_grid is None:
        tau0_grid = np.logspace(-14.5, -12.0, 120)
    if sigxi_grid is None:
        sigxi_grid = np.logspace(-6.0, -3.0, 120)
    Tau, Sig = np.meshgrid(tau0_grid, sigxi_grid, indexing='xy')
    lhs = (Sig**2)*Tau
    rhs = C*Sloc_fc/np.sqrt(M*(M-1)/2.0 * Nseg)
    detectable = lhs >= rhs

    plt.figure(figsize=(3.5, 2.6), dpi=200)
    cs = plt.contourf(Tau, Sig, detectable, levels=[-0.5,0.5,1.5], alpha=0.8)
    plt.contour(Tau, Sig, lhs, levels=[rhs], colors='k', linewidths=1.0)
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel(r"$\tau_0$ [s]")
    plt.ylabel(r"$\sigma_\xi$")
    plt.title("Detectable region (shaded)")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out

def main():
    p = argparse.ArgumentParser(description="Temporal Locations (TL) — spectra, cross-spectra, sensitivity.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fs", type=float, default=1e4, help="Sampling rate [Hz]")
    p.add_argument("--tau0", type=float, default=1e-13, help="Fundamental timescale [s]")
    p.add_argument("--sigma_xi", type=float, default=1e-4, help="Temporal fluctuation std")
    p.add_argument("--M", type=int, default=32, help="Number of channels (cross-spectra demo)")
    p.add_argument("--Nseg", type=int, default=512, help="Welch segments (sensitivity model)")
    p.add_argument("--Sloc_fc", type=float, default=1e-27, help="Per-channel local-noise PSD at fc [s^3]")
    args = p.parse_args()

    print(f"[info] fs={args.fs:.3g} Hz, tau0={args.tau0:.3g} s, sigma_xi={args.sigma_xi:.3g}, seed={args.seed}")
    f1 = figure2_jitter_spectrum(args.fs, args.tau0, args.sigma_xi, seed=args.seed, out="figure2_jitter_spectrum.pdf")
    f2a, f2b = fig2_cross_spectra(args.M, args.fs, args.tau0, args.sigma_xi, seed=args.seed, out="fig2-cross-spectra.pdf")
    f3 = fig3_sensitivity(args.fs, args.M, args.Nseg, args.Sloc_fc, out="fig3-sensitivity.pdf")
    print("[ok] wrote:", f1, f2a, f2b, f3)

if __name__ == "__main__":
    main()
