"""
FT Sensor Periodic Noise Analysis and Removal
==============================================
Uses the first portion of the recording (the idle / pre-contact segment,
where the sensor should ideally read zero) as the noise reference.
From that segment the script:

  1. Characterises noise – DC bias, RMS, and per-frequency PSD via Welch
  2. Identifies dominant periodic noise frequencies relative to the idle floor
  3. Applies four denoising methods:
       a. DC bias subtraction only
       b. Spectral subtraction  – subtract idle noise PSD from full signal
       c. Wiener filter         – data-driven optimal filter built from idle PSD
       d. Butterworth low-pass  – retain only the band below detected noise floor
  4. Evaluates each method by measuring how close the denoised idle segment
     is to zero (lower residual RMS = better noise removal)
  5. Exports cleaned CSVs and diagnostic plots

Usage:
  python analyze_periodic_noise.py [CSV_FILE]
         [--idle-duration S]    seconds of idle data to use (default: 0.5)
         [--channel CHANNEL]    channel shown in the detailed plot
         [--output-dir DIR]
         [--lowpass-cutoff HZ]  Butterworth cutoff (default: auto)

Default CSV:
  tactile_data/ur5/tactip-127/surface-zRxy-new-speed-1/run_5/ft_sensor_data.csv
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ──────────────────────────────────────────────────────────────────────────────
CHANNELS = ["force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z"]
CHANNEL_UNITS = {
    "force_x": "N", "force_y": "N", "force_z": "N",
    "torque_x": "Nm", "torque_y": "Nm", "torque_z": "Nm",
}


# ──────────────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────────────
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=0, names=["timestamp"] + CHANNELS)
    return df.dropna().sort_values("timestamp").reset_index(drop=True)


def describe_sampling(df: pd.DataFrame) -> dict:
    dt = np.diff(df["timestamp"].values)
    return {
        "n_samples":      len(df),
        "duration_s":     df["timestamp"].iloc[-1] - df["timestamp"].iloc[0],
        "dt_mean_ms":     dt.mean() * 1e3,
        "dt_median_ms":   np.median(dt) * 1e3,
        "dt_std_ms":      dt.std() * 1e3,
        "dt_min_ms":      dt.min() * 1e3,
        "dt_max_ms":      dt.max() * 1e3,
        "fs_mean_hz":     1.0 / dt.mean(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Resampling
# ──────────────────────────────────────────────────────────────────────────────
def resample_uniform(df: pd.DataFrame, fs_target: float = None):
    """Linear interpolation onto a uniform time grid at fs_target Hz."""
    t = df["timestamp"].values
    if fs_target is None:
        fs_target = 1.0 / np.median(np.diff(t))

    t_uni = np.arange(t[0], t[-1], 1.0 / fs_target)
    out = {"timestamp": t_uni}
    for ch in CHANNELS:
        f = interp1d(t, df[ch].values, kind="linear",
                     bounds_error=False, fill_value="extrapolate")
        out[ch] = f(t_uni)

    return pd.DataFrame(out), fs_target


# ──────────────────────────────────────────────────────────────────────────────
# Idle-segment noise characterisation
# ──────────────────────────────────────────────────────────────────────────────
def extract_idle(df_uni: pd.DataFrame, idle_duration_s: float):
    """Return the first `idle_duration_s` seconds of the uniform-grid data."""
    t = df_uni["timestamp"].values
    mask = t <= t[0] + idle_duration_s
    n_idle = mask.sum()
    if n_idle < 16:
        raise ValueError(
            f"Idle segment has only {n_idle} samples "
            f"(requested {idle_duration_s} s). "
            "Use a longer --idle-duration or check the recording.")
    return df_uni[mask].reset_index(drop=True), n_idle


def noise_stats(idle_df: pd.DataFrame, fs: float) -> dict:
    """
    Per-channel noise characterisation from the idle segment.

    Returns a dict with keys:
      dc_bias   – mean value (should be ~0 but rarely is)
      rms       – RMS of the raw idle signal
      psd_f     – frequency axis of Welch PSD  (shape [K])
      psd       – one-sided PSD array           (shape [K])
    """
    stats = {}
    for ch in CHANNELS:
        sig = idle_df[ch].values.astype(float)
        dc = sig.mean()
        rms = np.sqrt(np.mean(sig ** 2))
        # Welch PSD – use at most 256-sample segments to resolve low frequencies
        nperseg = min(256, len(sig) // 4)
        f_w, psd_w = signal.welch(sig, fs=fs, nperseg=nperseg,
                                   window="hann", scaling="density")
        stats[ch] = {"dc_bias": dc, "rms": rms, "psd_f": f_w, "psd": psd_w}
    return stats


def noise_dominant_freqs(noise_st: dict, n_top: int = 10) -> dict:
    """
    For each channel, return the top-n_top frequencies in the noise PSD
    (above 1 Hz to skip DC artefacts).
    """
    result = {}
    for ch in CHANNELS:
        psd_f = noise_st[ch]["psd_f"]
        psd   = noise_st[ch]["psd"]
        mask  = psd_f >= 1.0
        idx   = np.argsort(psd[mask])[::-1][:n_top]
        result[ch] = psd_f[mask][idx]
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Denoising methods
# ──────────────────────────────────────────────────────────────────────────────
def apply_dc_subtraction(sig: np.ndarray, dc: float) -> np.ndarray:
    """Remove the DC bias estimated from the idle segment."""
    return sig - dc


def apply_spectral_subtraction(sig: np.ndarray, fs: float,
                                noise_psd_f: np.ndarray,
                                noise_psd: np.ndarray,
                                oversub: float = 1.5) -> np.ndarray:
    """
    Power-spectrum subtraction (Boll, 1979).

    H(f)² = max( |X(f)|² − α·N(f) , β·|X(f)|² ) / |X(f)|²

    where  α = oversub (over-subtraction factor, > 1 reduces musical noise)
           β = 0.01    (spectral floor to prevent negative power)
    """
    n = len(sig)
    win = np.hanning(n)
    X = np.fft.rfft(sig * win)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    # Interpolate noise PSD to match the full-signal frequency axis
    noise_interp = np.interp(freqs, noise_psd_f, noise_psd, left=0.0, right=0.0)

    X_pow = np.abs(X) ** 2
    clean_pow = np.maximum(X_pow - oversub * noise_interp * (n / 2),
                            0.01 * X_pow)
    gain = np.sqrt(clean_pow / (X_pow + 1e-30))

    X_clean = X * gain
    # Correct for Hann window amplitude
    reconstructed = np.fft.irfft(X_clean, n=n) / (win.mean() + 1e-12)
    return reconstructed


def apply_wiener(sig: np.ndarray, fs: float,
                 noise_psd_f: np.ndarray,
                 noise_psd: np.ndarray) -> np.ndarray:
    """
    Wiener filter in the frequency domain.

    H(f) = (S_x(f) − S_n(f)) / S_x(f)
         = max(1 − S_n(f)/S_x(f), 0)

    S_x is estimated from the full signal via Welch; S_n is the idle PSD.
    """
    n = len(sig)
    # Estimate signal PSD (includes noise)
    nperseg = min(512, n // 4)
    f_x, psd_x = signal.welch(sig, fs=fs, nperseg=nperseg,
                                window="hann", scaling="density")

    # Interpolate noise PSD to signal PSD frequency axis
    noise_on_sig = np.interp(f_x, noise_psd_f, noise_psd, left=0.0, right=0.0)

    # Wiener gain (floored at 0)
    gain_psd = np.maximum(1.0 - noise_on_sig / (psd_x + 1e-30), 0.0)

    # Apply gain in rfft domain
    win = np.hanning(n)
    X = np.fft.rfft(sig * win)
    freqs_fft = np.fft.rfftfreq(n, d=1.0 / fs)
    gain_fft = np.interp(freqs_fft, f_x, gain_psd, left=gain_psd[0],
                          right=gain_psd[-1])
    X_clean = X * gain_fft

    reconstructed = np.fft.irfft(X_clean, n=n) / (win.mean() + 1e-12)
    return reconstructed


def apply_lowpass(sig: np.ndarray, fs: float,
                  cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Butterworth low-pass filter, zero-phase."""
    cutoff_hz = min(cutoff_hz, fs / 2 * 0.99)
    b, a = signal.butter(order, cutoff_hz, btype="low", fs=fs)
    return signal.filtfilt(b, a, sig)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation: residual in the idle window (lower → better)
# ──────────────────────────────────────────────────────────────────────────────
def idle_residual_rms(denoised: np.ndarray, n_idle: int) -> float:
    """RMS of the denoised signal over the idle segment (should be ≈ 0)."""
    return float(np.sqrt(np.mean(denoised[:n_idle] ** 2)))


def idle_snr_db(original: np.ndarray, denoised: np.ndarray,
                n_idle: int) -> float:
    """
    SNR improvement relative to idle baseline.

    original_noise_power = var of original signal in idle window
    residual_noise_power  = var of denoised signal in idle window
    improvement = 10·log10(original / residual)  [positive = better]
    """
    orig_noise_var = np.var(original[:n_idle]) + 1e-30
    resid_var      = np.var(denoised[:n_idle]) + 1e-30
    return 10.0 * np.log10(orig_noise_var / resid_var)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────
def _fft_mag(arr: np.ndarray, fs: float):
    n = len(arr)
    win = np.hanning(n)
    X = np.fft.rfft(arr * win)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, np.abs(X) / (n / 2)


def plot_sampling_intervals(df: pd.DataFrame, out_dir: Path):
    dt_ms = np.diff(df["timestamp"].values) * 1e3
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(dt_ms, lw=0.5, color="steelblue")
    axes[0].axhline(np.median(dt_ms), color="red", ls="--",
                    label=f"Median {np.median(dt_ms):.2f} ms")
    axes[0].set_xlabel("Sample index"); axes[0].set_ylabel("Δt (ms)")
    axes[0].set_title("Inter-sample interval over time"); axes[0].legend(fontsize=8)

    axes[1].hist(dt_ms, bins=80, color="steelblue", edgecolor="white", lw=0.3)
    axes[1].axvline(np.median(dt_ms), color="red", ls="--")
    axes[1].set_xlabel("Δt (ms)"); axes[1].set_ylabel("Count")
    axes[1].set_title("Histogram of inter-sample intervals")
    fig.tight_layout()
    p = out_dir / "sampling_intervals.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Saved: {p}")


def plot_noise_characterisation(noise_st: dict, idle_df: pd.DataFrame,
                                 t_idle: np.ndarray, fs: float,
                                 out_dir: Path):
    """
    For each channel: idle time trace + Welch PSD of the noise reference.
    """
    fig, axes = plt.subplots(len(CHANNELS), 2,
                              figsize=(14, 2.5 * len(CHANNELS)))
    for i, ch in enumerate(CHANNELS):
        st = noise_st[ch]
        sig_idle = idle_df[ch].values

        ax_t = axes[i, 0]
        ax_t.plot(t_idle - t_idle[0], sig_idle, lw=0.6, color="steelblue")
        ax_t.axhline(st["dc_bias"], color="red", ls="--", lw=0.8,
                     label=f"DC bias = {st['dc_bias']:.4f}")
        ax_t.axhline(0, color="black", lw=0.5)
        ax_t.set_title(f"{ch}  –  idle signal  (RMS = {st['rms']:.4e} "
                       f"{CHANNEL_UNITS.get(ch,'')})")
        ax_t.set_xlabel("Time (s)"); ax_t.legend(fontsize=7)

        ax_f = axes[i, 1]
        ax_f.semilogy(st["psd_f"], st["psd"], lw=0.7, color="darkorange")
        ax_f.set_title(f"{ch}  –  noise PSD (Welch)")
        ax_f.set_xlabel("Frequency (Hz)"); ax_f.set_ylabel("PSD")
        ax_f.grid(True, alpha=0.3)

    fig.suptitle("Noise characterisation from idle segment",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = out_dir / "noise_characterisation.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Saved: {p}")


def plot_denoising_comparison(t: np.ndarray, original: np.ndarray,
                               results: dict, channel: str,
                               fs: float, n_idle: int,
                               out_dir: Path):
    methods = list(results.keys())
    colors  = plt.cm.tab10(np.linspace(0, 0.9, len(methods)))

    fig = plt.figure(figsize=(16, 4 + 3 * len(methods)))
    gs  = gridspec.GridSpec(len(methods) + 1, 2, figure=fig,
                             hspace=0.55, wspace=0.35)

    # row 0: original
    ax_t0 = fig.add_subplot(gs[0, 0])
    ax_f0 = fig.add_subplot(gs[0, 1])
    ax_t0.plot(t - t[0], original, lw=0.5, color="gray")
    ax_t0.axvspan(0, (t[n_idle - 1] - t[0]), alpha=0.12, color="orange",
                  label="Idle reference")
    ax_t0.set_title(f"Original – {channel}")
    ax_t0.set_xlabel("Time (s)"); ax_t0.set_ylabel(CHANNEL_UNITS.get(channel, ""))
    ax_t0.legend(fontsize=7)

    f_o, m_o = _fft_mag(original, fs)
    ax_f0.semilogy(f_o, m_o, lw=0.6, color="gray")
    ax_f0.set_title("Original spectrum"); ax_f0.set_xlabel("Frequency (Hz)")
    ax_f0.grid(True, alpha=0.3)

    for i, (mname, den) in enumerate(results.items()):
        ax_t = fig.add_subplot(gs[i + 1, 0])
        ax_f = fig.add_subplot(gs[i + 1, 1])
        col  = colors[i]

        ax_t.plot(t - t[0], original, lw=0.4, color="lightgray",
                  label="Original")
        ax_t.plot(t - t[0], den, lw=0.6, color=col, label=mname)
        ax_t.axvspan(0, (t[n_idle - 1] - t[0]), alpha=0.12, color="orange")
        snr = idle_snr_db(original, den, n_idle)
        rms = idle_residual_rms(den, n_idle)
        ax_t.set_title(f"{mname}  |  idle SNR improvement: {snr:+.1f} dB"
                       f"  |  idle residual RMS: {rms:.3e}")
        ax_t.set_xlabel("Time (s)"); ax_t.set_ylabel(CHANNEL_UNITS.get(channel, ""))
        ax_t.legend(fontsize=7)

        f_d, m_d = _fft_mag(den, fs)
        ax_f.semilogy(f_o, m_o, lw=0.5, color="lightgray", label="Original")
        ax_f.semilogy(f_d, m_d, lw=0.6, color=col, label=mname)
        ax_f.set_title(f"Spectrum after {mname}")
        ax_f.set_xlabel("Frequency (Hz)"); ax_f.legend(fontsize=7)
        ax_f.grid(True, alpha=0.3)

    fig.suptitle(f"Denoising comparison – {channel}",
                 fontsize=13, fontweight="bold", y=1.001)
    p = out_dir / f"denoising_{channel}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p}")


def plot_all_channels_summary(t: np.ndarray, df_uni: pd.DataFrame,
                               denoised_all: dict, best_method: str,
                               n_idle: int, out_dir: Path):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    for i, ch in enumerate(CHANNELS):
        ax = axes[i]
        ax.plot(t - t[0], df_uni[ch].values, lw=0.5, color="lightgray",
                label="Original")
        ax.plot(t - t[0], denoised_all[ch][best_method], lw=0.7,
                color="steelblue", label=best_method)
        ax.axvspan(0, t[n_idle - 1] - t[0], alpha=0.12, color="orange",
                   label="Idle ref" if i == 0 else "")
        ax.set_title(ch); ax.set_ylabel(CHANNEL_UNITS.get(ch, ""))
        ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
    axes[-2].set_xlabel("Time (s)"); axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"All channels – Original vs {best_method}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = out_dir / "all_channels_summary.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Saved: {p}")


def plot_idle_before_after(t_idle: np.ndarray, idle_df: pd.DataFrame,
                            denoised_all: dict, best_method: str,
                            n_idle: int, out_dir: Path):
    """Show the idle segment alone, before and after best denoising."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex=True)
    axes = axes.flatten()
    for i, ch in enumerate(CHANNELS):
        ax = axes[i]
        orig_idle = idle_df[ch].values
        den_idle  = denoised_all[ch][best_method][:n_idle]
        ax.plot(t_idle - t_idle[0], orig_idle, lw=0.6, color="lightgray",
                label="Original idle")
        ax.plot(t_idle - t_idle[0], den_idle, lw=0.7, color="steelblue",
                label=f"After {best_method}")
        ax.axhline(0, color="red", lw=0.8, ls="--", label="Ideal = 0")
        ax.set_title(f"{ch}  (residual RMS = {idle_residual_rms(denoised_all[ch][best_method], n_idle):.3e})")
        ax.set_ylabel(CHANNEL_UNITS.get(ch, ""))
        ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
    axes[-2].set_xlabel("Time (s)"); axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Idle segment: original vs denoised (should be ≈ 0)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = out_dir / "idle_before_after.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Saved: {p}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def analyse(csv_path: str,
            idle_duration_s: float = 0.2,
            target_channel: str = "force_z",
            output_dir: str = "noise_analysis_output",
            lowpass_cutoff: float = None):

    csv_path = Path(csv_path)
    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("\n=== Step 1: Load Data ===")
    df   = load_data(str(csv_path))
    stat = describe_sampling(df)
    print(f"  Samples     : {stat['n_samples']}")
    print(f"  Duration    : {stat['duration_s']:.4f} s")
    print(f"  Mean Fs     : {stat['fs_mean_hz']:.1f} Hz")
    print(f"  Δt mean/std : {stat['dt_mean_ms']:.3f} / {stat['dt_std_ms']:.3f} ms")
    print(f"  Δt min/max  : {stat['dt_min_ms']:.3f} / {stat['dt_max_ms']:.3f} ms")
    plot_sampling_intervals(df, out_dir)

    # ── 2. Resample ───────────────────────────────────────────────────────────
    print("\n=== Step 2: Resample to Uniform Grid ===")
    df_uni, fs = resample_uniform(df)
    t = df_uni["timestamp"].values
    print(f"  Uniform Fs  : {fs:.1f} Hz  ({len(df_uni)} samples)")

    # ── 3. Extract idle segment & characterise noise ──────────────────────────
    print(f"\n=== Step 3: Noise Characterisation (first {idle_duration_s} s) ===")
    idle_df, n_idle = extract_idle(df_uni, idle_duration_s)
    t_idle = idle_df["timestamp"].values
    noise_st = noise_stats(idle_df, fs)

    print(f"  Idle samples: {n_idle}  ({t_idle[-1] - t_idle[0]:.4f} s)")
    print(f"\n  {'Channel':12s}  {'DC bias':>12s}  {'Noise RMS':>12s}")
    print(f"  {'-'*40}")
    for ch in CHANNELS:
        st = noise_st[ch]
        print(f"  {ch:12s}  {st['dc_bias']:>12.6f}  {st['rms']:>12.6f}")

    plot_noise_characterisation(noise_st, idle_df, t_idle, fs, out_dir)

    # Dominant noise frequencies across all channels (union, for display)
    dom_freqs = noise_dominant_freqs(noise_st, n_top=8)
    print("\n  Top noise frequencies per channel (from idle PSD):")
    for ch in CHANNELS:
        print(f"    {ch:12s}: {np.round(dom_freqs[ch], 1)} Hz")

    # ── 4. Choose low-pass cutoff ─────────────────────────────────────────────
    if lowpass_cutoff is None:
        # Heuristic: frequency below which noise PSD is flat – use 10 % of Nyquist
        lowpass_cutoff = min(0.10 * fs / 2, 50.0)
    print(f"\n  Low-pass cutoff : {lowpass_cutoff:.1f} Hz")

    # ── 5. Denoise ────────────────────────────────────────────────────────────
    print("\n=== Step 4: Denoising ===")
    denoised_all: dict[str, dict[str, np.ndarray]] = {ch: {} for ch in CHANNELS}

    for ch in CHANNELS:
        sig     = df_uni[ch].values.astype(float)
        st      = noise_st[ch]
        nf, np_ = st["psd_f"], st["psd"]

        den = {}
        den["DC Subtraction"]        = apply_dc_subtraction(sig, st["dc_bias"])
        den["Spectral Subtraction"]  = apply_spectral_subtraction(
                                            sig, fs, nf, np_, oversub=1.5)
        den["Wiener Filter"]         = apply_wiener(sig, fs, nf, np_)
        den["Butterworth LP"]        = apply_lowpass(sig, fs, lowpass_cutoff)
        denoised_all[ch] = den

    # Print per-channel metrics (idle segment residual)
    print(f"\n  {'Method':24s}", end="")
    for ch in CHANNELS:
        print(f"  {ch:>14s}", end="")
    print()
    print(f"  {'-'*120}")

    methods_list = list(next(iter(denoised_all.values())).keys())
    # Show original idle RMS as baseline
    print(f"  {'[Original idle RMS]':24s}", end="")
    for ch in CHANNELS:
        print(f"  {noise_st[ch]['rms']:>14.4e}", end="")
    print()

    for mname in methods_list:
        print(f"  {mname:24s}", end="")
        for ch in CHANNELS:
            rms = idle_residual_rms(denoised_all[ch][mname], n_idle)
            print(f"  {rms:>14.4e}", end="")
        print()

    print(f"\n  SNR improvement over idle segment (dB, higher = better):")
    print(f"  {'Method':24s}", end="")
    for ch in CHANNELS:
        print(f"  {ch:>14s}", end="")
    print()
    print(f"  {'-'*120}")
    for mname in methods_list:
        print(f"  {mname:24s}", end="")
        for ch in CHANNELS:
            snr = idle_snr_db(df_uni[ch].values, denoised_all[ch][mname], n_idle)
            print(f"  {snr:>+14.1f}", end="")
        print()

    # ── 6. Pick best method (highest mean SNR across all channels) ─────────────
    method_mean_snr = {}
    for mname in methods_list:
        vals = [idle_snr_db(df_uni[ch].values, denoised_all[ch][mname], n_idle)
                for ch in CHANNELS]
        method_mean_snr[mname] = np.mean(vals)
    best_method = max(method_mean_snr, key=method_mean_snr.get)
    print(f"\n  Best overall method: {best_method} "
          f"(mean SNR improvement {method_mean_snr[best_method]:+.1f} dB)")

    # ── 7. Plots ──────────────────────────────────────────────────────────────
    print("\n=== Step 5: Plots ===")
    plot_denoising_comparison(
        t, df_uni[target_channel].values,
        denoised_all[target_channel], target_channel, fs, n_idle, out_dir)
    plot_all_channels_summary(t, df_uni, denoised_all, best_method,
                               n_idle, out_dir)
    plot_idle_before_after(t_idle, idle_df, denoised_all, best_method,
                            n_idle, out_dir)

    # ── 8. Save cleaned CSVs ──────────────────────────────────────────────────
    print("\n=== Step 6: Save Cleaned Data ===")
    for mname in methods_list:
        safe = mname.lower().replace(" ", "_")
        fpath = out_dir / f"ft_sensor_denoised_{safe}.csv"
        df_out = pd.DataFrame({"timestamp": t})
        for ch in CHANNELS:
            df_out[ch] = denoised_all[ch][mname]
        df_out.to_csv(fpath, index=False, float_format="%.9f")
        print(f"  Saved: {fpath}")

    ref_path = out_dir / "ft_sensor_uniform_original.csv"
    df_uni.to_csv(ref_path, index=False, float_format="%.9f")
    print(f"  Saved: {ref_path}")

    # ── 9. Text report ────────────────────────────────────────────────────────
    rpt = out_dir / "noise_analysis_report.txt"
    with open(rpt, "w") as fh:
        fh.write("FT Sensor Periodic Noise Analysis Report\n")
        fh.write("=" * 70 + "\n\n")
        fh.write(f"Source file    : {csv_path}\n")
        fh.write(f"Samples        : {stat['n_samples']}\n")
        fh.write(f"Duration       : {stat['duration_s']:.4f} s\n")
        fh.write(f"Uniform Fs     : {fs:.1f} Hz\n")
        fh.write(f"Idle segment   : first {idle_duration_s} s "
                 f"({n_idle} samples)\n\n")

        fh.write("Noise statistics from idle segment\n")
        fh.write("-" * 50 + "\n")
        fh.write(f"  {'Channel':12s}  {'DC bias':>12s}  {'RMS':>12s}\n")
        for ch in CHANNELS:
            st = noise_st[ch]
            fh.write(f"  {ch:12s}  {st['dc_bias']:>12.6f}  {st['rms']:>12.6f}\n")

        fh.write(f"\nLow-pass cutoff : {lowpass_cutoff:.1f} Hz\n")
        fh.write(f"Best method     : {best_method}\n\n")

        fh.write("Idle residual RMS per method (lower = better)\n")
        fh.write("-" * 70 + "\n")
        fh.write(f"  {'Method':24s}")
        for ch in CHANNELS:
            fh.write(f"  {ch:>14s}")
        fh.write("\n")
        fh.write(f"  {'[Original]':24s}")
        for ch in CHANNELS:
            fh.write(f"  {noise_st[ch]['rms']:>14.4e}")
        fh.write("\n")
        for mname in methods_list:
            fh.write(f"  {mname:24s}")
            for ch in CHANNELS:
                fh.write(f"  {idle_residual_rms(denoised_all[ch][mname], n_idle):>14.4e}")
            fh.write("\n")

        fh.write("\nSNR improvement (dB) per method\n")
        fh.write("-" * 70 + "\n")
        fh.write(f"  {'Method':24s}")
        for ch in CHANNELS:
            fh.write(f"  {ch:>14s}")
        fh.write("\n")
        for mname in methods_list:
            fh.write(f"  {mname:24s}")
            for ch in CHANNELS:
                fh.write(f"  {idle_snr_db(df_uni[ch].values, denoised_all[ch][mname], n_idle):>+14.1f}")
            fh.write("\n")

    print(f"  Saved: {rpt}")
    print("\n=== Analysis complete ===")
    print(f"Output directory: {out_dir.resolve()}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    default_csv = (
        "tactile_data/ur5/tactip-127/"
        "surface-zRxy-10Jun-calibrate-1/run_0/ft_sensor_data.csv"
    )
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv_file", nargs="?", default=default_csv,
                   help="Path to ft_sensor_data.csv")
    p.add_argument("--idle-duration", type=float, default=0.1,
                   help="Length of the pre-contact idle window in seconds "
                        "(default: 0.1). Sensor should read ~0 in this window.")
    p.add_argument("--channel", default="force_z", choices=CHANNELS,
                   help="Channel shown in the detailed comparison plot")
    p.add_argument("--output-dir", default="noise_analysis_output",
                   help="Directory for plots and cleaned CSVs")
    p.add_argument("--lowpass-cutoff", type=float, default=None,
                   help="Butterworth low-pass cutoff in Hz "
                        "(default: 10%% of Nyquist, max 50 Hz)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyse(
        csv_path=args.csv_file,
        idle_duration_s=args.idle_duration,
        target_channel=args.channel,
        output_dir=args.output_dir,
        lowpass_cutoff=args.lowpass_cutoff,
    )
