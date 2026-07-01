"""
Frequency alignment analysis: compare dominant frequencies of SSIM, Force Z
and TCP_Z in the T_START–T_END window for each run.

Outputs per-run FFT comparison plots and a summary CSV.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


T_START = 1
T_END = 8

C_TCP  = "#2196F3"
C_SSIM = "#F3A221"
C_FZ   = "#E53935"


def load_signals(robot_path: str, ssim_path: str, ft_path: str):
    robot = pd.read_csv(robot_path)
    ssim  = pd.read_csv(ssim_path)
    ft    = pd.read_csv(ft_path)

    robot["t_rel"] = robot["timestamp"] - robot["timestamp"].iloc[0]
    ssim["t_rel"]  = ssim["timestamp"]  - ssim["timestamp"].iloc[0]
    ft["t_rel"]    = ft["timestamp"]    - ft["timestamp"].iloc[0]

    robot_w = robot[(robot["t_rel"] >= T_START) & (robot["t_rel"] <= T_END)].copy()
    ssim_w  = ssim[(ssim["t_rel"]   >= T_START) & (ssim["t_rel"]  <= T_END)].copy()
    ft_w    = ft[(ft["t_rel"]       >= T_START) & (ft["t_rel"]    <= T_END)].copy()

    t_r    = robot_w["t_rel"].values
    tcpz   = robot_w["actual_TCP_pose_2"].values
    t_s    = ssim_w["t_rel"].values
    ssim_v = ssim_w["ssim"].values
    t_f    = ft_w["t_rel"].values
    fz     = -ft_w["force_z"].values

    return t_r, tcpz, t_s, ssim_v, t_f, fz


def compute_fft(t: np.ndarray, signal: np.ndarray):
    fs = 1.0 / np.diff(t).mean()
    sig = signal - signal.mean()
    win = np.hanning(len(sig))
    n = len(sig)
    X = np.fft.rfft(sig * win)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(X) / (n / 2)
    return freqs, mag, fs


def dominant_freq(freqs: np.ndarray, mag: np.ndarray, f_min=0.1, f_max=10.0):
    mask = (freqs >= f_min) & (freqs <= f_max)
    idx = np.argmax(mag[mask])
    return freqs[mask][idx], mag[mask][idx]


def xcorr(t_ref: np.ndarray, ref: np.ndarray, t_sig: np.ndarray, sig: np.ndarray):
    """Interpolate sig onto t_ref, normalise, cross-correlate. Returns metrics + arrays."""
    sig_i = np.interp(t_ref, t_sig, sig)

    def _norm(x):
        x = x - x.mean()
        s = x.std()
        return x / s if s > 0 else x

    ref_n = _norm(ref)
    sig_n = _norm(sig_i)

    n = len(ref_n)
    corr = np.correlate(ref_n, sig_n, mode="full") / n
    dt = np.diff(t_ref).mean()
    lag_times = np.arange(-(n - 1), n) * dt

    peak_idx   = np.argmax(corr)
    lag_s      = lag_times[peak_idx]
    corr_peak  = corr[peak_idx]
    corr_zero  = corr[n - 1]

    return sig_n, ref_n, lag_times, corr, lag_s, corr_peak, corr_zero


def analyse_run(robot_path: str, ssim_path: str, ft_path: str, output_path: str):
    t_r, tcpz, t_s, ssim_v, t_f, fz = load_signals(robot_path, ssim_path, ft_path)

    freqs_z, mag_z, fs_z = compute_fft(t_r, tcpz)
    freqs_s, mag_s, fs_s = compute_fft(t_s, ssim_v)
    freqs_f, mag_f, fs_f = compute_fft(t_f, fz)

    f_peak_z, _ = dominant_freq(freqs_z, mag_z)
    f_peak_s, _ = dominant_freq(freqs_s, mag_s)
    f_peak_f, _ = dominant_freq(freqs_f, mag_f)

    ssim_n, tcp_n, lag_t, corr_s, lag_s, _, cz_s = xcorr(t_r, tcpz, t_s, ssim_v)
    fz_n,   _,     _,     corr_f, lag_f, _, cz_f = xcorr(t_r, tcpz, t_f, fz)

    period_est = 1.0 / max(f_peak_z, 1e-6)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # ── panel 1: raw time domain ──
    ax = axes[0]
    ax2 = ax.twinx()
    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("axes", 1.08))

    ax.plot(t_r, tcpz, color=C_TCP,  lw=1.2, alpha=0.8, label="TCP_Z")
    ax2.plot(t_s, ssim_v, color=C_SSIM, lw=1.1, alpha=0.8, label="SSIM")
    ax3.plot(t_f, fz,     color=C_FZ,   lw=1.1, alpha=0.8, label="Force Z")

    ax.set_ylabel("TCP_Z (m)",  color=C_TCP)
    ax2.set_ylabel("SSIM",      color=C_SSIM)
    ax3.set_ylabel("Force Z (N)", color=C_FZ)
    ax2.tick_params(axis="y", colors=C_SSIM)
    ax3.tick_params(axis="y", colors=C_FZ)

    ax.set_title(f"Time domain  [{T_START}–{T_END} s]  "
                 f"TCP_Z fs≈{fs_z:.0f} Hz  SSIM fs≈{fs_s:.0f} Hz  FZ fs≈{fs_f:.0f} Hz")
    ax.grid(True, linestyle="--", alpha=0.4)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    h3, l3 = ax3.get_legend_handles_labels()
    ax.legend(h1 + h2 + h3, l1 + l2 + l3, loc="upper right", fontsize=9)

    # ── panel 2: normalised overlay + xcorr insets ──
    ax = axes[1]
    ax.plot(t_r, tcp_n,  color=C_TCP,  lw=1.2, alpha=0.85, label="TCP_Z (norm)")
    ax.plot(t_r, ssim_n, color=C_SSIM, lw=1.1, alpha=0.85, ls="--",
            label=f"SSIM (norm)   r(0)={cz_s:.3f}  lag={lag_s*1000:.1f} ms")
    ax.plot(t_r, fz_n,   color=C_FZ,   lw=1.1, alpha=0.85, ls=":",
            label=f"Force Z (norm) r(0)={cz_f:.3f}  lag={lag_f*1000:.1f} ms")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title("Waveform overlay (all normalised to TCP_Z grid)")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    # inset: xcorr for SSIM and Fz
    win = 2 * period_est
    mask_lag = np.abs(lag_t) <= win
    ax_ins = ax.inset_axes([0.72, 0.05, 0.27, 0.45])
    ax_ins.plot(lag_t[mask_lag] * 1000, corr_s[mask_lag], color=C_SSIM, lw=1.0,
                label=f"SSIM  {lag_s*1000:.1f} ms")
    ax_ins.plot(lag_t[mask_lag] * 1000, corr_f[mask_lag], color=C_FZ,   lw=1.0,
                label=f"Fz    {lag_f*1000:.1f} ms")
    ax_ins.axvline(lag_s * 1000, color=C_SSIM, ls="--", lw=0.8)
    ax_ins.axvline(lag_f * 1000, color=C_FZ,   ls="--", lw=0.8)
    ax_ins.axvline(0, color="gray", ls=":", lw=0.8)
    ax_ins.set_xlabel("Lag (ms)", fontsize=7)
    ax_ins.set_ylabel("XCorr",   fontsize=7)
    ax_ins.tick_params(labelsize=7)
    ax_ins.legend(fontsize=7)
    ax_ins.grid(True, linestyle="--", alpha=0.3)

    # ── panel 3: FFT magnitude ──
    ax = axes[2]
    f_max_plot = 10.0
    m = freqs_z <= f_max_plot
    ax.plot(freqs_z[m], mag_z[m], color=C_TCP,  lw=1.2,
            label=f"TCP_Z  peak={f_peak_z:.3f} Hz")
    m = freqs_s <= f_max_plot
    ax.plot(freqs_s[m], mag_s[m], color=C_SSIM, lw=1.1, alpha=0.85,
            label=f"SSIM   peak={f_peak_s:.3f} Hz")
    m = freqs_f <= f_max_plot
    ax.plot(freqs_f[m], mag_f[m], color=C_FZ,   lw=1.1, alpha=0.85,
            label=f"Fz     peak={f_peak_f:.3f} Hz")
    for f, c in [(f_peak_z, C_TCP), (f_peak_s, C_SSIM), (f_peak_f, C_FZ)]:
        ax.axvline(f, color=c, ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title("FFT magnitude spectrum")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        f"TCP_Z={f_peak_z:.3f} Hz  SSIM={f_peak_s:.3f} Hz  Fz={f_peak_f:.3f} Hz  |  "
        f"SSIM r(0)={cz_s:.3f} lag={lag_s*1000:.1f} ms  |  "
        f"Fz   r(0)={cz_f:.3f} lag={lag_f*1000:.1f} ms",
        fontsize=10, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return {
        "f_peak_tcpz_hz":  round(f_peak_z, 4),
        "f_peak_ssim_hz":  round(f_peak_s, 4),
        "f_peak_fz_hz":    round(f_peak_f, 4),
        "corr_zero_ssim":  round(float(cz_s), 4),
        "lag_ms_ssim":     round(float(lag_s * 1000), 2),
        "corr_zero_fz":    round(float(cz_f), 4),
        "lag_ms_fz":       round(float(lag_f * 1000), 2),
    }


def plot_summary(summary_df: pd.DataFrame, out_path: str):
    datasets = summary_df["dataset"].unique()
    n_ds = len(datasets)

    fig, axes = plt.subplots(3, n_ds, figsize=(7 * n_ds, 11),
                             gridspec_kw={"height_ratios": [2, 1.2, 1.2]})
    if n_ds == 1:
        axes = axes.reshape(3, 1)

    for col, dataset in enumerate(datasets):
        sub = summary_df[summary_df["dataset"] == dataset].copy()
        sub["run_num"] = sub["run"].apply(lambda r: int(r.split("_")[1]))
        sub = sub.sort_values("run_num")
        x = sub["run_num"].to_numpy()
        short = dataset.split("/")[-1]
        # ── row 0: peak frequencies ──
        ax = axes[0, col]
        ax.plot(x, sub["f_peak_tcpz_hz"].to_numpy(), marker="o", color=C_TCP,  lw=1.4, ms=6, label="TCP_Z")
        ax.plot(x, sub["f_peak_ssim_hz"].to_numpy(), marker="s", color=C_SSIM, lw=1.4, ms=6, ls="--", label="SSIM")
        ax.plot(x, sub["f_peak_fz_hz"].to_numpy(),   marker="^", color=C_FZ,   lw=1.4, ms=6, ls=":",  label="Force Z")
        ax.set_title(short, fontsize=10, fontweight="bold")
        ax.set_ylabel("Dominant frequency (Hz)")
        ax.set_xticks(x)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        all_f = sub[["f_peak_tcpz_hz", "f_peak_ssim_hz", "f_peak_fz_hz"]].values
        mean_f = all_f.mean()
        ax.set_ylim(mean_f * 0.98, mean_f * 1.02)

        # ── row 1: r(0) comparison ──
        ax1 = axes[1, col]
        ax1.plot(x, sub["corr_zero_ssim"].to_numpy(), marker="s", color=C_SSIM, lw=1.3, ms=6, label="r(0) SSIM")
        ax1.plot(x, sub["corr_zero_fz"].to_numpy(),   marker="^", color=C_FZ,   lw=1.3, ms=6, label="r(0) Fz")
        ax1.axhline(0, color="gray", ls=":", lw=0.8)
        ax1.set_ylim(-1.05, 1.05)
        ax1.set_ylabel("r(0) vs TCP_Z")
        ax1.set_xticks(x)
        ax1.legend(fontsize=9)
        ax1.grid(True, linestyle="--", alpha=0.4)

        # ── row 2: lag comparison ──
        ax2 = axes[2, col]
        ax2.plot(x, sub["lag_ms_ssim"].to_numpy(), marker="s", color=C_SSIM, lw=1.3, ms=6, label="lag SSIM")
        ax2.plot(x, sub["lag_ms_fz"].to_numpy(),   marker="^", color=C_FZ,   lw=1.3, ms=6, label="lag Fz")
        ax2.axhline(0, color="gray", ls=":", lw=0.8)
        ax2.set_ylabel("Time lag vs TCP_Z (ms)")
        ax2.set_xlabel("Run")
        ax2.set_xticks(x)
        ax2.legend(fontsize=9)
        ax2.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        "Frequency & waveform alignment: SSIM and Force Z vs TCP_Z",
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved summary plot → {out_path}")


if __name__ == "__main__":
    base_dir = "./tactile_data/ur5/tactip-127"
    datasets = [
        # "surface-zRxy-June19-calibration",
        "surface-zRxy-June19-calibration-without_skin",
    ]

    all_rows = []
    for dataset in datasets:
        target_folder = f"{base_dir}/{dataset}"
        if not os.path.isdir(target_folder):
            continue
        runs = sorted(
            [d for d in os.listdir(target_folder)
             if os.path.isdir(os.path.join(target_folder, d)) and d.startswith("run_")],
            key=lambda x: int(x.split("_")[1])
        )
        output_dir = f"{target_folder}/output"
        os.makedirs(output_dir, exist_ok=True)
        for run in tqdm(runs, desc=dataset):
            folder     = os.path.join(target_folder, run)
            robot_path = f"{folder}/robot_data.csv"
            ssim_path  = f"{folder}/ssim.csv"
            ft_path    = f"{folder}/ft_sensor_data.csv"
            if not all(os.path.exists(p) for p in [robot_path, ssim_path, ft_path]):
                continue
            out_png = f"{output_dir}/{run}_raw_data.png"
            metrics = analyse_run(robot_path, ssim_path, ft_path, out_png)
            all_rows.append({"dataset": dataset, "run": run, **metrics})
            print(f"  {run}: TCP_Z={metrics['f_peak_tcpz_hz']:.3f}  "
                  f"SSIM={metrics['f_peak_ssim_hz']:.3f}  Fz={metrics['f_peak_fz_hz']:.3f} Hz  |  "
                  f"SSIM r(0)={metrics['corr_zero_ssim']:.3f} lag={metrics['lag_ms_ssim']:.1f} ms  |  "
                  f"Fz   r(0)={metrics['corr_zero_fz']:.3f} lag={metrics['lag_ms_fz']:.1f} ms")
