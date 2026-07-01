"""
Batch FT Sensor Denoising: Butterworth Low-Pass + Kalman Filter
Processes all run_i folders in-place:
  - Backs up ft_sensor_data.csv -> ft_sensor_data.csv.bak
  - Writes denoised result back to ft_sensor_data.csv

Usage:
  python denoise_runs.py [DATASET_DIR]
         [--lowpass-cutoff HZ]   (default: auto = 10% of Nyquist, max 50 Hz)
         [--butter-order N]      (default: 4)
         [--q-ratio R]           (default: 1e-3)

Default DATASET_DIR:
  tactile_data/ur5/tactip-127/surface-zRxy-June19-classic
"""

import argparse
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore", category=RuntimeWarning)

CHANNELS = ["force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z"]


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=0, names=["timestamp"] + CHANNELS)
    return df.dropna().sort_values("timestamp").reset_index(drop=True)


def resample_uniform(df: pd.DataFrame):
    t = df["timestamp"].values
    fs = 1.0 / np.median(np.diff(t))
    t_uni = np.arange(t[0], t[-1], 1.0 / fs)
    out = {"timestamp": t_uni}
    for ch in CHANNELS:
        f = interp1d(t, df[ch].values, kind="linear",
                     bounds_error=False, fill_value="extrapolate")
        out[ch] = f(t_uni)
    return pd.DataFrame(out), float(fs)


def butterworth_lp(sig: np.ndarray, fs: float, cutoff_hz: float, order: int) -> np.ndarray:
    cutoff_hz = min(cutoff_hz, fs / 2.0 * 0.99)
    b, a = signal.butter(order, cutoff_hz, btype="low", fs=fs)
    return signal.filtfilt(b, a, sig)


def kalman_1d(sig: np.ndarray, q_var: float, r_var: float) -> np.ndarray:
    n = len(sig)
    x, p = float(sig[0]), float(r_var)
    out = np.empty(n)
    for k in range(n):
        p_pred = p + q_var
        kk = p_pred / (p_pred + r_var)
        x = x + kk * (sig[k] - x)
        p = (1.0 - kk) * p_pred
        out[k] = x
    return out


def denoise_channel(sig: np.ndarray, fs: float,
                    cutoff_hz: float, order: int, q_ratio: float) -> np.ndarray:
    bw = butterworth_lp(sig, fs, cutoff_hz, order)
    if q_ratio > 0:
        r_var = float(np.var(sig - bw)) + 1e-30
        return kalman_1d(bw, q_var=r_var * q_ratio, r_var=r_var)
    else:
        return bw


def process_run(csv_path: Path, cutoff_hz, butter_order: int, q_ratio: float):
    bak_path = csv_path.with_suffix(".csv.bak")

    # Skip if already processed (bak already exists)
    if bak_path.exists():
        print(f"  [skip] {csv_path.parent.name}  (bak exists)")
        return

    df = load_data(csv_path)
    df_uni, fs = resample_uniform(df)
    t = df_uni["timestamp"].values

    cutoff = cutoff_hz if cutoff_hz is not None else min(0.10 * fs / 2.0, 50.0)

    df_out = pd.DataFrame({"timestamp": t})
    for ch in CHANNELS:
        df_out[ch] = denoise_channel(df_uni[ch].values.astype(float),
                                     fs, cutoff, butter_order, q_ratio)

    shutil.copy2(csv_path, bak_path)
    df_out.to_csv(csv_path, index=False, float_format="%.9f")
    print(f"  [done] {csv_path.parent.name}  ({len(df)} → {len(df_uni)} samples @ {fs:.0f} Hz, cutoff {cutoff:.1f} Hz)")


def restore_run(csv_path: Path):
    bak_path = csv_path.with_suffix(".csv.bak")

    if not bak_path.exists():
        print(f"  [miss] {csv_path.parent.name}  (no ft_sensor_data.csv.bak)")
        return

    shutil.copy2(bak_path, csv_path)
    bak_path.unlink()
    print(f"  [restored] {csv_path.parent.name}")


def main():
    default_dir = "tactile_data/ur5/tactip-127/surface-zRxy-June19-calibration"
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset_dir", nargs="?", default=default_dir)
    p.add_argument("--lowpass-cutoff", type=float, default=50)
    p.add_argument("--butter-order", type=int, default=4)
    p.add_argument("--q-ratio", type=float, default=0)
    p.add_argument("--restore", action="store_true", default=True,
                    help="Restore ft_sensor_data.csv from ft_sensor_data.csv.bak instead of denoising")
    args = p.parse_args()

    dataset_dir = Path(args.dataset_dir)
    run_dirs = sorted(dataset_dir.glob("run_*"), key=lambda d: int(d.name.split("_")[1]))

    print(f"Dataset : {dataset_dir.resolve()}")
    print(f"Runs    : {len(run_dirs)}")
    if args.restore:
        print("Mode    : restore from .bak\n")
        for run_dir in run_dirs:
            csv_path = run_dir / "ft_sensor_data.csv"
            restore_run(csv_path)
        print("\nDone.")

    print(f"Settings: order={args.butter_order}, q_ratio={args.q_ratio:.1e}, "
          f"cutoff={'auto' if args.lowpass_cutoff is None else f'{args.lowpass_cutoff} Hz'}\n")


    for run_dir in run_dirs:
        csv_path = run_dir / "ft_sensor_data.csv"
        if not csv_path.exists():
            print(f"  [miss] {run_dir.name}  (no ft_sensor_data.csv)")
            continue
        process_run(csv_path, args.lowpass_cutoff, args.butter_order, args.q_ratio)

    print("\nDone.")


if __name__ == "__main__":
    main()
