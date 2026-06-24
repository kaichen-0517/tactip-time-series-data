import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


NOISE_DURATION = 0.2  # seconds

COLORS = ["#2196F3", "#8721F3", "#21F391", "#2C21F3", "#F321A6", "#F3C621"]
LABELS = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
UNITS = ["N", "N", "N", "Nm", "Nm", "Nm"]
CHANNELS = ["force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z"]


def plot_noise_waveform(df_noise, t_noise, output_path):
    fig, axes = plt.subplots(6, 1, figsize=(10, 15), sharex=True)
    fig.suptitle(f"Noise Waveform (first {NOISE_DURATION}s)", fontsize=14)

    for i, ax in enumerate(axes):
        ax.plot(t_noise, df_noise[CHANNELS[i]], color=COLORS[i], linewidth=0.8)
        mean_val = df_noise[CHANNELS[i]].mean()
        ax.axhline(mean_val, color="red", linestyle="--", linewidth=1,
                   label=f"mean={mean_val:.4f}")
        ax.set_ylabel(f"{LABELS[i]} ({UNITS[i]})", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved noise waveform: {output_path}")


def plot_denoised(df, t_rel, noise_means, output_path):
    fig, axes = plt.subplots(6, 1, figsize=(12, 15), sharex=True)
    fig.suptitle("Denoised Force/Torque Signal (mean noise removed)", fontsize=14)

    for i, ax in enumerate(axes):
        denoised = df[CHANNELS[i]] - noise_means[i]
        ax.plot(t_rel, denoised, color=COLORS[i], linewidth=0.6)
        ax.axvline(NOISE_DURATION, color="gray", linestyle=":", linewidth=1,
                   label=f"noise window end ({NOISE_DURATION}s)")
        ax.set_ylabel(f"{LABELS[i]} ({UNITS[i]})", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved denoised waveform: {output_path}")


def main(csv_file, output_dir):
    df = pd.read_csv(csv_file)

    t0 = df["timestamp"].iloc[0]
    df["t_rel"] = df["timestamp"] - t0

    mask_noise = df["t_rel"] < NOISE_DURATION
    df_noise = df[mask_noise]
    t_noise = df_noise["t_rel"].values

    print(f"Total samples: {len(df)}, noise window samples: {len(df_noise)}")
    print(f"Signal duration: {df['t_rel'].iloc[-1]:.4f}s")

    noise_means = [df_noise[ch].mean() for ch in CHANNELS]
    print("Noise means:")
    for label, mean in zip(LABELS, noise_means):
        print(f"  {label}: {mean:.6f}")

    plot_noise_waveform(df_noise, t_noise,
                        os.path.join(output_dir, "ft_noise_waveform.png"))
    plot_denoised(df, df["t_rel"].values, noise_means,
                  os.path.join(output_dir, "ft_denoised.png"))

    df_out = df.copy()
    for ch in CHANNELS:
        df_out[ch] = df_out[ch] - df_out[ch].iloc[:len(df_noise)].mean()
    df_out.drop(columns=["t_rel"]).to_csv(
        os.path.join(output_dir, "ft_sensor_data_denoised.csv"), index=False)
    print(f"Saved denoised CSV: {os.path.join(output_dir, 'ft_sensor_data_denoised.csv')}")


if __name__ == "__main__":
    run_dir = "./tactile_data/ur5/tactip-127/surface-zRxy-10Jun-speed-1/run_0"
    csv_file = os.path.join(run_dir, "ft_sensor_data.csv")
    main(csv_file, run_dir)
