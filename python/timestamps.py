import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def load_data(robot_path: str, ssim_path: str):
    robot = pd.read_csv(robot_path).copy()
    ssim = pd.read_csv(ssim_path).copy()

    robot["t_rel"] = robot["timestamp"] - robot["timestamp"].iloc[0]
    ssim["t_rel"] = ssim["timestamp"] - ssim["timestamp"].iloc[0] + 0.008
    return robot, ssim


def align_interpolated(robot: pd.DataFrame, ssim: pd.DataFrame) -> pd.DataFrame:
    """Align SSIM onto the robot time grid using linear interpolation."""
    f_ssim = interp1d(
        ssim["t_rel"], ssim["ssim"],
        kind="linear", bounds_error=False, fill_value=np.nan,
    )
    ssim_on_robot_grid = f_ssim(robot["t_rel"])

    return pd.DataFrame({
        "t": robot["t_rel"],
        "tcp_z": robot["actual_TCP_pose_2"],
        "ssim": ssim_on_robot_grid,
    })



def plot_results(
    robot: pd.DataFrame,
    ssim_raw: pd.DataFrame,
    aligned_interp: pd.DataFrame,
    output_path: str = None,
):
    """Single-panel plot: raw and aligned TCP_z + SSIM on dual y-axes."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax_r = ax.twinx()

    ax.scatter(robot["t_rel"], robot["actual_TCP_pose_2"],
               s=8, color="#2196F3", alpha=0.4, label="TCP_z raw")
    ax.scatter(aligned_interp["t"], aligned_interp["tcp_z"],
               s=8, color="#0D47A1", alpha=0.8, marker="x", label="TCP_z aligned")
    ax_r.scatter(ssim_raw["t_rel"], ssim_raw["ssim"],
                 s=8, color="#F3A221", alpha=0.4, label="SSIM raw")
    ax_r.scatter(aligned_interp["t"], aligned_interp["ssim"],
                 s=8, color="#E65100", alpha=0.8, marker="x", label="SSIM aligned (interp)")

    ax.set_title("TCP_z & SSIM — raw vs aligned (linear interpolation)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("TCP_z (m)", color="#2196F3")
    ax_r.set_ylabel("SSIM", color="#F3A221")
    ax.grid(True, linestyle="--", alpha=0.4)

    handles = ax.get_legend_handles_labels()
    handles_r = ax_r.get_legend_handles_labels()
    ax.legend(handles[0] + handles_r[0], handles[1] + handles_r[1], loc="upper right", fontsize=9)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"\n  Plot saved → {output_path}")
    else:
        plt.show()


def compute_correlation(aligned: pd.DataFrame) -> float:
    valid = aligned["ssim"].notna()
    if valid.sum() < 2:
        return float("nan")
    return np.corrcoef(aligned.loc[valid, "tcp_z"], aligned.loc[valid, "ssim"])[0, 1]


def main():
    base_dir = "./tactile_data/ur5/tactip-127"
    target_folder = f"{base_dir}/surface-zRxy-10Jun-calibrate-1/run_3"

    robot, ssim = load_data(
        f"{target_folder}/robot_data.csv",
        f"{target_folder}/ssim.csv",
    )

    print(f"Robot data : {robot.shape}  duration: {robot['t_rel'].iloc[-1]:.3f} s")
    print(f"SSIM data  : {ssim.shape}  duration: {ssim['t_rel'].iloc[-1]:.3f} s")

    print(f"\nTCP_pose_2 minimum  t_rel = {robot['t_rel'][robot['actual_TCP_pose_2'].idxmin()]:.4f}"
          f"  value = {robot['actual_TCP_pose_2'].min():.6f}")
    print(f"SSIM minimum        t_rel = {ssim['t_rel'][ssim['ssim'].idxmin()]:.4f}"
          f"  value = {ssim['ssim'].min():.6f}")

    aligned_interp = align_interpolated(robot, ssim)
    corr_interp = compute_correlation(aligned_interp)

    print("\n--- Alignment (linear interpolation, tcp_z vs ssim) ---")
    print(f"  Correlation = {corr_interp:.4f}  "
          f"valid points = {aligned_interp['ssim'].notna().sum()}")

    plot_results(
        robot, ssim,
        aligned_interp,
        output_path=f"{target_folder}/timestamp_alignment.png",
    )


if __name__ == "__main__":
    main()
