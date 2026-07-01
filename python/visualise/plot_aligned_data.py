from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def rescale_to_match(series: pd.Series, ref: pd.Series) -> pd.Series:
    """Linearly rescale `series` so its min/max span matches `ref`'s amplitude."""
    lo, hi = series.min(), series.max()
    ref_lo, ref_hi = ref.min(), ref.max()
    if hi == lo:
        return series * 0 + ref_lo
    return (series - lo) / (hi - lo) * (ref_hi - ref_lo) + ref_lo


def resolve_tcp_z_column(columns):
    for c in columns:
        if c.lower() == "actual_tcp_z":
            return c
    if "actual_TCP_pose_2" in columns:
        return "actual_TCP_pose_2"
    raise KeyError("No actual_tcp_z / actual_TCP_pose_2 column found in aligned_data.csv")


def plot_run(run_dir: Path, output_dir: Path):
    aligned_path = run_dir / "aligned_data.csv"
    ssim_path = run_dir / "ssim.csv"
    if not aligned_path.exists() or not ssim_path.exists():
        print(f"[SKIP] {run_dir.name}: missing aligned_data.csv or ssim.csv")
        return

    aligned = pd.read_csv(aligned_path)
    ssim = pd.read_csv(ssim_path)

    tcp_z_col = resolve_tcp_z_column(aligned.columns)

    data = aligned[["time_series_images", "t_rel", tcp_z_col, "force_z"]].merge(
        ssim[["time_series_images", "ssim"]], on="time_series_images", how="left"
    ).sort_values("t_rel")

    fig, ax_left = plt.subplots(figsize=(10, 6))
    ax_ssim = ax_left.twinx()

    xlim = (data["t_rel"].min(), data["t_rel"].max())
    ax_left.set_xlim(*xlim)

    force_z = -data["force_z"]
    tcp_z_scaled = rescale_to_match(data[tcp_z_col], force_z)

    line_tcp = ax_left.scatter(data["t_rel"], tcp_z_scaled, color="tab:blue",
                              label=f"{tcp_z_col} (scaled to force_z amplitude)", s=10)
    line_force = ax_left.scatter(data["t_rel"], force_z, color="tab:orange", label="force_z", s=10)
    line_ssim = ax_ssim.scatter(data["t_rel"], data["ssim"], color="tab:green", label="ssim", s=10)

    ax_left.set_xlabel("t_rel (s)")
    ax_left.set_ylabel("force_z (N)")
    ax_ssim.set_ylabel("ssim", color="tab:green")
    ax_ssim.tick_params(axis="y", colors="tab:green")

    ax_left.grid(True)
    ax_left.legend(handles=[line_tcp, line_force, line_ssim], loc="upper right")

    fig.suptitle(run_dir.name)
    fig.tight_layout()

    out_path = f"{output_dir}/{run_dir.name.split('/')[-1]}_aligned_data.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] saved {out_path}")


def main():
    
    base_dir = "./tactile_data/ur5/tactip-127/surface-zRxy-June19-calibration-without_skin"
    output_dir = Path(f"{base_dir}/output")
    output_dir.mkdir(exist_ok=True)
    base_dir = Path(base_dir)

    run_dirs = sorted(base_dir.glob("run_*"), key=lambda p: int(p.name.split("_")[1]))
    for run_dir in run_dirs:
        plot_run(run_dir, output_dir)


if __name__ == "__main__":
    main()
