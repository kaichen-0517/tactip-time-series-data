""" Visualise FT sensor force XYZ overlaid in one subplot. """

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


def main(csv_file, output_file):
    df = pd.read_csv(csv_file)

    fx = df["force_x"]
    fy = df["force_y"]
    fz = df["force_z"]

    colors = ["#2196F3", "#8721F3", "#21F391"]
    labels = ["Fx", "Fy", "Fz"]

    fig, ax = plt.subplots(figsize=(max(10, len(fx) * 0.001), 5))

    for data, color, label in zip([fx, fy, fz], colors, labels):
        ax.scatter(range(len(data)), data - data.mean(), s=10, color=color, alpha=0.7, label=label)

    ax.set_ylabel("Force (N)", fontsize=11)
    ax.set_xlabel("Sample", fontsize=11)
    ax.set_title("Force XYZ", fontsize=14, pad=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()


if __name__ == "__main__":
    base_dir = f"./tactile_data/ur5/tactip-127"
    target_folder = f"{base_dir}/surface-zRxy-June19-calibration"
    for item in tqdm(os.listdir(target_folder), total=len(os.listdir(target_folder))):
        folder = os.path.join(target_folder, item)
        if os.path.isdir(folder) and item.startswith("run_"):
            csv = f"{folder}/ft_sensor_data.csv.bak"
            output = f"{folder}/force_xyz.png"
            main(csv, output)
