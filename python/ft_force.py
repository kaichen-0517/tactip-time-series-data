""" Visualise FT sensor data. """

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


def main(csv_file, output_file):
    df = pd.read_csv(csv_file)

    fx = df["force_x"]
    fy = df["force_y"]
    fz = df["force_z"]
    tx = df["torque_x"]
    ty = df["torque_y"]
    tz = df["torque_z"]

    data_sets = [fx, fy, fz, tx, ty, tz]
    colors = ["#2196F3", "#8721F3", "#21F391", "#2C21F3", "#F321A6", "#F3C621"]
    labels = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
    units = ["N", "N", "N", "Nmm", "Nmm", "Nmm"]

    fig, axes = plt.subplots(6, 1, figsize=(max(10, len(fx) * 0.02), 15), sharex=True)

    for i, ax in enumerate(axes):
        ax.scatter(range(len(data_sets[i])), data_sets[i], s=10, color=colors[i], alpha=0.7)

        # Labeling each subplot
        ax.set_ylabel(f"{labels[i]} ({units[i]})", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.4)

        # Title only on the top plot, or use fig.suptitle
        if i == 0:
            ax.set_title("actual_TCP_pose", fontsize=14, pad=12)
        fig.tight_layout()

    plt.savefig(output_file, dpi=150)
    # plt.show()


if __name__ == "__main__":
    base_dir = f"./tactile_data/ur5/tactip-127"
    target_folder = f"{base_dir}/surface-zRxy-June19-calibration"
    for item in tqdm(os.listdir(target_folder), total=len(os.listdir(target_folder))):
        folder = os.path.join(target_folder, item)
        if os.path.isdir(folder) and item.startswith("run_"):
            csv = f"{folder}/ft_sensor_data.csv"
            output = f"{folder}/force_torque.png"
            main(csv, output)
    
    # csv = f"{target_folder}/run_0/ft_sensor_data.csv"
    # output = f"{target_folder}/run_0/force_torque.png"     
    # main(csv, output)
