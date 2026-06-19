import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def main(csv_file, output_file):
    df = pd.read_csv(csv_file)

    x = df["actual_TCP_pose_0"]
    y = df["actual_TCP_pose_1"]
    z = df["actual_TCP_pose_2"]
    Rx = df["actual_TCP_pose_3"]
    Ry = df["actual_TCP_pose_4"]
    Rz = df["actual_TCP_pose_5"]

    data_sets = [x, y, z, Rx, Ry, Rz]
    colors = ["#2196F3", "#8721F3", "#21F391", "#2C21F3", "#F321A6", "#F3C621"]
    labels = ["X-axis", "Y-axis", "Z-axis", "Rx-axis", "Ry-axis", "Rz-axis"]
    units = ["m", "m", "m", "rad", "rad", "rad"]
    
    fig, axes = plt.subplots(6, 1, figsize=(max(10, len(x) * 0.1), 10), sharex=True)


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
    target_folder = f"{base_dir}/surface-zRxy-10Jun-calibrate-1"
    for item in tqdm(os.listdir(target_folder), total=len(os.listdir(target_folder))):
        folder = os.path.join(target_folder, item)
        if os.path.isdir(folder) and item.startswith("run_"):
            csv = output = f"{folder}/robot_data.csv"
            output = f"{folder}/tcp_pose.png"
            main(csv, output)