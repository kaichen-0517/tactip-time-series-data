import os
import zipfile
from tqdm import tqdm

LIST = ["image_data_aligned.csv", "ft_sensor_data_aligned.csv", "robot_data_aligned.csv",
        "image_data_aligned.parquet", "robot_aligned.parquet", "ft_sensor_aligned.parquet"]


target_folder = "./tactile_data/ur5/tactip-127/surface-zRxy-June19-classic"

# ft / image / robot
for item in tqdm(os.listdir(target_folder), total=len(os.listdir(target_folder))):
    full_path = os.path.join(target_folder, item)
    if os.path.isdir(full_path) and item.startswith("run_"):
        for file in os.listdir(full_path):
            if file in LIST:
                os.remove(f"{full_path}/{file}")
