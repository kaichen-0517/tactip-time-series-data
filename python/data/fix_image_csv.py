import pandas as pd
import os
from tqdm import tqdm


target_folder = "./tactile_data/ur5/tactip-127/surface-zRxy-new-speed-1"

for item in tqdm(os.listdir(target_folder), total=len(os.listdir(target_folder))):
    full_path = os.path.join(target_folder, item)
    if os.path.isdir(full_path) and item.startswith("run_"):
        df = pd.read_csv(f"{full_path}/image_data.csv")

        df['time_series_images'] = df.apply(
            lambda row: f"time_series_images/image_{int(row['frame_id'])}.png", 
            axis=1
        )
        df.to_csv(f"{full_path}/image_data.csv", index=False)