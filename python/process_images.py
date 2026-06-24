""" Unzip and process images. """

import zipfile
from tqdm import tqdm
import os
import cv2

UNZIP = True
PROCESS = True

base_dir = f"./temp/tactile_data/ur5/tactip-127"
target_folder = f"{base_dir}/surface-zRxy-June19-calibration-without_skin"
for item in tqdm(os.listdir(target_folder), total=len(os.listdir(target_folder))):
    folder = os.path.join(target_folder, item)
    if os.path.isdir(folder) and item.startswith("run_"):
        output_folder = f'{folder}/time_series_images'
        os.makedirs(output_folder, exist_ok=True)
        if UNZIP:
            with zipfile.ZipFile(f'{folder}/time_series_images_bak.zip', 'r') as zip_ref:
                zip_ref.extractall(output_folder)
            
        if PROCESS:
            target_size = (800, 450)

            for filename in os.listdir(output_folder):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img = cv2.imread(os.path.join(output_folder, filename))
                    if img is None:
                        continue
                    
                    if len(img.shape) == 3:
                        if img.shape[2] == 2:
                            img = img[:, :, 0] # Blue channel BGR
                    img = cv2.resize(img, target_size)
                    
                    cv2.imwrite(os.path.join(output_folder, filename), img)