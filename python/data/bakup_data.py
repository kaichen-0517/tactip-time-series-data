import os
import zipfile
from tqdm import tqdm


def zip_csv_files(folder_path, output_zip_path):
    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(".csv"):
                file_path = os.path.join(folder_path, file_name)
                zipf.write(file_path, arcname=file_name)


target_folder = "./tactile_data/ur5/tactip-127/surface-zRxy-10Jun-speed-2"

# target.csv
zip_csv_files(target_folder, f"{target_folder}/target.zip")

# ft / image / robot
for item in tqdm(os.listdir(target_folder), total=len(os.listdir(target_folder))):
    full_path = os.path.join(target_folder, item)
    if os.path.isdir(full_path) and item.startswith("run_"):
        zip_csv_files(full_path, f"{full_path}/csv.zip")
