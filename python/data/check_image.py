import csv
import os
from pathlib import Path
from tqdm import tqdm


def verify_images_with_csv(path, img_column="time_series_images"):
    """Check that image filenames in a CSV column match files on disk."""
    csv_file = Path(f"{path}/image_data.csv")
    img_dir = Path(f"{path}/time_series_images")

    if not csv_file.exists():
        print(f"Error: CSV file not found -> {csv_file}")
        return
    if not img_dir.exists():
        print(f"Error: Image directory not found -> {img_dir}")
        return

    actual_images = {
        f.name
        for f in img_dir.iterdir()
        if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg"}
    }

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if img_column not in reader.fieldnames:
            print(f"Error: Column '{img_column}' not found. Available: {reader.fieldnames}")
            return
        csv_images = [os.path.basename(row[img_column]) for row in reader]

    if len(csv_images) != len(actual_images):
        print(f"MISMATCH detected in {path}.")
        print(f"CSV records : {len(csv_images)}")
        print(f"Disk images : {len(actual_images)}")


if __name__ == "__main__":
    target_folder = f"./tactile_data/ur5/tactip-127/surface-zRxy-10Jun-calibrate-1"
    for item in tqdm(os.listdir(target_folder), total=len(os.listdir(target_folder))):
        full_path = os.path.join(target_folder, item)
        if os.path.isdir(full_path) and item.startswith("run_"):
            verify_images_with_csv(full_path)
