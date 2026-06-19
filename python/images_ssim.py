import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
import pandas as pd
from tqdm import tqdm

def compute_ssim_curve(folder_path, output_path=None):
    png_files = sorted(glob.glob(os.path.join(folder_path, "*.png")),
                   key=lambda f: int(os.path.splitext(os.path.basename(f))[0].split("_")[-1]))

    if len(png_files) < 2:
        return
    
    ref_img = io.imread(png_files[0])
    ref_name = os.path.basename(png_files[0])
    if ref_img.ndim == 3:
        ref_gray = color.rgb2gray(ref_img[..., :3])
    else:
        ref_gray = ref_img.astype(np.float64)

    ssim_values = []
    filenames = []
    
    for i, fpath in tqdm(enumerate(png_files), total=len(png_files), desc='Processing'):
        img = io.imread(fpath)
        name = os.path.basename(fpath)

        if img.ndim == 3:
            gray = color.rgb2gray(img[..., :3])
        else:
            gray = img.astype(np.float64)

        if gray.shape != ref_gray.shape:
            gray = resize(gray, ref_gray.shape, anti_aliasing=True)

        val = ssim(ref_gray, gray, data_range=1.0)
        ssim_values.append(val)
        filenames.append(name)
        # print(f"  [{i+1:3d}/{len(png_files)}] {name}  SSIM = {val:.4f}")

    fig, ax = plt.subplots(figsize=(max(10, len(png_files) * 0.1), 5))
    ax.scatter(range(len(ssim_values)), ssim_values, s=10, color="#2196F3", alpha=0.7)

    ax.set_title("Image SSIM", fontsize=14, pad=12)
    ax.set_xlabel("Index", fontsize=11)
    ax.set_ylabel("Value", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    if output_path is None:
        output_path = os.path.join(folder_path, "ssim_curve.png")
    fig.savefig(output_path, dpi=150)

    return ssim_values, filenames


if __name__ == "__main__":
    base_dir = f"./tactile_data/ur5/tactip-127"
    target_folder = f"{base_dir}/surface-zRxy-June19-calibration"
    for item in tqdm(os.listdir(target_folder), total=len(os.listdir(target_folder))):
        folder = os.path.join(target_folder, item)
        if os.path.isdir(folder) and item.startswith("run_"):
            ssim_values, _ = compute_ssim_curve(f"{folder}/time_series_images", f"{folder}/ssim_curve.png")
            data = pd.read_csv(f"{folder}/image_data.csv")
            ts = data['timestamp']
            images = data['time_series_images']
            
            rows = np.column_stack([ts, ssim_values, images])
            df = pd.DataFrame(rows, columns=['timestamp', 'ssim', 'time_series_images'])
            df.to_csv(f"{folder}/ssim.csv", index=False)
        