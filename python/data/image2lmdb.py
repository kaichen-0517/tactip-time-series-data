import os
import cv2
import lmdb
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import re

REFERENCE = None


GB_SIZE = 1024 * 1024 * 1024
    
def scan_dataset_folders(base_dir):
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    tasks = []
    target_paths = []
    for item in os.listdir(base_dir):
        full_path = os.path.join(base_dir, item).replace('\\', '/')
        if os.path.isdir(full_path) and item.startswith("run_"):
            if REFERENCE and f"run_{REFERENCE}" in full_path:
                print(f"Skipped {full_path}")
                continue
            target_paths.append(f"{full_path}/time_series_images") 
    target_paths.sort(key=lambda p: int(re.search(r'run_(\d+)', p).group(1)))
    
    for target in target_paths:
        img_files = sorted(
            [f for f in os.listdir(target) if f.lower().endswith(valid_extensions)],
            key=lambda f: int(re.search(r'\d+', f).group()) if re.search(r'\d+', f) else 0
        )
        run_prefix = re.search(r'run_(\d+)', target).group(0)
        for idx, filename in enumerate(img_files):
            file_path = os.path.join(target, filename).replace('\\', '/')
            key_str = f"{run_prefix}/image_{idx:06d}"
            tasks.append((key_str, file_path))
    return tasks

def process_image_task(key_str, file_path, data_queue, lossless_png=True):
    try:
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if img is None: return

        if lossless_png:
            _, img_encode = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        else:
            _, img_encode = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
        final_key = f"{key_str}/img".encode('ascii')
        data_queue.put((final_key, img_encode.tobytes()))
    except Exception as e:
        print(f"\n[Error] Failed: {file_path} {str(e)}")

def lmdb_writer(env, data_queue, batch_size=100, total_count=0):
    pbar = tqdm(total=total_count, desc="LMDB Writing", unit="frame")
    txn = env.begin(write=True)
    count = 0

    try:
        while True:
            key, img_bytes = data_queue.get()
            data_queue.task_done()

            if key is None:
                break

            txn.put(key, img_bytes)
            count += 1
            pbar.update(1)

            if count % batch_size == 0:
                txn.commit()
                txn = env.begin(write=True)

        txn.commit()
    except Exception:
        txn.abort()
        raise
    finally:
        pbar.close()

def convert_folder_to_lmdb(base_dir, lmdb_output_path, max_workers=8, batch_size=100, lossless_png=True):
    tasks = scan_dataset_folders(base_dir)
    total_images = len(tasks)
    print(total_images)
    if total_images == 0: return
    
    env = lmdb.open(lmdb_output_path, map_size=int(250*GB_SIZE), writemap=True)
    data_queue = queue.Queue(maxsize=128)
    writer_thread = threading.Thread(target=lmdb_writer, args=(env, data_queue, batch_size, total_images))
    writer_thread.start()
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for key_str, file_path in tasks:
                executor.submit(process_image_task, key_str, file_path, data_queue, lossless_png)
        # sentinel sent only after all executor threads have finished
        data_queue.put((None, None))
        writer_thread.join()
    finally:
        env.close()
    
    
if __name__ == "__main__": 
    base_dir = f"./tactile_data/ur5/tactip-127/surface-zRxy-new-speed-1"   
    convert_folder_to_lmdb(base_dir, base_dir)
    # out = scan_dataset_folders(base_dir)
    # print(out)