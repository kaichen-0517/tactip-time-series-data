import os
import cv2

image_folder = r"tactip\surface-zRxy\run_29\time_series_images"

valid_extensions = (".png", ".jpg", ".jpeg", ".bmp")
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)]

image_files.sort(key=lambda x: int("".join(filter(str.isdigit, x)) or 0))

if not image_files:
    exit()

window_name = "Image Viewer"

for idx, file_name in enumerate(image_files):
    full_path = os.path.join(image_folder, file_name)

    img = cv2.imread(full_path)
    if img is None:
        continue

    display_text = f"[{idx + 1}/{len(image_files)}] {file_name}"
    cv2.putText(img, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow(window_name, img)
    key = cv2.waitKey(1)
    if key == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()
