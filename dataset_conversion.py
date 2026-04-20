import os
import shutil
from tqdm import tqdm

# -------- PATHS --------
ROOT = "/Users/apple/Documents/CSUF/Spring 26/529/Circuit-Board-Fault-Detection-using-ML-main/DeepPCB-master/PCBData/"
OUTPUT = "/Users/apple/Documents/CSUF/Spring 26/529/Circuit-Board-Fault-Detection-using-ML-main/DeepPCB-master/output"

GOOD_DIR = os.path.join(OUTPUT, "good")
BAD_DIR = os.path.join(OUTPUT, "bad")

# os.makedirs(GOOD_DIR, exist_ok=True)
# os.makedirs(BAD_DIR, exist_ok=True)
#
# groups = [g for g in os.listdir(ROOT) if g.startswith("group")]
#
# good_count = 0
# bad_count = 0
#
# for group in tqdm(groups):
#     group_path = os.path.join(ROOT, group)
#
#     # Find image folder and annotation folder
#     subfolders = os.listdir(group_path)
#
#     img_folder = None
#     ann_folder = None
#
#     for sub in subfolders:
#         if sub.endswith("_not"):
#             ann_folder = os.path.join(group_path, sub)
#         else:
#             img_folder = os.path.join(group_path, sub)
#
#     if not img_folder or not ann_folder:
#         continue
#
#     image_files = os.listdir(img_folder)
#
#     for file in image_files:
#         if file.endswith("_temp.jpg"):
#             # -------- GOOD --------
#             src = os.path.join(img_folder, file)
#             dst = os.path.join(GOOD_DIR, file)
#             shutil.copy(src, dst)
#             good_count += 1
#
#         elif file.endswith("_test.jpg"):
#             base_name = file.replace("_test.jpg", "")
#             ann_file = base_name + ".txt"
#             ann_path = os.path.join(ann_folder, ann_file)
#
#             # -------- BAD (only if annotation exists) --------
#             if os.path.exists(ann_path):
#                 src = os.path.join(img_folder, file)
#                 dst = os.path.join(BAD_DIR, file)
#                 shutil.copy(src, dst)
#                 bad_count += 1
#
# print("✅ Conversion Complete")
# print(f"Good images: {good_count}")
# print(f"Bad images: {bad_count}")


import os
import cv2
import numpy as np

IMG_SIZE = 224

good_dir = "dataset/good"
bad_dir = "dataset/bad"

X = []
y = []

# Create lookup for templates
temp_map = {}

for file in os.listdir(good_dir):
    base = file.replace("_temp.jpg", "")
    temp_map[base] = file

# Process bad images and match with template
for file in os.listdir(bad_dir):
    base = file.replace("_test.jpg", "")

    if base not in temp_map:
        continue

    test_path = os.path.join(bad_dir, file)
    temp_path = os.path.join(good_dir, temp_map[base])

    test_img = cv2.imread(test_path)
    temp_img = cv2.imread(temp_path)

    if test_img is None or temp_img is None:
        continue

    test_img = cv2.resize(test_img, (IMG_SIZE, IMG_SIZE))
    temp_img = cv2.resize(temp_img, (IMG_SIZE, IMG_SIZE))

    # 🔥 KEY IMPROVEMENT
    diff = cv2.absdiff(test_img, temp_img)

    # BAD sample (defect)
    X.append(diff)
    y.append(1)

    # GOOD sample
    X.append(temp_img)
    y.append(0)

X = np.array(X) / 255.0
y = np.array(y)

print("Dataset:", X.shape, y.shape)