import os
import shutil
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# PATHS (EDIT THIS)
IMAGE_ROOT = r"D:\HCL Guvi\Landmark"
CSV_PATH = r"D:\HCL Guvi\Landmark\train.csv"
OUTPUT_DIR = r"D:\HCL Guvi\Landmark\dataset_filtered"

# LOAD CSV
print("Loading CSV...")
df = pd.read_csv(CSV_PATH, usecols=["id", "landmark_id"])

# map image_id -> landmark_id
id_to_landmark = dict(zip(df["id"].astype(str), df["landmark_id"]))

# STEP 1: COUNT IMAGES PER LANDMARK
print("Counting images per landmark...")

landmark_count = defaultdict(int)

all_images = []

for root, _, files in os.walk(IMAGE_ROOT):
    for f in files:
        if f.lower().endswith(".jpg"):
            img_id = os.path.splitext(f)[0]

            if img_id in id_to_landmark:
                landmark_id = id_to_landmark[img_id]
                landmark_count[landmark_id] += 1
                all_images.append((root, f, landmark_id))

# STEP 2: FILTER LANDMARKS >= 20 IMAGES
valid_landmarks = {k for k, v in landmark_count.items() if v >= 20}

print(f"Total landmarks found: {len(landmark_count)}")
print(f"Valid landmarks (>=20 images): {len(valid_landmarks)}")

# STEP 3: CREATE OUTPUT FOLDERS
os.makedirs(OUTPUT_DIR, exist_ok=True)

# STEP 4: COPY FILES
print("Copying images...")

copied = 0

for root, file, landmark_id in tqdm(all_images):

    if landmark_id in valid_landmarks:

        src = os.path.join(root, file)
        dst_folder = os.path.join(OUTPUT_DIR, str(landmark_id))
        os.makedirs(dst_folder, exist_ok=True)

        dst = os.path.join(dst_folder, file)

        shutil.copy2(src, dst)
        copied += 1

print("\nDONE")
print("Total images copied:", copied)
print("Total classes:", len(valid_landmarks))