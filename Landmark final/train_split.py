import os
import shutil
import random
from tqdm import tqdm

SOURCE_DIR = r"D:\HCL Guvi\Landmark\dataset_filtered"
OUTPUT_DIR = r"D:\HCL Guvi\Landmark\dataset_split"

TRAIN_RATIO = 0.8

random.seed(42)

os.makedirs(OUTPUT_DIR, exist_ok=True)

train_root = os.path.join(OUTPUT_DIR, "train")
test_root = os.path.join(OUTPUT_DIR, "test")

os.makedirs(train_root, exist_ok=True)
os.makedirs(test_root, exist_ok=True)

classes = [d for d in os.listdir(SOURCE_DIR)
           if os.path.isdir(os.path.join(SOURCE_DIR, d))]

print(f"Found {len(classes)} classes")

for cls in tqdm(classes):

    cls_path = os.path.join(SOURCE_DIR, cls)

    images = [
        f for f in os.listdir(cls_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if len(images) < 2:
        continue

    random.shuffle(images)

    split_idx = int(len(images) * TRAIN_RATIO)

    train_imgs = images[:split_idx]
    test_imgs = images[split_idx:]

    train_cls_dir = os.path.join(train_root, cls)
    test_cls_dir = os.path.join(test_root, cls)

    os.makedirs(train_cls_dir, exist_ok=True)
    os.makedirs(test_cls_dir, exist_ok=True)

    for img in train_imgs:
        shutil.copy2(
            os.path.join(cls_path, img),
            os.path.join(train_cls_dir, img)
        )

    for img in test_imgs:
        shutil.copy2(
            os.path.join(cls_path, img),
            os.path.join(test_cls_dir, img)
        )

print("\nDataset split completed!")