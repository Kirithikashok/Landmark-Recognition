# Landmark-Recognition

## Project Overview

This project implements a Landmark Recognition System. The model is trained on a subset of the Google Landmark Dataset and predicts the landmark present in an uploaded image.

---

## Features

- Landmark image classification
- Fine-tuning for improved accuracy
- Data augmentation
- Top-5 prediction support
- GUI-based image testing

---

## Dataset

### Dataset Source

Google Landmark Dataset v2

To download the datasert:

Run download-dataset.sh file to download the dataset

---

## Required Metadata Files

### 1. train.csv

Contains image ID and landmark ID mapping.

Download:

https://s3.amazonaws.com/google-landmark/metadata/train.csv


---

### 2. train_clean.csv

Contains verified image-to-landmark mappings.

Download:

https://s3.amazonaws.com/google-landmark/metadata/train_clean.csv


---

### 3. train_label_to_category.csv

Maps landmark IDs to category IDs.

Download:

https://s3.amazonaws.com/google-landmark/metadata/train_label_to_category.csv



---

## Download Size

Approximate size per archive:

```text
 1GB
```

### Recommended Downloads

| Files | Approx Size |
|---------|------------|
| images_000 only | 1 GB |
| images_000 - images_0020 | 20 GB |
| images_000 - images_040 | 40 GB |
| Full Dataset | 500+ GB |

---

## Dataset Used In This Project

Downloaded:

```text
archive: 20
Size: 19.7 GB
Files: 165,320
Folders: 177
```

---

## Dataset Statistics

After mapping image IDs to landmark IDs:

```text
Downloaded Images : 165,320
Matched Images    : 165,320
Unique Landmarks  : 75,058
```

Filtered dataset:

```text
Classes Selected : 260
Minimum Images Per Class : 20
```

---

## Project Workflow

### Step 1

Download metadata files.

### Step 2

Download image archives using:

```bash
bash download-dataset.sh train (value)
```

### Step 3

Extract image archives.

### Step 4

Match image IDs with landmark IDs using:

- train.csv
- train_clean.csv

### Step 5 (optional)

Filter landmarks containing at least 20 images.

### Step 6

Create dataset folders.

Example:

```text
dataset_filtered/
├── 27/
├── 1924/
├── 10618/
├── 20409/
└── ...
```

Each folder name represents a Landmark ID.

### Step 7

Split dataset:

```text
80% Train
20% Test
```

Structure:

```text
dataset_split/
│
├── train/
│   ├── 27/
│   ├── 1924/
│   └── ...
│
└── test/
    ├── 27/
    ├── 1924/
    └── ...
```

### Step 8

Train model using any of the following models:

- MobileNetV2
- EfficientNetB0
- EfficientNetB3
- ResNet50

### Step 9

Fine-tune selected layers.

### Step 10

Save trained model.

Example:

```text
landmark_model_final.keras
```

### Step 11

Test using GUI.

Upload image and predict landmark.

---

## Technologies Used

### Language

- Python 3

### Framework

- TensorFlow
- Keras

### Libraries

- NumPy
- Pandas
- Matplotlib
- Pillow
- Tkinter

---

## Model Architecture

### Data Augmentation

```python
RandomFlip
RandomRotation
RandomZoom
RandomContrast
```

### Backbone Networks

- MobileNetV2
- EfficientNetB0
- EfficientNetB3
- ResNet50

### Classification Head

```python
GlobalAveragePooling2D()
Dropout(0.4)
Dense(num_classes, activation="softmax")
```

---

## Future Improvements

- Increase number of landmark classes
- Download larger dataset portion
- Implement image retrieval
- Use FAISS similarity search
- Deploy using Flask

---
