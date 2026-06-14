import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import json

# LOAD MODEL
MODEL_PATH = "best_efficientnetb3.keras"

model = tf.keras.models.load_model(MODEL_PATH)

with open("class_names.json", "r") as f:
    class_names = json.load(f)

IMG_SIZE = (300, 300)

# PREDICT FUNCTION
def predict_image():

    file_path = filedialog.askopenfilename(
        filetypes=[
            ("Images", "*.jpg *.jpeg *.png *.bmp")
        ]
    )

    if not file_path:
        return

    # show image
    img = Image.open(file_path)
    img_preview = img.copy()
    img_preview.thumbnail((300, 300))

    photo = ImageTk.PhotoImage(img_preview)

    image_label.config(image=photo)
    image_label.image = photo

    # preprocess
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)

    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)

    # predict
    preds = model.predict(arr, verbose=0)[0]

    top5_idx = np.argsort(preds)[-5:][::-1]

    result_text = "\nTop 5 Predictions\n\n"

    for i, idx in enumerate(top5_idx, start=1):

        landmark_id = class_names[idx]
        confidence = preds[idx] * 100

        result_text += (
            f"{i}. Landmark ID: {landmark_id}\n"
            f"   Confidence: {confidence:.2f}%\n\n"
        )

    result_label.config(text=result_text)

# GUI
root = tk.Tk()
root.title("Landmark Recognition")

root.geometry("700x700")

title = tk.Label(
    root,
    text="Landmark Recognition System",
    font=("Arial", 16, "bold")
)
title.pack(pady=10)

btn = tk.Button(
    root,
    text="Upload Image",
    font=("Arial", 12),
    command=predict_image
)
btn.pack(pady=10)

image_label = tk.Label(root)
image_label.pack()

result_label = tk.Label(
    root,
    text="Upload an image",
    justify="left",
    font=("Arial", 11)
)
result_label.pack(pady=10)

root.mainloop()