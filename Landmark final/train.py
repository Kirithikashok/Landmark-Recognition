import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt

# =========================
# PATHS
# =========================
TRAIN_DIR = r"D:\HCL Guvi\Landmark\dataset_split\train"
TEST_DIR  = r"D:\HCL Guvi\Landmark\dataset_split\test"

IMG_SIZE = (300, 300)
BATCH_SIZE = 16
EPOCHS = 15

# =========================
# LOAD DATA
# =========================
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Classes:", num_classes)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# =========================
# DATA AUGMENTATION
# =========================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
])

# =========================
# BASE MODEL (EfficientNetB3)
# =========================
base_model = EfficientNetB3(
    weights="imagenet",
    include_top=False,
    input_shape=(300, 300, 3)
)

base_model.trainable = False

# =========================
# MODEL ARCHITECTURE
# =========================
inputs = layers.Input(shape=(300, 300, 3))

x = data_augmentation(inputs)

# EfficientNet preprocessing
x = preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.4)(x)

outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

# =========================
# COMPILE (PHASE 1)
# =========================
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)
    ]
)

# =========================
# CALLBACKS
# =========================
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=4,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_efficientnetb3.keras",
    monitor="val_accuracy",
    save_best_only=True
)

# =========================
# TRAINING PHASE 1
# =========================
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

# =========================
# FINE TUNING
# =========================
print("\nStarting fine-tuning...")

base_model.trainable = True

# Freeze earlier layers
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)
    ]
)

history_fine = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=10,
    callbacks=[early_stop, checkpoint]
)

# =========================
# SAVE FINAL MODEL
# =========================
model.save("efficientnetb3_landmark1.keras")

# =========================
# PLOT RESULTS
# =========================
plt.figure(figsize=(10,5))

plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("EfficientNetB3 Training")
plt.legend()

plt.show()