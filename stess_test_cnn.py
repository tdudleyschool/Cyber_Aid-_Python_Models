import tensorflow as tf
import os
import cv2
import numpy as np
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# -------------------------------
# Upload / Load CNN
# -------------------------------
# If using .h5 single file:
cnn = tf.keras.models.load_model("models/cnn_model.h5", compile=False)

# -------------------------------
# Config
# -------------------------------
img_width, img_height = 60, 60
base_path = "demo"
label_map = {"benign": 0, "malignant": 1}

# -------------------------------
# Image processing function
# -------------------------------
def process_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype(np.float32) / 255.0
    return img

# -------------------------------
# Gather all images and labels
# -------------------------------
all_images, all_labels = [], []
for class_name, label in label_map.items():
    folder = os.path.join(base_path, class_name)
    for entry in os.scandir(folder):
        if entry.is_file() and entry.name.lower().endswith(('.png', '.jpg', '.jpeg')):
            all_images.append(entry.path)
            all_labels.append(label)

all_images = np.array(all_images)
all_labels = np.array(all_labels)
print(f"Total images found: {len(all_images)}")

# -------------------------------
# Stress Test Loop
# -------------------------------
batch_sizes = range(100, 1000, 100)
results = []

for n in batch_sizes:
    idx = np.random.choice(len(all_images), n, replace=False)
    X_batch_paths = all_images[idx]
    y_batch = all_labels[idx]

    # Preprocess images
    X_batch = np.array([process_image(p) for p in X_batch_paths])

    # Ensure 3 channels (RGB) if CNN expects it
    if X_batch.ndim == 3:  # grayscale
        X_batch = np.expand_dims(X_batch, axis=-1)
        X_batch = np.repeat(X_batch, 3, axis=-1)

    # Predict
    start = time.time()
    y_pred_prob = cnn.predict(X_batch, verbose=0)
    cnn_time = time.time() - start

    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    # Metrics
    metrics = {
        "batch_size": n,
        "cnn_accuracy": accuracy_score(y_batch, y_pred),
        "cnn_precision": precision_score(y_batch, y_pred),
        "cnn_recall": recall_score(y_batch, y_pred),
        "cnn_f1": f1_score(y_batch, y_pred),
        "cnn_avg_time": cnn_time / n
    }

    results.append(metrics)
    print(f"Batch {n} done: CNN avg time per image {metrics['cnn_avg_time']:.5f}s")

# Save results
df_results = pd.DataFrame(results)
print("\nCNN Stress Test Summary:")
print(df_results)
df_results.to_csv("cnn_stress_test_results.csv", index=False)

import matplotlib.pyplot as plt

# Assuming df_results is the DataFrame from your stress test
metrics_to_plot = [
    "cnn_accuracy", "cnn_precision", "cnn_recall", "cnn_f1", "cnn_avg_time"
]

plt.figure(figsize=(18, 12))

for i, metric in enumerate(metrics_to_plot, 1):
    plt.subplot(3, 4, i)
    plt.plot(df_results["batch_size"], df_results[metric], marker='o', linestyle='-')
    plt.title(metric.replace("_", " ").title())
    plt.xlabel("Batch Size")
    plt.ylabel(metric.split("_")[-1].title())
    plt.grid(True)

plt.tight_layout()
plt.show()