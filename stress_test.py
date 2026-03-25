# -*- coding: utf-8 -*-
"""
Stress Test using updated Logistic Regression and Naive Bayes Pipelines
Includes memory usage tracking with tracemalloc for peak memory
"""

import os
import pickle
import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from skimage.feature import hog
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import tracemalloc  # NEW: for memory tracking

# ========================= #
# !!!!!Changes That Need To Be Made!!!!!:
# 1. Combine Models Onto One Plot. So it should show NB, and LogRegression on 1 matplotlib plot
# 2. Change the rate of increase of request (batch) currenlty at adding 100 per round making plot noisy. increase to 500 or 1000

# ===================== ##
# Paths & Config        ##
# ===================== ##
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
demo_path = os.path.join(BASE_DIR, "demo")
model_dir = os.path.join(BASE_DIR, "new_models")
output_dir = os.path.join(BASE_DIR, "stress_test")
os.makedirs(output_dir, exist_ok=True)

label_map = {"benign": 0, "malignant": 1}

# ================================ ##
# Load Pipelines For Models        ##
# ================================ ##
with open(os.path.join(model_dir, "logreg_pipeline.pkl"), "rb") as f:
    logreg_pipeline = pickle.load(f)

with open(os.path.join(model_dir, "naive_bayes_pipeline.pkl"), "rb") as f:
    nb_pipeline = pickle.load(f)

print(" Pipelines loaded successfully")

# ===================================== ##
# Image Processing and Applying HOG     ##
# ===================================== ##
def process_image(args, pipeline):
    path, label = args
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, pipeline["img_size"])
    img = img.astype(np.float32) / 255.0
    features = hog(img, **pipeline["hog_config"])
    return features, label

# ===================== ##
# Gather all demo images ##
# ===================== ##
all_images = []
all_labels = []

for class_name, label in label_map.items():
    folder = os.path.join(demo_path, class_name)
    for entry in os.scandir(folder):
        if entry.is_file() and entry.name.lower().endswith(('.png', '.jpg', '.jpeg')):
            all_images.append(entry.path)
            all_labels.append(label)

all_images = np.array(all_images)
all_labels = np.array(all_labels)
print(f"Total demo images found: {len(all_images)}")

# ===================== ##
# Stress Test Loop       ##
# ===================== ##
batch_sizes = range(100, 9701, 100)
results = []
memory_usage = []

for n in batch_sizes:
    idx = np.random.choice(len(all_images), n, replace=False)
    batch_paths = all_images[idx]
    batch_labels = all_labels[idx]

    # Start memory tracking
    tracemalloc.start()

    # Prepare batch features for both pipelines
    X_logreg = np.array([process_image((p, 0), logreg_pipeline)[0] for p in batch_paths])
    X_nb = np.array([process_image((p, 0), nb_pipeline)[0] for p in batch_paths])

    # Scale
    X_logreg = logreg_pipeline["scaler"].transform(X_logreg)
    X_nb = nb_pipeline["scaler"].transform(X_nb)

    # PCA if needed
    if logreg_pipeline["use_pca"]:
        X_logreg = logreg_pipeline["pca"].transform(X_logreg)
    if nb_pipeline["use_pca"]:
        X_nb = nb_pipeline["pca"].transform(X_nb)

    # Logistic Regression
    start = time.time()
    y_pred_logreg = logreg_pipeline["model"].predict(X_logreg)
    logreg_time = (time.time() - start) / n

    # Naive Bayes
    start = time.time()
    y_pred_nb = nb_pipeline["model"].predict(X_nb)
    nb_time = (time.time() - start) / n

    # Capture peak memory for this batch
    current, peak = tracemalloc.get_traced_memory()
    peak_mb = peak / (1024**2)  # convert to MB
    memory_usage.append(peak_mb)
    tracemalloc.stop()

    metrics = {
        "batch_size": n,
        "logreg_accuracy": accuracy_score(batch_labels, y_pred_logreg),
        "logreg_precision": precision_score(batch_labels, y_pred_logreg),
        "logreg_recall": recall_score(batch_labels, y_pred_logreg),
        "logreg_f1": f1_score(batch_labels, y_pred_logreg),
        "logreg_time": logreg_time,
        "nb_accuracy": accuracy_score(batch_labels, y_pred_nb),
        "nb_precision": precision_score(batch_labels, y_pred_nb),
        "nb_recall": recall_score(batch_labels, y_pred_nb),
        "nb_f1": f1_score(batch_labels, y_pred_nb),
        "nb_time": nb_time
    }

    results.append(metrics)
    print(f"Batch {n} done: LR avg {logreg_time:.5f}s, NB avg {nb_time:.5f}s, Peak Memory {peak_mb:.2f} MB")

# ===================== ##
# Save Results & Plots   ##
# ===================== ##
df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(output_dir, "stress_test_results.csv"), index=False)
print(" Stress test results saved.")

# Plot metrics
metrics_to_plot = [
    "logreg_accuracy", "logreg_precision", "logreg_recall", "logreg_f1", "logreg_time",
    "nb_accuracy", "nb_precision", "nb_recall", "nb_f1", "nb_time"
]

plt.figure(figsize=(18, 12))
for i, metric in enumerate(metrics_to_plot, 1):
    plt.subplot(3, 4, i)
    line_color = 'blue' if 'time' in metric else 'red'
    plt.plot(df_results["batch_size"], df_results[metric], linestyle='-', color=line_color)
    plt.title(metric.replace("_", " ").title())
    plt.xlabel("Batch Size")
    plt.ylabel(metric.split("_")[-1].title())
    plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "stress_test_plot.png"))
print(f"✅ Stress test plot saved at {output_dir}")

# Plot memory usage
plt.figure(figsize=(10, 6))
plt.plot(list(batch_sizes), memory_usage, linestyle='-', color='green')
plt.title("Peak Memory Usage per Batch Size")
plt.xlabel("Batch Size")
plt.ylabel("Memory Used (MB)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "stress_test_memory.png"))
print(f"✅ Peak memory usage plot saved at {output_dir}")