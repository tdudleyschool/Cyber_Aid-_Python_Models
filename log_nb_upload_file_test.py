# -*- coding: utf-8 -*-
"""
This tests was designed to demonstrate how to upload the already saved logistic regression and naive bayes models and run them on the test dataset.
"""

import os
import pickle
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from skimage.feature import hog
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report

# ===================== ##
# Paths                ##
# ===================== ##
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
alzimerz_test_path = os.path.join(BASE_DIR, "dataset", "test")
model_dir = os.path.join(BASE_DIR, "new_models")

# ===================== ##
# Load Pipelines        ##
# ===================== ##
with open(os.path.join(model_dir, "logreg_pipeline.pkl"), "rb") as f:
    logreg_pipeline = pickle.load(f)

with open(os.path.join(model_dir, "naive_bayes_pipeline.pkl"), "rb") as f:
    nb_pipeline = pickle.load(f)

print("Pipelines loaded successfully")

# ===================== ##
# Image Processing      ##
# ===================== ##
def process_image(args, pipeline):
    path, label = args

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, pipeline["img_size"])
    img = img.astype(np.float32) / 255.0

    # HOG feature extraction using the saved config
    features = hog(img, **pipeline["hog_config"])
    return features, label


def load_images_parallel(base_path, pipeline):
    label_map = {"benign": 0, "malignant": 1}
    tasks = []

    for class_name, label in label_map.items():
        folder = os.path.join(base_path, class_name)
        if not os.path.isdir(folder):
            continue
        for entry in os.scandir(folder):
            if entry.is_file() and entry.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                tasks.append((entry.path, label))

    #deterministic ordering though wondering if this is needed
    tasks.sort(key=lambda x: x[0])

    print(f"Found {len(tasks)} images in {base_path}")

    # Determine feature size dynamically
    sample_img = cv2.imread(tasks[0][0], cv2.IMREAD_GRAYSCALE)
    sample_img = cv2.resize(sample_img, pipeline["img_size"])
    sample_features = hog(sample_img, **pipeline["hog_config"])

    X = np.empty((len(tasks), len(sample_features)), dtype=np.float32)
    y = np.empty(len(tasks), dtype=np.int8)

    max_workers = min(32, cpu_count())
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, (features, label) in enumerate(
            executor.map(lambda args: process_image(args, pipeline), tasks)
        ):
            X[i] = features
            y[i] = label

    return X, y


# ===================== ##
# Evaluation Function   ##
# ===================== ##
def evaluate_pipeline(pipeline, name):
    print(f"\n===== {name} =====")

    X_test, y_test = load_images_parallel(alzimerz_test_path, pipeline)

    # Apply scaler
    X_test = pipeline["scaler"].transform(X_test)

    # Apply PCA if needed
    if pipeline["use_pca"]:
        X_test = pipeline["pca"].transform(X_test)

    # Predict
    y_pred = pipeline["model"].predict(X_test)

    # Metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


# ===================== ##
# Run Tests             ##
# ===================== ##
evaluate_pipeline(logreg_pipeline, "Logistic Regression")
evaluate_pipeline(nb_pipeline, "Naive Bayes")