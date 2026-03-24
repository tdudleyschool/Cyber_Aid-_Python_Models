# -*- coding: utf-8 -*-
"""
Alzheimer's Image Classification with Logistic Regression and Naive Bayes
Uses HOG features for better linear separability in LR and PCA+HOG for NB.

Dataset structure:
dataset/
    train/
        benign/
        malignant/
    test/
        benign/
        malignant/
"""

import os
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report

# ===================== ##
# Local Dataset Paths   ##
# ===================== ##
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
alzimerz_train_path = os.path.join(BASE_DIR, "dataset", "train")
alzimerz_test_path = os.path.join(BASE_DIR, "dataset", "test")

# ================== ##
# HOG Configuration  ##
# ================== ##
hog_config = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": 'L2-Hys',
    "feature_vector": True   # ensures consistent output
}

# ================== ##
# Image Processing   ##
# ================== ##
def process_image(args, use_hog=True):
    """
    Load, resize, normalize image and optionally extract HOG features
    """
    path, label, img_size = args

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0

    if use_hog:
        features = hog(img, **hog_config)   # <-- use shared config
        return features, label
    else:
        return img.flatten(), label

def load_images_parallel(base_path, img_size=(60, 60), use_hog=True):
    """
    Load images in parallel using threads, extract HOG or raw features
    """
    label_map = {"benign": 0, "malignant": 1}
    tasks = []

    for class_name, label in label_map.items():
        folder = os.path.join(base_path, class_name)
        if not os.path.isdir(folder):
            continue
        for entry in os.scandir(folder):
            if entry.is_file() and entry.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                tasks.append((entry.path, label, img_size))

    # 🔥 CRITICAL FIX: deterministic ordering
    tasks.sort(key=lambda x: x[0])

    print(f"Found {len(tasks)} images in {base_path}")

    # Pre-allocate array
    if use_hog:
        sample_img = cv2.imread(tasks[0][0], cv2.IMREAD_GRAYSCALE)
        sample_img = cv2.resize(sample_img, img_size)
        sample_features = hog(
            sample_img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )
        X = np.empty((len(tasks), len(sample_features)), dtype=np.float32)
    else:
        X = np.empty((len(tasks), img_size[0]*img_size[1]), dtype=np.float32)

    y = np.empty(len(tasks), dtype=np.int8)

    max_workers = min(32, cpu_count())
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, (features, label) in enumerate(executor.map(lambda args: process_image(args, use_hog), tasks)):
            X[i] = features
            y[i] = label

    return X, y

# ================== ##
# Load Dataset       ##
# ================== ##
X_train, y_train = load_images_parallel(alzimerz_train_path, img_size=(60, 60), use_hog=True)
X_test, y_test   = load_images_parallel(alzimerz_test_path, img_size=(60, 60), use_hog=True)

print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)
print("Unique labels in train:", np.unique(y_train))
print("Positive ratio:", np.mean(y_train))

# ============================== ##
# Feature Scaling                ##
# ============================== ##
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ============================== ##
# PCA for Naive Bayes            ##
# ============================== ##
pca = PCA(n_components=200, random_state=42)  # NB uses PCA with 200 components
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

print("After PCA (for NB):")
print("Train:", X_train_pca.shape)
print("Test:", X_test_pca.shape)

# ============================== ##
# Model Training & Evaluation    ##
# ============================== ##
# Logistic Regression on HOG features (no PCA)
logreg = LogisticRegression(max_iter=4000, class_weight='balanced')
logreg.fit(X_train_scaled, y_train)
y_pred_logreg = logreg.predict(X_test_scaled)

print("\nLogistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred_logreg))
print("Sensitivity (Recall):", recall_score(y_test, y_pred_logreg))
print("F1 Score:", f1_score(y_test, y_pred_logreg))
print(classification_report(y_test, y_pred_logreg))

# Naive Bayes on PCA(HOG) features
nb = GaussianNB()
nb.fit(X_train_pca, y_train)
y_pred_nb = nb.predict(X_test_pca)

print("\nNaive Bayes:")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Sensitivity (Recall):", recall_score(y_test, y_pred_nb))
print("F1 Score:", f1_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))


# ============================== ##
# Save Models                     ##
# ============================== ##


# ============================== ##
# Create Model Directory         ##
# ============================== ##
model_dir = os.path.join(BASE_DIR, "new_models")

# Create folder if it doesn't exist
os.makedirs(model_dir, exist_ok=True)

# ============================== ##
# Save Full Pipelines            ##
# ============================== ##
import pickle

logreg_pipeline = {
    "model": logreg,
    "scaler": scaler,
    "use_pca": False,
    "pca": None,
    "hog_config": hog_config,
    "img_size": (60, 60)
}

nb_pipeline = {
    "model": nb,
    "scaler": scaler,
    "use_pca": True,
    "pca": pca,
    "hog_config": hog_config,
    "img_size": (60, 60)
}

# File paths inside new_models folder
logreg_path = os.path.join(model_dir, "logreg_pipeline.pkl")
nb_path = os.path.join(model_dir, "naive_bayes_pipeline.pkl")

# Save pipelines
with open(logreg_path, "wb") as f:
    pickle.dump(logreg_pipeline, f)

with open(nb_path, "wb") as f:
    pickle.dump(nb_pipeline, f)

print(f"\n✅ Models saved to: {model_dir}")