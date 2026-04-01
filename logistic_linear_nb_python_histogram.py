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
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

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
# Logistic Regression (180x180)
X_train_lr, y_train = load_images_parallel(alzimerz_train_path, img_size=(180, 180), use_hog=True)
X_test_lr, y_test   = load_images_parallel(alzimerz_test_path, img_size=(180, 180), use_hog=True)

# Naive Bayes (60x60)  ← KEEP ORIGINAL. test dataet is the same
X_train_nb, _ = load_images_parallel(alzimerz_train_path, img_size=(60, 60), use_hog=True)
X_test_nb, _  = load_images_parallel(alzimerz_test_path, img_size=(60, 60), use_hog=True)

print("Train_lr:", X_train_lr.shape, y_train.shape)
print("Test_lr:", X_test_lr.shape, y_test.shape)
print("Train_nb:", X_train_nb.shape, y_train.shape)
print("Test_nb:", X_test_nb.shape, X_test_lr.shape)
print("Unique labels in train:", np.unique(y_train))
print("Positive ratio:", np.mean(y_train))

# ============================== ##
# Feature Scaling                ##
# ============================== ##
# Logistic Regression scaler (180x180)
scaler_lr = StandardScaler()
X_train_lr_scaled = scaler_lr.fit_transform(X_train_lr)
X_test_lr_scaled  = scaler_lr.transform(X_test_lr)

# Naive Bayes scaler (60x60)
scaler_nb = StandardScaler()
X_train_nb_scaled = scaler_nb.fit_transform(X_train_nb)
X_test_nb_scaled  = scaler_nb.transform(X_test_nb)

# ============================== ##
# PCA for Naive Bayes            ##
# ============================== ##
pca = PCA(n_components=200, random_state=42)  # NB uses PCA with 200 components
X_train_pca = pca.fit_transform(X_train_nb_scaled)
X_test_pca  = pca.transform(X_test_nb_scaled)

print("After PCA (for NB):")
print("Train:", X_train_pca.shape)
print("Test:", X_test_pca.shape)

# plts

# ============================== ##
# Model Training & Evaluation    ##
# ============================== ##
# Logistic Regression on HOG features (no PCA)
logreg = LogisticRegression(max_iter=4000, class_weight='balanced')
logreg.fit(X_train_lr_scaled, y_train)
y_pred_logreg = logreg.predict(X_test_lr_scaled)

y_prob_logreg = logreg.predict_proba(X_test_lr_scaled)[:, 1]
auc_logreg = roc_auc_score(y_test, y_prob_logreg)
fpr_logreg, tpr_logreg, _ = roc_curve(y_test, y_prob_logreg)

print("\nLogistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred_logreg))
print("Sensitivity (Recall):", recall_score(y_test, y_pred_logreg))
print("F1 Score:", f1_score(y_test, y_pred_logreg))
print("AUC-ROC:", auc_logreg)
print(classification_report(y_test, y_pred_logreg))

# Naive Bayes on PCA(HOG) features
nb = GaussianNB()
nb.fit(X_train_pca, y_train)
y_pred_nb = nb.predict(X_test_pca)

y_prob_nb = nb.predict_proba(X_test_pca)[:, 1]
auc_nb = roc_auc_score(y_test, y_prob_nb)
fpr_nb, tpr_nb, _ = roc_curve(y_test, y_prob_nb)

print("\nNaive Bayes:")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Sensitivity (Recall):", recall_score(y_test, y_pred_nb))
print("F1 Score:", f1_score(y_test, y_pred_nb))
print("AUC-ROC:", auc_nb)
print(classification_report(y_test, y_pred_nb))

plt.figure(figsize=(8, 6))
plt.plot(fpr_logreg, tpr_logreg, label=f"Logistic Regression (AUC = {auc_logreg:.4f})")
plt.plot(fpr_nb, tpr_nb, label=f"Naive Bayes (AUC = {auc_nb:.4f})")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

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
    "scaler": scaler_lr,
    "use_pca": False,
    "pca": None,
    "hog_config": hog_config,
    "img_size": (180, 180)   # ✅ FIXED
}

nb_pipeline = {
    "model": nb,
    "scaler": scaler_nb,
    "use_pca": True,
    "pca": pca,
    "hog_config": hog_config,
    "img_size": (60, 60)     # unchanged
}

# File paths inside new_models folder
logreg_path = os.path.join(model_dir, "logreg_pipeline.pkl")
nb_path = os.path.join(model_dir, "naive_bayes_pipeline.pkl")

# Save pipelines
with open(logreg_path, "wb") as f:
    pickle.dump(logreg_pipeline, f)

with open(nb_path, "wb") as f:
    pickle.dump(nb_pipeline, f)

print(f"\n Models saved to: {model_dir}")