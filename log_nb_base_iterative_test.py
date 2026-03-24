# -*- coding: utf-8 -*-
"""
Hyperparameter sweep for Alzheimer's image classification
- Sweep image sizes from 10x10 to 180x180
- Sweep PCA n_components from 10 to 1000
- Record Accuracy, Precision, F1 for Logistic Regression and Naive Bayes
"""

import os
import numpy as np
import cv2
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, f1_score

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(BASE_DIR, "dataset", "train")
test_path  = os.path.join(BASE_DIR, "dataset", "test")

# ---------------- Image Loading ----------------
def process_image(args):
    path, label, img_size = args
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img, label

def load_images_parallel(base_path, img_size=(60,60)):
    label_map = {"benign":0, "malignant":1}
    tasks=[]
    for cname,label in label_map.items():
        folder=os.path.join(base_path,cname)
        if not os.path.isdir(folder): continue
        for entry in os.scandir(folder):
            if entry.is_file() and entry.name.lower().endswith(('.png','.jpg','.jpeg')):
                tasks.append((entry.path,label,img_size))

    X = np.empty((len(tasks), img_size[0], img_size[1]), dtype=np.float32)
    y = np.empty(len(tasks), dtype=np.int8)
    max_workers = min(32, cpu_count())
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i,(img,label) in enumerate(executor.map(process_image,tasks)):
            X[i]=img
            y[i]=label
    return X, y

# ---------------- Evaluation Function ----------------
def evaluate_models(X_train_flat, X_test_flat, n_pca=100):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled  = scaler.transform(X_test_flat)

    pca = PCA(n_components=n_pca, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    # Logistic Regression
    logreg = LogisticRegression(max_iter=2000,class_weight='balanced')
    logreg.fit(X_train_pca, y_train)
    y_pred_logreg = logreg.predict(X_test_pca)

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train_pca, y_train)
    y_pred_nb = nb.predict(X_test_pca)

    metrics = {
        "logreg_acc": accuracy_score(y_test, y_pred_logreg),
        "logreg_prec": precision_score(y_test, y_pred_logreg),
        "logreg_f1": f1_score(y_test, y_pred_logreg),
        "nb_acc": accuracy_score(y_test, y_pred_nb),
        "nb_prec": precision_score(y_test, y_pred_nb),
        "nb_f1": f1_score(y_test, y_pred_nb)
    }
    return metrics

# ---------------- Sweep 1: Image Size ----------------
img_sizes = list(range(10, 181, 10))
results_imgsize = []

for size in img_sizes:
    print(f"Processing image size: {size}x{size}")
    X_train, y_train = load_images_parallel(train_path, img_size=(size,size))
    X_test, y_test   = load_images_parallel(test_path, img_size=(size,size))

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat  = X_test.reshape(X_test.shape[0], -1)

    # n_components = min(100, number of features)
    n_pca = min(100, X_train_flat.shape[1])
    metrics = evaluate_models(X_train_flat, X_test_flat, n_pca=n_pca)
    metrics["img_size"] = size
    results_imgsize.append(metrics)

df_imgsize = pd.DataFrame(results_imgsize)
df_imgsize.to_csv("results_image_size.csv", index=False)
print("Saved results_image_size.csv")

# ---------------- Sweep 2: PCA Components ----------------
# Use fixed image size 60x60 for PCA sweep
X_train, y_train = load_images_parallel(train_path, img_size=(60,60))
X_test, y_test   = load_images_parallel(test_path, img_size=(60,60))
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat  = X_test.reshape(X_test.shape[0], -1)

pca_values = list(range(10, min(1000, X_train_flat.shape[1])+1, 10))
results_pca = []

for n in pca_values:
    print(f"Processing PCA n_components: {n}")
    metrics = evaluate_models(X_train_flat, X_test_flat, n_pca=n)
    metrics["n_pca"] = n
    results_pca.append(metrics)

df_pca = pd.DataFrame(results_pca)
df_pca.to_csv("results_pca.csv", index=False)
print("Saved results_pca.csv")