#When writing the test for this dataset this should be similar to the file "log_nb_upload_file_test.py" below are some differences
# !!!! DO NOT EDIT the code in "log_nb_upload_file_test.py" just use it as a reference for this file since it will be very similar. !!!
#
# -you will have to change the path to the demo_realworld_ratio dataset path. model folder should be the same being new models
# -The load pipeline and image processing stuff should be the same
# -The evaluation part will mostly be the same. The only metric you will evaluate is the precision. other things don't need to be evaluated. 

import os
import pickle
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from skimage.feature import hog
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    classification_report, roc_auc_score
)

# ===================== ##
# Paths                ##
# ===================== ##
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
realworld_test_path = realworld_test_path = os.path.join(BASE_DIR, "demo_realworld_ratio")
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
            if entry.is_file() and entry.name.lower().endswith((".png", ".jpg", ".jpeg")):
                tasks.append((entry.path, label))

    tasks.sort(key=lambda x: x[0])

    print(f"Found {len(tasks)} images in {base_path}")

    if len(tasks) == 0:
        raise ValueError(f"No images found in {base_path}")

    # Feature size detection
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
    print(f"\n===== {name} on Real-World Dataset =====")

    X_test, y_test = load_images_parallel(realworld_test_path, pipeline)

    # Scale
    X_test = pipeline["scaler"].transform(X_test)

    # PCA (if used)
    if pipeline["use_pca"]:
        X_test = pipeline["pca"].transform(X_test)

    # Predictions
    y_pred = pipeline["model"].predict(X_test)

    # Probabilities (for AUROC)
    if hasattr(pipeline["model"], "predict_proba"):
        y_prob = pipeline["model"].predict_proba(X_test)[:, 1]
        print("AUROC:", roc_auc_score(y_test, y_prob))

    # Metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))



# ===================== ##
# Run Tests (LR + NB)   ##
# ===================== ##
print("\nRunning Logistic Regression Test...")
evaluate_pipeline(logreg_pipeline, "Logistic Regression")

print("\nRunning Naive Bayes Test...")
evaluate_pipeline(nb_pipeline, "Naive Bayes")