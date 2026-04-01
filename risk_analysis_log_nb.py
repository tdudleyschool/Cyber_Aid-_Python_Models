# Risk Test Following This Formula
# Risk(t) = d*C_FN*(1-TPR(t)) + (1-d)*C_FP*FPR(t)
# where:
#   d = Prevalence of the condition (proportion of actual positives in the population)
#   C_FN = Cost of a false negative (missing a positive case)
#   C_FP = Cost of a false positive (incorrectly identifying a negative case as positive)
#   TPR(t) = True Positive Rate at threshold t (sensitivity)
#   1-TPR(t) is the false negative rate
#   FPR(t) = False Positive Rate at threshold t (1 - specificity)
#   t = Threshold for classification (between 0 and 1) for python models

# We will iterate over several thresholds, and over several cost scenarios

import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from skimage.feature import hog
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, confusion_matrix

# ===================== ##
# Setting Up Paths      ##
# ===================== ##

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # sets base path to the current file's directory
REALWORLD_TEST_DIR = os.path.join(BASE_DIR, "demo_realworld_ratio")  # sets path to the test dataset
MODEL_DIR = os.path.join(BASE_DIR, "new_models")  # sets path to the saved models

# ===================== ##
# Load Model Pipelines  ##
# ===================== ##

# NOTE: The pipeline contains the following:
#    : "model": the trained model (Logistic Regression or Naive Bayes)
#    : "scaler": the StandardScaler used for feature scaling
#    : "freature_extractor": the function used to extract features like PCA and HOG
#    : "image size"

# Think of a pipeline as a struct from C++. Contains collection of variables in this case related to this ML project

with open(os.path.join(MODEL_DIR, "logreg_pipeline.pkl"), "rb") as f:
    logreg_pipeline = pickle.load(f)

with open(os.path.join(MODEL_DIR, "naive_bayes_pipeline.pkl"), "rb") as f:
    nb_pipeline = pickle.load(f)

print("Models Loaded Successfully")

# ===================== ##
# Image Processing      ##
# ===================== ##

def process_image(path, label, pipeline):
    '''
    Input: path to image, label of image, pipeline containing model and preprocessing info
    Output: (Old was just resized and normalized images and label)
          : Now it still resizes, normalizes and applies HOG to it extracting those features. All setup in the pipeline from the training file to maintain consistency
          : Additionally HOG does flatten the images so we don't need to do that separately
    '''
    # Load image in grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # Resize image to the size specified in the pipeline
    img = cv2.resize(img, pipeline["img_size"])
    # Normalize pixel values to [0, 1]
    img = img.astype(np.float32) / 255.0
    # HOG feature extraction based on pipeline config
    features = hog(img, **pipeline["hog_config"])
    return features, label


def load_images_parallel(base_path, pipeline):
    '''
    Purpose: Load and process a lot of images
    Input: base path to dataset, pipeline containing model and preprocessing info
    Output: list of features and labels for all images in the dataset
    '''

    # Label mapping attached to folder names
    label_map = {"benign": 0, "malignant": 1}

    # Tasks for processing images in parallel
    tasks = []

    # Loop through label_map to find folders and images. Only loop two times for benign and malignant
    for class_name, label in label_map.items():
        # Set folder to path of dataset + class name
        folder = os.path.join(base_path, class_name)

        # Edge case if folder doesn't exist
        if not os.path.isdir(folder):
            continue
        
        # Loop through images in folder and add to tasks list with path, label and pipeline
        for entry in os.scandir(folder):
            # Only if image
            if entry.is_file() and entry.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                # We need to pass the pipeline to the process_image function so it can use the correct image size and hog config
                tasks.append((entry.path, label, pipeline))

    # Reporting how many images there were before processing
    print(f"Found {len(tasks)} images in {base_path}")

    # Determine feature size dynamically by processing one image. This is important for pre-allocating the feature array
    sample_img = cv2.imread(tasks[0][0], cv2.IMREAD_GRAYSCALE)
    sample_img = cv2.resize(sample_img, pipeline["img_size"])
    # This will give number of features based on HOG config
    sample_features = hog(sample_img, **pipeline["hog_config"])

    # Preallocate np arrays for features and labels based on number of tasks and feature size
    X = np.empty((len(tasks), len(sample_features)), dtype=np.float32)
    y = np.empty(len(tasks), dtype=np.int8)

    # Process images using threads. Utilize min threads on CPU
    max_workers = min(32, cpu_count())
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # executor.map applies process_image function to each task in parallel.
        for i, (features, label) in enumerate(executor.map(lambda args: process_image(*args), tasks)):
            X[i] = features
            y[i] = label
    
    return X, y


# ===================== ##
# Risk Evaluation       ##
# ===================== ##

def risk_evaluation(pipeline, X_test, y_test, t, C_FN, C_FP, d=0.1):
    '''
    Formula: Risk(t) = d*C_FN*(1-TPR(t)) + (1-d)*C_FP*FPR(t)
    d = prevalence of condition
    C_FN = cost of false negative
    C_FP = cost of false positive
    t = threshold for classification

    Input:
         pipeline: (basically model and its parts)
         X and y features from dataset preprocessed
         threshold for classification
         prevalence of condition (d)
         C_FN and C_FP costs
    Output:
         risk score based on the formula
    '''
    # Apply scaler
    X_test_scaled = pipeline["scaler"].transform(X_test)

    # Apply PCA if needed (primarily for Naive Bayes)
    if pipeline["use_pca"]:
        X_test_scaled = pipeline["pca"].transform(X_test_scaled)
    
    # Getting probabilities of prediction allowing us to do the threshold
    y_prob = pipeline["model"].predict_proba(X_test_scaled)[:, 1]
    # Apply threshold to get binary predictions
    y_pred = (y_prob >= t).astype(int)

    # Calculate (1-TPR) AKA FNR and FPR for the formula
    TP = np.sum((y_test == 1) & (y_pred == 1))
    FN = np.sum((y_test == 1) & (y_pred == 0))
    FP = np.sum((y_test == 0) & (y_pred == 1))
    TN = np.sum((y_test == 0) & (y_pred == 0))

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0

    risk = d * C_FN * (1 - TPR) + (1 - d) * C_FP * FPR

    return risk


# ====================== ##
# Running Risk Test Loop ##
# ====================== ##

X_test_nb, y_test_nb  = load_images_parallel(REALWORLD_TEST_DIR, nb_pipeline)
X_test_lr, y_test_lr  = load_images_parallel(REALWORLD_TEST_DIR, logreg_pipeline)

# Define cost scenarios
Test_C_FP = 1
Test_C_FN = [2, 5, 10, 20]  # FN are much more costly for medical diagnosis so we will vary that more

# Defining threshold range from 0.0 to 1.0 with steps of 0.1
thresholds = np.arange(0.0, 1.01, 0.1)

# Dictionary to store risk results per model per cost scenario for matplotlib
Logreg_risk_results = {C_FN: [] for C_FN in Test_C_FN}
NB_risk_results = {C_FN: [] for C_FN in Test_C_FN}

print("\nStarting Risk Evaluation Loop...")
for C_FN in Test_C_FN:
    for t in thresholds:
        logreg_risk = risk_evaluation(logreg_pipeline, X_test_lr, y_test_lr, t, C_FN=C_FN, C_FP=Test_C_FP)
        nb_risk = risk_evaluation(nb_pipeline, X_test_nb, y_test_nb, t, C_FN=C_FN, C_FP=Test_C_FP)

        Logreg_risk_results[C_FN].append(logreg_risk)
        NB_risk_results[C_FN].append(nb_risk)

print("Risk Evaluation Loop Completed.")
print("Processing results and generating plots...")
#print risk results. print the minimum risk and threshold that achieves it for each model and cost scenario for each model
for C_FN in Test_C_FN:
    logreg_min_risk = min(Logreg_risk_results[C_FN])
    logreg_best_threshold = thresholds[np.argmin(Logreg_risk_results[C_FN])]
    nb_min_risk = min(NB_risk_results[C_FN])
    nb_best_threshold = thresholds[np.argmin(NB_risk_results[C_FN])]

    print(f"C_FN={C_FN}: Logistic Regression Min Risk={logreg_min_risk:.4f} at Threshold={logreg_best_threshold:.2f}")
    print(f"C_FN={C_FN}: Naive Bayes Min Risk={nb_min_risk:.4f} at Threshold={nb_best_threshold:.2f}")


# Plot risk vs threshold for each model
# Logistic Regression Plot
plt.figure(figsize=(12, 6))
for C_FN in Test_C_FN:
    plt.plot(thresholds, Logreg_risk_results[C_FN], label=f'C_FN={C_FN}')
plt.xlabel('Threshold')
plt.ylabel('Risk')
plt.title('Risk vs Threshold for Logistic Regression')
plt.legend()
plt.grid(True)
plt.show()

# Naive Bayes Plot
plt.figure(figsize=(12, 6))
for C_FN in Test_C_FN:
    plt.plot(thresholds, NB_risk_results[C_FN], label=f'C_FN={C_FN}')
plt.xlabel('Threshold')
plt.ylabel('Risk')
plt.title('Risk vs Threshold for Naive Bayes')
plt.legend()
plt.grid(True)
plt.show()