# train_bls_models.py

import os
import cv2
import numpy as np
from recognition import train_bls
from joblib import dump

# ------------------------
# Step 1: Load dataset
# ------------------------
def load_dataset(folder_path):
    X, y = [], []
    for label in sorted(os.listdir(folder_path)):
        label_path = os.path.join(folder_path, label)
        if not os.path.isdir(label_path):
            continue
        for file in os.listdir(label_path):
            img_path = os.path.join(label_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (32, 32))
            X.append(img.flatten() / 255.0)
            y.append(label)
    return np.array(X), np.array(y)

# ------------------------
# Step 2: Load letters and digits
# ------------------------
X_letters, y_letters = load_dataset("dataset_letters")
X_digits, y_digits = load_dataset("dataset_digits")

# ------------------------
# Step 3: Train BLS models
# ------------------------
bls_letters = train_bls(X_letters, y_letters)
bls_digits = train_bls(X_digits, y_digits)

# Save the models
os.makedirs("models", exist_ok=True)
dump(bls_letters, "models/sae_bls_letter.pkl")
dump(bls_digits, "models/sae_bls_digit.pkl")

print("✅ BLS models trained and saved successfully!")
