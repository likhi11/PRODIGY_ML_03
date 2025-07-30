import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Path to dataset (You should unzip 'train' folder from Kaggle dataset and provide its path)
DATA_DIR = "dataset/train"
IMG_SIZE = 64

def load_data():
    X, y = [], []
    for img_name in os.listdir(DATA_DIR):
        label = 0 if 'cat' in img_name else 1
        img_path = os.path.join(DATA_DIR, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).flatten()
            X.append(img)
            y.append(label)
    return np.array(X), np.array(y)

print("Loading data...")
X, y = load_data()
print(f"Total images: {len(X)}")

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training SVM
print("Training SVM model...")
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Testing
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")
