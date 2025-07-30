
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import joblib

# Load and preprocess images
def load_images(folder, label, img_size=(64, 64), limit=1000):
    images, labels = [], []
    count = 0
    for fname in os.listdir(folder):
        if fname.endswith('.jpg') and count < limit:
            img = load_img(os.path.join(folder, fname), target_size=img_size)
            img_array = img_to_array(img) / 255.0
            images.append(img_array.flatten())
            labels.append(label)
            count += 1
    return images, labels

# Paths (you should download and extract the dataset from Kaggle first)
cat_dir = 'data/cats'
dog_dir = 'data/dogs'

cat_images, cat_labels = load_images(cat_dir, 0)
dog_images, dog_labels = load_images(dog_dir, 1)

X = np.array(cat_images + dog_images)
y = np.array(cat_labels + dog_labels)

# Dimensionality reduction
pca = PCA(n_components=100)
X_pca = pca.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# SVM classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(clf, 'svm_model.pkl')
joblib.dump(pca, 'pca_transform.pkl')

# Visualization
def visualize_samples(images, labels, preds=None):
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i].reshape(64, 64, 3))
        title = 'Cat' if labels[i] == 0 else 'Dog'
        if preds is not None:
            title += f" / {'Cat' if preds[i] == 0 else 'Dog'}"
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Visualize some test images with predictions
test_images = X_test @ pca.components_ + pca.mean_
visualize_samples(test_images[:10].reshape(-1, 64, 64, 3), y_test[:10], y_pred[:10])
