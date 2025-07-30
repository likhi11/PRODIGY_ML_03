
# Task-03: Dogs vs Cats Image Classification using SVM

## Dataset
Download from: https://www.kaggle.com/c/dogs-vs-cats/data
Extract the dataset such that you have folders:
```
data/
  cats/
    cat.0.jpg
    ...
  dogs/
    dog.0.jpg
    ...
```

## Setup
```bash
pip install numpy matplotlib scikit-learn tensorflow joblib
```

## Run
```bash
python svm_dog_vs_cat.py
```
This script:
- Loads images from the dataset (1000 cats and 1000 dogs)
- Preprocesses and flattens them
- Uses PCA for dimensionality reduction
- Trains an SVM classifier
- Evaluates and saves the model
- Visualizes predictions
