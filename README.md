# Iris Data Classification Using K-Nearest Neighbors (KNN)

## Project Overview

This project implements a basic supervised machine learning classification model using the Iris dataset. The objective is to classify iris flowers into their respective species based on flower measurements.

The project follows the Input-Process-Output (IPO) framework and demonstrates the complete machine learning workflow from data preparation to model evaluation.

---

## Objective

Build a classification model that:

- Loads and understands a dataset
- Preprocesses and scales data
- Splits data into training and testing sets
- Trains a K-Nearest Neighbors (KNN) model
- Makes predictions on unseen data
- Evaluates performance using a Confusion Matrix and F1 Score

---

## Dataset

The project uses the built-in Iris dataset from Scikit-Learn.

### Dataset Information

- Total Samples: 150
- Classes:
  - Setosa
  - Versicolor
  - Virginica
- Features:
  - Sepal Length (cm)
  - Sepal Width (cm)
  - Petal Length (cm)
  - Petal Width (cm)

---

## Technologies Used

- Python 3
- Pandas
- NumPy
- Scikit-Learn

---

## Machine Learning Pipeline

### 1. Data Loading

The Iris dataset is loaded using:

```python
from sklearn.datasets import load_iris
```

### 2. Feature Scaling

The StandardScaler is applied to normalize feature values.

Benefits:

- Mean = 0
- Variance = 1
- Prevents large-value features from dominating distance calculations

```python
from sklearn.preprocessing import StandardScaler
```

### 3. Train-Test Split

The dataset is shuffled and split into:

- Training Set: 80%
- Testing Set: 20%

```python
train_test_split(
    test_size=0.2,
    shuffle=True,
    random_state=42
)
```

### 4. Model Training

A K-Nearest Neighbors classifier is used with:

```python
KNeighborsClassifier(n_neighbors=5)
```

Reason:

- K = 1 may overfit
- Very large K may underfit
- K = 5 provides a good balance

### 5. Prediction

The trained model predicts the species of unseen flowers from the testing dataset.

### 6. Evaluation

The model is evaluated using:

- Confusion Matrix
- F1 Score

---

## Results

### Confusion Matrix

```text
[[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]
```

### F1 Score

```text
1.0
```

### Interpretation

- All test samples were classified correctly.
- No False Positives or False Negatives occurred.
- The model achieved perfect classification performance on the test dataset.

---

## How to Run

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Git Bash:

```bash
source venv/Scripts/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Project

```bash
python main.py
```

---

## Concepts Demonstrated

- Supervised Learning
- Data Preprocessing
- Feature Scaling
- Train-Test Split
- K-Nearest Neighbors (KNN)
- Classification
- Model Evaluation
- Confusion Matrix
- F1 Score

---

## Author

Freeman Wisdom
