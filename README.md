# 🍷 Wine Quality Classification using Support Vector Machine (SVM)

## 📌 Project Overview

This project focuses on predicting wine quality using the **Support Vector Machine (SVM)** classification algorithm.

The model analyzes physicochemical properties of white wine samples to classify their quality scores.

The dataset contains several chemical features, such as alcohol content, acidity, density, and pH, which are used to predict wine quality.

## 🎯 Project Objectives

- Explore and understand the Wine Quality dataset.
- Perform data inspection and feature selection.
- Prepare training and testing datasets.
- Apply feature scaling using StandardScaler.
- Build an SVM classification model.
- Evaluate model performance using classification metrics.

## 📊 Dataset

**Dataset:** Wine Quality - White Wine

The dataset contains physicochemical properties of white wine samples.

### Features

The model uses 11 input features:

| Feature | Description |
|---|---|
| fixed acidity | Fixed acidity |
| volatile acidity | Volatile acidity |
| citric acid | Citric acid content |
| residual sugar | Residual sugar |
| chlorides | Chloride content |
| free sulfur dioxide | Free sulfur dioxide |
| total sulfur dioxide | Total sulfur dioxide |
| density | Wine density |
| pH | Acidity level |
| sulphates | Sulphates content |
| alcohol | Alcohol content |

**Target Variable:** `quality`

The target represents the wine quality score.

## 🛠️ Technologies & Libraries

- Python
- NumPy
- Pandas
- Matplotlib
- SciPy
- Scikit-learn
- Jupyter Notebook

## ⚙️ Project Workflow

### 1. Data Loading
Load the Wine Quality dataset using Pandas.

### 2. Exploratory Data Analysis (EDA)
Inspect dataset structure, data types, and quality distribution.

### 3. Data Visualization
Create scatter plots to explore the relationship between alcohol content, volatile acidity, and wine quality.

### 4. Feature Selection
Select 11 physicochemical properties as input features (`X`) and use wine quality as the target (`y`).

### 5. Train-Test Split
Split the dataset into:
- 80% Training Data
- 20% Testing Data

### 6. Feature Scaling
Apply StandardScaler to standardize the input features before training the SVM model.

### 7. Model Training

Use Support Vector Machine with an RBF kernel.

```python
from sklearn import svm

clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
```

### 8. Model Prediction

Generate predictions using the testing dataset.

```python
yhat = clf.predict(X_test)
```

### 9. Model Evaluation

Evaluate the classification model using:

- Confusion Matrix
- Classification Report
- Precision
- Recall
- F1-Score

## 🤖 Machine Learning Algorithm

**Support Vector Machine (SVM)**

SVM is a supervised machine learning algorithm used for classification tasks.

In this project, the RBF (Radial Basis Function) kernel is used to learn nonlinear decision boundaries between
