<h1 align="center">📉 Customer Churn Prediction using ANN</h1>

<p align="center">
  Predicting whether a customer will leave a business using deep learning.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9-blue?logo=python">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow">
  <img src="https://img.shields.io/badge/Scikit--learn-1.x-green?logo=scikit-learn">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey">
</p>

---

## 🚀 Project Overview

Customer churn is a major concern for subscription-based and service businesses. This project leverages an **Artificial Neural Network (ANN)** to accurately predict the likelihood of a customer leaving (churning) based on their historical data.

> **Goal:** Build and evaluate a robust ANN model that predicts customer churn to help businesses retain valuable customers.

---

## 📊 Dataset Description

- **Source**: [Kaggle - Churn Modeling Dataset](https://www.kaggle.com/datasets/shubhendra7/customer-churn-prediction)
- **Size**: 10,000 records
- **Target Variable**: `Exited` (1 = Churned, 0 = Stayed)
- **Features include**:
  - Demographics: `Geography`, `Gender`, `Age`
  - Customer Info: `CreditScore`, `Balance`, `Tenure`, `EstimatedSalary`
  - Services: `NumOfProducts`, `HasCrCard`, `IsActiveMember`

---

## 🧠 Machine Learning Pipeline

### 📌 1. Data Preprocessing
- Dropped: `RowNumber`, `CustomerId`, `Surname`
- Label encoding: `Gender`, One-hot encoding: `Geography`
- Train-Test Split (80-20)
- Feature scaling using `StandardScaler`

### 🤖 2. Building the ANN
- 2 Hidden Layers with ReLU Activation
- Output Layer with Sigmoid Activation
- Optimizer: Adam
- Loss: Binary Crossentropy

### 🏋️ 3. Model Training
```python
model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_test, y_test))
