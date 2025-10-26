# Fraud Detection using PyOD AutoEncoder

This project demonstrates how to detect fraudulent credit card transactions using the **PyOD AutoEncoder** deep learning model.

## Overview
The model uses reconstruction errors to identify anomalies (potential frauds) in an anonymized credit card dataset. PyOD simplifies the process of building deep-learning-based outlier detection models.

## Steps
1. Preprocess and normalize data
2. Train AutoEncoder on non-fraudulent samples
3. Compute reconstruction errors
4. Classify points with high error as fraud

## Setup
```bash
pip install -r requirements.txt
python fraud_detection_autoencoder.py
```

## Dataset
Use the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud). Place `creditcard.csv` in the project folder.

## Output
- ROC-AUC and classification metrics printed in terminal
- Histogram visualization of reconstruction errors
