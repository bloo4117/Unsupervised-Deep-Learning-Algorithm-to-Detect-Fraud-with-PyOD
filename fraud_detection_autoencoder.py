# fraud_detection_autoencoder.py
# Author: [Your Name]
# Course: MSCS 633 - Advanced Artificial Intelligence
# Assignment: Fraud Detection using PyOD AutoEncoder

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from pyod.models.auto_encoder import AutoEncoder

# Load dataset (download from Kaggle: https://www.kaggle.com/mlg-ulb/creditcardfraud)
data = pd.read_csv('creditcard.csv')

# Preprocessing
X = data.drop(columns=['Class'])
y = data['Class']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build AutoEncoder model
clf = AutoEncoder(hidden_neurons=[64, 32, 32, 64],
                  epochs=30,
                  batch_size=32,
                  dropout_rate=0.1,
                  contamination=0.002,  # Estimated fraud ratio
                  random_state=42)

clf.fit(X_train)

# Predict
y_test_pred = clf.predict(X_test)
y_test_scores = clf.decision_function(X_test)

# Evaluation
print("ROC AUC Score:", roc_auc_score(y_test, y_test_scores))
print(classification_report(y_test, y_test_pred))

# Visualization
plt.figure(figsize=(8, 5))
plt.hist(y_test_scores[y_test == 0], bins=50, alpha=0.6, label='Normal')
plt.hist(y_test_scores[y_test == 1], bins=50, alpha=0.6, label='Fraud')
plt.title('Reconstruction Error Distribution')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.legend()
plt.show()
