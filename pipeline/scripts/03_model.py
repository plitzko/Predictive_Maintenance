#!/usr/bin/env python3
"""
Step 4 - Baseline ML Model (XGBoost Classifier)
Trains XGBoost, generates classification report, feature importance, and confusion matrix
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import xgboost as xgb

print("=== Running 03_model.py ===")

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12})
HM_RED = '#E8352A'

# Load dataset
df = pd.read_csv('data/engine_health/engine_data.csv')

# Prepare features and target
target_col = 'Engine Condition'
X = df.drop(columns=[target_col])
y = df[target_col]

# Encode target if necessary
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)
    class_names = le.classes_
else:
    class_names = ['Faulty', 'Healthy'] if y.nunique() == 2 else [str(i) for i in sorted(y.unique())]

feature_names = X.columns.tolist()

print(f"Features: {len(feature_names)}")
print(f"Classes: {class_names}")
print(f"Dataset size: {len(df)} samples")
print()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print()

# Train XGBoost
print("Training XGBoost classifier...")
model = xgb.XGBClassifier(
    n_estimators=100,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=class_names)

# Save report
output_file = 'outputs/03_model_report.txt'
with open(output_file, 'w') as f:
    # Duplicate output to both console and file
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, text):
            for file in self.files:
                file.write(text)
        def flush(self):
            for file in self.files:
                file.flush()

    sys.stdout = Tee(sys.stdout, f)

    print("="*80)
    print("XGBOOST BASELINE MODEL REPORT")
    print("="*80)
    print()
    print(f"Model: XGBoost Classifier (n_estimators=100)")
    print(f"Test Set Size: {len(X_test)} samples")
    print()
    print("CLASSIFICATION REPORT")
    print("-" * 80)
    print(report)
    print()
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print()
    print("="*80)

# Reset stdout
sys.stdout = sys.__stdout__

print(f"Report saved to {output_file}")
print()

### CHART 6 - Feature Importance
print("Creating Chart 6: Feature Importance...")
importance = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values('Importance', ascending=True)

# Get top 10 or all if less
top_n = min(10, len(feature_importance_df))
top_features = feature_importance_df.tail(top_n)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top_features['Feature'], top_features['Importance'], color=HM_RED, alpha=0.8, edgecolor='black')
ax.set_xlabel('Importance', fontsize=13, weight='bold')
ax.set_ylabel('Feature', fontsize=13, weight='bold')
ax.set_title('Feature Importance – XGBoost', fontsize=16, weight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('outputs/charts/06_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: outputs/charts/06_feature_importance.png")

### CHART 7 - Confusion Matrix
print("Creating Chart 7: Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar_kws={"shrink": 0.8},
            xticklabels=class_names, yticklabels=class_names, ax=ax,
            linewidths=1, linecolor='black')
ax.set_xlabel('Vorhergesagte Klasse', fontsize=13, weight='bold')
ax.set_ylabel('Tatsächliche Klasse', fontsize=13, weight='bold')
ax.set_title('Konfusionsmatrix – XGBoost Baseline', fontsize=16, weight='bold', pad=20)
plt.tight_layout()
plt.savefig('outputs/charts/07_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: outputs/charts/07_confusion_matrix.png")

print("\n=== Done. Model trained and results saved ===")
