#!/usr/bin/env python3
"""
Step 5 - Anomaly Detection Preview (Isolation Forest)
Detects anomalies in engine data and visualizes results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

print("=== Running 04_anomaly.py ===")

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12})
HM_RED = '#E8352A'

# Load dataset
df = pd.read_csv('data/engine_health/engine_data.csv')

# Select only numeric features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
X = df[numeric_cols]

print(f"Dataset shape: {df.shape}")
print(f"Numeric features: {numeric_cols}")
print()

# Train Isolation Forest
print("Training Isolation Forest (contamination=0.05)...")
iso_forest = IsolationForest(
    contamination=0.05,
    random_state=42,
    n_estimators=100
)
df['anomaly'] = iso_forest.fit_predict(X)
df['anomaly_score'] = iso_forest.score_samples(X)

# Count anomalies
n_anomalies = (df['anomaly'] == -1).sum()
n_normal = (df['anomaly'] == 1).sum()
pct_anomalies = (n_anomalies / len(df)) * 100

print(f"Total samples: {len(df)}")
print(f"Normal samples: {n_normal} ({100-pct_anomalies:.2f}%)")
print(f"Anomalies detected: {n_anomalies} ({pct_anomalies:.2f}%)")
print()

### CHART 8 - Anomaly Score Distribution
print("Creating Chart 8: Anomaly Score Distribution...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df['anomaly_score'], bins=50, color=HM_RED, alpha=0.7, edgecolor='black')
ax.axvline(df[df['anomaly'] == -1]['anomaly_score'].max(),
           color='red', linestyle='--', linewidth=2,
           label=f'Anomaly Threshold (top {pct_anomalies:.1f}%)')
ax.set_xlabel('Anomalie-Score', fontsize=13, weight='bold')
ax.set_ylabel('Häufigkeit', fontsize=13, weight='bold')
ax.set_title('Isolation Forest – Anomalie-Score Verteilung', fontsize=16, weight='bold', pad=20)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/charts/08_anomaly_scores.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: outputs/charts/08_anomaly_scores.png")

### CHART 9 - Anomaly Scatter Plot
print("Creating Chart 9: Anomaly Scatter Plot...")
# Pick 2 features with highest variance for visualization
variances = df[numeric_cols].var().sort_values(ascending=False)
feature_x = variances.index[0]
feature_y = variances.index[1]

fig, ax = plt.subplots(figsize=(10, 8))

# Plot normal points first (in background)
normal_data = df[df['anomaly'] == 1]
ax.scatter(normal_data[feature_x], normal_data[feature_y],
          c='gray', alpha=0.4, s=30, label='Normal', edgecolors='none')

# Plot anomalies on top (in foreground)
anomaly_data = df[df['anomaly'] == -1]
ax.scatter(anomaly_data[feature_x], anomaly_data[feature_y],
          c=HM_RED, alpha=0.8, s=50, label='Anomalie', edgecolors='black', linewidth=0.5)

ax.set_xlabel(feature_x, fontsize=13, weight='bold')
ax.set_ylabel(feature_y, fontsize=13, weight='bold')
ax.set_title(f'Anomalieerkennung: {feature_x} vs {feature_y}', fontsize=16, weight='bold', pad=20)
ax.legend(fontsize=12, loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/charts/09_anomaly_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: outputs/charts/09_anomaly_scatter.png")

print("\n=== Done. Anomaly detection complete ===")
