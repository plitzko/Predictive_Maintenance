#!/usr/bin/env python3
"""
Step 3 - Exploratory Data Analysis (EDA)
Produces 5 visualization charts saved to outputs/charts/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

print("=== Running 02_eda.py ===")

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12})
HM_RED = '#E8352A'

# Load dataset
df = pd.read_csv('data/engine_health/engine_data.csv')

# Map numeric target to text labels
df['Engine Status'] = df['Engine Condition'].map({0: 'Faulty', 1: 'Healthy'})
target_col = 'Engine Status'

# Identify numeric features (exclude target)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'Engine Condition' in numeric_cols:
    numeric_cols.remove('Engine Condition')

print(f"Target column: {target_col}")
print(f"Numeric features: {numeric_cols}")
print()

### CHART 1 - Label Distribution
print("Creating Chart 1: Label Distribution...")
fig, ax = plt.subplots(figsize=(10, 6))
counts = df[target_col].value_counts()
ax.bar(counts.index, counts.values, color=HM_RED, alpha=0.8, edgecolor='black')
ax.set_xlabel('Fahrzeuggesundheitsklasse', fontsize=14, weight='bold')
ax.set_ylabel('Anzahl', fontsize=14, weight='bold')
ax.set_title('Verteilung der Fahrzeuggesundheitsklassen', fontsize=16, weight='bold', pad=20)
# Add count labels on bars
for i, (label, count) in enumerate(counts.items()):
    ax.text(i, count + 50, str(count), ha='center', fontsize=12, weight='bold')
plt.tight_layout()
plt.savefig('outputs/charts/01_label_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: outputs/charts/01_label_distribution.png")

### CHART 2 - Feature Distributions (Histograms)
print("Creating Chart 2: Feature Histograms...")
n_features = len(numeric_cols)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
axes = axes.flatten() if n_features > 1 else [axes]

for i, col in enumerate(numeric_cols):
    axes[i].hist(df[col], bins=30, color=HM_RED, alpha=0.7, edgecolor='black')
    axes[i].set_xlabel(col, fontsize=11, weight='bold')
    axes[i].set_ylabel('Häufigkeit', fontsize=11)
    axes[i].set_title(col, fontsize=12, weight='bold')
    axes[i].grid(True, alpha=0.3)

# Hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

fig.suptitle('Sensorwerte-Verteilung (alle Features)', fontsize=16, weight='bold', y=1.0)
plt.tight_layout()
plt.savefig('outputs/charts/02_feature_histograms.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: outputs/charts/02_feature_histograms.png")

### CHART 3 - Correlation Heatmap
print("Creating Chart 3: Correlation Heatmap...")
fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Feature-Korrelationsmatrix', fontsize=16, weight='bold', pad=20)
plt.tight_layout()
plt.savefig('outputs/charts/03_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: outputs/charts/03_correlation_heatmap.png")

### CHART 4 - Boxplots by Health Class
print("Creating Chart 4: Boxplots by Class...")
# Select top 4 features with highest variance
variances = df[numeric_cols].var().sort_values(ascending=False)
top_4_features = variances.head(4).index.tolist()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, feature in enumerate(top_4_features):
    df.boxplot(column=feature, by=target_col, ax=axes[i], patch_artist=True)
    axes[i].set_title(feature, fontsize=12, weight='bold')
    axes[i].set_xlabel('Gesundheitsklasse', fontsize=11, weight='bold')
    axes[i].set_ylabel(feature, fontsize=11)
    axes[i].get_figure().suptitle('')  # Remove default title

fig.suptitle('Sensordaten nach Gesundheitsklasse', fontsize=16, weight='bold', y=0.995)
plt.tight_layout()
plt.savefig('outputs/charts/04_boxplots_by_class.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: outputs/charts/04_boxplots_by_class.png")

### CHART 5 - Class Separation Scatter
print("Creating Chart 5: Class Separation Scatter...")
# Find 2 features that best separate classes (highest between-class variance)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_numeric = le.fit_transform(df[target_col])

# Calculate between-class variance for each feature
between_class_vars = {}
for col in numeric_cols:
    class_means = df.groupby(target_col)[col].mean()
    overall_mean = df[col].mean()
    class_counts = df[target_col].value_counts()
    bcv = sum(class_counts[cls] * (class_means[cls] - overall_mean)**2 for cls in class_means.index)
    between_class_vars[col] = bcv

# Get top 2 features
sorted_features = sorted(between_class_vars.items(), key=lambda x: x[1], reverse=True)
feature_x, feature_y = sorted_features[0][0], sorted_features[1][0]

fig, ax = plt.subplots(figsize=(10, 8))
colors = {label: HM_RED if i == 0 else '#2E86AB' for i, label in enumerate(df[target_col].unique())}
for label in df[target_col].unique():
    subset = df[df[target_col] == label]
    ax.scatter(subset[feature_x], subset[feature_y],
              label=label, alpha=0.6, s=50, color=colors[label], edgecolor='black', linewidth=0.5)

ax.set_xlabel(feature_x, fontsize=13, weight='bold')
ax.set_ylabel(feature_y, fontsize=13, weight='bold')
ax.set_title(f'Klassentrennung: {feature_x} vs {feature_y}', fontsize=16, weight='bold', pad=20)
ax.legend(title='Gesundheitsklasse', fontsize=11, title_fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/charts/05_scatter_class_separation.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: outputs/charts/05_scatter_class_separation.png")

print("\n=== Done. Charts saved to outputs/charts/ ===")
