#!/usr/bin/env python3
"""
Step 2 - Initial Data Inspection
Prints dataset overview and saves to outputs/01_inspection_report.txt
"""

import sys
import pandas as pd
import numpy as np

print("=== Running 01_inspect.py ===")

# Load dataset
df = pd.read_csv('data/engine_health/engine_data.csv')

# Redirect stdout to file
output_file = 'outputs/01_inspection_report.txt'
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
    print("DATASET INSPECTION REPORT - Engine Health Dataset")
    print("="*80)
    print()

    print("1. DATASET SHAPE")
    print("-" * 40)
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print()

    print("2. COLUMN NAMES AND DATA TYPES")
    print("-" * 40)
    print(df.dtypes)
    print()

    print("3. FIRST 5 ROWS")
    print("-" * 40)
    print(df.head())
    print()

    print("4. MISSING VALUES")
    print("-" * 40)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0] if missing.sum() > 0 else "No missing values found!")
    print()

    print("5. TARGET LABEL DISTRIBUTION")
    print("-" * 40)
    # Find the target column (string/object type with reasonable unique values)
    target_col = None
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].nunique() <= 10:
            target_col = col
            break

    if target_col:
        print(f"Target Column: {target_col}")
        print(df[target_col].value_counts())
        print()
        print("Percentages:")
        print(df[target_col].value_counts(normalize=True).mul(100).round(2))
    else:
        print("No clear target column found (looking for categorical with <=10 unique values)")
    print()

    print("6. DESCRIPTIVE STATISTICS")
    print("-" * 40)
    print(df.describe())
    print()

    print("="*80)
    print("INSPECTION COMPLETE")
    print("="*80)

# Reset stdout
sys.stdout = sys.__stdout__

print(f"\n=== Done. Report saved to {output_file} ===")
