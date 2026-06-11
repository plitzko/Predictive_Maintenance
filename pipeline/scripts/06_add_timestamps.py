"""
06_add_timestamps.py
Phase 2, Schritt 1: Synthetische Zeitstempel erzeugen
Adds truck_id, timestamp, route_type, load_pct to the engine dataset.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=== Running 06_add_timestamps.py ===")

# Load original data
df = pd.read_csv("data/engine_health/engine_data.csv")
print(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")

n_rows = len(df)
n_trucks = 10
np.random.seed(42)

# Assign truck_id round-robin
df["truck_id"] = [f"LKW-{(i % n_trucks) + 1:02d}" for i in range(n_rows)]

# Assign timestamps: each truck gets a measurement every 30 minutes
# Starting from 2024-01-01 00:00:00
start_date = datetime(2024, 1, 1, 0, 0, 0)
interval = timedelta(minutes=30)

# For each row, calculate the measurement index for that truck
# Row i belongs to truck (i % n_trucks), and is measurement number (i // n_trucks)
timestamps = []
for i in range(n_rows):
    measurement_idx = i // n_trucks
    ts = start_date + measurement_idx * interval
    timestamps.append(ts)

df["timestamp"] = timestamps

# Assign route_type randomly
route_types = ["Autobahn", "Stadtverkehr", "Landstrasse"]
df["route_type"] = np.random.choice(route_types, size=n_rows)

# Assign load_pct (30-100%)
df["load_pct"] = np.random.randint(30, 101, size=n_rows)

# Save
output_path = "data/engine_health/engine_data_with_timestamps.csv"
df.to_csv(output_path, index=False)

# Print summary
print(f"\nAnzahl Zeilen: {len(df)}")
print(f"Zeitspanne: {df['timestamp'].min()} bis {df['timestamp'].max()}")
print(f"\nErste 3 Zeilen (neue Spalten):")
print(df[["truck_id", "timestamp", "route_type", "load_pct"]].head(3).to_string(index=False))

print("\n=== Done. Outputs saved to data/engine_health/engine_data_with_timestamps.csv ===")
