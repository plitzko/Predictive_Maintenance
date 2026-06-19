"""
10_alert_demo.py
Phase 2, Schritt 5: Alert-Logik mit DTC-Integration
Demonstrates the alert system with DTC codes for fleet managers.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import os

print("=== Running 10_alert_demo.py ===")

# Load final dataset
df = pd.read_csv("data/engine_health/engine_data_final.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
print(f"Loaded {len(df)} rows.")

# Define features (Model B: sensors + weather)
sensor_cols = ["Engine rpm", "Lub oil pressure", "Fuel pressure",
               "Coolant pressure", "lub oil temp", "Coolant temp"]
weather_cols = ["temperature_c", "precipitation_mm"]
features_B = sensor_cols + weather_cols
target = "Engine Condition"

# Train Model B
X = df[features_B]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = xgb.XGBClassifier(
    n_estimators=100,
    random_state=42,
    eval_metric="logloss",
    verbosity=0
)
model.fit(X_train, y_train)

# Get predictions and confidence for test set
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Train Isolation Forest for anomaly scores
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(X_train[sensor_cols])
anomaly_scores = iso_forest.decision_function(X_test[sensor_cols])

# Create test dataframe with predictions
test_df = df.iloc[X_test.index].copy()
test_df["prediction"] = y_pred
test_df["confidence_faulty"] = y_proba[:, 0]  # probability of class 0 (Faulty)
test_df["confidence_healthy"] = y_proba[:, 1]  # probability of class 1 (Healthy)
test_df["anomaly_score"] = anomaly_scores


# Alert classification logic
def classify_alert(row):
    prediction = row["prediction"]
    anomaly_score = row["anomaly_score"]
    confidence = row["confidence_faulty"]

    if prediction == 0 and confidence > 0.7:
        return "KRITISCH"
    elif prediction == 0 or anomaly_score < -0.1:
        return "WARNUNG"
    else:
        return "OK"


test_df["alert_level"] = test_df.apply(classify_alert, axis=1)

# Count alert levels
alert_counts = test_df["alert_level"].value_counts()
print(f"\nAlert-Verteilung im Testset:")
for level in ["KRITISCH", "WARNUNG", "OK"]:
    count = alert_counts.get(level, 0)
    print(f"  {level}: {count}")

# Select 5 representative examples
critical = test_df[test_df["alert_level"] == "KRITISCH"].head(2)
warning = test_df[test_df["alert_level"] == "WARNUNG"].head(2)
ok = test_df[test_df["alert_level"] == "OK"].head(1)

examples = pd.concat([critical, warning, ok])

# Calculate z-scores for display
means = df[sensor_cols].mean()
stds = df[sensor_cols].std()

# Format alert output
alert_symbol = {"KRITISCH": "🔴", "WARNUNG": "🟡", "OK": "🟢"}

output_lines = []
output_lines.append("=" * 60)
output_lines.append("  FLEET ALERT SYSTEM - Demo-Ausgabe")
output_lines.append("  Phase 2: DTC-Integration + Wetter-Kontext")
output_lines.append("=" * 60)

for idx, row in examples.iterrows():
    z_scores = {}
    for col in sensor_cols:
        z = (row[col] - means[col]) / stds[col]
        z_scores[col] = z

    alert = row["alert_level"]
    symbol = alert_symbol[alert]

    output_lines.append("")
    output_lines.append("=" * 56)
    output_lines.append(f"LKW-ID:      {row['truck_id']}")
    output_lines.append(f"Zeitstempel: {row['timestamp']}")
    output_lines.append(f"Wetter:      {row['temperature_c']:.0f} C, {row['precipitation_mm']:.0f}mm Niederschlag")
    output_lines.append(f"Beladung:    {row['load_pct']}%")
    output_lines.append(f"Route:       {row['route_type']}")
    output_lines.append("-" * 56)
    output_lines.append(f"Alert:       {symbol} {alert}")
    output_lines.append(f"DTC-Code:    {row['primary_dtc_code']}")
    output_lines.append(f"Fehler:      {row['primary_dtc_german']}")
    output_lines.append(f"Massnahme:   {row['primary_dtc_action']}")
    output_lines.append("-" * 56)
    output_lines.append("Sensordaten:")

    for col in sensor_cols:
        z = z_scores[col]
        val = row[col]
        # Determine unit
        if "rpm" in col.lower():
            unit = ""
        elif "pressure" in col.lower():
            unit = " bar"
        elif "temp" in col.lower():
            unit = " C"
        else:
            unit = ""

        # Flag if anomalous
        if abs(z) > 2.0:
            flag = " ⚠️"
        else:
            flag = ""

        if abs(z) > 1.5:
            z_display = f"(z-Score: {z:+.1f}){flag}"
        else:
            z_display = "(normal)"

        output_lines.append(f"  {col:20s}: {val:8.1f}{unit} {z_display}")

    output_lines.append("-" * 56)
    if alert in ["KRITISCH", "WARNUNG"]:
        output_lines.append("Einsparung bei verhinderter Panne: ~600 EUR")
    output_lines.append("=" * 56)

# Add summary
output_lines.append("\n")
output_lines.append("=" * 56)
output_lines.append("  ZUSAMMENFASSUNG")
output_lines.append("=" * 56)
output_lines.append(f"  Testset-Groesse: {len(test_df)} Fahrzeug-Messungen")
output_lines.append(f"  KRITISCH:  {alert_counts.get('KRITISCH', 0)}")
output_lines.append(f"  WARNUNG:   {alert_counts.get('WARNUNG', 0)}")
output_lines.append(f"  OK:        {alert_counts.get('OK', 0)}")
output_lines.append(f"\n  Geschaetzte Einsparung (alle kritischen Alerts):")
n_critical = alert_counts.get('KRITISCH', 0)
output_lines.append(f"  {n_critical} x 600 EUR = {n_critical * 600:,.0f} EUR")
output_lines.append("=" * 56)

output_text = "\n".join(output_lines)
print(f"\n{output_text}")

# Save
os.makedirs("outputs", exist_ok=True)
with open("outputs/11_alert_demo.txt", "w") as f:
    f.write(output_text)

print("\n=== Done. Outputs saved to outputs/11_alert_demo.txt ===")
