"""
08_dtc_mapping.py
Phase 2, Schritt 3: DTC-Mapping (OBD-II Fehlercodes)
Maps anomalous sensor readings to standardized diagnostic trouble codes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("=== Running 08_dtc_mapping.py ===")

# DTC Lookup Table
DTC_MAPPING = {
    "Engine rpm": {
        "code": "P0219",
        "description": "Engine Overspeed Condition",
        "german": "Motordrehzahl kritisch erhoht",
        "action": "Motor sofort abschalten, Drehzahlregler pruefen"
    },
    "Lub oil pressure": {
        "code": "P0520",
        "description": "Engine Oil Pressure Sensor/Switch Circuit Malfunction",
        "german": "Schmieroeldrueck kritisch niedrig",
        "action": "Sofortiger Motorstopp, Oelspiegel und Pumpe pruefen"
    },
    "Fuel pressure": {
        "code": "P0087",
        "description": "Fuel Rail/System Pressure Too Low",
        "german": "Kraftstoffdruck zu niedrig",
        "action": "Kraftstofffilter und -pumpe pruefen"
    },
    "Coolant pressure": {
        "code": "P0191",
        "description": "Fuel Rail Pressure Sensor Circuit Range/Performance",
        "german": "Kuehlmitteldruck ausserhalb Normbereich",
        "action": "Kuehlsystem und Drucksensor pruefen"
    },
    "lub oil temp": {
        "code": "P0524",
        "description": "Engine Oil Pressure Too Low",
        "german": "Schmieroel-Temperatur kritisch hoch",
        "action": "Motor abkuehlen lassen, Oelqualitaet pruefen"
    },
    "Coolant temp": {
        "code": "P0217",
        "description": "Engine Overtemperature Condition",
        "german": "Motorueberhitzung erkannt",
        "action": "Sofortiger Halt, Kuehlsystem pruefen"
    }
}

# Load enriched data
df = pd.read_csv("data/engine_health/engine_data_enriched.csv")
print(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")

# Sensor feature columns
sensor_cols = ["Engine rpm", "Lub oil pressure", "Fuel pressure",
               "Coolant pressure", "lub oil temp", "Coolant temp"]

# Calculate z-scores for each sensor feature
z_scores = pd.DataFrame()
for col in sensor_cols:
    z_scores[col] = (df[col] - df[col].mean()) / df[col].std()

# Find the most anomalous feature per row (highest absolute z-score)
abs_z = z_scores.abs()
df["most_anomalous_feature"] = abs_z.idxmax(axis=1)
df["max_z_score"] = abs_z.max(axis=1)

# Map to DTC codes
df["primary_dtc_code"] = df["most_anomalous_feature"].map(
    lambda f: DTC_MAPPING[f]["code"]
)
df["primary_dtc_german"] = df["most_anomalous_feature"].map(
    lambda f: DTC_MAPPING[f]["german"]
)
df["primary_dtc_action"] = df["most_anomalous_feature"].map(
    lambda f: DTC_MAPPING[f]["action"]
)

# Save final dataset
output_path = "data/engine_health/engine_data_final.csv"
df.to_csv(output_path, index=False)
print(f"Saved final dataset: {output_path} ({len(df)} rows x {len(df.columns)} columns)")

# === Generate DTC distribution report ===
os.makedirs("outputs", exist_ok=True)

report_lines = []
report_lines.append("=== DTC-Verteilung (OBD-II Fehlercodes) ===\n")

# Overall DTC distribution
dtc_counts = df["primary_dtc_code"].value_counts()
dtc_pct = df["primary_dtc_code"].value_counts(normalize=True) * 100

report_lines.append("Haeufigkeit jedes DTC-Codes (gesamt):")
report_lines.append("-" * 50)
for code in dtc_counts.index:
    feature = [k for k, v in DTC_MAPPING.items() if v["code"] == code][0]
    report_lines.append(
        f"  {code} ({DTC_MAPPING[feature]['german']}): "
        f"{dtc_counts[code]:,} ({dtc_pct[code]:.1f}%)"
    )

# DTC by Engine Condition
report_lines.append("\n\nDTC-Verteilung nach Fahrzeugzustand:")
report_lines.append("-" * 50)

for condition, label in [(0, "Faulty"), (1, "Healthy")]:
    subset = df[df["Engine Condition"] == condition]
    counts = subset["primary_dtc_code"].value_counts()
    pct = subset["primary_dtc_code"].value_counts(normalize=True) * 100
    report_lines.append(f"\n  {label} (n={len(subset)}):")
    for code in counts.index:
        feature = [k for k, v in DTC_MAPPING.items() if v["code"] == code][0]
        report_lines.append(f"    {code}: {counts[code]:,} ({pct[code]:.1f}%)")

# DTC for anomalies (if anomaly_label column exists)
if "anomaly_label" in df.columns:
    report_lines.append("\n\nTop-3 DTC-Codes bei Isolation-Forest-Anomalien:")
    report_lines.append("-" * 50)
    anomalies = df[df["anomaly_label"] == -1]
    anom_counts = anomalies["primary_dtc_code"].value_counts().head(3)
    for code, count in anom_counts.items():
        feature = [k for k, v in DTC_MAPPING.items() if v["code"] == code][0]
        report_lines.append(f"  {code} ({DTC_MAPPING[feature]['german']}): {count}")
else:
    report_lines.append("\n\n(anomaly_label Spalte nicht vorhanden - Isolation Forest Analyse uebersprungen)")

report_text = "\n".join(report_lines)
print(f"\n{report_text}")

with open("outputs/09_dtc_distribution.txt", "w") as f:
    f.write(report_text)

# === Generate DTC distribution chart ===
os.makedirs("outputs/charts", exist_ok=True)

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))

# Prepare grouped data
plot_data = df.groupby(["primary_dtc_code", "Engine Condition"]).size().unstack(fill_value=0)
plot_data.columns = ["Faulty", "Healthy"]

# Sort by total count
plot_data["total"] = plot_data.sum(axis=1)
plot_data = plot_data.sort_values("total", ascending=True)
plot_data = plot_data.drop(columns="total")

# Add German labels to y-axis
y_labels = []
for code in plot_data.index:
    feature = [k for k, v in DTC_MAPPING.items() if v["code"] == code][0]
    y_labels.append(f"{code}\n{DTC_MAPPING[feature]['german']}")

# Plot
bar_width = 0.35
y_pos = np.arange(len(plot_data))

bars1 = ax.barh(y_pos - bar_width/2, plot_data["Faulty"], bar_width,
                label="Faulty", color="#E8352A")
bars2 = ax.barh(y_pos + bar_width/2, plot_data["Healthy"], bar_width,
                label="Healthy", color="#9CA3AF")

ax.set_yticks(y_pos)
ax.set_yticklabels(y_labels, fontsize=9)
ax.set_xlabel("Anzahl Datenpunkte")
ax.set_title("OBD-II Fehlercodeverteilung nach Fahrzeugzustand", fontsize=13, fontweight="bold")
ax.legend(loc="lower right")

plt.tight_layout()
plt.savefig("outputs/charts/10_dtc_distribution.png", dpi=300, bbox_inches="tight")
plt.close()
print("\nChart gespeichert: outputs/charts/10_dtc_distribution.png")

print("\n=== Done. Outputs saved to outputs/ ===")
