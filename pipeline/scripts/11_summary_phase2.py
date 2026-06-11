"""
11_summary_phase2.py
Phase 2, Schritt 6: Zusammenfassung Phase 2
Generates the final summary report for Phase 2.
"""

import pandas as pd
import os

print("=== Running 11_summary_phase2.py ===")

# Load final dataset
df = pd.read_csv("data/engine_health/engine_data_final.csv")
print(f"Loaded {len(df)} rows x {len(df.columns)} columns.")

# Load model comparison results
model_report = ""
try:
    with open("outputs/10_model_comparison.txt", "r") as f:
        model_report = f.read()
except FileNotFoundError:
    print("Warning: outputs/10_model_comparison.txt not found. Run 09_model_with_weather.py first.")

# Load alert demo results
alert_report = ""
try:
    with open("outputs/11_alert_demo.txt", "r") as f:
        alert_report = f.read()
except FileNotFoundError:
    print("Warning: outputs/11_alert_demo.txt not found. Run 10_alert_demo.py first.")

# Extract model metrics from report (parse the text)
baseline_acc = "N/A"
weather_acc = "N/A"
context_acc = "N/A"
baseline_recall = "N/A"
weather_recall = "N/A"
context_recall = "N/A"

lines = model_report.split("\n")
current_model = None
for line in lines:
    if "Modell A" in line:
        current_model = "A"
    elif "Modell B" in line:
        current_model = "B"
    elif "Modell C" in line:
        current_model = "C"

    if "Accuracy:" in line and "Faulty Recall:" in line:
        parts = line.split("|")
        acc_part = parts[0].split("Accuracy:")[1].strip() if "Accuracy:" in parts[0] else "N/A"
        recall_part = parts[1].split("Faulty Recall:")[1].strip() if len(parts) > 1 and "Faulty Recall:" in parts[1] else "N/A"
        if current_model == "A":
            baseline_acc = acc_part
            baseline_recall = recall_part
        elif current_model == "B":
            weather_acc = acc_part
            weather_recall = recall_part
        elif current_model == "C":
            context_acc = acc_part
            context_recall = recall_part

# DTC analysis
sensor_cols = ["Engine rpm", "Lub oil pressure", "Fuel pressure",
               "Coolant pressure", "lub oil temp", "Coolant temp"]

DTC_MAPPING = {
    "Engine rpm": {"code": "P0219", "german": "Motordrehzahl kritisch erhoht"},
    "Lub oil pressure": {"code": "P0520", "german": "Schmieroeldrueck kritisch niedrig"},
    "Fuel pressure": {"code": "P0087", "german": "Kraftstoffdruck zu niedrig"},
    "Coolant pressure": {"code": "P0191", "german": "Kuehlmitteldruck ausserhalb Normbereich"},
    "lub oil temp": {"code": "P0524", "german": "Schmieroel-Temperatur kritisch hoch"},
    "Coolant temp": {"code": "P0217", "german": "Motorueberhitzung erkannt"},
}

# Most common DTC overall
dtc_counts = df["primary_dtc_code"].value_counts()
most_common_code = dtc_counts.index[0]
least_common_code = dtc_counts.index[-1]

# Get descriptions
most_common_desc = ""
least_common_desc = ""
for feat, info in DTC_MAPPING.items():
    if info["code"] == most_common_code:
        most_common_desc = info["german"]
    if info["code"] == least_common_code:
        least_common_desc = info["german"]

# Most common DTC in Faulty class
faulty_dtc = df[df["Engine Condition"] == 0]["primary_dtc_code"].value_counts()
faulty_most_common = faulty_dtc.index[0]
faulty_pct = faulty_dtc.iloc[0] / len(df[df["Engine Condition"] == 0]) * 100

# Alert counts from alert_report
kritisch_count = "N/A"
warnung_count = "N/A"
ok_count = "N/A"
for line in alert_report.split("\n"):
    if "KRITISCH:" in line and "Alert" not in line and "symbol" not in line:
        try:
            kritisch_count = line.split(":")[1].strip()
        except (IndexError, ValueError):
            pass
    elif "WARNUNG:" in line and "Alert" not in line:
        try:
            warnung_count = line.split(":")[1].strip()
        except (IndexError, ValueError):
            pass
    elif "OK:" in line and "Alert" not in line and "=" not in line:
        try:
            ok_count = line.split(":")[1].strip()
        except (IndexError, ValueError):
            pass

# Build summary
summary_lines = []
summary_lines.append("=== Phase 2 Zusammenfassung - Datenanreicherung ===\n")

summary_lines.append("Datenstrategie:")
summary_lines.append("  Quellen kombiniert: Kaggle (Kerndaten) + OpenWeatherMap (Wetter) + DTC-Mapping (CarAPI-Ersatz)")
summary_lines.append(f"  Finale Datensatzgroesse: {len(df):,} Zeilen x {len(df.columns)} Features")
summary_lines.append("  Neue Features: timestamp, truck_id, route_type, load_pct, temperature_c,")
summary_lines.append("                 precipitation_mm, primary_dtc_code, primary_dtc_german")

summary_lines.append(f"\nModellvergleich:")
summary_lines.append(f"  Baseline (6 Features):    Accuracy {baseline_acc} | Faulty Recall {baseline_recall}")
summary_lines.append(f"  Mit Wetter (8 Features):  Accuracy {weather_acc} | Faulty Recall {weather_recall}")
summary_lines.append(f"  Mit Kontext (9 Features): Accuracy {context_acc} | Faulty Recall {context_recall}")

summary_lines.append(f"\nDTC-Analyse:")
summary_lines.append(f"  Haeufigster DTC: {most_common_code} - {most_common_desc} "
                     f"({dtc_counts.iloc[0]:,} Faelle, {dtc_counts.iloc[0]/len(df)*100:.1f}%)")
summary_lines.append(f"  Seltenster DTC: {least_common_code} - {least_common_desc} "
                     f"({dtc_counts.iloc[-1]:,} Faelle)")
summary_lines.append(f"  DTC bei Faulty-Fahrzeugen: Dominiert von {faulty_most_common} ({faulty_pct:.1f}%)")

summary_lines.append(f"\nAlert-System:")
summary_lines.append(f"  Kritische Alerts im Testset: {kritisch_count}")
summary_lines.append(f"  Warnungen im Testset: {warnung_count}")
summary_lines.append(f"  OK im Testset: {ok_count}")

summary_lines.append("""
Fazit fuer Praesentation:
  Die Kombination aus drei Datenquellen erlaubt es, aus einem einfachen
  Healthy/Faulty-Label einen actionable Alert mit konkretem Fehlercode,
  Handlungsempfehlung und Wetterkontext zu machen - genau das, was
  Flottenmanager Thomas und Werkstattleiter Stefan brauchen.

  Der technische Mehrwert:
  - DTC-Codes machen Alerts fuer Werkstaetten verstaendlich (Industriestandard)
  - Wetterkontext hilft bei Ursachenanalyse (Kaelte -> Oelviskositaet)
  - Routentyp und Beladung ermoeglichen situative Bewertung
  - 3-Stufen-Alertsystem (OK/Warnung/Kritisch) priorisiert Handlungsbedarf""")

summary_text = "\n".join(summary_lines)
print(f"\n{summary_text}")

# Save
os.makedirs("outputs", exist_ok=True)
with open("outputs/12_summary_phase2.txt", "w") as f:
    f.write(summary_text)

print("\n=== Done. Outputs saved to outputs/12_summary_phase2.txt ===")
