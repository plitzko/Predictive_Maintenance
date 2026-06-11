"""
09_model_with_weather.py
Phase 2, Schritt 4: Modell neu trainieren mit Wetter-Features
Compares baseline (6 features) vs +weather (8) vs +context (9).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import os

print("=== Running 09_model_with_weather.py ===")

# Load final dataset
df = pd.read_csv("data/engine_health/engine_data_final.csv")
print(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")

# Define feature sets
sensor_cols = ["Engine rpm", "Lub oil pressure", "Fuel pressure",
               "Coolant pressure", "lub oil temp", "Coolant temp"]

weather_cols = ["temperature_c", "precipitation_mm"]

# Encode route_type
le = LabelEncoder()
df["route_type_encoded"] = le.fit_transform(df["route_type"])
# Autobahn=0, Landstrasse=1, Stadtverkehr=2

context_cols = ["load_pct", "route_type_encoded"]

target = "Engine Condition"

# Feature sets for 3 models
features_A = sensor_cols
features_B = sensor_cols + weather_cols
features_C = sensor_cols + weather_cols + context_cols

# Train/test split (same for all models)
X = df[features_C]  # superset of all features
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

results = {}

for name, features in [("A", features_A), ("B", features_B), ("C", features_C)]:
    print(f"\nTraining Modell {name} ({len(features)} Features)...")

    model = xgb.XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric="logloss",
        verbosity=0
    )

    model.fit(X_train[features], y_train)
    y_pred = model.predict(X_test[features])

    acc = accuracy_score(y_test, y_pred)
    prec_faulty = precision_score(y_test, y_pred, pos_label=0)
    rec_faulty = recall_score(y_test, y_pred, pos_label=0)
    f1_faulty = f1_score(y_test, y_pred, pos_label=0)

    # Feature importance
    importance = dict(zip(features, model.feature_importances_))
    top_10 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

    results[name] = {
        "accuracy": acc,
        "precision_faulty": prec_faulty,
        "recall_faulty": rec_faulty,
        "f1_faulty": f1_faulty,
        "top_features": top_10,
        "n_features": len(features)
    }

    print(f"  Accuracy: {acc:.4f} | Faulty Recall: {rec_faulty:.4f} | Faulty F1: {f1_faulty:.4f}")

# === Generate comparison report ===
os.makedirs("outputs", exist_ok=True)

report_lines = []
report_lines.append("=== Modellvergleich ===\n")

report_lines.append(f"Modell A (Baseline, {results['A']['n_features']} Features):")
report_lines.append(f"  Accuracy: {results['A']['accuracy']*100:.2f}%  |  "
                    f"Faulty Recall: {results['A']['recall_faulty']*100:.2f}%  |  "
                    f"Faulty F1: {results['A']['f1_faulty']:.4f}")
report_lines.append(f"  Top-Features: {', '.join([f'{f[0]} ({f[1]:.3f})' for f in results['A']['top_features'][:5]])}")

report_lines.append(f"\nModell B (+ Wetter, {results['B']['n_features']} Features):")
report_lines.append(f"  Accuracy: {results['B']['accuracy']*100:.2f}%  |  "
                    f"Faulty Recall: {results['B']['recall_faulty']*100:.2f}%  |  "
                    f"Faulty F1: {results['B']['f1_faulty']:.4f}")
delta_b = (results['B']['accuracy'] - results['A']['accuracy']) * 100
report_lines.append(f"  Verbesserung vs. Baseline: {delta_b:+.2f} Prozentpunkte Accuracy")
report_lines.append(f"  Top-Features: {', '.join([f'{f[0]} ({f[1]:.3f})' for f in results['B']['top_features'][:5]])}")

report_lines.append(f"\nModell C (+ Wetter + Kontext, {results['C']['n_features']} Features):")
report_lines.append(f"  Accuracy: {results['C']['accuracy']*100:.2f}%  |  "
                    f"Faulty Recall: {results['C']['recall_faulty']*100:.2f}%  |  "
                    f"Faulty F1: {results['C']['f1_faulty']:.4f}")
delta_c = (results['C']['accuracy'] - results['A']['accuracy']) * 100
report_lines.append(f"  Verbesserung vs. Baseline: {delta_c:+.2f} Prozentpunkte Accuracy")
report_lines.append(f"  Top-Features: {', '.join([f'{f[0]} ({f[1]:.3f})' for f in results['C']['top_features'][:5]])}")

# Fazit
report_lines.append("\nFazit:")
if delta_c > 1:
    report_lines.append(f"  Die Datenanreicherung verbessert das Modell um {delta_c:.2f} Prozentpunkte.")
    report_lines.append("  Wetter- und Kontextdaten liefern einen messbaren Mehrwert fuer die Vorhersage.")
elif delta_c > 0:
    report_lines.append(f"  Marginale Verbesserung ({delta_c:.2f} PP). Die Wetter-Features haben geringen")
    report_lines.append("  direkten Einfluss, da die synthetischen Daten keinen echten kausalen Zusammenhang")
    report_lines.append("  mit dem Engine Condition haben. In der Realitaet waere der Effekt staerker")
    report_lines.append("  (Kaelte beeinflusst Oelviskositaet, Naesse beeinflusst Bremsweg).")
else:
    report_lines.append("  Keine signifikante Verbesserung durch synthetische Wetter-Features.")
    report_lines.append("  Dies ist erwartbar, da die synthetischen Daten keinen echten kausalen")
    report_lines.append("  Zusammenhang mit dem Engine Condition haben.")
    report_lines.append("  Der Mehrwert liegt im Kontext fuer das Alert-System (Wetter bei Panne).")

report_text = "\n".join(report_lines)
print(f"\n{report_text}")

with open("outputs/10_model_comparison.txt", "w") as f:
    f.write(report_text)

# === Generate comparison chart ===
os.makedirs("outputs/charts", exist_ok=True)

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

models = ["Baseline\n(6 Features)", "+ Wetter\n(8 Features)", "+ Kontext\n(9 Features)"]
colors = ["#9CA3AF", "#1D4ED8", "#E8352A"]

# Panel 1: Accuracy
accs = [results[m]["accuracy"] * 100 for m in ["A", "B", "C"]]
bars1 = axes[0].bar(models, accs, color=colors, edgecolor="black", linewidth=0.5)
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_title("Accuracy")
axes[0].set_ylim(0, 100)
# Add value labels
for bar, val in zip(bars1, accs):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

# Panel 2: Faulty Recall
recalls = [results[m]["recall_faulty"] * 100 for m in ["A", "B", "C"]]
bars2 = axes[1].bar(models, recalls, color=colors, edgecolor="black", linewidth=0.5)
axes[1].set_ylabel("Faulty Recall (%)")
axes[1].set_title("Faulty-Klasse Recall")
axes[1].set_ylim(0, 100)
# Add value labels
for bar, val in zip(bars2, recalls):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

fig.suptitle("Modellvergleich: Baseline vs. Angereicherte Daten", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/charts/11_model_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("\nChart gespeichert: outputs/charts/11_model_comparison.png")

print("\n=== Done. Outputs saved to outputs/ ===")
