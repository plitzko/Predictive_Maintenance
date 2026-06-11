"""
15_anomaly_isolation_forest.py
Phase 4: Anomalieerkennung auf Simulator-Daten mit Isolation Forest

Trainiert ein Isolation-Forest-Modell auf den Healthy-Phasen aller LKW
(per default die ersten 14 Tage pro Truck, in denen alle Trucks im
Mittel noch ueber dem 0.5er Health-Threshold liegen) und scort danach
den gesamten Simulator-Datensatz.

Wichtig: Weder `health_score` noch `Engine Condition` werden als
Feature uebergeben. Das Modell lernt unsupervised aus reinen
Sensoren + Aux + Wetter + Kontext, was eine "normale" LKW-Messung
ist, und markiert Abweichungen davon als Anomalie. Der Vergleich
gegen `Engine Condition` dient nur als externe Validierung.

Ab Phase 4.2 wird bevorzugt engine_data_features.csv gelesen. Wenn die Datei
nicht existiert, bleibt der Fallback auf engine_data_simulated.csv erhalten.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler

RNG_SEED = 42
TRAIN_DAYS = 14            # erste 14 Tage pro Truck als "healthy" Anker
CONTAMINATION = 0.10       # erwarteter Anomalie-Anteil im Trainings-Sample
N_ESTIMATORS = 200

SENSOR_COLS = [
    "Engine rpm", "Lub oil pressure", "Fuel pressure",
    "Coolant pressure", "lub oil temp", "Coolant temp",
]
AUX_COLS = ["tire_pressure_bar", "brake_fluid_pct", "odometer_km"]
WEATHER_COLS = ["temperature_c", "precipitation_mm"]
CONTEXT_COLS = ["load_pct", "route_type_encoded"]


def load_input() -> tuple[pd.DataFrame, Path]:
    candidates = [
        Path("data/engine_health/engine_data_features.csv"),
        Path("data/engine_health/engine_data_simulated.csv"),
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            validate_schema(df, path)
            print(f"Loaded {len(df):,} rows from {path}. Columns: {len(df.columns)}")
            return df, path

    raise FileNotFoundError(
        "Weder engine_data_features.csv noch engine_data_simulated.csv gefunden. "
        "Bitte zuerst `python scripts/13_simulator.py` und optional "
        "`python scripts/17_feature_engineering.py` ausfuehren."
    )


def validate_schema(df: pd.DataFrame, source_path: Path) -> None:
    """Pruefe, dass alle erforderlichen Spalten und ein Mindestumfang
    pro LKW vorhanden sind. Bei fehlenden Spalten oder zu kurzer
    Healthy-Phase wird ein klarer Fehler geworfen."""
    required = set(
        ["truck_id", "timestamp", "Engine Condition"]
        + SENSOR_COLS + AUX_COLS + WEATHER_COLS + ["load_pct", "route_type"]
    )
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"{source_path} ist fuer Skript 15 nicht geeignet. "
            f"Fehlende Spalten: {missing}. Bitte scripts/13_simulator.py laufen lassen."
        )

    timespans = (
        df.groupby("truck_id")["timestamp"]
        .agg(lambda s: (s.max() - s.min()).total_seconds() / 86_400)
    )
    too_short = timespans[timespans < TRAIN_DAYS]
    if not too_short.empty:
        print(f"WARNUNG: {len(too_short)} LKW haben weniger als {TRAIN_DAYS} "
              f"Tage Daten und werden im Trainings-Anker unterrepraesentiert "
              f"sein:")
        for tid, days in too_short.items():
            print(f"  {tid}: {days:.1f} Tage")


def engineered_feature_cols(df: pd.DataFrame, source_path: Path) -> list[str]:
    """Isolation Forest nutzt FE-Spalten, aber nie health_score."""
    if source_path.name != "engine_data_features.csv":
        return []

    patterns = [
        "_roll6h_mean",
        "_roll24h_mean",
        "_roll24h_std",
        "_delta1",
        "_dev_baseline",
    ]
    load_flags = {"load_low", "load_med", "load_high"}
    cols = [
        col for col in df.columns
        if any(pattern in col for pattern in patterns) or col in load_flags
    ]
    return sorted(cols)


def main() -> None:
    print("=== Running 15_anomaly_isolation_forest.py ===")
    df, source = load_input()

    le = LabelEncoder()
    df["route_type_encoded"] = le.fit_transform(df["route_type"])

    extra_features = engineered_feature_cols(df, source)
    features = SENSOR_COLS + AUX_COLS + WEATHER_COLS + CONTEXT_COLS + extra_features
    if extra_features:
        print(f"Feature-Engineering-Spalten: {len(extra_features)}")
    df = df.sort_values(["truck_id", "timestamp"]).reset_index(drop=True)

    # Healthy-Anker: erste TRAIN_DAYS Tage pro Truck
    train_mask = df.groupby("truck_id")["timestamp"].transform(
        lambda s: s < (s.min() + pd.Timedelta(days=TRAIN_DAYS))
    )
    train_df = df[train_mask]
    print(f"Trainings-Pool: {len(train_df):,} Zeilen "
          f"(erste {TRAIN_DAYS} Tage je LKW)")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features])
    X_all = scaler.transform(df[features])

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RNG_SEED,
        n_jobs=-1,
    )
    model.fit(X_train)

    df["anomaly_score"] = model.decision_function(X_all)
    df["is_anomaly"] = (model.predict(X_all) == -1).astype(int)

    # Kalibrierter Threshold: 5% der Trainings-Daten als Anomalie
    train_scores = model.decision_function(X_train)
    threshold = float(np.quantile(train_scores, 0.05))
    df["is_anomaly_cal"] = (df["anomaly_score"] < threshold).astype(int)

    n_anom_default = int(df["is_anomaly"].sum())
    n_anom_cal = int(df["is_anomaly_cal"].sum())
    print(f"\nAnomalien (default Threshold)   : "
          f"{n_anom_default:,} ({n_anom_default/len(df):.1%})")
    print(f"Anomalien (kalibriert auf 5% TR): "
          f"{n_anom_cal:,} ({n_anom_cal/len(df):.1%})")

    # Externe Validierung gegen Engine Condition
    cond_anom = df.groupby("Engine Condition")["is_anomaly_cal"].mean()
    print("\nValidierung gegen Engine Condition:")
    print(f"  Healthy (1) wird flagg'ed: {cond_anom.get(1, 0):.1%}")
    print(f"  Faulty  (0) wird flagg'ed: {cond_anom.get(0, 0):.1%}")
    if cond_anom.get(0, 0) > cond_anom.get(1, 0):
        print("  -> Faulty-Quote hoeher als Healthy-Quote: Modell lernt sinnvoll.")
    else:
        print("  -> Warnung: Faulty-Quote nicht hoeher als Healthy-Quote.")

    # Anomalie-Rate pro Truck
    per_truck = df.groupby("truck_id").agg(
        rows=("is_anomaly_cal", "size"),
        anomalies=("is_anomaly_cal", "sum"),
        anomaly_rate=("is_anomaly_cal", "mean"),
        faulty_rate=("Engine Condition", lambda s: (s == 0).mean()),
    )
    per_truck["anomaly_rate"] = (per_truck["anomaly_rate"] * 100).round(2)
    per_truck["faulty_rate"] = (per_truck["faulty_rate"] * 100).round(2)
    per_truck = per_truck.sort_values("anomaly_rate", ascending=False)
    print(f"\nAnomalie-Rate pro Truck (top 5):")
    print(per_truck.head(5).to_string())

    # ----- Outputs -----
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/charts", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)

    # Report
    report_path = "outputs/15_anomaly_isolation_forest.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Anomalieerkennung mit Isolation Forest ===\n\n")
        f.write(f"Quelle: {source}\n")
        f.write(f"Trainings-Anker: erste {TRAIN_DAYS} Tage je LKW "
                f"({len(train_df):,} Zeilen)\n")
        f.write(f"Modell: IsolationForest(n_estimators={N_ESTIMATORS}, "
                f"contamination={CONTAMINATION}, seed={RNG_SEED})\n\n")
        f.write(f"Features ({len(features)}, ohne health_score und "
                f"Engine Condition):\n")
        f.write(f"  {', '.join(features)}\n\n")
        f.write("Ergebnisse:\n")
        f.write(f"  Anomalien (default Threshold): "
                f"{n_anom_default:,} ({n_anom_default/len(df):.1%})\n")
        f.write(f"  Anomalien (kalibriert):        "
                f"{n_anom_cal:,} ({n_anom_cal/len(df):.1%})\n")
        f.write(f"  Healthy-Anomalie-Rate: {cond_anom.get(1,0):.1%}\n")
        f.write(f"  Faulty-Anomalie-Rate : {cond_anom.get(0,0):.1%}\n\n")
        f.write("Anomalie-Rate pro Truck:\n")
        f.write(per_truck.to_string())
        f.write("\n\nMethodischer Hinweis: Die Anker-Phase (erste 14 Tage)\n")
        f.write("ist eine Heuristik. In Realitaet sollten Healthy-Anker aus\n")
        f.write("validierten Wartungs-/Inspektionszeitfenstern stammen.\n")
    print(f"\nReport: {report_path}")

    # Modell + Scaler persistieren
    model_path = "outputs/models/anomaly_isolation_forest_v1.pkl"
    joblib.dump(
        {
            "model": model, "scaler": scaler, "features": features,
            "label_encoder": le, "threshold": threshold,
            "source": str(source), "train_days": TRAIN_DAYS,
            "contamination": CONTAMINATION,
        },
        model_path,
    )
    print(f"Modell:  {model_path}")

    # Anomalie-CSV
    out_csv = "data/engine_health/engine_data_anomaly.csv"
    df[["truck_id", "timestamp", "anomaly_score",
        "is_anomaly", "is_anomaly_cal"]].to_csv(out_csv, index=False)
    print(f"Output:  {out_csv}")

    # Visualisierungen
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Anomalie-Rate pro Truck vs. Faulty-Rate
    sorted_trucks = per_truck.index.tolist()
    x = np.arange(len(sorted_trucks))
    w = 0.4
    axes[0].bar(x - w/2, per_truck["anomaly_rate"], width=w,
                color="#E53935", label="Anomalie-Rate")
    axes[0].bar(x + w/2, per_truck["faulty_rate"], width=w,
                color="#9CA3AF", label="Faulty-Rate")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(sorted_trucks, rotation=45, fontsize=9)
    axes[0].set_ylabel("Rate [%]")
    axes[0].set_title("Anomalie-Rate vs. Faulty-Rate pro LKW")
    axes[0].legend()

    # Panel 2: Anomalie-Score-Verlauf von 3 LKW
    show_trucks = ["LKW-07", "LKW-04", "LKW-06"]  # high/low/no faulty
    for tid, color in zip(show_trucks, ["#E53935", "#FB8C00", "#43A047"]):
        sub = df[df["truck_id"] == tid].sort_values("timestamp")
        daily = sub.set_index("timestamp")["anomaly_score"].resample("D").mean()
        axes[1].plot(daily.index, daily.values, label=tid,
                     color=color, linewidth=1.6)
    axes[1].axhline(threshold, color="black", linestyle="--",
                    linewidth=1, alpha=0.6, label="Threshold")
    axes[1].set_xlabel("Datum")
    axes[1].set_ylabel("Anomalie-Score (Tagesmittel)")
    axes[1].set_title("Anomalie-Score-Verlauf (3 ausgewaehlte LKW)")
    axes[1].legend()
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, fontsize=8)

    fig.suptitle("Isolation Forest auf Simulator-Daten",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    chart_path = "outputs/charts/18_anomaly_isolation_forest.png"
    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Chart:   {chart_path}")

    print("\n=== Done. ===")


if __name__ == "__main__":
    main()
