"""
16_classification_xgboost.py
Phase 4: 3-Klassen-Klassifikation auf Simulator-Daten mit XGBoost

Klassifiziert jede Messung als OK, WARNUNG oder KRITISCH. Das Label
wird aus dem health_score abgeleitet:
  - health_score > 0.7  -> OK
  - 0.5 < health_score <= 0.7 -> WARNUNG
  - health_score <= 0.5 -> KRITISCH

Ehrlicher Reality-Check: health_score wird BEWUSST NICHT als Feature
uebergeben. Damit testen wir, ob das Modell aus den realen Sensoren,
Aux-Sensoren, Wetter- und Kontextdaten alleine die Klasse vorhersagen
kann.

GroupShuffleSplit ueber Trucks: 8 LKW im Training, 2 LKW im Test.
So lernt das Modell nicht "Truck-IDs auswendig", sondern muss aus
Sensorpattern generalisieren.

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
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

RNG_SEED = 42
CLASS_LABELS = ["KRITISCH", "WARNUNG", "OK"]
CLASS_INDEX = {label: i for i, label in enumerate(CLASS_LABELS)}

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
    """Pruefe, dass alle erforderlichen Spalten vorhanden sind und
    mindestens drei Klassen abgeleitet werden koennen."""
    required = set(
        ["truck_id", "timestamp", "health_score", "route_type"]
        + SENSOR_COLS + AUX_COLS + WEATHER_COLS + ["load_pct"]
    )
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"{source_path} ist fuer Skript 16 nicht geeignet. "
            f"Fehlende Spalten: {missing}. Bitte scripts/13_simulator.py laufen lassen."
        )
    if df["truck_id"].nunique() < 3:
        raise ValueError(
            f"{source_path} hat zu wenige Trucks ({df['truck_id'].nunique()}) "
            f"fuer einen sinnvollen GroupShuffleSplit."
        )


def engineered_feature_cols(df: pd.DataFrame, source_path: Path) -> list[str]:
    """Nutze alle neuen Feature-Engineering-Spalten, aber nie health_score."""
    if source_path.name != "engine_data_features.csv":
        return []

    patterns = [
        "_roll6h_mean",
        "_roll24h_mean",
        "_roll24h_std",
        "_delta1",
        "_lag1",
        "_lag12",
        "_lag48",
        "_dev_baseline",
    ]
    explicit_cols = {"load_low", "load_med", "load_high", "hour_of_day", "day_of_week"}
    cols = [
        col for col in df.columns
        if any(pattern in col for pattern in patterns) or col in explicit_cols
    ]
    return sorted(cols)


def derive_label(df: pd.DataFrame) -> pd.Series:
    label = pd.Series(index=df.index, dtype="object")
    label[df["health_score"] > 0.7] = "OK"
    label[(df["health_score"] > 0.5) & (df["health_score"] <= 0.7)] = "WARNUNG"
    label[df["health_score"] <= 0.5] = "KRITISCH"
    return label


def main() -> None:
    print("=== Running 16_classification_xgboost.py ===")
    df, source = load_input()

    le = LabelEncoder()
    df["route_type_encoded"] = le.fit_transform(df["route_type"])
    df["alert_class"] = derive_label(df)
    y_str = df["alert_class"]
    y = y_str.map(CLASS_INDEX).astype(int)
    print("\nKlassenverteilung (gesamt):")
    for cls in CLASS_LABELS:
        n = (y_str == cls).sum()
        print(f"  {cls:9s}: {n:7,} ({n/len(df):.1%})")

    extra_features = engineered_feature_cols(df, source)
    features = SENSOR_COLS + AUX_COLS + WEATHER_COLS + CONTEXT_COLS + extra_features
    print(f"\nFeatures ({len(features)}, ohne health_score):")
    print(f"  {', '.join(features)}")
    if extra_features:
        print(f"  Feature-Engineering-Spalten: {len(extra_features)}")

    X = df[features]
    groups = df["truck_id"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RNG_SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    train_trucks = sorted(df.iloc[train_idx]["truck_id"].unique())
    test_trucks = sorted(df.iloc[test_idx]["truck_id"].unique())
    print(f"\nGroupShuffleSplit:")
    print(f"  Train-Trucks ({len(train_trucks)}): {', '.join(train_trucks)}")
    print(f"  Test-Trucks  ({len(test_trucks)}): {', '.join(test_trucks)}")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=len(CLASS_LABELS),
        eval_metric="mlogloss",
        random_state=RNG_SEED,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
    )
    print("\nTrainiere XGBoost-Klassifikator (300 Baeume, max_depth=6)...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    recall_macro = recall_score(y_test, y_pred, average="macro")
    recall_critical = recall_score(
        y_test, y_pred,
        labels=[CLASS_INDEX["KRITISCH"]],
        average="macro",
    )

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    cm_df = pd.DataFrame(
        cm,
        index=[f"actual_{c}" for c in CLASS_LABELS],
        columns=[f"pred_{c}" for c in CLASS_LABELS],
    )

    print(f"\nEvaluation auf Group-Test ({len(y_test):,} Samples):")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Macro-F1        : {f1_macro:.4f}")
    print(f"  Macro-Recall    : {recall_macro:.4f}")
    print(f"  Recall KRITISCH : {recall_critical:.4f}  "
          f"(wichtigste Metrik fuer Predictive Maintenance)")

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_LABELS,
        digits=3,
    ))
    print("\nConfusion Matrix:")
    print(cm_df.to_string())

    importance = sorted(
        zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True
    )
    print("\nTop-5 Feature Importances:")
    for feat, val in importance[:5]:
        print(f"  {feat:25s}: {val:.4f}")

    # ----- Outputs -----
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/charts", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)

    report_path = "outputs/16_classification_xgboost.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== 3-Klassen-Klassifikation mit XGBoost ===\n\n")
        f.write(f"Quelle: {source}\n")
        f.write("Label: aus health_score abgeleitet "
                "(>0.7 OK, 0.5..0.7 WARNUNG, <=0.5 KRITISCH)\n")
        f.write("Wichtig: health_score selbst ist NICHT als Feature dabei.\n\n")
        f.write("Modell: xgb.XGBClassifier(n_estimators=300, max_depth=6, "
                f"random_state={RNG_SEED})\n")
        f.write(f"Features ({len(features)}): {', '.join(features)}\n\n")
        f.write(f"Train-Trucks: {', '.join(train_trucks)}\n")
        f.write(f"Test-Trucks : {', '.join(test_trucks)}\n\n")
        f.write("Ergebnisse:\n")
        f.write(f"  Accuracy        : {acc:.4f}\n")
        f.write(f"  Macro-F1        : {f1_macro:.4f}\n")
        f.write(f"  Macro-Recall    : {recall_macro:.4f}\n")
        f.write(f"  Recall KRITISCH : {recall_critical:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(
            y_test, y_pred,
            labels=[0, 1, 2],
            target_names=CLASS_LABELS,
            digits=3,
        ))
        f.write("\nConfusion Matrix:\n")
        f.write(cm_df.to_string())
        f.write("\n\nTop-5 Feature Importances:\n")
        for feat, val in importance[:5]:
            f.write(f"  {feat:25s}: {val:.4f}\n")
        f.write("\nMethodischer Hinweis: Die Klassengrenzen (0.5/0.7) sind\n")
        f.write("Heuristiken aus dem MVP-Schwellenwert. health_score wird\n")
        f.write("bewusst NICHT als Feature uebergeben, damit das Modell die\n")
        f.write("Klasse aus realen Sensoren ableiten muss. Faellt der\n")
        f.write("Recall-KRITISCH unter ca. 0.6, ist das ein Indiz, dass die\n")
        f.write("Sensor-Drift im Simulator zu schwach ist.\n")
    print(f"\nReport: {report_path}")

    model_path = "outputs/models/classification_xgboost_v1.pkl"
    joblib.dump(
        {
            "model": model, "features": features,
            "label_encoder": le, "class_labels": CLASS_LABELS,
            "source": str(source),
        },
        model_path,
    )
    print(f"Modell: {model_path}")

    # Visualisierung: Confusion Matrix + Feature Importance
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=axes[0],
        cbar=False,
    )
    axes[0].set_xlabel("Vorhersage")
    axes[0].set_ylabel("Tatsaechlich")
    axes[0].set_title(f"Confusion Matrix (Acc={acc:.2f})")

    top_n = 8
    top_features = importance[:top_n]
    names = [f[0] for f in top_features][::-1]
    values = [f[1] for f in top_features][::-1]
    axes[1].barh(names, values, color="#1D4ED8", edgecolor="black", linewidth=0.5)
    axes[1].set_xlabel("Feature Importance")
    axes[1].set_title(f"Top-{top_n} Feature Importances")

    fig.suptitle("XGBoost auf Simulator-Daten (ohne health_score)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    chart_path = "outputs/charts/19_classification_xgboost.png"
    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Chart:  {chart_path}")

    print("\n=== Done. ===")


if __name__ == "__main__":
    main()
