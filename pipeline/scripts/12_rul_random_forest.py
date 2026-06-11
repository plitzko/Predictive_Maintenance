"""
12_rul_random_forest.py
Phase 4: RUL-Prognose (Remaining Useful Life) aus simulierten Verschleisskurven

Ab Phase 4.2 liest dieses Skript bevorzugt engine_data_features.csv und faellt
auf engine_data_simulated.csv zurueck, falls Feature Engineering noch nicht
ausgefuehrt wurde. Das RUL-Label wird nicht mehr aus max_z_score abgeleitet,
sondern aus dem zeitlichen health_score-Verlauf pro LKW:

    RUL = Tage bis health_score zum ersten Mal unter 0.5 faellt.

Wenn ein LKW im 90-Tage-Fenster nicht unter 0.5 faellt, wird die RUL aus dem
beobachteten Health-Score-Trend extrapoliert und als zensiertes Label markiert.
Diese Extrapolation ist methodisch riskant und dient nur dem Hochschul-MVP.
Echte Wartungs- oder Ausfalldaten muessen das Label in produktiven Szenarien
ersetzen.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder


RNG_SEED = 42
FAILURE_THRESHOLD = 0.5
MAX_CENSORED_RUL_DAYS = 365.0

SENSOR_COLS = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp",
]
WEATHER_COLS = ["temperature_c", "precipitation_mm"]
AUX_COLS = ["tire_pressure_bar", "brake_fluid_pct", "odometer_km"]
CONTEXT_COLS = ["load_pct", "route_type_encoded"]
# health_score ist bewusst KEIN Feature: das RUL-Label ist direkt daraus
# abgeleitet, als Feature waere es Ziel-Leakage (gleiches Prinzip wie in
# 16_classification_xgboost.py).
STATE_COLS: list[str] = []


def load_input() -> tuple[pd.DataFrame, Path]:
    """Lade bevorzugt Feature-Daten, sonst den reinen Simulator-Datensatz."""
    candidates = [
        Path("data/engine_health/engine_data_features.csv"),
        Path("data/engine_health/engine_data_simulated.csv"),
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            print(f"Loaded {len(df):,} rows from {path}. Columns: {len(df.columns)}")
            return df, path

    raise FileNotFoundError(
        "Weder engine_data_features.csv noch engine_data_simulated.csv gefunden. "
        "Bitte zuerst `python scripts/13_simulator.py` und optional "
        "`python scripts/17_feature_engineering.py` ausfuehren."
    )


def validate_schema(df: pd.DataFrame, source_path: Path) -> None:
    required = set(
        ["truck_id", "timestamp", "route_type", "load_pct", "health_score"]
        + SENSOR_COLS
        + WEATHER_COLS
        + AUX_COLS
    )
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"{source_path} ist fuer Phase-4-RUL nicht geeignet. "
            f"Fehlende Spalten: {missing}. Bitte scripts/13_simulator.py laufen lassen."
        )


def engineered_feature_cols(df: pd.DataFrame, source_path: Path) -> list[str]:
    """Ergaenze Feature-Engineering-Spalten nur bei engine_data_features.csv."""
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
    load_flags = {"load_low", "load_med", "load_high"}
    cols = [
        col for col in df.columns
        if any(pattern in col for pattern in patterns) or col in load_flags
    ]
    return sorted(cols)


def add_rul_labels(df: pd.DataFrame) -> pd.DataFrame:
    labeled = df.copy()
    labeled["timestamp"] = pd.to_datetime(labeled["timestamp"])
    labeled = labeled.sort_values(["truck_id", "timestamp"]).reset_index(drop=True)
    labeled["rul_days"] = np.nan
    labeled["rul_label_source"] = ""

    for truck_id, group in labeled.groupby("truck_id", sort=False):
        idx = group.index
        failure_rows = group[group["health_score"] < FAILURE_THRESHOLD]

        if not failure_rows.empty:
            first_failure_ts = failure_rows["timestamp"].iloc[0]
            seconds_to_failure = (
                first_failure_ts - group["timestamp"]
            ).dt.total_seconds()
            rul_days = np.maximum(seconds_to_failure / 86_400, 0.0)
            labeled.loc[idx, "rul_days"] = rul_days
            labeled.loc[idx, "rul_label_source"] = "observed_first_threshold"
            continue

        duration_days = max(
            (group["timestamp"].iloc[-1] - group["timestamp"].iloc[0]).total_seconds()
            / 86_400,
            1.0,
        )
        health_drop = group["health_score"].iloc[0] - group["health_score"].iloc[-1]
        wear_per_day = max(health_drop / duration_days, 1e-6)
        extrapolated = (group["health_score"] - FAILURE_THRESHOLD) / wear_per_day
        extrapolated = extrapolated.clip(lower=0.0, upper=MAX_CENSORED_RUL_DAYS)
        labeled.loc[idx, "rul_days"] = extrapolated
        labeled.loc[idx, "rul_label_source"] = "extrapolated_censored"

    labeled["rul_days"] = labeled["rul_days"].round(2)
    return labeled


def safe_mae(y_true: pd.Series, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(mean_absolute_error(y_true, y_pred))


def main() -> None:
    print("=== Running 12_rul_random_forest.py ===")

    df, source_path = load_input()
    validate_schema(df, source_path)

    df = add_rul_labels(df)
    le = LabelEncoder()
    df["route_type_encoded"] = le.fit_transform(df["route_type"])

    extra_features = engineered_feature_cols(df, source_path)
    features = SENSOR_COLS + WEATHER_COLS + AUX_COLS + CONTEXT_COLS + STATE_COLS
    features += extra_features
    X = df[features]
    y = df["rul_days"]
    groups = df["truck_id"]

    print("\nRUL-Label aus health_score-Verlauf:")
    print(
        f"  observed_first_threshold: "
        f"{(df['rul_label_source'] == 'observed_first_threshold').sum():,} rows"
    )
    print(
        f"  extrapolated_censored   : "
        f"{(df['rul_label_source'] == 'extrapolated_censored').sum():,} rows"
    )
    print(f"  mean={y.mean():.1f}d  min={y.min():.1f}d  max={y.max():.1f}d")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RNG_SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"\nTrainiere RandomForestRegressor auf {len(features)} Features...")
    if extra_features:
        print(f"  Feature-Engineering-Spalten: {len(extra_features)}")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        random_state=RNG_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    test_condition = df.loc[X_test.index, "Engine Condition"]
    mae_healthy = safe_mae(y_test[test_condition == 1], y_pred[test_condition == 1])
    mae_faulty = safe_mae(y_test[test_condition == 0], y_pred[test_condition == 0])

    print(f"\nEvaluation auf Group-Test-Set ({len(y_test):,} Samples):")
    print(f"  Test-Trucks   : {', '.join(sorted(df.loc[X_test.index, 'truck_id'].unique()))}")
    print(f"  MAE gesamt    : {mae:.2f} Tage")
    print(f"  MAE healthy   : {mae_healthy:.2f} Tage")
    print(f"  MAE faulty    : {mae_faulty:.2f} Tage")
    print(f"  R^2           : {r2:.4f}")

    importance = sorted(
        zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True
    )
    print("\nTop-5 Feature Importances:")
    for feat, val in importance[:5]:
        print(f"  {feat:25s}: {val:.4f}")

    os.makedirs("outputs", exist_ok=True)
    report_path = "outputs/13_rul_random_forest.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== RUL-Prognose mit Random Forest ===\n\n")
        f.write(f"Quelle: {source_path}\n")
        f.write("Label: Tage bis health_score erstmals unter 0.5 faellt\n")
        f.write("Methodisches Risiko: synthetische Verschleisskurven und zensierte Labels\n")
        f.write("koennen Scheingenauigkeit erzeugen; echte Ausfallzeitpunkte fehlen.\n\n")
        f.write("Modell: RandomForestRegressor (sklearn)\n")
        f.write("  n_estimators = 200\n")
        f.write("  max_depth    = 15\n")
        f.write(f"  Features     = {len(features)} ({', '.join(features)})\n\n")
        f.write(f"Trainings-Set: {len(X_train)} Samples\n")
        f.write(f"Test-Set     : {len(X_test)} Samples\n\n")
        f.write("Ergebnisse:\n")
        f.write(f"  MAE gesamt  : {mae:.2f} Tage\n")
        f.write(f"  MAE healthy : {mae_healthy:.2f} Tage\n")
        f.write(f"  MAE faulty  : {mae_faulty:.2f} Tage\n")
        f.write(f"  R^2         : {r2:.4f}\n\n")
        f.write("Top-5 Feature Importances:\n")
        for feat, val in importance[:5]:
            f.write(f"  {feat:25s}: {val:.4f}\n")

    print(f"\nReport gespeichert: {report_path}")

    os.makedirs("outputs/models", exist_ok=True)
    model_path = "outputs/models/rul_random_forest_v1.pkl"
    joblib.dump(
        {
            "model": model,
            "features": features,
            "label_encoder": le,
            "source": str(source_path),
            "label_definition": "days_until_first_health_score_below_0.5",
        },
        model_path,
    )
    print(f"Modell gespeichert: {model_path}")

    os.makedirs("outputs/charts", exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(y_test, y_pred, alpha=0.25, s=10, c="#1D4ED8")
    max_lim = max(30, min(MAX_CENSORED_RUL_DAYS, float(max(y_test.max(), y_pred.max()))))
    lims = [0, max_lim]
    axes[0].plot(lims, lims, color="#E8352A", linestyle="--", linewidth=1.5, label="ideal")
    axes[0].set_xlabel("Tatsaechliche RUL (Tage)")
    axes[0].set_ylabel("Vorhergesagte RUL (Tage)")
    axes[0].set_title(f"Predicted vs. Actual  (MAE={mae:.1f}d, R^2={r2:.2f})")
    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)
    axes[0].legend()

    top_n = 8
    top_features = importance[:top_n]
    names = [f[0] for f in top_features][::-1]
    values = [f[1] for f in top_features][::-1]
    axes[1].barh(names, values, color="#4AD386", edgecolor="black", linewidth=0.5)
    axes[1].set_xlabel("Feature Importance")
    axes[1].set_title(f"Top-{top_n} Feature Importances")

    fig.suptitle("RUL-Prognose mit Random Forest", fontsize=14, fontweight="bold")
    plt.tight_layout()
    chart_path = "outputs/charts/12_rul_random_forest.png"
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Chart gespeichert: {chart_path}")

    out_csv = "data/engine_health/engine_data_with_rul.csv"
    df_out = df.copy()
    df_out["rul_predicted"] = model.predict(df_out[features]).round().astype(int)
    df_out.to_csv(out_csv, index=False)
    print(f"Dataset mit RUL-Spalten gespeichert: {out_csv}")

    print("\n=== Done. ===")


if __name__ == "__main__":
    main()
