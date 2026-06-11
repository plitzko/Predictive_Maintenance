"""
17_feature_engineering.py
Phase 4.2: Feature Engineering fuer Simulator-Zeitreihen

Dieses Skript erweitert engine_data_simulated.csv um die im Pflichtenheft
geforderten Zeitreihenfeatures: Rolling Means, Rolling Std, Deltas, Lags,
Beladungs-Flags, fahrzeugindividuelle Healthy-Baseline-Abweichungen und
einfache Zeitfeatures.

Wichtig: Outlier-Capping der definierten Originalsensoren erfolgt vor allen
Rolling-/Delta-/Lag-Berechnungen. Zeilen werden nicht gedroppt; entstehende
NaNs am Anfang jedes LKW-Verlaufs werden pro LKW aufgefuellt.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/engine_health/engine_data_simulated.csv")
OUTPUT_PATH = Path("data/engine_health/engine_data_features.csv")

ROLLING_WINDOWS = {
    "6h": 12,
    "24h": 48,
}
LAG_STEPS = {
    "lag1": 1,
    "lag12": 12,
    "lag48": 48,
}
BASELINE_ROWS = 48

SENSOR_COLS = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp",
]

REQUIRED_COLS = [
    "truck_id",
    "timestamp",
    "route_type",
    "load_pct",
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp",
    "tire_pressure_bar",
    "brake_fluid_pct",
    "odometer_km",
    "temperature_c",
    "precipitation_mm",
    "health_score",
    "Engine Condition",
    "most_anomalous_feature",
    "max_z_score",
    "primary_dtc_code",
    "primary_dtc_german",
    "primary_dtc_action",
]

CAP_LIMITS = {
    "Coolant temp": (40.0, 110.0),
    "Lub oil pressure": (0.5, 8.0),
    "Engine rpm": (400.0, 2400.0),
}


def load_input() -> pd.DataFrame:
    """Lade den Simulator-Datensatz und pruefe das erwartete Phase-4-Schema."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "engine_data_simulated.csv nicht gefunden. Bitte zuerst "
            "`python scripts/13_simulator.py` ausfuehren."
        )

    df = pd.read_csv(INPUT_PATH)
    validate_schema(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def validate_schema(df: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLS) - set(df.columns))
    if missing:
        raise ValueError(
            f"{INPUT_PATH} ist fuer Feature Engineering nicht geeignet. "
            f"Fehlende Spalten: {missing}. Bitte scripts/13_simulator.py laufen lassen."
        )


def cap_sensor_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Begrenze definierte Sensoren auf fachlich plausible Wertebereiche."""
    capped = df.copy()
    cap_counts: dict[str, int] = {}

    for col, (lower, upper) in CAP_LIMITS.items():
        before = capped[col].copy()
        capped[col] = capped[col].clip(lower=lower, upper=upper)
        cap_counts[col] = int((before != capped[col]).sum())

    return capped, cap_counts


def add_rolling_features(df: pd.DataFrame, feature_cols: list[str]) -> None:
    for sensor in SENSOR_COLS:
        grouped = df.groupby("truck_id", sort=False)[sensor]

        roll6_col = f"{sensor}_roll6h_mean"
        df[roll6_col] = (
            grouped.rolling(window=ROLLING_WINDOWS["6h"], min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        feature_cols.append(roll6_col)

        roll24_col = f"{sensor}_roll24h_mean"
        df[roll24_col] = (
            grouped.rolling(window=ROLLING_WINDOWS["24h"], min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        feature_cols.append(roll24_col)

        std24_col = f"{sensor}_roll24h_std"
        df[std24_col] = (
            grouped.rolling(window=ROLLING_WINDOWS["24h"], min_periods=2)
            .std()
            .reset_index(level=0, drop=True)
        )
        feature_cols.append(std24_col)


def add_delta_lag_features(df: pd.DataFrame, feature_cols: list[str]) -> None:
    for sensor in SENSOR_COLS:
        grouped = df.groupby("truck_id", sort=False)[sensor]

        delta_col = f"{sensor}_delta1"
        df[delta_col] = grouped.diff(1)
        feature_cols.append(delta_col)

        for suffix, periods in LAG_STEPS.items():
            lag_col = f"{sensor}_{suffix}"
            df[lag_col] = grouped.shift(periods)
            feature_cols.append(lag_col)


def add_load_flags(df: pd.DataFrame, feature_cols: list[str]) -> None:
    df["load_low"] = (df["load_pct"] <= 40).astype(int)
    df["load_med"] = ((df["load_pct"] > 40) & (df["load_pct"] <= 70)).astype(int)
    df["load_high"] = (df["load_pct"] > 70).astype(int)
    feature_cols.extend(["load_low", "load_med", "load_high"])


def add_baseline_deviations(df: pd.DataFrame, feature_cols: list[str]) -> None:
    baseline = (
        df.groupby("truck_id", sort=False)
        .head(BASELINE_ROWS)
        .groupby("truck_id")[SENSOR_COLS]
        .mean()
    )

    for sensor in SENSOR_COLS:
        baseline_col = f"{sensor}_baseline"
        dev_col = f"{sensor}_dev_baseline"
        df[baseline_col] = df["truck_id"].map(baseline[sensor])
        df[dev_col] = df[sensor] - df[baseline_col]
        df.drop(columns=[baseline_col], inplace=True)
        feature_cols.append(dev_col)


def add_time_features(df: pd.DataFrame, feature_cols: list[str]) -> None:
    df["hour_of_day"] = df["timestamp"].dt.hour.astype(int)
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype(int)
    feature_cols.extend(["hour_of_day", "day_of_week"])


def impute_feature_nans(df: pd.DataFrame, feature_cols: list[str]) -> int:
    nan_before = int(df[feature_cols].isna().sum().sum())
    if nan_before == 0:
        return 0

    df[feature_cols] = df.groupby("truck_id", sort=False)[feature_cols].bfill()

    remaining = df[feature_cols].isna().sum()
    for col in remaining[remaining > 0].index:
        fill_value = df[col].mean()
        if pd.isna(fill_value):
            fill_value = 0.0
        df[col] = df[col].fillna(fill_value)

    return nan_before


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int], int, list[str]]:
    feature_df = df.sort_values(["truck_id", "timestamp"]).reset_index(drop=True)
    feature_df, cap_counts = cap_sensor_outliers(feature_df)

    engineered_cols: list[str] = []
    add_rolling_features(feature_df, engineered_cols)
    add_delta_lag_features(feature_df, engineered_cols)
    add_load_flags(feature_df, engineered_cols)
    add_baseline_deviations(feature_df, engineered_cols)
    add_time_features(feature_df, engineered_cols)

    imputed_nans = impute_feature_nans(feature_df, engineered_cols)
    feature_df["timestamp"] = feature_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    return feature_df, cap_counts, imputed_nans, engineered_cols


def print_summary(
    input_cols: int,
    output_cols: int,
    cap_counts: dict[str, int],
    imputed_nans: int,
    engineered_cols: list[str],
) -> None:
    total_capped = sum(cap_counts.values())
    examples = {
        "Rolling Means": [col for col in engineered_cols if "_roll" in col and "_mean" in col][:4],
        "Rolling Std": [col for col in engineered_cols if "_roll24h_std" in col][:3],
        "Deltas": [col for col in engineered_cols if "_delta1" in col][:3],
        "Lags": [col for col in engineered_cols if "_lag" in col][:4],
        "Baseline": [col for col in engineered_cols if "_dev_baseline" in col][:3],
        "Beladung": ["load_low", "load_med", "load_high"],
        "Zeit": ["hour_of_day", "day_of_week"],
    }

    print("=== Running 17_feature_engineering.py ===")
    print(f"Input:  {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Spalten: {input_cols} -> {output_cols}")
    print(f"Gecappte Werte gesamt: {total_capped:,}")
    for col, count in cap_counts.items():
        print(f"  {col:16s}: {count:,}")
    print(f"Imputierte NaNs: {imputed_nans:,}")
    print("\nBeispiel-Features:")
    for category, cols in examples.items():
        print(f"  {category:14s}: {', '.join(cols)}")
    print("\n=== Done. ===")


def main() -> None:
    df = load_input()
    input_cols = len(df.columns)
    feature_df, cap_counts, imputed_nans, engineered_cols = build_features(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(OUTPUT_PATH, index=False)

    print_summary(
        input_cols=input_cols,
        output_cols=len(feature_df.columns),
        cap_counts=cap_counts,
        imputed_nans=imputed_nans,
        engineered_cols=engineered_cols,
    )


if __name__ == "__main__":
    main()
