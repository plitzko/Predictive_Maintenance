"""
Adapter: Pipeline-Daten → PREMA-CSVs

Liest bevorzugt engine_data_with_rul.csv, faellt auf engine_data_simulated.csv
und zuletzt auf engine_data_final.csv zurueck. Schreibt die vier CSV-Dateien,
die PREMA erwartet: fleet.csv, timeseries.csv, alerts.csv, truck_alerts.csv

Aufruf:
    python generate_from_tracking.py
Oder automatisch aus load_data() in app.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_PIPELINE_DATA = _HERE.parent / "pipeline" / "data" / "engine_health"

SOURCE_CANDIDATES = [
    _PIPELINE_DATA / "engine_data_with_rul.csv",
    _PIPELINE_DATA / "engine_data_simulated.csv",
    _PIPELINE_DATA / "engine_data_final.csv",
]

OUTPUT_DIR = _HERE

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

# Fuer Simulator-Daten: Snapshot-Fenster, das eine gute Demo-Verteilung liefert.
# Tag 53 von 90 ergibt ~2 KRITISCH, 4 WARNUNG, 4 OK.
# 53 Tage * 48 Messungen/Tag = 2544 Schritte pro LKW.
_SNAPSHOT_STEPS = 53 * 48

# ---------------------------------------------------------------------------
# Statische Mappings
# ---------------------------------------------------------------------------
DRIVER_MAP = {
    "LKW-01": "M. Weber",  "LKW-02": "J. Bauer",  "LKW-03": "S. Klein",
    "LKW-04": "T. Holm",   "LKW-05": "A. Graf",    "LKW-06": "P. Roth",
    "LKW-07": "K. Lang",   "LKW-08": "F. Wolf",    "LKW-09": "R. Vogel",
    "LKW-10": "H. Stein",
}

# Nur fuer Fallback engine_data_final.csv benoetigt
_START_KM = {
    "LKW-01": 287_400, "LKW-02": 198_650, "LKW-03": 142_300,
    "LKW-04":  98_120, "LKW-05": 215_870, "LKW-06": 156_440,
    "LKW-07": 234_910, "LKW-08": 112_580, "LKW-09": 178_220,
    "LKW-10": 134_750,
}
_KM_PER_STEP: dict[str, float] = {
    "Autobahn": 50.0, "Landstrasse": 30.0, "Stadtverkehr": 15.0
}

_STATUS_ORDER = {"KRITISCH": 0, "WARNUNG": 1, "OK": 2}
_REVERSE_STATUS = {v: k for k, v in _STATUS_ORDER.items()}

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _classify_status_hs(health_score: float) -> str:
    """Status basierend auf health_score (bevorzugt fuer Simulator-Daten)."""
    if health_score < 0.25:
        return "KRITISCH"
    if health_score < 0.65:
        return "WARNUNG"
    return "OK"


def _classify_status_fallback(condition: int, z_score: float) -> str:
    """Status aus Engine Condition + z-Score (Fallback fuer engine_data_final.csv)."""
    if condition == 0 and z_score > 2.5:
        return "KRITISCH"
    if condition == 0 or z_score > 1.5:
        return "WARNUNG"
    return "OK"


def _rul_hours_from_condition(g: pd.DataFrame) -> pd.Series:
    """Schaetzt RUL (h) aus dem rollenden Fehler-Anteil (Fallback)."""
    faulty_roll = (1 - g["Engine Condition"]).rolling(48, min_periods=1).mean()
    return ((1 - faulty_roll) * 2400).clip(24, 2400).round(0).astype(int)


def _brake_fluid_from_condition(g: pd.DataFrame) -> pd.Series:
    """Schaetzt Bremsfluessigkeit aus dem rollenden Fehler-Anteil (Fallback)."""
    faulty_ratio = (1 - g["Engine Condition"]).rolling(48, min_periods=1).mean()
    return (90 - faulty_ratio * 80).clip(5, 95).round(1)


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------

def main() -> None:
    src_path = None
    for candidate in SOURCE_CANDIDATES:
        if candidate.exists():
            src_path = candidate
            break

    if src_path is None:
        print("[generate_from_tracking] Keine Quelldatei gefunden – uebersprungen.")
        return

    print(f"[generate_from_tracking] Lese {src_path.name}")
    src = pd.read_csv(src_path, parse_dates=["timestamp"])
    src = src.sort_values(["truck_id", "timestamp"]).reset_index(drop=True)

    has_health_score    = "health_score" in src.columns
    has_direct_sensors  = "tire_pressure_bar" in src.columns
    has_odometer        = "odometer_km" in src.columns
    has_rul_predicted   = "rul_predicted" in src.columns

    # Snapshot-Fenster: nur erste _SNAPSHOT_STEPS Zeilen pro LKW verwenden,
    # damit die Demo-Verteilung (2-3 KRITISCH, ~4 WARNUNG, ~4 OK) stimmt.
    if has_health_score:
        src = src.groupby("truck_id").head(_SNAPSHOT_STEPS).reset_index(drop=True)

    # Timestamps relativ zu jetzt verschieben (letzter Datenpunkt = jetzt)
    delta = pd.Timestamp.now().floor("30min") - src["timestamp"].max()
    src["timestamp"] = src["timestamp"] + delta

    rng = np.random.default_rng(42)

    # ── Sensor-Spalten normalisieren ──────────────────────────────────────────
    if has_health_score:
        # Bremsfluessigkeit aus health_score skalieren: health=1 → 95%, health=0 → 5%
        # Damit zeigen degradierte LKW visuell niedrige Bremsfl.-Werte im Dashboard.
        src["brake_fluid_pct"] = (src["health_score"] * 90 + 5).clip(5, 95).round(1)
    elif "brake_fluid_pct" not in src.columns:
        src["brake_fluid_pct"] = src.groupby("truck_id", group_keys=False).apply(
            _brake_fluid_from_condition, include_groups=False
        )

    if has_direct_sensors:
        src["tire_fl_bar"] = (src["tire_pressure_bar"] + rng.normal(0, 0.05, len(src))).clip(6.0, 10.2).round(1)
        src["tire_fr_bar"] = (src["tire_pressure_bar"] + rng.normal(0, 0.05, len(src))).clip(6.0, 10.2).round(1)
    else:
        src["tire_fl_bar"] = (8.0 + rng.normal(0, 0.15, len(src))).clip(7.0, 9.0).round(1)
        src["tire_fr_bar"] = (8.0 + rng.normal(0, 0.15, len(src))).clip(7.0, 9.0).round(1)

    if has_odometer:
        src["km_total"] = src["odometer_km"].round(0).astype(int)
    else:
        src["km_total"] = 0
        for tid, grp in src.groupby("truck_id"):
            start = _START_KM.get(tid, 100_000)
            step_km = grp["route_type"].map(_KM_PER_STEP).fillna(30.0)
            src.loc[grp.index, "km_total"] = (start + step_km.cumsum()).astype(int)

    if has_rul_predicted:
        src["rul_hours"] = (src["rul_predicted"] * 24).clip(12, 2400).round(0).astype(int)
    else:
        src["rul_hours"] = src.groupby("truck_id", group_keys=False).apply(
            _rul_hours_from_condition, include_groups=False
        )

    src["lkw_id"]           = src["truck_id"]
    src["driver"]           = src["truck_id"].map(DRIVER_MAP)
    src["motor_temp_c"]     = src["Coolant temp"].round(1)
    src["oil_pressure_bar"] = src["Lub oil pressure"].round(2)
    src["max_z_abs"]        = src["max_z_score"].abs().fillna(0)

    if has_health_score:
        src["status"] = src["health_score"].map(_classify_status_hs)
    else:
        src["status"] = src.apply(
            lambda r: _classify_status_fallback(r["Engine Condition"], r["max_z_abs"]), axis=1
        )

    # ── fleet.csv: schlechtester Status der letzten 48 Schritte (= 24 h) je LKW
    FLEET_COLS = [
        "lkw_id", "driver", "status", "motor_temp_c", "brake_fluid_pct",
        "oil_pressure_bar", "tire_fl_bar", "tire_fr_bar", "rul_hours",
        "km_total", "load_pct",
    ]
    fleet_rows = []
    for _, grp in src.groupby("truck_id"):
        recent = grp.tail(48)
        worst_order = recent["status"].map(_STATUS_ORDER).min()
        last_row = grp.iloc[-1].copy()
        last_row["status"] = _REVERSE_STATUS[worst_order]
        fleet_rows.append(last_row[FLEET_COLS])

    pd.DataFrame(fleet_rows).to_csv(OUTPUT_DIR / "fleet.csv", index=False)

    # ── timeseries.csv: letzte 144 Schritte je LKW = 72 h ────────────────────
    ts_rows = src.groupby("truck_id", group_keys=False).tail(144)
    ts_rows[["lkw_id", "timestamp", "brake_fluid_pct", "motor_temp_c"]].to_csv(
        OUTPUT_DIR / "timeseries.csv", index=False
    )

    # ── alerts.csv: kritische/warnende Ereignisse der letzten 7 Tage ──────────
    cutoff_7d = src["timestamp"].max() - pd.Timedelta(days=7)
    if has_health_score:
        fault_mask = (src["timestamp"] >= cutoff_7d) & (src["health_score"] < 0.65)
    else:
        fault_mask = (
            (src["timestamp"] >= cutoff_7d) &
            ((src["Engine Condition"] == 0) | (src["max_z_abs"] > 1.5))
        )
    fault = src[fault_mask].copy()
    fault["severity"] = (
        fault["health_score"].map(_classify_status_hs)
        if has_health_score
        else fault.apply(lambda r: _classify_status_fallback(r["Engine Condition"], r["max_z_abs"]), axis=1)
    )
    fault["message"] = fault.apply(
        lambda r: (
            r["primary_dtc_german"]
            if pd.notna(r.get("primary_dtc_german")) and str(r.get("primary_dtc_german", "")).strip()
            else f"Anomalie: {r['most_anomalous_feature']}"
        ),
        axis=1,
    )
    fault["source"] = fault["severity"].map({
        "KRITISCH": "XGBoost-Klassifikation",
        "WARNUNG":  "XGBoost-Klassifikation",
        "INFO":     "Isolation Forest",
    }).fillna("Isolation Forest")
    fault["savings_eur"] = fault["severity"].map(
        {"KRITISCH": 600, "WARNUNG": 400, "INFO": 200}
    ).fillna(200).astype(int)

    fault.sort_values("timestamp", ascending=False)[
        ["timestamp", "severity", "lkw_id", "message", "source", "savings_eur"]
    ].head(50).to_csv(OUTPUT_DIR / "alerts.csv", index=False)

    # ── truck_alerts.csv: letzte 30 Tage je LKW (fuer Detail-View) ───────────
    cutoff_30d = src["timestamp"].max() - pd.Timedelta(days=30)
    if has_health_score:
        truck_mask = (src["timestamp"] >= cutoff_30d) & (src["health_score"] < 0.5)
    else:
        truck_mask = (
            (src["timestamp"] >= cutoff_30d) &
            (src["Engine Condition"] == 0)
        )
    truck_fault = src[truck_mask].copy()
    truck_fault["severity"] = (
        truck_fault["health_score"].map(_classify_status_hs)
        if has_health_score
        else truck_fault.apply(lambda r: _classify_status_fallback(r["Engine Condition"], r["max_z_abs"]), axis=1)
    )
    truck_fault["message"] = truck_fault.apply(
        lambda r: (
            r["primary_dtc_german"]
            if pd.notna(r.get("primary_dtc_german")) and str(r.get("primary_dtc_german", "")).strip()
            else f"Anomalie: {r['most_anomalous_feature']}"
        ),
        axis=1,
    )
    truck_fault.sort_values("timestamp", ascending=False)[
        ["timestamp", "lkw_id", "severity", "message"]
    ].head(200).to_csv(OUTPUT_DIR / "truck_alerts.csv", index=False)

    print("[generate_from_tracking] fleet.csv, timeseries.csv, alerts.csv, truck_alerts.csv generiert.")


if __name__ == "__main__":
    main()
