"""
Adapter: Pipeline-Daten -> PREMA-CSVs

Liest bevorzugt engine_data_with_rul.csv, faellt auf engine_data_simulated.csv
und zuletzt auf engine_data_final.csv zurueck. Schreibt die Dateien, die das
Dashboard erwartet:

    fleet.csv         Status-Snapshot je LKW (geglaettet ueber 24 h)
    timeseries.csv    Sensor-Zeitreihen der letzten 72 h je LKW
    alerts.csv        Alert-Feed der letzten 7 Tage (Ereignisse, dedupliziert)
    truck_alerts.csv  Alert-Historie der letzten 30 Tage je LKW
    replay.csv        Zukunfts-Messungen nach dem Snapshot (Live-Demo-Modus)
    replay_alerts.csv Alert-Ereignisse im Replay-Zeitraum (Live-Demo-Modus)
    metrics.json      ML-Metriken aus den Pipeline-Reports (falls vorhanden)

Alert-Logik: Ein Alert wird nur bei einem Severity-Wechsel je LKW erzeugt
(Zustandsuebergang), mit einer Re-Arm-Sperre von 6 h pro Severity-Stufe.
Damit entsteht ein lesbarer Feed statt einer Meldung pro 30-Minuten-Messung.

Severity-Quellen (FA-3, FA-4, FA-5):
    KRITISCH / WARNUNG  -> XGBoost-Klassifikation (health_score-Schwellen)
    INFO                -> Isolation Forest (Anomalie bei Status OK)

Aufruf:
    python generate_from_tracking.py
Oder automatisch aus load_data() in app.py.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_PIPELINE_DATA = _HERE.parent / "pipeline" / "data" / "engine_health"
_PIPELINE_OUTPUTS = _HERE.parent / "pipeline" / "outputs"

SOURCE_CANDIDATES = [
    _PIPELINE_DATA / "engine_data_with_rul.csv",
    _PIPELINE_DATA / "engine_data_simulated.csv",
    _PIPELINE_DATA / "engine_data_final.csv",
]

OUTPUT_DIR = _HERE

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

# Snapshot-Fenster fuer Simulator-Daten (Tag X von 90, 48 Messungen/Tag).
# Tag 49 ist der fruehste Tag, an dem LKW-01 (Demo-Szenario FA-7) KRITISCH
# ist. Verteilung dort: 5 KRITISCH, 3 WARNUNG, 2 OK.
_SNAPSHOT_DAY = 49
_SNAPSHOT_STEPS = _SNAPSHOT_DAY * 48

# Isolation-Forest-Anomalie: |z| oberhalb dieser Schwelle bei Status OK
# erzeugt einen INFO-Alert. 3.2 = empirisches 99%-Quantil der OK-Zeilen,
# d.h. nur das ungewoehnlichste 1 % der Normalmessungen gilt als Anomalie.
# Für Replay-Bereich (Zukunfts-Messungen) lockerer einstellen (2.5) damit die
# Demo lebendiger wirkt und mehr sichtbare Aktivität zeigt.
_ANOMALY_Z_INFO = 3.2
_ANOMALY_Z_INFO_REPLAY = 2.5

# Re-Arm-Sperre: gleicher LKW + gleiche Severity fruehestens nach 6 h erneut.
_ALERT_REARM = pd.Timedelta(hours=6)

# Geschaetzte Einsparung in EUR pro Stunde vermiedener Standzeit (FA-5).
_SAVINGS_EUR_H = {"KRITISCH": 600, "WARNUNG": 400, "INFO": 0}

_GENERIC_RECOMMENDATION = {
    "KRITISCH": "Fahrzeug aus dem Verkehr ziehen, Werkstatt sofort",
    "WARNUNG": "Wartungstermin innerhalb 14 Tagen einplanen",
    "INFO": "Beobachten, keine sofortige Massnahme noetig",
}

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

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _classify_status_hs(health_score: float) -> str:
    """Status basierend auf health_score - identische Grenzen wie 16_classification_xgboost.py."""
    if health_score <= 0.5:
        return "KRITISCH"
    if health_score <= 0.7:
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


def _weather_label(temp_c, precip_mm) -> str:
    """Leitet eine lesbare Wetterlage aus Temperatur und Niederschlag ab (FA-2)."""
    if pd.isna(temp_c):
        return ""
    if pd.notna(precip_mm) and precip_mm > 0:
        return "Schneefall" if temp_c <= 0.5 else "Regen"
    return "Frost" if temp_c <= 0 else "Trocken"


def _build_alert_events(src: pd.DataFrame) -> pd.DataFrame:
    """Erzeugt deduplizierte Alert-Ereignisse aus den klassifizierten Messreihen.

    Ein Ereignis entsteht beim Wechsel der Severity eines LKW (Zustands-
    uebergang). Gleiche Severity desselben LKW wird fruehestens nach
    _ALERT_REARM erneut gemeldet.
    """
    events = []
    for _, grp in src.groupby("truck_id"):
        prev_sev = None
        last_emit: dict[str, pd.Timestamp] = {}
        for row in grp.itertuples():
            sev = row.alert_severity
            if not isinstance(sev, str):
                prev_sev = None
                continue
            changed = sev != prev_sev
            prev_sev = sev
            if not changed:
                continue
            last = last_emit.get(sev)
            if last is not None and row.timestamp - last < _ALERT_REARM:
                continue
            last_emit[sev] = row.timestamp
            events.append(row.Index)
    return src.loc[events].copy()


def _parse_report_metrics() -> dict:
    """Liest ML-Metriken aus den Text-Reports der Pipeline (FA-6)."""
    metrics: dict = {}
    patterns = {
        _PIPELINE_OUTPUTS / "13_rul_random_forest.txt": {
            "rul_mae_days": r"MAE gesamt\s*:\s*([\d.]+)",
            "rul_r2": r"R\^2\s*:\s*([\d.]+)",
        },
        _PIPELINE_OUTPUTS / "16_classification_xgboost.txt": {
            "clf_accuracy": r"Accuracy\s*:\s*([\d.]+)",
            "clf_macro_f1": r"Macro-F1\s*:\s*([\d.]+)",
            "clf_recall_critical": r"Recall KRITISCH\s*:\s*([\d.]+)",
        },
    }
    for path, keys in patterns.items():
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for key, rx in keys.items():
            m = re.search(rx, text)
            if m:
                metrics[key] = float(m.group(1))
    if (_PIPELINE_OUTPUTS / "15_anomaly_isolation_forest.txt").exists():
        metrics["anomaly_model"] = "Isolation Forest (kalibriert, Baseline 14 Tage)"
    return metrics


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
        print("[generate_from_tracking] Keine Quelldatei gefunden - uebersprungen.")
        return

    print(f"[generate_from_tracking] Lese {src_path.name}")
    src = pd.read_csv(src_path, parse_dates=["timestamp"])
    src = src.sort_values(["truck_id", "timestamp"]).reset_index(drop=True)

    has_health_score    = "health_score" in src.columns
    has_direct_sensors  = "tire_pressure_bar" in src.columns
    has_odometer        = "odometer_km" in src.columns
    has_rul_predicted   = "rul_predicted" in src.columns
    has_weather         = "temperature_c" in src.columns

    # Snapshot-Grenze: Tag _SNAPSHOT_DAY ist "jetzt". Zeilen danach werden
    # nicht verworfen, sondern speisen den Live-Replay-Modus des Dashboards
    # (replay.csv / replay_alerts.csv).
    if has_health_score:
        snap_mask = src.groupby("truck_id").cumcount() < _SNAPSHOT_STEPS
    else:
        snap_mask = pd.Series(True, index=src.index)

    # Timestamps relativ zu jetzt verschieben (letzter Snapshot-Datenpunkt
    # = jetzt; alles danach liegt bewusst in der Zukunft).
    delta = pd.Timestamp.now().floor("30min") - src.loc[snap_mask, "timestamp"].max()
    src["timestamp"] = src["timestamp"] + delta
    now_anchor = src.loc[snap_mask, "timestamp"].max()

    rng = np.random.default_rng(42)

    # -- Sensor-Spalten normalisieren ----------------------------------------
    if has_health_score:
        # Bremsfluessigkeit aus health_score skalieren: health=1 -> 95%, health=0 -> 5%
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
    src["driver"]           = src["truck_id"].map(DRIVER_MAP).fillna("n. n.")
    src["motor_temp_c"]     = src["Coolant temp"].round(1)
    src["oil_pressure_bar"] = src["Lub oil pressure"].round(2)
    src["max_z_abs"]        = src["max_z_score"].abs().fillna(0)

    if has_health_score:
        src["status"] = src["health_score"].map(_classify_status_hs)
    else:
        src["status"] = src.apply(
            lambda r: _classify_status_fallback(r["Engine Condition"], r["max_z_abs"]), axis=1
        )

    # Alert-Severity je Messzeile (FA-3 + FA-4):
    #   KRITISCH/WARNUNG aus der Klassifikation, INFO aus dem Isolation
    #   Forest (Anomalie trotz Status OK), sonst keine Meldung (NaN).
    src["alert_severity"] = src["status"].where(src["status"] != "OK")
    # Snapshot: strenge Schwelle (3.2 = 99%-Quantil)
    info_mask = (src["status"] == "OK") & (src["max_z_abs"] > _ANOMALY_Z_INFO) & snap_mask
    src.loc[info_mask, "alert_severity"] = "INFO"
    # Replay: lockere Schwelle (2.5) für lebendiger wirkende Demo
    if has_health_score:
        info_mask_replay = (src["status"] == "OK") & (src["max_z_abs"] > _ANOMALY_Z_INFO_REPLAY) & (~snap_mask)
        src.loc[info_mask_replay, "alert_severity"] = "INFO"

    # -- fleet.csv: Status je LKW geglaettet ueber die letzten 48 Schritte ----
    # Median des health_scores der letzten 24 h verhindert, dass ein einzelner
    # Ausreisser-Messwert ein Fahrzeug auf KRITISCH eskaliert.
    FLEET_COLS = [
        "lkw_id", "driver", "status", "motor_temp_c", "brake_fluid_pct",
        "oil_pressure_bar", "tire_fl_bar", "tire_fr_bar", "rul_hours",
        "km_total", "load_pct",
    ]
    snap = src[snap_mask]
    fleet_rows = []
    for _, grp in snap.groupby("truck_id"):
        recent = grp.tail(48)
        last_row = grp.iloc[-1].copy()
        if has_health_score:
            last_row["status"] = _classify_status_hs(recent["health_score"].median())
        else:
            worst = recent["status"].map(_STATUS_ORDER).min()
            last_row["status"] = {v: k for k, v in _STATUS_ORDER.items()}[worst]
        last_row["rul_hours"] = int(recent["rul_hours"].median())
        fleet_rows.append(last_row[FLEET_COLS])

    pd.DataFrame(fleet_rows).to_csv(OUTPUT_DIR / "fleet.csv", index=False)

    # -- timeseries.csv: letzte 144 Schritte je LKW = 72 h --------------------
    ts_rows = snap.groupby("truck_id", group_keys=False).tail(144)
    ts_rows[["lkw_id", "timestamp", "brake_fluid_pct", "motor_temp_c", "oil_pressure_bar"]].to_csv(
        OUTPUT_DIR / "timeseries.csv", index=False
    )

    # -- Alert-Ereignisse (dedupliziert) fuer Feed und Historie ---------------
    events = _build_alert_events(src)

    def _message(r) -> str:
        if r["alert_severity"] == "INFO":
            feat = r.get("most_anomalous_feature", "")
            return f"Anomalie erkannt: {feat} (z={r['max_z_abs']:.1f})"
        dtc = r.get("primary_dtc_german")
        if pd.notna(dtc) and str(dtc).strip():
            return str(dtc)
        feat = r.get("most_anomalous_feature", "Sensorwerte")
        return f"Auffaelliges Muster: {feat}"

    def _recommendation(r) -> str:
        action = r.get("primary_dtc_action")
        if r["alert_severity"] != "INFO" and pd.notna(action) and str(action).strip():
            return str(action)
        return _GENERIC_RECOMMENDATION[r["alert_severity"]]

    events["severity"] = events["alert_severity"]
    events["message"] = events.apply(_message, axis=1)
    events["recommendation"] = events.apply(_recommendation, axis=1)
    events["source"] = np.where(
        events["severity"] == "INFO", "Isolation Forest", "XGBoost-Klassifikation"
    )
    events["savings_eur"] = events["severity"].map(_SAVINGS_EUR_H).astype(int)
    if "primary_dtc_code" in events.columns:
        events["dtc_code"] = events["primary_dtc_code"].fillna("")
        events.loc[events["severity"] == "INFO", "dtc_code"] = ""
    else:
        events["dtc_code"] = ""
    if has_weather:
        events["temperature_c"] = events["temperature_c"].round(1)
        events["weather"] = events.apply(
            lambda r: _weather_label(r["temperature_c"], r.get("precipitation_mm")), axis=1
        )
    else:
        events["temperature_c"] = np.nan
        events["weather"] = ""
    if "route_type" not in events.columns:
        events["route_type"] = ""
    events["load_pct"] = events.get("load_pct", pd.Series(np.nan, index=events.index)).round(0)

    ALERT_COLS = [
        "timestamp", "severity", "lkw_id", "message", "source", "savings_eur",
        "dtc_code", "recommendation", "temperature_c", "weather", "load_pct", "route_type",
    ]

    # alerts.csv: Feed der letzten 7 Tage (nur bis zum Snapshot-Zeitpunkt)
    cutoff_7d = now_anchor - pd.Timedelta(days=7)
    feed = events[(events["timestamp"] >= cutoff_7d) & (events["timestamp"] <= now_anchor)]
    feed.sort_values("timestamp", ascending=False)[ALERT_COLS].head(120).to_csv(
        OUTPUT_DIR / "alerts.csv", index=False
    )

    # truck_alerts.csv: Historie der letzten 30 Tage je LKW (fuer Detail-View)
    cutoff_30d = now_anchor - pd.Timedelta(days=30)
    history = events[(events["timestamp"] >= cutoff_30d) & (events["timestamp"] <= now_anchor)]
    history.sort_values("timestamp", ascending=False)[
        ["timestamp", "lkw_id", "severity", "message", "dtc_code", "recommendation"]
    ].head(400).to_csv(OUTPUT_DIR / "truck_alerts.csv", index=False)

    # -- Replay-Daten: Messungen nach dem Snapshot fuer den Live-Demo-Modus ---
    # Status und RUL werden wie in fleet.csv ueber 24 h (48 Schritte) geglaettet,
    # damit der Replay nahtlos am statischen Snapshot ansetzt.
    REPLAY_COLS = ["lkw_id", "timestamp", "status", "brake_fluid_pct",
                   "motor_temp_c", "oil_pressure_bar", "rul_hours"]
    future = src[~snap_mask]
    if has_health_score and not future.empty:
        roll_health = (src.groupby("truck_id")["health_score"]
                       .rolling(48, min_periods=1).median()
                       .reset_index(level=0, drop=True))
        roll_rul = (src.groupby("truck_id")["rul_hours"]
                    .rolling(48, min_periods=1).median()
                    .reset_index(level=0, drop=True))
        replay = future[["lkw_id", "timestamp", "brake_fluid_pct",
                         "motor_temp_c", "oil_pressure_bar"]].copy()
        replay["status"] = roll_health.loc[future.index].map(_classify_status_hs)
        replay["rul_hours"] = roll_rul.loc[future.index].round(0).astype(int)
        replay[REPLAY_COLS].to_csv(OUTPUT_DIR / "replay.csv", index=False)
        events[events["timestamp"] > now_anchor].sort_values("timestamp")[ALERT_COLS].to_csv(
            OUTPUT_DIR / "replay_alerts.csv", index=False
        )
    else:
        pd.DataFrame(columns=REPLAY_COLS).to_csv(OUTPUT_DIR / "replay.csv", index=False)
        pd.DataFrame(columns=ALERT_COLS).to_csv(OUTPUT_DIR / "replay_alerts.csv", index=False)

    # -- metrics.json: ML-Guete aus den Pipeline-Reports (FA-6) ---------------
    metrics = _parse_report_metrics()
    metrics.update({
        "source_file": src_path.name,
        "snapshot_day": _SNAPSHOT_DAY if has_health_score else None,
        "n_trucks": int(src["truck_id"].nunique()),
        "n_rows": int(snap_mask.sum()),
        "n_replay_rows": int((~snap_mask).sum()),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    })
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    n_feed = len(feed)
    print(f"[generate_from_tracking] fleet.csv, timeseries.csv, alerts.csv "
          f"({n_feed} Ereignisse/7d), truck_alerts.csv, metrics.json generiert.")


if __name__ == "__main__":
    main()
