"""
13_simulator.py
Phase 2, Schritt 8: Python-basierter OBD-II-Telemetrie-Simulator

Dieses Skript erzeugt einen chargenweisen Telemetriedatensatz fuer 10 virtuelle
LKW auf Basis der realen Sensorverteilungen aus dem KIT/Kaggle Automotive Engine
Health Datensatz. Anders als die Schritte 06-08 modelliert es pro LKW eine
zeitliche Verschleisskurve: Beladung, Routentyp und Wetter beeinflussen die
Health-Score-Entwicklung kausal, und die Motorsensoren driften entlang dieser
Verschleisskurve von Healthy- zu Faulty-Verteilungen.

Hinweis zur Pipeline-Kompatibilitaet:
Der Output enthaelt alle Spalten, die 09_model_with_weather.py,
10_alert_demo.py, 11_summary_phase2.py, 12_rul_random_forest.py und app.py aus
engine_data_final.csv erwarten. Ab Phase 4 ist engine_data_simulated.csv die
primaere Datenquelle fuer RUL und Dashboard; die bestehenden Skripte 06-08
bleiben als dokumentierte Legacy-Anreicherung erhalten.

Methodischer Hinweis:
Die Verschleisskurven sind synthetisch und aus statischen KIT-Verteilungen
kalibriert. Der Faulty-Anteil wird absichtlich nahe an der KIT-Referenz
eingestellt, ist aber kein empirisch validierter Ausfallprozess.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


RNG_SEED = 42
N_TRUCKS = 10
N_DAYS = 90
MEASUREMENTS_PER_DAY = 48
START_TS = pd.Timestamp("2024-01-01 00:00:00")

SENSOR_COLS = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp",
]

OUTPUT_COLS = [
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

DTC_MAPPING = {
    "Engine rpm": {
        "code": "P0219",
        "german": "Motordrehzahl kritisch erhoeht",
        "action": "Motorlast reduzieren, Drehzahlregler pruefen",
    },
    "Lub oil pressure": {
        "code": "P0520",
        "german": "Schmieroeldruck kritisch niedrig",
        "action": "Motor stoppen, Oelspiegel und Oelpumpe pruefen",
    },
    "Fuel pressure": {
        "code": "P0087",
        "german": "Kraftstoffdruck zu niedrig",
        "action": "Kraftstofffilter, Leitungen und Pumpe pruefen",
    },
    "Coolant pressure": {
        "code": "P0191",
        "german": "Kuehlmitteldruck ausserhalb Normbereich",
        "action": "Kuehlsystem und Drucksensor pruefen",
    },
    "lub oil temp": {
        "code": "P0524",
        "german": "Schmieroel-Temperatur kritisch hoch",
        "action": "Motor abkuehlen lassen, Oelqualitaet pruefen",
    },
    "Coolant temp": {
        "code": "P0217",
        "german": "Motorueberhitzung erkannt",
        "action": "Sofortiger Halt, Kuehlsystem pruefen",
    },
}

ROUTE_CONFIG = {
    "Autobahn": {
        "prob": 0.42,
        "load_mean": 76,
        "load_std": 9,
        "wear_factor": 1.15,
        "km_per_step_mean": 40,
        "km_per_step_std": 6,
        "brake_factor": 0.75,
    },
    "Landstrasse": {
        "prob": 0.33,
        "load_mean": 58,
        "load_std": 11,
        "wear_factor": 1.00,
        "km_per_step_mean": 28,
        "km_per_step_std": 5,
        "brake_factor": 1.00,
    },
    "Stadtverkehr": {
        "prob": 0.25,
        "load_mean": 43,
        "load_std": 10,
        "wear_factor": 1.20,
        "km_per_step_mean": 14,
        "km_per_step_std": 4,
        "brake_factor": 1.45,
    },
}


@dataclass
class TruckState:
    truck_id: str
    health_score: float
    driving_style: float
    base_ttf_days: float
    odometer_km: float
    tire_pressure_base: float
    brake_fluid_pct: float
    base_wear_per_step: float


def load_sensor_profiles(input_path: Path) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    df = pd.read_csv(input_path)
    missing = sorted(set(SENSOR_COLS + ["Engine Condition"]) - set(df.columns))
    if missing:
        raise ValueError(f"Input-Datei ohne erwartete Spalten: {missing}")

    profiles: dict[str, pd.Series] = {}
    grouped = df.groupby("Engine Condition")[SENSOR_COLS]
    profiles["healthy_mean"] = grouped.mean().loc[1]
    profiles["healthy_std"] = grouped.std().loc[1].replace(0, 1e-6)
    profiles["faulty_mean"] = grouped.mean().loc[0]
    profiles["faulty_std"] = grouped.std().loc[0].replace(0, 1e-6)
    profiles["min"] = df[SENSOR_COLS].min()
    profiles["max"] = df[SENSOR_COLS].max()
    return df, profiles


def generate_weather(rng: np.random.Generator) -> pd.DataFrame:
    dates = pd.date_range(START_TS.date(), periods=N_DAYS, freq="D")
    day_offsets = np.arange(N_DAYS)

    temp_base = 12.5 - 12.5 * np.cos(2 * np.pi * (day_offsets - 15) / 365)
    temperature = np.clip(temp_base + rng.normal(0, 3, size=N_DAYS), -15, 38)

    has_rain = rng.random(N_DAYS) < 0.20
    precipitation = np.where(has_rain, rng.uniform(1, 15, size=N_DAYS), 0.0)

    return pd.DataFrame(
        {
            "date": dates.date,
            "temperature_c": np.round(temperature, 1),
            "precipitation_mm": np.round(precipitation, 1),
        }
    )


def sample_daily_routes(rng: np.random.Generator) -> dict[str, list[str]]:
    route_names = list(ROUTE_CONFIG)
    route_probs = np.array([ROUTE_CONFIG[name]["prob"] for name in route_names])
    route_probs = route_probs / route_probs.sum()
    routes: dict[str, list[str]] = {}

    for i in range(1, N_TRUCKS + 1):
        routes[f"LKW-{i:02d}"] = rng.choice(route_names, size=N_DAYS, p=route_probs).tolist()
    return routes


def calibrate_ttf_to_row_faulty_share(
    base_ttfs: np.ndarray,
    target_faulty_share: float,
) -> np.ndarray:
    """Scale nominal TTFs so the expected row-level faulty share matches KIT."""
    target_faulty_share = float(np.clip(target_faulty_share, 0.05, 0.85))

    def expected_share(scale: float) -> float:
        crossing_days = base_ttfs * scale
        faulty_days = np.clip(N_DAYS - crossing_days, 0, N_DAYS)
        return float(np.mean(faulty_days / N_DAYS))

    low, high = 0.05, 5.0
    for _ in range(80):
        mid = (low + high) / 2
        if expected_share(mid) > target_faulty_share:
            low = mid
        else:
            high = mid

    return base_ttfs * ((low + high) / 2)


def init_trucks(rng: np.random.Generator, target_faulty_share: float) -> list[TruckState]:
    weibull_shape = 2.0
    weibull_scale = N_DAYS / (-np.log(1 - target_faulty_share)) ** (1 / weibull_shape)
    base_ttfs = rng.weibull(weibull_shape, size=N_TRUCKS) * weibull_scale
    base_ttfs = calibrate_ttf_to_row_faulty_share(base_ttfs, target_faulty_share)

    trucks: list[TruckState] = []
    for i, base_ttf_days in enumerate(base_ttfs, start=1):
        initial_health = rng.uniform(0.88, 0.99)
        driving_style = rng.lognormal(mean=0.0, sigma=0.12)
        base_ttf_steps = max(base_ttf_days * MEASUREMENTS_PER_DAY, 1.0)
        expected_stress_multiplier = 1.45
        base_wear = (initial_health - 0.5) / (base_ttf_steps * expected_stress_multiplier)

        trucks.append(
            TruckState(
                truck_id=f"LKW-{i:02d}",
                health_score=initial_health,
                driving_style=driving_style,
                base_ttf_days=base_ttf_days,
                odometer_km=rng.uniform(85_000, 650_000),
                tire_pressure_base=rng.normal(8.7, 0.22),
                brake_fluid_pct=rng.uniform(90, 99),
                base_wear_per_step=base_wear,
            )
        )
    return trucks


def route_load(rng: np.random.Generator, route_type: str) -> float:
    cfg = ROUTE_CONFIG[route_type]
    load = rng.normal(cfg["load_mean"], cfg["load_std"])
    return float(np.clip(load, 20, 100))


def step_distance(rng: np.random.Generator, route_type: str, driving_style: float) -> float:
    cfg = ROUTE_CONFIG[route_type]
    distance = rng.normal(cfg["km_per_step_mean"], cfg["km_per_step_std"])
    return float(np.clip(distance * (0.95 + 0.06 * driving_style), 2, 55))


def update_health(
    health_score: float,
    base_wear_per_step: float,
    route_type: str,
    load_pct: float,
    precipitation_mm: float,
    driving_style: float,
    rng: np.random.Generator,
) -> float:
    cfg = ROUTE_CONFIG[route_type]
    load_factor = 0.70 + (load_pct / 100) ** 1.45
    rain_factor = 1.0 + min(precipitation_mm, 20) * 0.006
    random_factor = np.clip(rng.normal(1.0, 0.04), 0.88, 1.14)
    wear = base_wear_per_step * cfg["wear_factor"] * load_factor
    wear *= rain_factor * driving_style * random_factor
    return float(np.clip(health_score - wear, 0.0, 1.0))


def sample_engine_sensors(
    rng: np.random.Generator,
    profiles: dict[str, pd.Series],
    health_score: float,
    load_pct: float,
    temperature_c: float,
    route_type: str,
) -> dict[str, float]:
    fault_weight = 1.0 - health_score
    mean = (1 - fault_weight) * profiles["healthy_mean"] + fault_weight * profiles["faulty_mean"]
    std = (1 - fault_weight) * profiles["healthy_std"] + fault_weight * profiles["faulty_std"]

    values = rng.normal(mean.to_numpy(), std.to_numpy())
    row = dict(zip(SENSOR_COLS, values))

    load_stress = (load_pct - 55) / 45
    city_stress = 1.0 if route_type == "Stadtverkehr" else 0.0
    heat_stress = max(temperature_c - 20, 0) / 20

    row["Engine rpm"] += 45 * load_stress + 60 * city_stress
    row["Lub oil pressure"] -= 0.20 * fault_weight + 0.08 * load_stress
    row["Fuel pressure"] += 0.35 * load_stress
    row["Coolant pressure"] += 0.18 * load_stress + 0.08 * heat_stress
    row["lub oil temp"] += 2.1 * fault_weight + 1.2 * load_stress + 0.7 * heat_stress
    row["Coolant temp"] += 4.5 * fault_weight + 2.0 * load_stress + 1.2 * heat_stress

    clipped = {}
    for col in SENSOR_COLS:
        lower = profiles["min"][col]
        upper = profiles["max"][col]
        margin = 0.08 * (upper - lower)
        clipped[col] = float(np.clip(row[col], lower - margin, upper + margin))
    return clipped


def update_auxiliary_sensors(
    truck: TruckState,
    route_type: str,
    load_pct: float,
    distance_km: float,
    health_score: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    cfg = ROUTE_CONFIG[route_type]
    load_stress = max(load_pct - 50, 0) / 50
    wear_progress = 1 - health_score

    tire_pressure = truck.tire_pressure_base
    tire_pressure -= 0.000003 * (truck.odometer_km - 85_000)
    tire_pressure -= 0.16 * load_stress + 0.24 * wear_progress
    tire_pressure += rng.normal(0, 0.05)

    brake_loss = 0.000020 * distance_km * cfg["brake_factor"] * (1 + 0.35 * load_stress)
    brake_loss += 0.00055 * wear_progress
    truck.brake_fluid_pct = float(np.clip(truck.brake_fluid_pct - brake_loss, 35, 100))

    return float(np.clip(tire_pressure, 6.0, 10.2)), truck.brake_fluid_pct


def assign_dtc(row: dict[str, float], profiles: dict[str, pd.Series]) -> dict[str, object]:
    z_scores = {
        col: (row[col] - profiles["healthy_mean"][col]) / profiles["healthy_std"][col]
        for col in SENSOR_COLS
    }
    most_anomalous_feature = max(z_scores, key=lambda col: abs(z_scores[col]))
    max_z_score = float(abs(z_scores[most_anomalous_feature]))

    result: dict[str, object] = {
        "most_anomalous_feature": most_anomalous_feature,
        "max_z_score": max_z_score,
        "primary_dtc_code": "",
        "primary_dtc_german": "",
        "primary_dtc_action": "",
    }

    if max_z_score > 2.0 or row["Engine Condition"] == 0:
        dtc = DTC_MAPPING[most_anomalous_feature]
        result.update(
            {
                "primary_dtc_code": dtc["code"],
                "primary_dtc_german": dtc["german"],
                "primary_dtc_action": dtc["action"],
            }
        )
    return result


def simulate() -> pd.DataFrame:
    rng = np.random.default_rng(seed=RNG_SEED)
    base_dir = Path(__file__).resolve().parents[1]
    input_path = base_dir / "data" / "engine_health" / "engine_data.csv"
    output_path = base_dir / "data" / "engine_health" / "engine_data_simulated.csv"

    raw_df, profiles = load_sensor_profiles(input_path)
    weather_df = generate_weather(rng)
    weather_by_date = weather_df.set_index("date").to_dict("index")
    daily_routes = sample_daily_routes(rng)
    source_faulty_share = float((raw_df["Engine Condition"] == 0).mean())
    trucks = init_trucks(rng, source_faulty_share)

    rows: list[dict[str, object]] = []
    for truck in trucks:
        for step in range(N_DAYS * MEASUREMENTS_PER_DAY):
            timestamp = START_TS + pd.Timedelta(minutes=30 * step)
            day_idx = step // MEASUREMENTS_PER_DAY
            date_key = timestamp.date()
            route_type = daily_routes[truck.truck_id][day_idx]
            weather = weather_by_date[date_key]

            load_pct = route_load(rng, route_type)
            distance_km = step_distance(rng, route_type, truck.driving_style)
            truck.odometer_km += distance_km

            truck.health_score = update_health(
                truck.health_score,
                truck.base_wear_per_step,
                route_type,
                load_pct,
                weather["precipitation_mm"],
                truck.driving_style,
                rng,
            )

            engine_values = sample_engine_sensors(
                rng,
                profiles,
                truck.health_score,
                load_pct,
                weather["temperature_c"],
                route_type,
            )
            tire_pressure, brake_fluid = update_auxiliary_sensors(
                truck,
                route_type,
                load_pct,
                distance_km,
                truck.health_score,
                rng,
            )

            row: dict[str, object] = {
                "truck_id": truck.truck_id,
                "timestamp": timestamp,
                "route_type": route_type,
                "load_pct": round(load_pct, 1),
                **engine_values,
                "tire_pressure_bar": round(tire_pressure, 2),
                "brake_fluid_pct": round(brake_fluid, 2),
                "odometer_km": round(truck.odometer_km, 1),
                "temperature_c": weather["temperature_c"],
                "precipitation_mm": weather["precipitation_mm"],
                "health_score": round(truck.health_score, 4),
                "Engine Condition": 1 if truck.health_score > 0.5 else 0,
            }
            row.update(assign_dtc(row, profiles))
            rows.append(row)

    simulated = pd.DataFrame(rows, columns=OUTPUT_COLS)
    simulated.to_csv(output_path, index=False)
    print_summary(simulated, raw_df, trucks, output_path)
    return simulated


def print_summary(
    simulated: pd.DataFrame,
    raw_df: pd.DataFrame,
    trucks: list[TruckState],
    output_path: Path,
) -> None:
    faulty_share = (simulated["Engine Condition"] == 0).mean()
    source_faulty_share = (raw_df["Engine Condition"] == 0).mean()
    dtc_count = simulated["primary_dtc_code"].ne("").sum()
    final_faulty_trucks = (
        simulated.sort_values("timestamp").groupby("truck_id").tail(1)["Engine Condition"].eq(0).sum()
    )

    print("=== Running 13_simulator.py ===")
    print(f"Output: {output_path}")
    print(f"Anzahl Zeilen: {len(simulated):,}")
    print(f"Anzahl LKW: {simulated['truck_id'].nunique()}")
    print(f"Zeitraum: {simulated['timestamp'].min()} bis {simulated['timestamp'].max()}")
    print(
        f"Faulty-Anteil simuliert: {faulty_share:.1%} "
        f"(KIT-Referenz: {source_faulty_share:.1%})"
    )
    print("Kalibrierung: Zeilen-Faulty-Anteil per TTF-Skalierung auf KIT-Referenz gezogen.")
    print(f"Faulty-LKW am Simulationsende: {final_faulty_trucks}/{N_TRUCKS}")
    print(f"DTC-Vergaben: {dtc_count:,} ({dtc_count / len(simulated):.1%})")

    print("\nMittelwerte je Truck: erster Tag vs. letzter Tag")
    sorted_df = simulated.sort_values("timestamp").copy()
    first_day = (
        sorted_df.groupby("truck_id")
        .head(MEASUREMENTS_PER_DAY)
        .groupby("truck_id")
        .agg(
            health_start=("health_score", "mean"),
            rpm_start=("Engine rpm", "mean"),
            oil_temp_start=("lub oil temp", "mean"),
        )
    )
    last_day = (
        sorted_df.groupby("truck_id")
        .tail(MEASUREMENTS_PER_DAY)
        .groupby("truck_id")
        .agg(
            health_end=("health_score", "mean"),
            rpm_end=("Engine rpm", "mean"),
            oil_temp_end=("lub oil temp", "mean"),
            load_end=("load_pct", "mean"),
            max_z_end=("max_z_score", "mean"),
        )
    )
    first_last = first_day.join(last_day)
    first_last = first_last.round(
        {
            "health_start": 3,
            "health_end": 3,
            "rpm_start": 1,
            "rpm_end": 1,
            "oil_temp_start": 1,
            "oil_temp_end": 1,
            "load_end": 1,
            "max_z_end": 2,
        }
    )
    ttf_lookup = {truck.truck_id: truck.base_ttf_days for truck in trucks}
    first_last["base_ttf_days"] = [
        round(ttf_lookup[truck_id], 1) for truck_id in first_last.index
    ]
    print(first_last.to_string())

    print("\nDTC-Verteilung:")
    dtc_counts = simulated.loc[simulated["primary_dtc_code"].ne(""), "primary_dtc_code"]
    if dtc_counts.empty:
        print("  Keine DTC-Vergaben")
    else:
        for code, count in dtc_counts.value_counts().items():
            print(f"  {code}: {count:,}")

    print("\n=== Done. ===")


if __name__ == "__main__":
    simulate()
