# PREMA – Predictive Maintenance Dashboard

Big-Data-Projekt, SS2026, Hochschule Muenchen, Team 1.

PREMA baut ein Predictive-Maintenance-MVP fuer LKW-Flotten. Es simuliert
OBD-II-nahe Telemetriedaten, erkennt kritische Zustaende, ordnet DTC-Codes zu
und schaetzt eine Remaining Useful Life (RUL) pro Fahrzeug. Das Ergebnis ist
ein interaktives Streamlit-Dashboard mit drei Ansichten: Flottenübersicht,
Fahrzeugdetail und Alert-Feed.

## Architektur

Zwei parallele Arbeitsstroeme wurden zusammengefuehrt:

- **Pipeline** (`pipeline/`): ML-Pipeline aus dem tracking-Teilprojekt —
  Simulator, Feature Engineering, RUL-Modell, Anomalie- und Klassifikations-
  Skripte.
- **Dashboard** (`app.py`): Streamlit-UI aus dem Predictive_Maintenance-
  Teilprojekt — drei Screens (Flotte, Fahrzeugdetail, Alerts), URL-Navigation,
  Custom CSS.

Der Adapter `data/generate_from_tracking.py` verbindet beide: er liest die
Pipeline-Ausgabe und schreibt die vier CSVs, die das Dashboard erwartet.

## Datenpfad (Phase 4)

```text
13_simulator.py
    → pipeline/data/engine_health/engine_data_simulated.csv
17_feature_engineering.py
    → pipeline/data/engine_health/engine_data_features.csv
12_rul_random_forest.py
    → pipeline/data/engine_health/engine_data_with_rul.csv
    → pipeline/outputs/models/rul_random_forest_v1.pkl
data/generate_from_tracking.py
    → data/fleet.csv
    → data/timeseries.csv
    → data/alerts.csv
    → data/truck_alerts.csv
app.py  (liest die vier CSVs, ruft Adapter beim Start automatisch auf)
```

Der Adapter bevorzugt `engine_data_with_rul.csv`, faellt auf
`engine_data_simulated.csv` und zuletzt auf `engine_data_final.csv` zurueck.
Fuer das Dashboard-Snapshot wird Tag 49 der Simulation gewaehlt: der
fruehste Tag, an dem LKW-01 (Demo-Szenario FA-7) KRITISCH ist.
Verteilung dort: 5 KRITISCH, 3 WARNUNG, 2 OK.

Der Flotten-Status je LKW wird ueber den Median des health_scores der
letzten 24 h geglaettet, damit einzelne Ausreisser-Messungen kein Fahrzeug
auf KRITISCH eskalieren.

### Alert-Logik

Ein Alert entsteht nur beim Severity-Wechsel eines LKW (Zustandsuebergang),
mit 6 h Re-Arm-Sperre pro Stufe — statt einer Meldung pro 30-Minuten-Messung.
Severity-Quellen:

- **KRITISCH / WARNUNG** — XGBoost-Klassifikation (health_score-Schwellen
  0.5 / 0.7), Meldung aus dem DTC-Mapping inkl. Code und
  Handlungsempfehlung.
- **INFO** — Isolation Forest: |z| > 2.5 bei Status OK.

Jeder Alert traegt Kontext aus der Anreicherung (FA-2): Temperatur,
Wetterlage, Beladung und Routentyp.

Zusaetzlich schreibt der Adapter `data/metrics.json` mit den ML-Metriken
aus den Pipeline-Reports (Accuracy, Recall KRITISCH, RUL-MAE, R²); das
Dashboard zeigt sie in der Flottenansicht unter "ML-Modellguete" (FA-6).

## Was die ML-Skripte messen (und was nicht)

- **Skript 12 (Random Forest, RUL)** schaetzt Tage bis zum Health-Score-
  Unterschreiten der Schadensschwelle. `health_score` ist als Feature dabei,
  was die Modell-Performance optimistisch ueberzeichnet (Tautologie-Risiko).
- **Skript 15 (Isolation Forest, Anomalien)** misst Abweichung vom
  Normalverhalten der ersten 14 Tage je LKW. Erkennt Sensor-Drift und Outlier,
  ist aber kein Beweis fuer einen tatsaechlichen Defekt.
- **Skript 16 (XGBoost, Klassifikation)** klassifiziert ohne `health_score`
  als Feature — der ehrlichste Reality-Check, ca. 0,85 Recall fuer kritische
  Zustaende.

Kurzform: Die Ergebnisse zeigen, dass der Simulator **in sich konsistent**
ist. Sie sind kein Beweis fuer reale Predictive-Maintenance-Signalstaerke.
Dafuer waeren echte Wartungs- und Ausfalldaten oder ein Datensatz wie NASA
CMAPSS noetig.

## Feature Engineering

`pipeline/scripts/17_feature_engineering.py` ist Pflicht-Stufe zwischen
Simulator und RUL-Modell. Ergaenzt werden:

- Rolling Means/Std pro LKW und Sensor fuer 6 h und 24 h.
- Deltas zur vorherigen 30-Minuten-Messung.
- Lags mit 1, 12 und 48 Schritten.
- One-Hot-Beladungsflags `load_low`, `load_med`, `load_high`.
- Healthy-Baseline-Abweichungen aus den ersten 24 Stunden je LKW.
- Zeitfeatures `hour_of_day` und `day_of_week`.
- Outlier-Capping fuer `Coolant temp`, `Lub oil pressure`, `Engine rpm`.

22 Eingabespalten → 75 Ausgabespalten, alle 43.200 Zeilen bleiben erhalten.

## Methodische Risiken

- RUL-Labels sind keine real beobachteten Restlebensdauern.
- Faulty-Anteil kalibriert auf KIT/Kaggle-Referenz (~36,9 %).
- Nicht beobachtete Ausfaelle im 90-Tage-Fenster werden extrapoliert.
- Backfill-Imputation erzeugt leichten Look-Ahead-Bias am LKW-Anfang.
- Alle Top-Features (`brake_fluid_pct`, `odometer_km`, `temperature_c`)
  stammen aus dem Simulator selbst, nicht aus echter Sensorik.

## Projektstruktur

```text
prema/
|-- app.py                          # Streamlit-Dashboard (3 Screens)
|-- requirements.txt
|-- runtime.txt                     # Python-Version fuer Streamlit Cloud
|-- docker-compose.yml              # ein Befehl: docker compose up
|-- README.md / TASKS.md
|-- .streamlit/
|   `-- config.toml                 # Streamlit-Theme (PREMA-Farben)
|
|-- docs/
|   |-- Pflichtenheft_Predictive_Maintenance.pdf
|   `-- Begleitende_Praesentation_Pflichtenheft.pdf
|
|-- docker/
|   |-- Dockerfile
|   `-- entrypoint.sh               # Pipeline-Lauf beim Container-Start
|
|-- tests/
|   `-- smoke_test.py               # AppTest: alle Views + Login-Flow
|
|-- data/
|   |-- generate_from_tracking.py   # Adapter Pipeline → Dashboard-CSVs
|   |-- fleet.csv                   # generiert
|   |-- timeseries.csv              # generiert
|   |-- alerts.csv                  # generiert
|   |-- truck_alerts.csv            # generiert
|   |-- metrics.json                # generiert (ML-Metriken fuer FA-6)
|   `-- feedback.csv                # Werkstatt-Feedback (FA-8)
|
`-- pipeline/
    |-- scripts/
    |   |-- 01_inspect.py           # EDA Phase 1
    |   |-- 02_eda.py
    |   |-- 03_model.py
    |   |-- 04_anomaly.py
    |   |-- 05_summary.py
    |   |-- 06_add_timestamps.py    # Legacy Phase 2
    |   |-- 07_add_weather.py       # Legacy Phase 2
    |   |-- 08_dtc_mapping.py       # Legacy Phase 2
    |   |-- 09_model_with_weather.py
    |   |-- 10_alert_demo.py
    |   |-- 11_summary_phase2.py
    |   |-- 12_rul_random_forest.py # Phase 4: RUL-Modell
    |   |-- 13_simulator.py         # Phase 4: primaere Datenquelle
    |   |-- 14_simulator_diagnostics.py
    |   |-- 15_anomaly_isolation_forest.py
    |   |-- 16_classification_xgboost.py
    |   `-- 17_feature_engineering.py # Phase 4: Pflicht-Stufe vor RUL
    |
    |-- data/engine_health/
    |   |-- engine_data.csv                  # Original KIT/Kaggle-Datensatz
    |   |-- engine_data_final.csv            # Phase-2-Anreicherung (Fallback)
    |   |-- engine_data_enriched.csv         # Zwischenprodukt Phase 2
    |   |-- engine_data_with_timestamps.csv  # Zwischenprodukt Phase 2
    |   |-- engine_data_simulated.csv        # Phase 4: Simulator-Output
    |   |-- engine_data_features.csv         # Phase 4: Feature-Engineering
    |   `-- engine_data_with_rul.csv         # Phase 4: RUL-Vorhersagen
    |
    `-- outputs/
        |-- 13_rul_random_forest.txt
        |-- charts/                 # 19 PNG-Visualisierungen
        `-- models/
            `-- rul_random_forest_v1.pkl
```

## Lokale Ausfuehrung

Voraussetzungen: Python 3.12, pip.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Phase-4-Pipeline (aus `pipeline/` starten):

```bash
cd pipeline
python scripts/13_simulator.py
python scripts/17_feature_engineering.py
python scripts/12_rul_random_forest.py
cd ..
```

Dashboard starten:

```bash
streamlit run app.py
```

Smoke-Tests (rendert alle Views + Login-Flow headless):

```bash
python tests/smoke_test.py
```

### Optionaler Passwortschutz (NFR Zugriffsschutz)

Ohne Konfiguration laeuft das Dashboard im offenen Demo-Modus. Fuer
rollenbasierten Login eine `.streamlit/secrets.toml` anlegen (nicht
committen):

```toml
[passwords]
fm = "..."   # Flottenmanager
wl = "..."   # Werkstattleiter
```

Hinweis: Nach dem Login traegt die URL-Navigation ein Hash-Token
(SHA-256 des Passworts, gekuerzt), damit die Anmeldung Link-Klicks und
Reloads uebersteht — Streamlit startet bei jeder Navigation eine neue
Session. Das Klartext-Passwort landet nie in der URL (MVP-Loesung, im
Zielbild ersetzt durch echte Session-Cookies).

Der Adapter `data/generate_from_tracking.py` wird beim Dashboard-Start
automatisch ausgefuehrt und aktualisiert die vier CSVs aus der Pipeline-Ausgabe.
Er kann auch manuell aufgerufen werden:

```bash
python data/generate_from_tracking.py
```

## Docker

```bash
docker compose up --build
```

Dashboard: <http://localhost:8501>

Der Container fuehrt beim Start automatisch den Simulator aus, falls
`engine_data_simulated.csv` fehlt. Danach Feature Engineering, falls
`engine_data_features.csv` fehlt oder veraltet ist. Dann RUL-Training,
falls `engine_data_with_rul.csv` fehlt oder veraltet ist.

Health-Check: `http://localhost:8501/_stcore/health`

## Kernergebnisse

- Simulator: 10 LKW, 90 Tage, 30-Minuten-Takt, 43.200 Zeilen.
- Zusatzsensoren: Reifendruck, Bremsfluessigkeit, Kilometerstand.
- Health Score: fahrzeugbezogene Verschleisskurve mit Last-, Routen- und
  Wettereinfluss. 9 von 10 LKW unterschreiten die Schadensschwelle (0,5)
  innerhalb von 90 Tagen.
- Feature Engineering: 22 → 75 Spalten, kein Zeilenverlust.
- RUL-Modell: MAE 8 Tage (Test-Set), R² 0,68. Top-Feature: health_score.
- XGBoost-Klassifikation (ohne health_score als Feature): Accuracy 0,71,
  Recall KRITISCH 0,85.
- Dashboard-Snapshot: Tag 49 der Simulation → 5 KRITISCH, 3 WARNUNG, 2 OK
  (fruehster Tag mit LKW-01 = KRITISCH fuer das Demo-Szenario FA-7).

## Datenquellen

1. KIT/Kaggle Automotive Vehicles Engine Health Dataset als Sensoranker.
2. Python-Simulator fuer Zeitreihen, Zusatzsensoren und Verschleisskurven.
3. Lokales DTC-Mapping fuer OBD-II-nahe Fehlercodes.

## Naechste sinnvolle Schritte

- Sauberer zeitlicher Train/Test-Split fuer RUL statt Zufallssplit.
- Sensitivitaetsanalyse der Simulatorparameter.
- Persistenter Storage (z. B. TimescaleDB) statt CSV.
- Validierung gegen echte Flotten-, Wartungs- oder Ausfalldaten.
