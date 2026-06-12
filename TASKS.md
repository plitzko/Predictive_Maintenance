# Aufgabenliste PREMA

Abgabe Abschlusspräsentation: **10.07.2026**
Zwischenmeilenstein MVP-Demo: **19.06.2026**

Legende: `[x]` erledigt · `[ ]` offen · (FA-x / NFR) = Referenz ins Pflichtenheft

---

## Team Backend

### Person A — Data Analytics & ML

#### Erledigt

- [x] EDA des KIT/Kaggle Engine-Health-Datensatzes — Inspektion, Verteilungen, Korrelationen (`01_inspect.py`, `02_eda.py`)
- [x] Baseline-Klassifikationsmodell Phase 1 (`03_model.py`)
- [x] Anomalieerkennung per z-Score als Vorstufe (`04_anomaly.py`, `05_summary.py`)
- [x] OBD-II-Simulator — 10 LKW, 90 Tage, 30-Minuten-Takt, 43.200 Zeilen, Seed 42, physikalische Kopplung Beladung ↔ Verschleiß (`13_simulator.py`) (FA-1, NFR Reproduzierbarkeit)
- [x] Simulator-Diagnostik — statistischer Nachweis der Verschleißkorrelationen (`14_simulator_diagnostics.py`) (FA-1 Testkriterium)
- [x] Feature Engineering — Rolling Means/Std (6 h/24 h), Deltas, Lags (1/12/48), Beladungs-Flags, Baseline-Abweichungen, Zeitfeatures; 22 → 75 Spalten ohne Zeilenverlust (`17_feature_engineering.py`)
- [x] RUL-Prognose mit Random Forest — MAE 8 Tage, R² 0,68, Modell-Artefakt `rul_random_forest_v1.pkl` (`12_rul_random_forest.py`) (FA-4)
- [x] Anomalieerkennung mit Isolation Forest — kalibriert auf 14-Tage-Baseline je LKW, ohne health_score als Feature (`15_anomaly_isolation_forest.py`) (FA-3)
- [x] 3-Klassen-Klassifikation mit XGBoost — Accuracy 0,71, Recall KRITISCH 0,85, bewusst ohne health_score als Feature (`16_classification_xgboost.py`) (FA-4)
- [x] XGBoost-Schwellen in den Datenpfad eingebunden — Status-Grenzen (health_score 0,5/0,7) identisch zwischen Skript 16 und Adapter (FA-4)
- [x] Isolation-Forest-Scores ins Dashboard überführt — anomale Messungen (|z| > 2,5 bei Status OK) als INFO-Alert, `source`-Feld unterscheidet „Isolation Forest" / „XGBoost-Klassifikation" (FA-3)
- [x] Status-Glättung — Flotten-Status aus Median des Health-Scores der letzten 24 h, einzelne Ausreißer eskalieren kein Fahrzeug mehr
- [x] ML-Metriken im Dashboard — Adapter parst die Reports aus Skript 12/15/16 nach `data/metrics.json`, Anzeige im Expander „ML-Modellgüte" (FA-6)

#### Offen

- [ ] Klassifikations-Accuracy auf ≥ 80 % heben — FA-4-Testkriterium fordert 80 %, aktuell 71 % (Recall KRITISCH 0,85 hält das Ziel von ≥ 85 % knapp)
- [ ] Threshold-Tuning — Recall für KRITISCH-Klasse maximieren, Sicherheit vor Precision; False-Positive-Rate < 10 % für Isolation Forest nachweisen (FA-3 Testkriterium)
- [ ] Zeitlicher Train/Test-Split für RUL statt Zufallssplit — Look-Ahead-Bias vermeiden (vgl. README „Methodische Risiken")
- [ ] Sensitivitätsanalyse der Simulatorparameter
- [ ] Nächtlicher Retraining-Job mit Qualitäts-Gate — Feedback-Labels aus `data/feedback.csv` einbeziehen, neues Modell nur bei gehaltenen Metriken aktivieren, Qualitätsreport protokollieren (FA-8)

### Person B — Datenanreicherung, Integration & Infrastruktur

#### Erledigt

- [x] Zeitstempel-Anreicherung des Rohdatensatzes (`06_add_timestamps.py`)
- [x] Wetterdaten als Verschleißfaktor — Temperatur und Niederschlag im Datenpfad, simuliert (`07_add_weather.py`, Phase 2) (FA-2, teilweise)
- [x] DTC-Mapping — lokale Zuordnung OBD-II-Fehlercodes mit deutscher Beschreibung und Handlungsempfehlung (`08_dtc_mapping.py`) (FA-2, teilweise)
- [x] Modellvergleich mit/ohne Wetterfeatures (`09_model_with_weather.py`)
- [x] Alert-Demo und Phase-2-Zusammenfassung (`10_alert_demo.py`, `11_summary_phase2.py`)
- [x] Adapter Pipeline → Dashboard — erzeugt `fleet.csv`, `timeseries.csv`, `alerts.csv`, `truck_alerts.csv`, `metrics.json`; Fallback-Kette über drei Quelldateien (`data/generate_from_tracking.py`)
- [x] Alert-Engine entspammt — Alert nur bei Severity-Wechsel je LKW (Zustandsübergang) mit 6-h-Re-Arm-Sperre statt einer Meldung pro 30-Minuten-Messung (FA-5)
- [x] Dreistufiges Warnsystem KRITISCH/WARNUNG/INFO inkl. Kosteneinsparung pro Alert (600/400/0 €/h) (FA-5)
- [x] DTC-Codes live im Alert-Feed — DTC-Code + Handlungsempfehlung bei KRITISCH/WARNUNG sichtbar (FA-2, FA-5)
- [x] Wetter- und Beladungskontext in Alert-Details — `temperature_c`, `weather`, `load_pct`, `route_type` in `alerts.csv` (FA-2)
- [x] Feedback-Persistenz — „Wartung bestätigt / Fehlalarm" pro Truck wird mit Zeitstempel und Status-Kontext in `data/feedback.csv` gespeichert (FA-8, Erfassungsseite)
- [x] Docker-Setup — `docker/Dockerfile`, `docker-compose.yml`, `docker/entrypoint.sh` mit automatischem Pipeline-Lauf bei fehlenden Artefakten (NFR Deployment)
- [x] Snapshot-Logik — Tag 49 der Simulation als Demo-Fenster (frühster Tag mit LKW-01 = KRITISCH), Zeitstempel relativ zu „jetzt" verschoben (FA-7)

#### Offen

- [ ] OpenWeatherMap-API in aktiven Datenpfad einbinden — Wetterdaten nachweislich per REST-API abrufen, Fallback-Wert bei API-Ausfall (FA-2 Testkriterium; aktuell simuliert)
- [ ] CarAPI-Anbindung prüfen — DTC-Datenbank per REST statt lokalem Mapping, mindestens 3 DTCs mit korrekter Beschreibung nachweisen (FA-2 Testkriterium; lokales Mapping als Fallback behalten)
- [ ] E-Mail-Versand für kritische Alerts — SMTP-Mock (`smtplib`), Versand erfolgreich protokollieren (FA-5 Testkriterium)
- [ ] TimescaleDB-Persistenz — Rohdaten und ML-Ergebnisse in lokaler TimescaleDB statt CSV; Dashboard-Zustand übersteht Neustarts (NFR Datenpersistenz; im MVP bewusst CSV, Entscheidung dokumentieren falls es dabei bleibt)
- [ ] Docker End-to-End prüfen — Pipeline (13→17→12) + Adapter + Dashboard laufen in einem `docker compose up` durch; Batch-Latenz < 30 s für 10 Fahrzeuge messen (NFR Latenz/Deployment)

---

## Team Frontend

### Person A — Screens, Navigation & Zugriff

#### Erledigt

- [x] Flottenübersicht (Screen 1) — Statusampel, 4 KPI-Karten, nach Priorität sortierte Tabelle, kritische Fahrzeuge oben (FA-6)
- [x] Alert-Feed (Screen 3) — chronologische Liste, Filter nach Schweregrad (ALLE/KRITISCH/WARNUNG/INFO), Quelle pro Alert, Link zur Detailansicht (FA-6)
- [x] URL-basierte Navigation — Deep-Links für Rolle, View, LKW und Filter; ungültige Parameter fallen sauber zurück (unbekannter LKW, ungültiger Filter, WL-Umleitung vom Alert-Feed)
- [x] Rollen-Auswahl-Screen — zwei Personas (Flottenmanager / Werkstattleiter) mit unterschiedlichen Sichtrechten: nur FM sieht Alert-Feed und Kosten-KPI, nur WL sieht Ø-RUL und Wartungsfeedback (NFR Zugriffsschutz)
- [x] Login-Screen — `st.secrets`-basierter Passwortschutz pro Rolle, offener Demo-Modus ohne Secrets; Anmeldung übersteht Navigation und Reload per Hash-Token in der URL (NFR Zugriffsschutz)
- [x] CSV-Export — Download-Buttons in Flottenübersicht und Alert-Feed (gefiltert), UTF-8-BOM für Excel (FA-6)
- [x] Kosteneinsparungs-Kalkulation — verhinderte Kosten in Flotten-KPI und pro Alert, zentrale Konstanten (600 €/h × 4 h Standzeit) (FA-5)
- [x] Feedback-Buttons in der Detailansicht für die WL-Rolle (FA-8, UI-Seite)

#### Offen

- [x] **Groq-Chatbot „PREMA Copilot" (LKW-Diagnose-Assistent)** — spezialisierter, thematisch eingeschränkter Chat-Assistent für Flottenmanager und Werkstattleiter in der Detailansicht (`chatbot.py`); verweigert Off-Topic-Anfragen und Prompt-Injection, kennt den Live-Zustand des gewählten LKW und hat Hintergrundwissen zu DTC-Codes, Sensorwerten und Wartungsempfehlungen

  **Teilaufgaben:**
  - [x] `groq`-Paket eingebunden, API-Key über `st.secrets` (`.streamlit/secrets.toml`, gitignored) oder `GROQ_API_KEY`; Modell `llama-3.3-70b-versatile`
  - [x] System-Prompt: Bot darf **nur** LKW-Diagnose, DTC-Codes, Sensoranomalien und Wartungsempfehlungen beantworten; Off-Topic und „ignoriere deine Anweisungen" werden in einem Satz abgelehnt (live getestet)
  - [x] Dynamische Kontextinjektion: Status, Sensorwerte, RUL inkl. Flottenrang, letzte 8 Alerts mit DTC sowie ML-Metriken werden pro Truck in den Prompt geladen
  - [x] Statisches Hintergrundwissen: DTC-Referenz, Sensor-Schwellwerte, Status-/RUL-Logik und Kostenmodell als autoritativer Prompt-Block (kein RAG im MVP)
  - [x] Rollenbasierte Antworttiefe: WL bekommt Prüfschritte und Rohwerte, FM betriebliche Konsequenzen und Handlungsempfehlung
  - [x] Chat-UI: `st.chat_message` / `st.chat_input` mit Streaming, Einstiegsfragen als Pills, eigenes Styling im PREMA-Design, in der Detailansicht
  - [x] Gesprächs-Reset je LKW und Rolle (`st.session_state`-Key pro Truck), „Neuer Chat"-Button

  **Offen:**
  - [ ] API-Key rotieren (wurde im Klartext geteilt) und in Streamlit Cloud Secrets hinterlegen

### Person B — Detailansicht, Visualisierung & Demo

#### Erledigt

- [x] Einzelfahrzeug-Detail (Screen 2) — Sensor-Balkenindikatoren mit Schwellenfarben, Fahrzeug-Metadaten, Empfehlungs-Banner je Status (FA-6)
- [x] Sensor-Zeitreihen — Bremsflüssigkeit (72 h) mit Warn-/Kritisch-Schwellen und Alert-Markern, Motortemperatur mit kritischer Schwelle und dynamischer y-Achse, Öldruck mit Warnschwelle (FA-6)
- [x] RUL-Balkendiagramm — Flottenvergleich mit hervorgehobenem Fahrzeug, statusfarbig (FA-6, Wireframe Screen 2)
- [x] Alert-Verlauf je Fahrzeug (30 Tage) inkl. DTC-Code und Handlungsempfehlung (FA-6)
- [x] Wireframe-Treue hergestellt — „Quelle pro Alert" sichtbar, 4 KPI-Karten im Alert-Feed, dreistufiges Farbsystem Grün/Orange/Rot (Pflichtenheft Abb. 2–4)
- [x] Custom CSS — Light/Dark-Mode über Streamlit-CSS-Variablen, KPI-Cards, Badges, Tabellen-Hover; Dark-Mode-Bug der Motortemperatur-Linie behoben
- [x] Deutsche Zahlenformatierung zentralisiert (`fmt_de`), Bug mit zerstörten Kommas in Meldungstexten behoben
- [x] Demo-Szenario LKW-01 — Durchklick für Präsentation: Flottenübersicht → KRITISCH-Alert → Detailansicht → Sensor-Zeitreihe → RUL → Alert-Historie; Snapshot-Tag so gewählt, dass LKW-01 KRITISCH ist (FA-7)
- [x] Smoke-Tests — `python tests/smoke_test.py` rendert alle Views inkl. Edge-Cases und testet den Login-Flow headless (NFR Testbarkeit)

#### Offen

- [ ] Demo-Generalprobe für den 19.06. — kompletten Durchklick mit Stoppuhr proben, Gesamtdurchlaufzeit < 2 min nachweisen (FA-7 Testkriterium)
- [ ] Screenshots/Folien der drei Screens für die Abschlusspräsentation am 10.07. aktualisieren (Abgleich mit `docs/Begleitende_Praesentation_Pflichtenheft.pdf`)
- [ ] Mobile/kleine Viewports prüfen — KPI-Grid und Tabellen-Layout bei schmalen Fenstern (nice-to-have, Demo läuft am Desktop)
