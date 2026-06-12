# Aufgabenliste PREMA

Abschlusspräsentation: **10.07.2026** · MVP-Demo: **19.06.2026**
Stand: 12.06.2026 · (FA-x / NFR) = Pflichtenheft · (P-FA-xx) = begleitende Präsentation

## Anforderungs-Überblick

| Anforderung | Status | Anmerkung |
| --- | --- | --- |
| FA-1 Simulator & Datenerfassung | ✅ | 10 LKW, 90 Tage, Seed 42, saisonal korrekt |
| FA-2 Externe APIs (Wetter, DTC) | ⚠️ | simuliert/lokal — echte REST-Calls offen |
| FA-3 Anomalieerkennung (Isolation Forest) | ⚠️ | Modell trainiert; Dashboard nutzt noch z-Score-Heuristik |
| FA-4 Klassifikation + RUL | ⚠️ | Recall KRITISCH 0,86 ✅ · Accuracy 76 % (Ziel 80) · RUL-Ziel verfehlt |
| FA-5 Alert-Engine (3 Stufen) | ⚠️ | Stufen + Kosten + DTC ✅ · E-Mail-Mock offen |
| FA-6 Streamlit-Dashboard | ✅ | übertroffen: + Wartungsplan, Live-Demo, RUL-Prognoselinie |
| FA-7 Demo-Szenario LKW-01 | ✅ | Generalprobe (< 2 min) offen |
| FA-8 Feedback & Retraining | ⚠️ | Erfassung ✅ · nächtlicher Retraining-Job offen |
| NFR Zugriffsschutz | ✅ | Rollen-Login, optionale Passwörter (st.secrets) |
| NFR Deployment (Docker) | ⚠️ | Setup ✅ · End-to-End-Nachweis offen |
| NFR Datenpersistenz (TimescaleDB) | ❌ | MVP bewusst CSV — Entscheidung dokumentieren |
| P-FA-10 Wartungsliste als CSV | ✅ | Export auf dem Wartungsplan-Screen |

---

## Team Backend

### Person A — Data Analytics & ML

#### Erledigt

- [x] EDA + Baseline-Modelle Phase 1 (`01`–`05`)
- [x] OBD-II-Simulator: 10 LKW, 90 Tage, Seed 42 (`13`) (FA-1)
- [x] Simulator-Diagnostik (`14`) (FA-1 Testkriterium)
- [x] Feature Engineering 22 → 75 Spalten (`17`)
- [x] Isolation Forest mit 14-Tage-Baseline (`15`) (FA-3)
- [x] XGBoost-Klassifikation ohne health_score: Accuracy 0,76, Recall KRITISCH 0,86 (`16`) (FA-4)
- [x] RUL-Ziel-Leakage entfernt — ehrliche Metriken: MAE 13,1 d, R² 0,45 (`12`) (FA-4)
- [x] Statusgrenzen validiert: WARNUNG→KRITISCH min. 14,6 d Vorlauf (FA-5)
- [x] INFO-Schwelle = 99-%-Quantil (z > 3,2; Replay 2,5) (FA-3)
- [x] Status-Glättung (24-h-Median) + ML-Metriken im Dashboard (FA-6)

#### Offen

- [ ] **ML-Modelle im Adapter anwenden** — `predict()` statt Schwellen-/z-Score-Heuristik (FA-3, FA-4)
- [ ] Accuracy auf ≥ 80 % heben (aktuell 76 %) (FA-4)
- [ ] RUL-Ziel < 15 % Abweichung — verbessern oder als MVP-Grenze dokumentieren (FA-4)
- [ ] Isolation-Forest-FPR < 10 % nachweisen (FA-3)
- [ ] Zeitlicher Train/Test-Split für RUL statt Zufallssplit
- [ ] Nächtlicher Retraining-Job mit Qualitäts-Gate (FA-8)

### Person B — Integration & Infrastruktur

#### Erledigt

- [x] Zeitstempel-/Wetter-/DTC-Anreicherung Phase 2 (`06`–`11`) (FA-2 teilw.)
- [x] Adapter Pipeline → Dashboard-CSVs (`data/generate_from_tracking.py`)
- [x] Alert-Engine: nur Severity-Wechsel + 6-h-Re-Arm, 3 Stufen, Kosten (FA-5)
- [x] DTC, Wetter und Beladung im Alert-Feed (FA-2)
- [x] Feedback-Persistenz `data/feedback.csv` (FA-8)
- [x] Docker-Setup mit automatischem Pipeline-Lauf (NFR Deployment)
- [x] Snapshot Tag 49 + Saison-Fix: Wetter passt zur realen Jahreszeit (FA-7)
- [x] Replay-Export `replay.csv`/`replay_alerts.csv` für die Live-Demo (FA-7)
- [x] Bremsschwellen konsistent zur health-Skalierung (68 %/50 %)

#### Offen

- [ ] OpenWeatherMap-API mit Fallback einbinden (FA-2 Testkriterium)
- [ ] CarAPI für DTCs prüfen, lokales Mapping als Fallback (FA-2 Testkriterium)
- [ ] E-Mail-Versand kritischer Alerts per SMTP-Mock (FA-5 Testkriterium)
- [ ] TimescaleDB-Persistenz oder CSV-Entscheidung dokumentieren (NFR)
- [ ] Docker End-to-End prüfen, Batch-Latenz < 30 s messen (NFR, P-NFA-02)

---

## Team Frontend

### Person A — Screens, Navigation & Zugriff

#### Erledigt

- [x] Flottenübersicht: KPI-Karten + priorisierte Statustabelle (FA-6)
- [x] Alert-Feed mit Filtern und Tages-Gruppierung (FA-6)
- [x] URL-Navigation: Deep-Links, Rollen-Guards, saubere Fallbacks
- [x] Rollenauswahl + Passwort-Login (st.secrets, Hash-Token) (NFR Zugriffsschutz)
- [x] Login-Redesign: Avatare, Check-Listen, dynamischer Sicherheitshinweis
- [x] CSV-Exporte Flotte + Alert-Feed (FA-6)
- [x] Wartungsplan als eigener Screen mit Nav-Badge (Persona Stefan)
- [x] Wartungsplan-CSV-Export (P-FA-10)
- [x] Feedback-Buttons in der Detailansicht (FA-8)

#### Offen

- [x] **Groq-Chatbot „PREMA Copilot" (LKW-Diagnose-Assistent)** — spezialisierter, thematisch eingeschränkter Chat-Assistent für Flottenmanager und Werkstattleiter (`chatbot.py`); globales Sidebar-Panel auf jedem Screen, öffnet über schwebenden »✦ Copilot«-Button; verweigert Off-Topic-Anfragen und Prompt-Injection, kennt Flotte + Alert-Feed und in der Detailansicht zusätzlich die Tiefendaten des gewählten LKW

  **Teilaufgaben:**
  - [x] `groq`-Paket eingebunden, API-Key über `st.secrets` (`.streamlit/secrets.toml`, gitignored) oder `GROQ_API_KEY`; Modell `llama-3.3-70b-versatile`
  - [x] System-Prompt: Bot darf **nur** LKW-Diagnose, DTC-Codes, Sensoranomalien und Wartungsempfehlungen beantworten; Off-Topic und „ignoriere deine Anweisungen" werden in einem Satz abgelehnt (live getestet)
  - [x] Dynamische Kontextinjektion: Flottentabelle + Alert-Feed immer; in der Detailansicht zusätzlich Sensorwerte, RUL mit Flottenrang und letzte 8 Alerts des Fokus-LKW; ML-Metriken
  - [x] Statisches Hintergrundwissen: DTC-Referenz, Sensor-Schwellwerte, Status-/RUL-Logik und Kostenmodell als autoritativer Prompt-Block (kein RAG im MVP)
  - [x] Rollenbasierte Antworttiefe: WL bekommt Prüfschritte und Rohwerte, FM betriebliche Konsequenzen und Handlungsempfehlung
  - [x] Chat-UI: Sidebar-Panel mit `st.chat_message` / `st.chat_input`, Streaming, rollenspezifische Einstiegsfragen als Pills, Kontext-Chip („Flotte" / „Flotte + LKW-xx"), PREMA-Design
  - [x] Gesprächsverlauf pro Rolle (`st.session_state`), „Neuer Chat"-Button; kein Auto-Scroll in der Detailansicht, da der Chat nicht mehr im Seiteninhalt liegt

  **Offen:**
  - [ ] API-Key rotieren (wurde im Klartext geteilt) und in Streamlit Cloud Secrets hinterlegen

### Person B — Detailansicht, Visualisierung & Demo

#### Erledigt

- [x] Detailansicht: Sensor-Indikatoren, Metadaten, Empfehlungs-Banner (FA-6)
- [x] Sensor-Zeitreihen mit Schwellen und Alert-Markern (FA-6)
- [x] RUL-Prognoselinie im Brems-Chart, 14-Tage-Horizont (FA-4, FA-6)
- [x] RUL-Flottenvergleich + Alert-Historie 30 Tage (FA-6)
- [x] Custom CSS: Light/Dark, Icons, Animationen, deutsche Formate
- [x] Demo-Szenario LKW-01 (FA-7)
- [x] Live-Demo-Zeitraffer mit Toasts, Start/Stopp in der Navigation (FA-7)
- [x] UI-Fehlerdurchgang: Doppel-Button, Toast-Loop, totes CSS behoben
- [x] Smoke-Tests `tests/smoke_test.py` (NFR Testbarkeit)

#### Offen

- [ ] Demo-Generalprobe für den 19.06. — Durchklick inkl. Live-Demo, < 2 min nachweisen (FA-7 Testkriterium)
