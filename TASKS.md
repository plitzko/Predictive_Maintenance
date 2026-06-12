# PREMA — Projektstand & Aufgaben

Predictive-Maintenance-MVP für LKW-Flotten · Big Data SS2026 · Team 1 · Stand: 12.06.2026

| Meilenstein | Termin |
| --- | --- |
| **MVP-Demo / Elevator Pitch** | **19.06.2026** |
| Abschlusspräsentation | 10.07.2026 |

## Teamaufteilung

| Team | Person | Schwerpunkt |
| --- | --- | --- |
| Backend | **Edel** (Person A) | Data Analytics & ML |
| Backend | **Walter** (Person B) | Integration & Infrastruktur |
| Frontend | **Ujjwal** (Person A) | Screens, Navigation & Zugriff |
| Frontend | **Max** (Person B) | Detailansicht, Visualisierung & Demo |

**Status-Legende:** ✅ Erledigt · 🔄 In Arbeit · ⬜ Geplant · 🎯 Zielbild (nach dem MVP)

---

## Projektstand auf einen Blick

| Bereich | Stand | Kurzfassung |
| --- | --- | --- |
| Datensimulation & Pipeline | ✅ | 10 LKW, 90 Tage, Seed 42 — reproduzierbar, Feature Engineering 22 → 75 Spalten |
| ML-Modelle | ✅ | Isolation Forest, XGBoost (Recall KRITISCH 0,86), Random-Forest-RUL — trainiert & evaluiert |
| Alert-Engine | ✅ | 3 Stufen, DTC-Codes, Handlungsempfehlung, Kosteneinsparung pro Alert |
| Dashboard | ✅ | 4 Screens, Rollen-Login, Live-Demo-Zeitraffer, CSV-Exporte, Light/Dark |
| Demo-Szenario (LKW-01) | 🔄 | Szenario steht — Generalprobe für den Pitch |
| Deployment | ✅ | `docker compose up` inkl. automatischem Pipeline-Lauf |

---

## Aufgaben bis zum Elevator Pitch (19.06.)

| # | Aufgabe | Wer | Warum wichtig für den Pitch |
| --- | --- | --- | --- |
| 1 | Demo-Generalprobe: Durchklick + Live-Demo unter 2 min | Max | Der Pitch steht und fällt mit diesem Ablauf |
| 2 | Mind. 3 verschiedene DTC-Codes im Alert-Feed sichtbar machen | Walter | Zeigt die Fehlercode-Anreicherung live |
| 3 | E-Mail-Versand kritischer Alerts (SMTP-Mock) | Walter | „Kritischer Zustand → Mail an den Flottenmanager" ist ein Aha-Moment |
| 4 | ML-`predict()` im Adapter statt Schwellen-Heuristik | Edel | Damit laufen die trainierten Modelle wirklich in der Demo |
| 5 | Letzter UI-Feinschliff (Texte, Übergänge, Wow-Effekt) | Max | Erster Eindruck beim Publikum |
| 6 | AI-Chatbot: Flotten-Fragen & DTC-Erklärungen im Dashboard | Ujjwal | Zusätzliches Nice-To-Have |

---

## Team Backend

### Edel — Data Analytics & ML

#### Erledigt (Edel)

- [x] EDA + Baseline-Modelle (Skripte `01`–`05`)
- [x] OBD-II-Simulator: 10 LKW, 90 Tage, Seed 42, Beladung↔Verschleiß physikalisch modelliert (FA-1)
- [x] Feature Engineering: 22 → 75 Spalten — Rolling Means, Deltas, Lags, Baselines (FA-1)
- [x] Isolation Forest mit 14-Tage-Baseline, False-Positive-Rate 9,1 % (< 10 % gefordert) (FA-3)
- [x] XGBoost-Klassifikation: Recall KRITISCH 0,86 (Ziel ≥ 0,85 erreicht), Accuracy 0,76 (FA-4)
- [x] RUL-Prognose (Random Forest): MAE 13,1 Tage — ohne Ziel-Leakage, ehrliche Metriken (FA-4)
- [x] Statusgrenzen validiert: WARNUNG → KRITISCH mit min. 14,6 Tagen Vorlauf (FA-5)
- [x] ML-Metriken im Dashboard sichtbar (FA-6)

#### Offen (Edel)

- [ ] 🔄 ML-Modelle per `predict()` in den Adapter einbinden — *bis zum Pitch (Fokus #4)*
- [ ] ⬜ MVP-Grenzen dokumentieren: Accuracy 76 % statt 80 %, RUL-Genauigkeit (FA-4)
- [ ] 🎯 Nächtlicher Retraining-Job mit Qualitäts-Gate (FA-8)
- [ ] 🎯 Skalierungstest mit 100 Fahrzeugen (NFR)

### Walter — Integration & Infrastruktur

#### Erledigt (Walter)

- [x] Wetter- und DTC-Anreicherung der Simulationsdaten (FA-2)
- [x] Adapter Pipeline → Dashboard-CSVs (`data/generate_from_tracking.py`)
- [x] Alert-Engine: 3 Stufen, nur bei Zustandswechsel, mit Kosten + DTC + Empfehlung (FA-5)
- [x] Feedback-Persistenz als Trainingslabel (`data/feedback.csv`) (FA-8)
- [x] Docker-Setup: ein Befehl, Pipeline läuft beim Start automatisch mit (NFR)
- [x] Demo-Snapshot Tag 49 + Saison-Fix: Wetter passt zur realen Jahreszeit (FA-7)
- [x] Replay-Export für den Live-Demo-Zeitraffer (FA-7)

#### Offen (Walter)

- [ ] 🔄 Mind. 3 DTC-Codes im Feed sichtbar — *bis zum Pitch (Fokus #2)*
- [ ] 🔄 E-Mail-Mock für kritische Alerts — *bis zum Pitch (Fokus #3)*
- [ ] ⬜ Echter OpenWeatherMap-Call mit Fallback (Vorlage in Skript `07` vorhanden) (FA-2)
- [ ] ⬜ Docker-Durchlauf einmal frisch auf zweitem Rechner verifizieren (NFR)
- [ ] ⬜ Entscheidung „CSV statt TimescaleDB im MVP" kurz im README begründen (NFR)
- [ ] 🎯 CarAPI-Anbindung (lokales DTC-Mapping ist der dokumentierte Fallback) (FA-2)

---

## Team Frontend

### Ujjwal — Screens, Navigation & Zugriff

#### Erledigt (Ujjwal)

- [x] Flottenübersicht: KPI-Karten + nach Dringlichkeit sortierte Statusampel (FA-6)
- [x] Alert-Feed mit Schweregrad-Filtern und Tages-Gruppierung (FA-6)
- [x] Kosteneinsparungs-KPI: vermiedene Ausfallkosten in € (FA-5)
- [x] Rollenbasierter Login (Flottenmanager / Werkstattleiter) mit optionalem Passwort (NFR)
- [x] URL-Navigation mit Deep-Links und Rollen-Guards
- [x] Wartungsplan-Screen mit priorisierter Terminliste + CSV-Export (P-FA-10)
- [x] CSV-Exporte für Flotte und Alert-Feed (FA-6)
- [x] Feedback-Buttons (bestätigt / Fehlalarm) in der Detailansicht (FA-8)

#### Offen (Ujjwal)

- [ ] 🔄 AI-Chatbot auf Groq-Basis: Flotten-Fragen & DTC-Erklärungen im Dashboard — *bis zum Pitch (Fokus #6)*

### Max — Detailansicht, Visualisierung & Demo

#### Erledigt (Max)

- [x] Fahrzeug-Detailansicht: Sensor-Indikatoren, Zeitreihen, Empfehlungs-Banner (FA-6)
- [x] RUL-Prognoselinie im Brems-Chart mit 14-Tage-Horizont (FA-4 + FA-6)
- [x] RUL-Flottenvergleich + 30-Tage-Alert-Historie je Fahrzeug (FA-6)
- [x] Live-Demo-Zeitraffer: 41 Simulationstage im Schnelldurchlauf, Alert-Toasts, Start/Stopp (FA-7)
- [x] Custom Design: Light/Dark-Mode, Animationen, deutsche Formate
- [x] Smoke-Tests: alle Views + Login-Flow laufen automatisiert durch (NFR)

#### Offen (Max)

- [ ] 🔄 Demo-Generalprobe < 2 min — *bis zum Pitch (Fokus #1)*
- [ ] 🔄 Letzter UI-Feinschliff: Texte, Übergänge, Wow-Effekt — *bis zum Pitch (Fokus #5)*

---

## Anforderungsabdeckung (Pflichtenheft)

| Anforderung | Status | Stand |
| --- | --- | --- |
| FA-1 Simulator & Datenerfassung | ✅ | übertrifft das Soll (2.160 simulierte Stunden statt 1.000, Diagnostik-Skript) |
| FA-2 Externe APIs (Wetter, DTC) | 🔄 | Daten & Fallback-Logik vorhanden, echte REST-Calls folgen nach dem MVP |
| FA-3 Anomalieerkennung | ✅ | Modell trainiert, FPR-Ziel < 10 % nachgewiesen; Adapter-Anschluss bis zum Pitch |
| FA-4 Klassifikation + RUL | 🔄 | Recall-Ziel erreicht; Accuracy/RUL als dokumentierte MVP-Grenzen |
| FA-5 Alert-Engine (3 Stufen) | 🔄 | Kernlogik fertig, E-Mail-Mock bis zum Pitch |
| FA-6 Streamlit-Dashboard | ✅ | übertrifft das Soll (4 statt 3 Screens, Live-Demo, ML-Metriken) |
| FA-7 Demo-Szenario LKW-01 | 🔄 | steht, Generalprobe bis zum Pitch |
| FA-8 Feedback & Retraining | 🔄 | Feedback-Erfassung fertig; automatisches Retraining als Ausbaustufe |
| NFR Reproduzierbarkeit, Zugriffsschutz, Testbarkeit, Deployment | ✅ | Seed 42, Rollen-Login, Smoke-Tests, Docker Compose |
| NFR Datenpersistenz (TimescaleDB) | 🎯 | bewusste MVP-Entscheidung: CSV — Begründung wird dokumentiert |

---

## Über das Pflichtenheft hinaus gebaut

- **Wartungsplan-Screen** (4. Ansicht) mit Terminvorschlag aus der RUL-Prognose — deckt die Werkstattleiter-Persona komplett ab
- **Live-Demo-Zeitraffer**: 41 Simulationstage im Schnelldurchlauf mit Alert-Benachrichtigungen
- **RUL-Prognoselinie** direkt im Sensor-Chart statt nur als Balken
- **ML-Modellgüte live im Dashboard** (Accuracy, Recall, MAE)
- **Status-Glättung** über 24-h-Median — keine Fehleskalation durch einzelne Ausreißer
- **Login übersteht Navigation & Reload** (Hash-Token) — über das geforderte Rollen-Login hinaus

## Bewusste MVP-Entscheidungen

- **CSV statt TimescaleDB** — kein DB-Betrieb nötig, Architektur erlaubt späteren Tausch
- **Simuliertes Wetter statt Live-API** — reproduzierbare Demo, echter API-Call als Ausbaustufe
- **Lokales DTC-Mapping statt CarAPI** — 6 realistische OBD-II-Codes reichen für den MVP
- **RUL je Fahrzeug statt je Komponente** — Vereinfachung, im Pitch transparent benannt
- **Kein Kafka/MQTT, keine Mobile App, kein Deep Learning** — laut Pflichtenheft explizit Zielbild
