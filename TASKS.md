# Aufgabenliste PREMA

Abgabe Abschlusspräsentation: **10.07.2026**
Zwischenmeilenstein MVP-Demo: **19.06.2026**

---

## Team Backend

### Person A

- [ ] XGBoost-Klassifikation in Datenpfad einbinden — Adapter soll XGBoost-Predictions als Status-Quelle nutzen statt Health-Score-Threshold (FA-4)
- [ ] Isolation-Forest-Scores ins Dashboard überführen — anomale Messungen als INFO-Alert in `alerts.csv`, inkl. `source`-Feld (`"Isolation Forest"` / `"XGBoost"`) (FA-3)
- [ ] Threshold-Tuning — Recall für KRITISCH-Klasse maximieren, Sicherheit vor Precision
- [ ] ML-Metriken im Dashboard anzeigen — Accuracy, Recall, MAE aus Skript 12, 15, 16 (FA-6)

### Person B

- [ ] OpenWeatherMap-API in aktiven Datenpfad einbinden — `07_add_weather.py` ist Phase-2-Legacy; Wetterdaten nachweislich per API abrufen (FA-2)
- [ ] DTC-Codes live im Alert-Feed — DTC-Code + Handlungsempfehlung bei KRITISCH/WARNUNG sichtbar (FA-2, FA-5)
- [ ] Wetter- und Beladungskontext in Alert-Details — `temperature`, `weather_condition`, `load_pct`, `route_type` in `alerts.csv` schreiben
- [ ] Feedback-Stub implementieren — Button „Wartung bestätigt / Fehlalarm" pro Truck, speichert in CSV (FA-8)
- [ ] Docker End-to-End prüfen — Pipeline (13→17→12) + Adapter + Dashboard laufen in einem `docker compose up` durch (NFR)

---

## Team Frontend

### Person A

- [ ] CSV-Export im Dashboard — Download-Button in Flottenübersicht und Alert-Feed (FA-6)
- [ ] Login-Screen — `st.secrets`-basierter Passwortschutz, zwei Rollen: Flottenmanager und Werkstattleiter (NFR)

### Person B

- [ ] Wireframe-Treue prüfen — Pflichtenheft Abb. 2–4 und Präsentationsfolien abgleichen; insbesondere „Quelle pro Alert" sichtbar machen und 4-KPI-Karten im Alert-Feed (FA-6)
- [ ] Demo-Szenario LKW-01 — Durchklick für Präsentation: Flottenübersicht → KRITISCH-Alert → Detailansicht → Sensor-Zeitreihe → RUL → Alert-Historie (FA-7)
