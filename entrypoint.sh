#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="/app/pipeline/data/engine_health"
SCRIPT_DIR="/app/pipeline"
SIMULATED="$DATA_DIR/engine_data_simulated.csv"
FEATURES="$DATA_DIR/engine_data_features.csv"
WITH_RUL="$DATA_DIR/engine_data_with_rul.csv"

# Phase 4: Simulator ist die primaere Pipeline-Quelle. 06-08 bleiben Legacy.
if [ ! -f "$SIMULATED" ]; then
    echo "[entrypoint] engine_data_simulated.csv fehlt -> erzeuge Simulator-Daten..."
    cd "$SCRIPT_DIR"
    python scripts/13_simulator.py
    cd /app
fi

# Feature Engineering neu ausfuehren, wenn es fehlt oder der Simulator neuer ist.
if [ ! -f "$FEATURES" ] || [ "$SIMULATED" -nt "$FEATURES" ]; then
    echo "[entrypoint] engine_data_features.csv fehlt/veraltet -> berechne Features..."
    cd "$SCRIPT_DIR"
    python scripts/17_feature_engineering.py
    cd /app
fi

# RUL neu trainieren, wenn es fehlt oder der Feature-Datensatz neuer ist.
if [ ! -f "$WITH_RUL" ] || [ "$FEATURES" -nt "$WITH_RUL" ]; then
    echo "[entrypoint] engine_data_with_rul.csv fehlt/veraltet -> trainiere RUL-Modell..."
    cd "$SCRIPT_DIR"
    python scripts/12_rul_random_forest.py
    cd /app
fi

echo "[entrypoint] Starte Streamlit auf Port ${STREAMLIT_SERVER_PORT:-8501}..."
exec streamlit run app.py
