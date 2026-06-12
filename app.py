"""
PREMA - Predictive Maintenance Dashboard (MVP Click-Demo)
Hochschule München | Big Data SS2026 | Team 1 (Predictive)

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    https://share.streamlit.io

Optionaler Passwortschutz (NFR Zugriffsschutz): In .streamlit/secrets.toml
    [passwords]
    fm = "..."
    wl = "..."
Ohne secrets.toml läuft das Dashboard im offenen Demo-Modus.
"""
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import altair as alt
import pandas as pd
import streamlit as st

# ============================================================================
# Konstanten (Schwellenwerte & Kostenmodell, vgl. Pflichtenheft FA-5/FA-7)
# ============================================================================
# Bremsflüssigkeit wird im Adapter aus dem Health-Score skaliert
# (brake = health*90+5): health 0.7 -> 68 %, health 0.5 -> 50 %.
# Die Schwellen entsprechen damit exakt den Statusgrenzen WARNUNG/KRITISCH.
BRAKE_WARN_PCT = 68      # Warnschwelle Bremsflüssigkeit (%)
BRAKE_CRIT_PCT = 50      # Kritische Schwelle Bremsflüssigkeit (%)
MOTOR_TEMP_CRIT_C = 95   # Kritische Motortemperatur (°C)
HEALTH_WARN = 0.7        # Health-Score-Schwelle WARNUNG (vgl. Adapter/XGBoost)
HEALTH_CRIT = 0.5        # Health-Score-Schwelle KRITISCH
OIL_PRESSURE_WARN_BAR = 2.5  # Warnschwelle Öldruck (bar)
COST_PER_HOUR_EUR = 600  # Kosten pro Stillstandsstunde (Pflichtenheft: 500-700 €)
HOURS_PER_BREAKDOWN = 4  # Angenommene Standzeit pro vermiedener Panne

FORECAST_HORIZON_H = 14 * 24   # RUL-Prognoselinie: max. 14 Tage in die Zukunft
REPLAY_TICK_SECONDS = 2.0      # Live-Demo: reale Sekunden pro Schritt
REPLAY_HOURS_PER_TICK = 12     # Live-Demo: simulierte Stunden pro Schritt
PLANNER_SAFETY = 0.8           # Wartungsplaner: Termin bei 80 % der Restlaufzeit

WEEKDAYS_DE = ["Montag", "Dienstag", "Mittwoch", "Donnerstag",
               "Freitag", "Samstag", "Sonntag"]

SEVERITIES = ("KRITISCH", "WARNUNG", "INFO")
ALERT_FILTERS = ("ALLE",) + SEVERITIES

# Statusfarben: Rot ist exklusiv für KRITISCH reserviert, die Marken-/
# Interaktionsfarbe (Blau) liegt als --p-brand im CSS.
COLOR_CRIT = "#E5484D"
COLOR_WARN = "#F59E0B"
COLOR_OK = "#10B981"
COLOR_INFO = "#3B82F6"

# ============================================================================
# Page config
# ============================================================================
st.set_page_config(
    page_title="PREMA – Predictive Maintenance",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"About": "PREMA MVP · HM Big Data SS2026"}
)

# ============================================================================
# Custom CSS - industrial/refined aesthetic
# ============================================================================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700&display=swap" rel="stylesheet">
<style>
    /* ── Design tokens – accent colors bleiben fix, alles andere
       nutzt Streamlits eigene CSS-Variablen (auto Light/Dark)      */
    :root {
        /* Marke & Interaktion (Navigation, aktive Zustände) */
        --p-brand:       #2563EB;
        --p-brand-light: #60A5FA;
        /* Statusfarben – Rot ist exklusiv für KRITISCH reserviert */
        --p-crit:    #E5484D;
        --p-ok:      #10B981;
        --p-warn:    #F59E0B;
        --p-info:    #3B82F6;
        /* Rahmen & Schatten: rgba-Basis funktioniert in beiden Modi */
        --p-border:  rgba(120,120,140,0.22);
        --p-shadow:  0 1px 3px rgba(0,0,0,0.07), 0 0 0 1px rgba(120,120,140,0.1);
        --p-shadow-hover: 0 2px 8px rgba(0,0,0,0.1), 0 0 0 1px rgba(120,120,140,0.16);
    }

    /* ── Fonts ── */
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
    h1,h2,h3,h4,h5,h6,.section-title,.kpi-label,.stButton>button {
        font-family: 'Plus Jakarta Sans','Inter',sans-serif;
    }

    /* ── Streamlit-Chrome verstecken ── */
    #MainMenu,footer,header { visibility: hidden; }
    .block-container { padding-top: 1rem; padding-bottom: 1.5rem; max-width: 1400px; }

    /* ── Header-Bar (immer dunkel) ── */
    .header-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.7rem 1.2rem;
        background: linear-gradient(90deg, #16161A 0%, #26262C 100%);
        color: #FFF;
        border-radius: 8px;
        margin-bottom: 1rem;
        font-family: 'Plus Jakarta Sans', sans-serif;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3), inset 0 -2px 0 var(--p-brand);
    }
    .header-brand {
        font-size: 1.25rem; font-weight: 700; letter-spacing: 0.08em;
        display: flex; align-items: center; gap: 0.65rem;
    }
    .header-brand-accent { color: var(--p-brand-light); }
    .header-logo { display: inline-flex; width: 20px; height: 20px; color: var(--p-brand-light); flex: none; }
    .header-logo svg { width: 100%; height: 100%; }
    .header-tagline {
        font-size: 0.6rem; font-weight: 600; letter-spacing: 0.16em;
        text-transform: uppercase; color: rgba(255,255,255,0.38);
        border-left: 1px solid rgba(255,255,255,0.16);
        padding-left: 0.65rem;
    }
    .header-user { font-size: 0.75rem; opacity: 0.75; letter-spacing: 0.05em;
        display: flex; align-items: center; gap: 0.4rem; }
    @media (max-width: 760px) { .header-tagline { display: none; } }

    /* ── Navigation ── */
    .nav-buttons {
        display: flex; gap: 0.8rem; margin-bottom: 1.2rem;
        align-items: center; flex-wrap: wrap;
    }
    .nav-button {
        display: inline-flex; align-items: center; gap: 0.5rem;
        padding: 0.6rem 1.2rem; border-radius: 6px;
        font-family: 'Plus Jakarta Sans','Inter',sans-serif;
        font-size: 0.88rem; font-weight: 700; letter-spacing: 0.05em;
        border: 2px solid transparent; cursor: pointer;
        transition: all 0.2s ease; text-decoration: none !important;
    }
    .nav-button,.nav-button *,.nav-button:visited,.nav-button:hover,.nav-button:active {
        text-decoration: none !important;
    }
    .nav-button.active,.nav-button.detail {
        background: var(--p-brand); color: #FFF; border-color: var(--p-brand);
    }
    .nav-button.detail { cursor: default; }
    .nav-button.inactive {
        background: rgba(120,120,140,0.1);
        color: var(--text-color);
        border-color: var(--p-border);
    }
    .nav-button.inactive:hover { border-color: var(--p-brand); }
    .nav-count {
        display: inline-flex; align-items: center; justify-content: center;
        min-width: 1.3rem; height: 1.3rem; padding: 0 0.3rem;
        border-radius: 99px; font-size: 0.68rem; font-weight: 700;
        background: var(--p-crit); color: #FFF;
    }
    .nav-button.active .nav-count { background: rgba(255,255,255,0.25); }
    /* Live-Demo: rechtsbündiger Pill in der Navigation (bewusst rot als
       Eyecatcher – einzige Ausnahme von »Rot nur für KRITISCH«) */
    .nav-button.demo {
        margin-left: auto;
        background: transparent; color: var(--p-crit);
        border: 1px solid var(--p-crit);
        font-size: 0.78rem; letter-spacing: 0.06em; font-weight: 700;
    }
    .nav-button.demo:hover {
        background: var(--p-crit); color: #FFF;
    }
    .nav-button.demo.running {
        background: var(--p-crit); color: #FFF; border-color: var(--p-crit);
    }
    .nav-button.demo.running .replay-dot { background: #FFF; }
    .nav-button.demo svg { width: 11px; height: 11px; flex: none; }

    /* ── KPI-Cards ── */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 0.8rem; margin-bottom: 1rem;
    }
    .kpi-card {
        background: var(--background-color);
        border: 1px solid var(--p-border);
        padding: 0.9rem 1rem; border-radius: 6px;
        box-shadow: var(--p-shadow);
        transition: box-shadow 0.2s ease;
    }
    .kpi-card:hover { box-shadow: var(--p-shadow-hover); }
    .kpi-card-header { display: flex; align-items: center; gap: 0.45rem; margin-bottom: 0.35rem; }
    .kpi-card-icon { display: inline-flex; width: 15px; height: 15px; color: var(--text-color); opacity: 0.55; flex: none; }
    .kpi-card-icon svg { width: 100%; height: 100%; }
    .kpi-card.critical .kpi-card-icon { color: var(--p-crit); opacity: 1; }
    .kpi-card.warning  .kpi-card-icon { color: var(--p-warn);   opacity: 1; }
    .kpi-card.ok       .kpi-card-icon { color: var(--p-ok);     opacity: 1; }
    .kpi-card.critical {
        background: color-mix(in srgb, #E5484D 6%, var(--background-color));
    }
    .kpi-card.warning {
        background: color-mix(in srgb, #F59E0B 6%, var(--background-color));
    }
    .kpi-card.ok {
        background: color-mix(in srgb, #10B981 6%, var(--background-color));
    }
    .kpi-link,.kpi-link *,.kpi-link:visited,.kpi-link:hover,.kpi-link:active {
        color: inherit !important; text-decoration: none !important;
    }
    .kpi-link { display: block; height: 100%; }
    .kpi-link .kpi-card { height: 100%; box-sizing: border-box; }
    .kpi-label {
        font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.12em;
        color: var(--text-color); opacity: 0.55;
        font-family: 'Plus Jakarta Sans','Inter',sans-serif; font-weight: 600;
    }
    .kpi-value {
        font-size: 1.9rem; font-weight: 700; line-height: 1.1;
        color: var(--text-color); font-family: 'Plus Jakarta Sans',sans-serif;
    }
    .kpi-sub { font-size: 0.7rem; color: var(--text-color); opacity: 0.55; margin-top: 0.2rem; }

    /* ── Status-Badges ── */
    .badge {
        display: inline-flex; align-items: center; justify-content: center;
        padding: 0.28rem 0.6rem; border-radius: 4px;
        font-size: 0.68rem; font-weight: 700; letter-spacing: 0.06em;
        font-family: 'Plus Jakarta Sans','Inter',sans-serif;
        width: 86px; min-width: 86px; text-align: center; box-sizing: border-box;
    }
    .badge-critical { background: var(--p-crit); color: #FFF; }
    .badge-warning  { background: var(--p-warn);   color: #FFF; }
    .badge-ok       { background: var(--p-ok);     color: #FFF; }
    .badge-info     { background: var(--p-info);   color: #FFF; }

    /* ── Sensor-Leiste ── */
    .sensor-row {
        display: grid;
        grid-template-columns: 1fr 130px 64px;
        gap: 0.7rem; align-items: center; padding: 0.5rem 0;
        border-bottom: 1px solid var(--p-border);
    }
    .sensor-label { font-size: 0.8rem; color: var(--text-color); font-weight: 500; }

    /* ── Status-Dot ── */
    .status-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; }
    .status-dot-kritisch { background: var(--p-crit); box-shadow: 0 0 0 3px rgba(229,72,77,0.18); }
    .status-dot-warnung  { background: var(--p-warn); box-shadow: 0 0 0 3px rgba(245,158,11,0.18); }
    .status-dot-ok       { background: var(--p-ok);   box-shadow: 0 0 0 3px rgba(16,185,129,0.18); }
    .sensor-bar-bg {
        position: relative; height: 6px;
        background: var(--secondary-background-color);
        border-radius: 3px;
    }
    .sensor-bar-fill { height: 100%; border-radius: 3px; }
    /* Schwellenmarker auf den Sensor-Balken (Warn-/Kritisch-Grenze) */
    .sensor-mark {
        position: absolute; top: -2px; bottom: -2px; width: 2px;
        border-radius: 1px; opacity: 0.65;
    }
    .sensor-mark-warn { background: var(--p-warn); }
    .sensor-mark-crit { background: var(--p-crit); }
    .sensor-value {
        font-variant-numeric: tabular-nums; font-size: 0.8rem;
        font-weight: 600; text-align: right; color: var(--text-color);
    }

    /* ── Alert-Feed ── */
    .alert-row {
        display: grid;
        grid-template-columns: 88px 86px 66px minmax(0,1fr) 78px;
        gap: 0.65rem; align-items: center; padding: 0.55rem 0.7rem;
        border-bottom: 1px solid var(--p-border);
        font-size: 0.82rem;
        transition: background 0.15s; cursor: pointer; color: var(--text-color);
    }
    .alert-row.history { grid-template-columns: 88px 86px minmax(0,1fr); cursor: default; }
    .alert-row:hover { background: var(--secondary-background-color); }
    .alert-time {
        font-variant-numeric: tabular-nums; color: var(--text-color);
        opacity: 0.55; font-size: 0.73rem; font-weight: 500;
    }
    .alert-truck  { font-variant-numeric: tabular-nums; font-weight:700; color:var(--text-color); }
    .alert-savings { text-align:right; font-variant-numeric: tabular-nums; color:var(--p-ok); font-weight:700; }
    .alert-savings.dash { color: var(--text-color); opacity: 0.3; font-weight: 400; }
    .alert-meta   { font-size:0.7rem; color:var(--text-color); opacity:0.45; margin-top:0.1rem; }
    .alert-reco   { font-size:0.73rem; color:var(--text-color); opacity:0.7; margin-top:0.15rem; }
    .alert-reco b { font-weight:600; }
    .alert-message { min-width:0; overflow-wrap:anywhere; color:var(--text-color); }

    /* ── Filter-Buttons ── */
    .filter-buttons {
        display: flex; gap: 0.55rem; margin: 0.2rem 0 0.8rem 0;
        align-items: center; flex-wrap: wrap;
    }
    .filter-link {
        display: inline-flex; align-items: center; justify-content: center;
        min-width: 88px; min-height: 2.1rem; padding: 0.38rem 0.65rem;
        border-radius: 4px; border: 1px solid var(--p-border);
        background: rgba(120,120,140,0.1); color: var(--text-color);
        font-family: 'Plus Jakarta Sans','Inter',sans-serif;
        font-size: 0.73rem; font-weight: 700; letter-spacing: 0.06em;
        text-decoration: none !important; transition: all 0.15s ease;
    }
    .filter-link:hover { border-color: var(--p-brand); }
    .filter-link.active {
        background: var(--p-brand); border-color: var(--p-brand);
        color: #FFF;
    }
    .filter-link,.filter-link:visited,.filter-link:hover,.filter-link:active {
        text-decoration: none !important;
    }

    /* ── Section-Titel ── */
    .section-title {
        font-size: 1.1rem; font-weight: 700; margin: 1rem 0 0.4rem 0;
        color: var(--text-color); letter-spacing: -0.01em;
        font-family: 'Plus Jakarta Sans',sans-serif;
        display: flex; align-items: center; gap: 0.6rem;
    }
    .section-icon { font-size: 1.2rem; }
    .section-sub {
        font-size: 0.72rem; color: var(--text-color); opacity: 0.55;
        margin-bottom: 0.6rem; font-family: 'Inter',sans-serif;
        letter-spacing: 0.04em; text-transform: uppercase;
    }

    /* ── Empfehlungs-Banner ── */
    .reco-banner {
        background: color-mix(in srgb, #E5484D 7%, var(--background-color));
        padding: 0.9rem 1rem; border-radius: 6px; margin-bottom: 1rem;
        box-shadow: var(--p-shadow);
    }
    .reco-banner.warning {
        background: color-mix(in srgb, #F59E0B 7%, var(--background-color));
    }
    .reco-banner.ok {
        background: color-mix(in srgb, #10B981 7%, var(--background-color));
    }
    .reco-title {
        font-weight: 700; font-size: 0.9rem; margin-bottom: 0.2rem;
        color: var(--text-color); font-family: 'Plus Jakarta Sans',sans-serif;
        display: flex; align-items: center; gap: 0.4rem;
    }
    .reco-text { font-size: 0.8rem; color: var(--text-color); opacity: 0.75; line-height: 1.4; }

    /* ── Tabellen-Container: ein HTML-Block für Header + Zeilen.
       Bewusst rahmenlos – nur die Trennlinien zwischen den Zeilen. ── */
    .table-card { margin-bottom: 0.6rem; }
    .table-card .row-link:last-child .truck-table-row,
    .table-card .row-link:last-child .alert-row,
    .table-card .row-link:last-child .plan-row,
    .table-card > .alert-row:last-child { border-bottom: none; }

    /* ── Flotten-Tabelle ── */
    .truck-table-header {
        display: grid;
        grid-template-columns: 28px minmax(70px,0.8fr) minmax(110px,1.2fr) minmax(85px,0.9fr) minmax(75px,0.8fr) minmax(100px,1fr) 86px;
        gap: 0.7rem; padding: 0.55rem 0.8rem;
        background: transparent;
        color: var(--text-color); opacity: 0.65;
        font-size: 0.68rem; text-transform: uppercase;
        letter-spacing: 0.08em; font-family: 'Plus Jakarta Sans','Inter',sans-serif;
        border-bottom: 2px solid var(--p-border); font-weight: 700;
    }
    .truck-table-row {
        display: grid;
        grid-template-columns: 28px minmax(70px,0.8fr) minmax(110px,1.2fr) minmax(85px,0.9fr) minmax(75px,0.8fr) minmax(100px,1fr) 86px;
        gap: 0.7rem; padding: 0.55rem 0.8rem;
        border-bottom: 1px solid var(--p-border);
        background: var(--background-color);
        font-size: 0.85rem; align-items: center;
        transition: background 0.15s ease;
        cursor: pointer; color: var(--text-color);
    }
    .truck-icon { font-size: 1.1rem; text-align: center; }
    .truck-table-row:hover {
        background: var(--secondary-background-color);
    }
    .truck-table-row.critical {
        background: color-mix(in srgb, #E5484D 9%, var(--background-color));
    }
    .truck-table-row.warning {
        background: color-mix(in srgb, #F59E0B 8%, var(--background-color));
    }
    .row-link,.row-link *,.row-link:visited,.row-link:hover,.row-link:active {
        color: inherit !important; text-decoration: none !important;
    }
    .row-link { display: block; }
    .truck-id { font-variant-numeric: tabular-nums; font-weight:700; color:var(--text-color); }

    /* ── Dark-Mode: stärkere Rahmen & Schatten ──────────────────── */
    [data-theme="dark"] {
        --p-border:  rgba(255,255,255,0.14);
        --p-shadow:  0 2px 8px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.07);
        --p-shadow-hover: 0 6px 18px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.12);
    }
    [data-theme="dark"] .kpi-card.critical { background: color-mix(in srgb, #E5484D 10%, var(--background-color)); }
    [data-theme="dark"] .kpi-card.warning  { background: color-mix(in srgb, #F59E0B 10%, var(--background-color)); }
    [data-theme="dark"] .kpi-card.ok       { background: color-mix(in srgb, #10B981 10%, var(--background-color)); }
    [data-theme="dark"] .truck-table-row.critical { background: color-mix(in srgb, #E5484D  8%, var(--background-color)); }
    [data-theme="dark"] .truck-table-row.warning  { background: color-mix(in srgb, #F59E0B  8%, var(--background-color)); }
    [data-theme="dark"] .sensor-bar-bg { background: rgba(255,255,255,0.1); }
    [data-theme="dark"] .filter-link   { background: rgba(255,255,255,0.07); }
    [data-theme="dark"] .nav-button.inactive { background: rgba(255,255,255,0.07); }

    /* ── Streamlit-Overrides ── */
    .stButton>button {
        font-family: 'Plus Jakarta Sans','Inter',sans-serif;
        font-size: 0.75rem; font-weight: 700; letter-spacing: 0.06em;
        white-space: nowrap; border-radius: 4px; transition: all 0.15s ease;
    }
    .stDownloadButton>button {
        font-family: 'Plus Jakarta Sans','Inter',sans-serif;
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.05em;
        border-radius: 4px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Plus Jakarta Sans','Inter',sans-serif;
        font-size: 0.78rem; font-weight: 600;
        letter-spacing: 0.06em; padding: 0.45rem 0.9rem;
    }

    /* ── Rollen-Auswahl ── */
    .role-screen { max-width: 680px; margin: 3rem auto; text-align: center; }
    .role-screen-brand {
        font-family: 'Plus Jakarta Sans', sans-serif; font-size: 2.1rem; font-weight: 700;
        letter-spacing: 0.18em; color: var(--text-color); margin-bottom: 0.25rem;
    }
    .role-screen-brand-accent { color: var(--p-brand); }
    .role-screen-tagline {
        font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase;
        color: var(--text-color); opacity: 0.35; margin-bottom: 2.6rem;
    }
    .role-screen-heading {
        font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.35rem; font-weight: 700;
        color: var(--text-color); margin-bottom: 0.3rem;
    }
    .role-screen-sub {
        font-size: 0.8rem; color: var(--text-color); opacity: 0.5; margin-bottom: 1.6rem;
    }
    .role-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.1rem; }
    .role-card {
        background: var(--background-color); border: 1px solid var(--p-border); border-radius: 12px;
        padding: 1.6rem 1.5rem 1.4rem; cursor: pointer; transition: all 0.18s ease;
        display: block; color: inherit; position: relative; overflow: hidden;
        text-align: left; box-shadow: var(--p-shadow);
    }
    .role-card:hover { transform: translateY(-3px); box-shadow: 0 12px 32px rgba(0,0,0,0.12); border-color: rgba(37,99,235,0.3); }
    .role-card,.role-card *,.role-card:visited,.role-card:hover,.role-card:active { text-decoration: none !important; color: inherit; }
    .role-card-head { display: flex; align-items: center; gap: 0.85rem; margin-bottom: 1.15rem; }
    .role-card-avatar {
        width: 46px; height: 46px; border-radius: 50%; flex: none;
        display: flex; align-items: center; justify-content: center;
        background: linear-gradient(135deg, var(--p-brand), #60A5FA);
        color: #FFF; font-weight: 800; font-size: 0.95rem;
        font-family: 'Plus Jakarta Sans', sans-serif;
        box-shadow: 0 4px 12px rgba(37,99,235,0.25);
    }
    .role-card-type {
        font-family: 'Plus Jakarta Sans', sans-serif; font-size: 0.6rem; font-weight: 700;
        letter-spacing: 0.18em; text-transform: uppercase; color: var(--p-brand); margin-bottom: 0.15rem;
    }
    .role-card-name {
        font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.15rem; font-weight: 700;
        color: var(--text-color); line-height: 1.2;
    }
    .role-card-features { list-style: none; padding: 0; margin: 0 0 1.4rem 0; }
    .role-card-features li {
        display: flex; align-items: center; gap: 0.5rem;
        font-size: 0.77rem; padding: 0.28rem 0; color: var(--text-color); opacity: 0.65;
    }
    .role-card-features li svg { width: 12px; height: 12px; color: var(--p-ok); flex: none; }
    .role-card-cta {
        display: inline-flex; align-items: center; gap: 0.35rem;
        font-family: 'Plus Jakarta Sans', sans-serif; font-size: 0.74rem; font-weight: 700;
        color: var(--p-brand); letter-spacing: 0.05em;
        border: 1px solid var(--p-brand); border-radius: 6px;
        padding: 0.45rem 0.95rem; transition: all 0.15s ease;
    }
    .role-card-cta svg { width: 12px; height: 12px; flex: none; }
    .role-card:hover .role-card-cta {
        background: var(--p-brand); color: #FFF !important;
    }
    .role-screen-note {
        display: flex; align-items: center; justify-content: center; gap: 0.4rem;
        font-size: 0.67rem; color: var(--text-color); opacity: 0.45; margin-top: 1.6rem;
        font-family: 'Inter', sans-serif;
    }
    .role-screen-note svg { width: 11px; height: 11px; flex: none; }

    /* ── Login-Karte (st.form gibt es nur auf der Login-Seite) ── */
    [data-testid="stForm"] {
        background: var(--background-color);
        border: 1px solid var(--p-border);
        border-radius: 12px;
        padding: 2rem 1.8rem 1.3rem;
        box-shadow: var(--p-shadow-hover);
        animation: fadeUp 0.35s ease both;
    }
    .login-avatar {
        width: 60px; height: 60px; border-radius: 50%;
        margin: 0 auto 0.85rem;
        display: flex; align-items: center; justify-content: center;
        background: linear-gradient(135deg, var(--p-brand), #60A5FA);
        color: #FFF; font-weight: 800; font-size: 1.25rem;
        font-family: 'Plus Jakarta Sans', sans-serif; letter-spacing: 0.02em;
        box-shadow: 0 6px 16px rgba(37,99,235,0.25);
    }
    .login-name {
        font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.25rem;
        font-weight: 700; color: var(--text-color); margin-bottom: 0.2rem;
    }
    .login-field-label {
        font-size: 0.64rem; text-transform: uppercase; letter-spacing: 0.14em;
        font-weight: 700; color: var(--text-color); opacity: 0.55;
        margin: 0.4rem 0 0.35rem 0.1rem;
        font-family: 'Plus Jakarta Sans','Inter',sans-serif;
    }
    .login-divider { height: 1px; background: var(--p-border); margin: 1.1rem 0 0.75rem; }
    .login-note {
        display: flex; align-items: center; justify-content: center; gap: 0.4rem;
        font-size: 0.67rem; color: var(--text-color); opacity: 0.5;
        font-family: 'Inter', sans-serif;
    }
    .login-note svg { width: 11px; height: 11px; flex: none; }

    /* ── Mikro-Animationen & Akzente ── */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(4px); }
        to   { opacity: 1; transform: none; }
    }
    .kpi-card, .reco-banner, .role-card, .table-card { animation: fadeUp 0.3s ease both; }
    @media (prefers-reduced-motion: reduce) {
        * { animation: none !important; transition: none !important; }
    }
    a:focus-visible, button:focus-visible {
        outline: 2px solid var(--p-brand); outline-offset: 2px;
    }

    .feed-day {
        font-size: 0.66rem; font-weight: 700; letter-spacing: 0.12em;
        text-transform: uppercase; color: var(--text-color); opacity: 0.6;
        padding: 0.55rem 0.8rem 0.4rem; margin: 0;
        background: var(--secondary-background-color);
        font-family: 'Plus Jakarta Sans','Inter',sans-serif;
    }

    /* ── Live-Demo-Leiste ── */
    .replay-bar {
        display: flex; align-items: center; gap: 0.7rem;
        padding: 0.7rem 1.1rem; margin-bottom: 0.9rem;
        border: 1px solid var(--p-brand); border-radius: 8px;
        background: color-mix(in srgb, #2563EB 7%, var(--background-color));
        font-size: 0.8rem; font-variant-numeric: tabular-nums;
        color: var(--text-color); font-weight: 600; letter-spacing: 0.05em;
    }
    .replay-dot {
        width: 10px; height: 10px; border-radius: 50%; flex: none;
        background: var(--p-brand); animation: replayPulse 1.4s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    @keyframes replayPulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.35; transform: scale(0.85); }
    }
    .replay-time { font-weight: 700; color: var(--p-brand); }
    .replay-progress {
        flex: 1; height: 6px; border-radius: 3px;
        background: rgba(120, 120, 140, 0.2); overflow: hidden;
    }
    .replay-progress-fill {
        height: 100%; border-radius: 3px;
        background: var(--p-brand);
        transition: width 0.4s ease;
    }

    /* ── Wartungsplaner ── */
    .plan-row {
        display: grid;
        grid-template-columns: 26px 86px 86px minmax(110px,0.8fr) minmax(160px,1.6fr) 96px;
        gap: 0.65rem; align-items: center; padding: 0.55rem 0.8rem;
        border-bottom: 1px solid var(--p-border);
        font-size: 0.82rem; color: var(--text-color);
        transition: background 0.15s; cursor: pointer;
    }
    .plan-row:hover { background: var(--secondary-background-color); }
    .plan-pos {
        font-variant-numeric: tabular-nums; font-weight: 700;
        color: var(--text-color); opacity: 0.45;
    }
    .plan-deadline { font-weight: 700; font-variant-numeric: tabular-nums; }
    .plan-deadline.now { color: var(--p-crit); }
    .plan-reco { font-size: 0.74rem; opacity: 0.7; min-width: 0; overflow-wrap: anywhere; }
    .plan-rul { text-align: right; font-variant-numeric: tabular-nums; opacity: 0.7; }

    /* ── Responsive: schmale Viewports ── */
    @media (max-width: 760px) {
        .role-grid { grid-template-columns: 1fr; }
        .role-screen { margin: 1.5rem auto; }
        .kpi-grid { grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }
        .truck-table-header, .truck-table-row { font-size: 0.72rem; gap: 0.4rem; }
        .alert-row { font-size: 0.74rem; gap: 0.4rem; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Helper functions
# ============================================================================
def fmt_de(value) -> str:
    """Ganzzahl mit deutschem Tausenderpunkt (12345 -> '12.345')."""
    try:
        return f"{int(round(float(value))):,}".replace(",", ".")
    except (TypeError, ValueError):
        return str(value)


_ICON_PATHS = {
    # Feather Icons (MIT) – monochrome Linien-Symbole, erben currentColor
    "truck": '<path d="M1 3h15v13H1z"/><path d="M16 8h4l3 3v5h-7V8z"/>'
             '<circle cx="5.5" cy="18.5" r="2.5"/><circle cx="18.5" cy="18.5" r="2.5"/>',
    "check-circle": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>'
                    '<polyline points="22 4 12 14.01 9 11.01"/>',
    "alert-triangle": '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 '
                      '1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
                      '<line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "alert-octagon": '<polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"/>'
                     '<line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>',
    "wrench": '<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1'
              '-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>',
    "info": '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/>'
            '<line x1="12" y1="8" x2="12.01" y2="8"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
    "lock": '<rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>'
            '<path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
    "check": '<polyline points="20 6 9 17 4 12"/>',
    "play": '<polygon points="6 4 20 12 6 20 6 4"/>',
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "calendar": '<rect x="3" y="4" width="18" height="18" rx="2" ry="2"/>'
                '<line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/>'
                '<line x1="3" y1="10" x2="21" y2="10"/>',
    "package": '<line x1="16.5" y1="9.4" x2="7.5" y2="4.21"/>'
               '<path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 '
               '1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>'
               '<polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/>',
    "arrow-right": '<line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>',
}


def icon(name: str) -> str:
    """Inline-SVG-Symbol (Feather-Stil) für KPI-Karten."""
    return ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round">{_ICON_PATHS[name]}</svg>')


def status_badge(status):
    cls = {"KRITISCH": "badge-critical", "WARNUNG": "badge-warning",
           "OK": "badge-ok", "INFO": "badge-info"}.get(status, "badge-info")
    return f'<span class="badge {cls}">{status}</span>'


def severity_color(severity):
    return {"KRITISCH": COLOR_CRIT, "WARNUNG": COLOR_WARN,
            "INFO": COLOR_INFO, "OK": COLOR_OK}.get(severity, "#8A8A8F")


def styled_chart(chart):
    """Einheitliches Altair-Theme: Inter, dezente Gridlines, keine Achsenlinien."""
    return chart.configure_view(strokeWidth=0).configure_axis(
        labelFont="Inter", titleFont="Inter",
        labelColor="#8A8A93", titleColor="#8A8A93",
        gridColor="rgba(120,120,140,0.15)",
        domainOpacity=0, tickOpacity=0,
    )


def gradient_area(df, x_enc, y_enc, color, baseline=None):
    """Verlaufsfläche unter einer Linie – gemeinsames Stilmittel aller Zeitreihen.

    baseline: unterer Rand der Fläche. Nötig, wenn die Y-Achse nicht bei 0
    beginnt – ohne y2 füllt Vega-Lite bis zur Null-Basislinie und die Fläche
    läuft unter den Diagrammbereich hinaus.
    """
    encodings = {"x": x_enc, "y": y_enc}
    if baseline is not None:
        encodings["y2"] = alt.datum(baseline)
    return alt.Chart(df).mark_area(
        color=alt.Gradient(
            gradient="linear",
            stops=[alt.GradientStop(color="transparent", offset=0),
                   alt.GradientStop(color=color, offset=1)],
            x1=1, x2=1, y1=1, y2=0,
        ),
        opacity=0.16,
    ).encode(**encodings)


def _role_param():
    """Hängt Rolle (und im Passwort-Modus das Auth-Token) an interne Links.

    Jeder Link-Klick lädt die Seite neu und startet eine frische
    Streamlit-Session — ohne Token in der URL wäre der Login nach jeder
    Navigation wieder weg.
    """
    r = st.session_state.get("role") or ""
    if not r:
        return ""
    token = _auth_token(r)
    if token and st.session_state.get(f"auth_{r}"):
        return f"&role={r}&auth={token}"
    return f"&role={r}"


def detail_href(truck_id):
    return f"?view=detail&truck={quote(str(truck_id))}{_role_param()}"


def fleet_href():
    return f"?view=fleet{_role_param()}"


def alerts_href(filter_severity):
    return f"?view=alerts&filter={quote(str(filter_severity))}{_role_param()}"


def plan_href():
    return f"?view=plan{_role_param()}"


def demo_href():
    return f"?view=fleet&demo=1{_role_param()}"


# ============================================================================
# Data loading
# ============================================================================
def _sync_from_tracking(data_dir: Path) -> str | None:
    """Generiert die CSVs aus dem tracking-Repo, falls die Quelldaten vorhanden sind.

    Gibt eine Fehlermeldung zurück statt sie anzuzeigen, damit der Aufrufer
    sie auch bei Cache-Treffern konsistent behandeln kann.
    """
    adapter = data_dir / "generate_from_tracking.py"
    tracking_dir = data_dir.parent / "pipeline" / "data" / "engine_health"
    if not adapter.exists() or not tracking_dir.exists():
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location("generate_from_tracking", adapter)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        mod.main()
        return None
    except Exception as e:
        return f"Tracking-Sync fehlgeschlagen: {e}"


@st.cache_data(ttl=300, show_spinner="Lade Flottendaten …")
def load_data():
    base = Path(__file__).parent / "data"
    sync_error = _sync_from_tracking(base)

    required = ["fleet.csv", "timeseries.csv", "alerts.csv", "truck_alerts.csv"]
    missing = [f for f in required if not (base / f).exists()]
    if missing:
        return None, None, None, None, None, None, None, None, sync_error, missing

    fleet = pd.read_csv(base / "fleet.csv")
    timeseries = pd.read_csv(base / "timeseries.csv", parse_dates=["timestamp"])
    alerts = pd.read_csv(base / "alerts.csv", parse_dates=["timestamp"])
    truck_alerts = pd.read_csv(base / "truck_alerts.csv", parse_dates=["timestamp"])

    # Replay-Daten (Live-Demo-Modus) sind optional: ohne sie läuft die App
    # unverändert, nur der Demo-Schalter fehlt.
    replay = pd.DataFrame()
    replay_alerts = pd.DataFrame()
    try:
        if (base / "replay.csv").exists():
            replay = pd.read_csv(base / "replay.csv", parse_dates=["timestamp"])
        if (base / "replay_alerts.csv").exists():
            replay_alerts = pd.read_csv(base / "replay_alerts.csv", parse_dates=["timestamp"])
    except (ValueError, OSError):
        replay, replay_alerts = pd.DataFrame(), pd.DataFrame()

    # RUL-Prognose-Verlauf (optional, vom Adapter erzeugt)
    rul_history = pd.DataFrame()
    try:
        if (base / "rul_history.csv").exists():
            rul_history = pd.read_csv(base / "rul_history.csv", parse_dates=["timestamp"])
    except (ValueError, OSError):
        rul_history = pd.DataFrame()

    metrics = {}
    metrics_path = base / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            metrics = {}

    return (fleet, timeseries, alerts, truck_alerts, replay, replay_alerts,
            rul_history, metrics, sync_error, [])


(fleet, timeseries, alerts, truck_alerts, replay, replay_alerts,
 rul_history, ml_metrics, _sync_error, _missing) = load_data()

if _missing:
    st.error(
        "Datendateien fehlen: " + ", ".join(_missing) +
        ". Bitte zuerst die Pipeline ausführen "
        "(`python data/generate_from_tracking.py` bzw. README, Abschnitt »Lokale Ausführung«)."
    )
    if _sync_error:
        st.warning(_sync_error)
    st.stop()

if _sync_error:
    st.warning(_sync_error)

# ============================================================================
# Session state & URL routing
# ============================================================================
if "role" not in st.session_state:
    st.session_state.role = None
if "view" not in st.session_state:
    st.session_state.view = "fleet"
if "selected_truck" not in st.session_state:
    st.session_state.selected_truck = None
if "alert_filter" not in st.session_state:
    st.session_state.alert_filter = "ALLE"

KNOWN_TRUCKS = set(fleet["lkw_id"])


def _password_for(role: str) -> str | None:
    """Optionales Passwort aus st.secrets (NFR Zugriffsschutz). None = Demo-Modus."""
    try:
        return st.secrets["passwords"][role]
    except Exception:
        return None


def _auth_token(role: str) -> str | None:
    """Hash-Token für die URL-Navigation im Passwort-Modus (MVP-Lösung,
    das Klartext-Passwort landet nie in der URL)."""
    pw = _password_for(role)
    if pw is None:
        return None
    return hashlib.sha256(pw.encode()).hexdigest()[:16]


query_role = st.query_params.get("role")
if query_role in ("fm", "wl"):
    st.session_state.role = query_role

# Rollen mit konfiguriertem Passwort erst nach Login freischalten.
# Ein gültiges Auth-Token in der URL zählt als Anmeldung, damit der Login
# Link-Klicks (= neue Streamlit-Session) übersteht.
if st.session_state.role:
    _r = st.session_state.role
    if _password_for(_r) is not None and not st.session_state.get(f"auth_{_r}"):
        if st.query_params.get("auth") == _auth_token(_r):
            st.session_state[f"auth_{_r}"] = True
        else:
            st.session_state.pending_role = _r
            st.session_state.role = None

query_view = st.query_params.get("view")
if query_view == "fleet":
    st.session_state.view = "fleet"
    st.session_state.selected_truck = None
elif query_view == "alerts":
    st.session_state.view = "alerts"
    qf = st.query_params.get("filter", "ALLE")
    st.session_state.alert_filter = qf if qf in ALERT_FILTERS else "ALLE"
elif query_view == "plan":
    st.session_state.view = "plan"
    st.session_state.selected_truck = None
elif query_view == "detail":
    query_truck = st.query_params.get("truck")
    if query_truck in KNOWN_TRUCKS:
        st.session_state.view = "detail"
        st.session_state.selected_truck = query_truck

# Rollen-Guards: Werkstattleiter hat keinen Alert-Feed, Flottenmanager
# keinen Wartungsplan – jeweils auf die Flottenübersicht umleiten.
if st.session_state.role == "wl" and st.session_state.view == "alerts":
    st.session_state.view = "fleet"
if st.session_state.role == "fm" and st.session_state.view == "plan":
    st.session_state.view = "fleet"

# ============================================================================
# Header
# ============================================================================
_role_labels = {
    "fm": "Thomas Müller · Flottenmanager",
    "wl": "Stefan Berger · Werkstattleiter",
}
# Header-Bar nur für angemeldete Nutzer: die Rollenauswahl-/Login-Seite hat
# ihr eigenes zentriertes Branding. Bewusst als Einzeiler – Zeilen, die nur
# aus Leerzeichen bestehen, beenden in Markdown den HTML-Block und ließen
# das schließende </div> als Code-Schnipsel erscheinen.
if st.session_state.role:
    _h_user = _role_labels.get(st.session_state.role, "")
    _switch_link = (
        ' &nbsp;<a href="?" target="_self" style="color:rgba(255,255,255,0.45);'
        'font-size:0.68rem;letter-spacing:0.05em;text-decoration:none!important;">Abmelden</a>'
    )
    st.markdown(
        '<div class="header-bar">'
        f'<div class="header-brand"><span class="header-logo">{icon("truck")}</span>'
        '<span>PRE<span class="header-brand-accent">MA</span></span>'
        '<span class="header-tagline">Predictive Maintenance</span></div>'
        f'<div class="header-user">{_h_user}{_switch_link}</div>'
        '</div>',
        unsafe_allow_html=True,
    )

if st.session_state.role:
    n_crit_alerts = int((alerts["severity"] == "KRITISCH").sum())
    _fnc = "nav-button active" if st.session_state.view == "fleet" else "nav-button inactive"
    _nav_html = f'<div class="nav-buttons"><a class="{_fnc}" href="{fleet_href()}" target="_self">FLOTTE</a>'
    if st.session_state.role == "fm":
        _anc = "nav-button active" if st.session_state.view == "alerts" else "nav-button inactive"
        _alerts_label = (f'ALERTS <span class="nav-count">{n_crit_alerts}</span>'
                         if n_crit_alerts else "ALERTS")
        _nav_html += f'<a class="{_anc}" href="{alerts_href("ALLE")}" target="_self">{_alerts_label}</a>'
    if st.session_state.role == "wl":
        _pnc = "nav-button active" if st.session_state.view == "plan" else "nav-button inactive"
        n_due = int(fleet["status"].isin(("KRITISCH", "WARNUNG")).sum())
        _plan_label = (f'WARTUNGSPLAN <span class="nav-count">{n_due}</span>'
                       if n_due else "WARTUNGSPLAN")
        _nav_html += f'<a class="{_pnc}" href="{plan_href()}" target="_self">{_plan_label}</a>'
    if st.session_state.view == "detail" and st.session_state.selected_truck:
        _nav_html += f'<span class="nav-button detail">{st.session_state.selected_truck}</span>'
    # Live-Demo: rechtsbündiger Pill, nur auf der Flottenansicht und wenn
    # der Adapter Replay-Daten erzeugt hat. Läuft über den Query-Parameter
    # ?demo=1, damit Start/Stopp wie die restliche Navigation funktioniert.
    if st.session_state.view == "fleet" and replay is not None and not replay.empty:
        if st.query_params.get("demo") == "1":
            _nav_html += (f'<a class="nav-button demo running" href="{fleet_href()}" target="_self">'
                          f'<span class="replay-dot"></span>DEMO BEENDEN</a>')
        else:
            _nav_html += (f'<a class="nav-button demo" href="{demo_href()}" target="_self">'
                          f'{icon("play")}LIVE-DEMO</a>')
    _nav_html += '</div>'
    st.markdown(_nav_html, unsafe_allow_html=True)

# ============================================================================
# Feedback helper (FA-8)
# ============================================================================
def save_feedback(truck_id: str, verdict: str, status: str) -> None:
    """Persistiert Werkstatt-Feedback als Trainingslabel für den Retraining-Loop."""
    feedback_path = Path(__file__).parent / "data" / "feedback.csv"
    new_row = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "lkw_id": truck_id,
        "verdict": verdict,
        "status": status,
    }])
    if feedback_path.exists():
        existing = pd.read_csv(feedback_path)
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_csv(feedback_path, index=False)


def last_feedback(truck_id: str):
    """Letzter Feedback-Eintrag für ein Fahrzeug (oder None)."""
    feedback_path = Path(__file__).parent / "data" / "feedback.csv"
    if not feedback_path.exists():
        return None
    try:
        df = pd.read_csv(feedback_path)
        # Timestamps stammen aus isoformat(); to_datetime mit coerce ist
        # robust gegen alte/abweichende Einträge (parse_dates wäre es nicht).
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    except (ValueError, OSError, KeyError):
        return None
    rows = df[df["lkw_id"] == truck_id]
    if rows.empty:
        return None
    return rows.sort_values("timestamp").iloc[-1]


# ============================================================================
# Shared: ML-Metriken (FA-6)
# ============================================================================
def render_ml_metrics():
    if not ml_metrics:
        return
    with st.expander("ML-Modellgüte · Pipeline-Reports"):
        c1, c2, c3, c4 = st.columns(4)
        if "clf_accuracy" in ml_metrics:
            c1.metric("Accuracy (XGBoost)", f"{ml_metrics['clf_accuracy']*100:.1f} %")
        if "clf_recall_critical" in ml_metrics:
            c2.metric("Recall KRITISCH", f"{ml_metrics['clf_recall_critical']*100:.1f} %",
                      help="Sicherheitspriorität: kein kritischer Zustand darf unerkannt bleiben (Ziel ≥ 85 %).")
        if "rul_mae_days" in ml_metrics:
            c3.metric("RUL-MAE", f"{ml_metrics['rul_mae_days']:.1f} Tage",
                      help="Mittlerer absoluter Fehler der Restlaufzeit-Prognose (Random Forest).")
        if "rul_r2" in ml_metrics:
            c4.metric("RUL-R²", f"{ml_metrics['rul_r2']:.2f}")
        sub = []
        if ml_metrics.get("anomaly_model"):
            sub.append(f"Anomalie-Modell: {ml_metrics['anomaly_model']}")
        if ml_metrics.get("source_file"):
            sub.append(f"Datenquelle: {ml_metrics['source_file']}")
        if ml_metrics.get("generated_at"):
            sub.append(f"Stand: {ml_metrics['generated_at'][:16].replace('T', ' ')}")
        if sub:
            st.caption(" · ".join(sub))


# ============================================================================
# SCREEN 1: FLEET OVERVIEW
# ============================================================================
def _overview_kpis(fdf):
    """KPI-Karten aus einem Flotten-Snapshot (statisch oder Replay-Stand)."""
    n_total = len(fdf)
    n_ok = int((fdf["status"] == "OK").sum())
    n_warn = int((fdf["status"] == "WARNUNG").sum())
    n_crit = int((fdf["status"] == "KRITISCH").sum())
    avoided_eur = n_crit * COST_PER_HOUR_EUR * HOURS_PER_BREAKDOWN
    avg_rul = int(fdf["rul_hours"].mean())

    # KPI-Karten 3 und 4 sind rollenspezifisch: nur der Flottenmanager hat
    # einen Alert-Feed, daher verlinken die Karten nur bei ihm dorthin;
    # der Werkstattleiter sieht statt der Kosten-KPI die Ø-Restlaufzeit.
    # Hinweis: keine Zeilen-Einrückung im HTML, sonst interpretiert Markdown
    # die Karten als Code-Block und das Grid verzieht sich.
    def _card(cls, icon_name, label, value, sub, href=None):
        card = (f'<div class="kpi-card {cls}">'
                f'<div class="kpi-card-header"><span class="kpi-card-icon">{icon(icon_name)}</span>'
                f'<div class="kpi-label">{label}</div></div>'
                f'<div class="kpi-value">{value}</div>'
                f'<div class="kpi-sub">{sub}</div></div>')
        if href:
            return f'<a class="kpi-link" href="{href}" target="_self">{card}</a>'
        return card

    is_wl = st.session_state.get("role") == "wl"
    cards = [
        _card("", "truck", "Fahrzeuge gesamt", n_total, "aktive Flotte"),
        _card("ok", "check-circle", "Status OK", n_ok, f"{n_ok / n_total * 100:.0f}% der Flotte"),
        _card("warning", "alert-triangle", "Warnung", n_warn, "Wartung in &lt; 14 Tagen",
              href=None if is_wl else alerts_href("WARNUNG")),
        _card("", "wrench", "Ø RUL Flotte", f"{fmt_de(avg_rul)} h", "Durchschn. Restlaufzeit")
        if is_wl else
        _card("critical", "alert-octagon", "Kritisch", n_crit,
              f"~ {fmt_de(avoided_eur)} EUR verhinderte Kosten",
              href=alerts_href("KRITISCH")),
    ]
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def _overview_table(fdf):
    """Flotten-Tabelle; gibt die sortierte Flotte zurück."""
    # Sort: critical first, then warning, then OK; innerhalb gleicher Stufe
    # entscheidet die kürzeste Restlaufzeit.
    status_order = {"KRITISCH": 0, "WARNUNG": 1, "OK": 2}
    fleet_sorted = fdf.copy()
    fleet_sorted["sort_key"] = fleet_sorted["status"].map(status_order)
    fleet_sorted = fleet_sorted.sort_values(["sort_key", "rul_hours"]).drop(columns=["sort_key"])

    # Header + Zeilen als ein HTML-Block, damit der Card-Container
    # (.table-card) alles umschließen kann.
    parts = ['<div class="table-card">'
             '<div class="truck-table-header">'
             '<div></div><div>LKW-ID</div><div>Fahrer</div><div>Motortemp.</div>'
             '<div>Bremse %</div><div>RUL (Stunden)</div><div>Status</div></div>']
    for _, truck in fleet_sorted.iterrows():
        row_class = "truck-table-row"
        if truck["status"] == "KRITISCH":
            row_class += " critical"
        elif truck["status"] == "WARNUNG":
            row_class += " warning"
        dot_cls = str(truck["status"]).lower()
        parts.append(
            f'<a class="row-link" href="{detail_href(truck["lkw_id"])}" target="_self">'
            f'<div class="{row_class}">'
            f'<div class="truck-icon"><span class="status-dot status-dot-{dot_cls}"></span></div>'
            f'<div class="truck-id">{truck["lkw_id"]}</div>'
            f'<div>{truck["driver"]}</div>'
            f'<div>{truck["motor_temp_c"]:.0f} °C</div>'
            f'<div>{truck["brake_fluid_pct"]:.0f} %</div>'
            f'<div>{fmt_de(truck["rul_hours"])} h</div>'
            f'<div>{status_badge(truck["status"])}</div>'
            f'</div></a>'
        )
    parts.append('</div>')
    st.markdown("".join(parts), unsafe_allow_html=True)
    return fleet_sorted


def render_maintenance_plan():
    """SCREEN (nur Werkstattleiter): priorisierte Terminliste aus Status + RUL.

    Terminvorschlag = jetzt + PLANNER_SAFETY * RUL, d.h. die Wartung liegt
    mit Sicherheitspuffer vor dem prognostizierten kritischen Zustand.
    """
    head_l, head_r = st.columns([5, 1], vertical_alignment="bottom")
    with head_l:
        st.markdown('<div class="section-title">Wartungsplan</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="section-sub">TERMINVORSCHLAG = {PLANNER_SAFETY:.0%} DER '
            f'PROGNOSTIZIERTEN RESTLAUFZEIT · PRIORISIERT NACH DRINGLICHKEIT</div>',
            unsafe_allow_html=True,
        )
    status_order = {"KRITISCH": 0, "WARNUNG": 1, "OK": 2}
    due = fleet[fleet["status"].isin(("KRITISCH", "WARNUNG"))].copy()
    due["sort_key"] = due["status"].map(status_order)
    due = due.sort_values(["sort_key", "rul_hours"]).drop(columns=["sort_key"])
    if due.empty:
        st.info("Keine Wartung fällig – alle Fahrzeuge im Normalbetrieb.")
        return

    # Konkrete Maßnahme aus dem letzten Alert des Fahrzeugs (DTC-Empfehlung)
    last_reco = {}
    if not truck_alerts.empty and "recommendation" in truck_alerts.columns:
        last = truck_alerts.sort_values("timestamp").groupby("lkw_id").tail(1)
        last_reco = dict(zip(last["lkw_id"], last["recommendation"]))

    # Einträge einmal berechnen – Bildschirm und CSV-Export (P-FA-10) nutzen
    # exakt dieselben Daten.
    now = datetime.now()
    entries = []
    for pos, (_, t) in enumerate(due.iterrows(), start=1):
        slack_h = float(t["rul_hours"]) * PLANNER_SAFETY
        deadline = now + timedelta(hours=slack_h)
        reco = last_reco.get(t["lkw_id"])
        if not isinstance(reco, str) or not reco.strip():
            reco = ("Fahrzeug aus dem Verkehr ziehen, Werkstatt sofort"
                    if t["status"] == "KRITISCH" else "Wartungstermin einplanen")
        entries.append({
            "Priorität": pos,
            "LKW-ID": t["lkw_id"],
            "Fahrer": t["driver"],
            "Status": t["status"],
            "RUL (Stunden)": int(t["rul_hours"]),
            "Termin bis": "SOFORT" if slack_h <= 24 else f"{deadline:%d.%m.%Y}",
            "Maßnahme": reco,
            "_deadline": deadline,
            "_sofort": slack_h <= 24,
        })

    with head_r:
        export = pd.DataFrame(entries).drop(columns=["_deadline", "_sofort"])
        st.download_button(
            "CSV-Export",
            icon=":material/download:",
            data=export.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"prema_wartungsplan_{now:%Y%m%d_%H%M}.csv",
            mime="text/csv",
            width="stretch",
        )

    rows = []
    for e in entries:
        if e["_sofort"]:
            deadline_html = '<div class="plan-deadline now">SOFORT</div>'
        else:
            deadline_html = (f'<div class="plan-deadline">bis '
                             f'{WEEKDAYS_DE[e["_deadline"].weekday()][:2]} {e["_deadline"]:%d.%m.}</div>')
        rows.append(
            f'<a class="row-link" href="{detail_href(e["LKW-ID"])}" target="_self">'
            f'<div class="plan-row">'
            f'<div class="plan-pos">{e["Priorität"]}</div>'
            f'<div class="truck-id">{e["LKW-ID"]}</div>'
            f'<div>{status_badge(e["Status"])}</div>'
            f'{deadline_html}'
            f'<div class="plan-reco">{e["Maßnahme"]}</div>'
            f'<div class="plan-rul">RUL {fmt_de(e["RUL (Stunden)"])} h</div>'
            f'</div></a>'
        )
    st.markdown('<div class="table-card">' + "".join(rows) + '</div>',
                unsafe_allow_html=True)


def _render_replay():
    """Live-Demo: spielt die simulierten Wochen nach dem Stichtag im Zeitraffer
    ab (REPLAY_HOURS_PER_TICK Stunden pro Tick). Nutzt replay.csv aus dem
    Adapter; Status/RUL sind dort wie in fleet.csv über 24 h geglättet."""
    all_ts = replay["timestamp"].drop_duplicates().sort_values().tolist()
    stride = max(REPLAY_HOURS_PER_TICK * 2, 1)  # 30-Minuten-Raster -> 2 Zeilen/h
    steps = all_ts[stride - 1::stride]
    if not steps or steps[-1] != all_ts[-1]:
        steps.append(all_ts[-1])
    start_ts = timeseries["timestamp"].max()

    @st.fragment(run_every=REPLAY_TICK_SECONDS)
    def _frame():
        i = min(st.session_state.get("replay_idx", 0), len(steps) - 1)
        ts = steps[i]
        prev_ts = steps[i - 1] if i > 0 else start_ts
        pct = (i + 1) / len(steps) * 100

        st.markdown(f"""
        <div class="replay-bar">
            <span class="replay-dot"></span>
            <span>LIVE-DEMO · ZEITRAFFER</span>
            <span class="replay-time">{WEEKDAYS_DE[ts.weekday()][:2]} {ts:%d.%m.%Y · %H:%M} Uhr</span>
            <div class="replay-progress"><div class="replay-progress-fill" style="width:{pct:.0f}%"></div></div>
        </div>
        """, unsafe_allow_html=True)

        # Benachrichtigungen: Toast wenn neue Alerts. Nur beim ersten Rendern
        # eines Schritts – am Demo-Ende läuft das Fragment weiter, ohne dass
        # sich der Schritt ändert, und würde sonst dieselben Toasts wiederholen.
        if not replay_alerts.empty and st.session_state.get("replay_toasted_i") != i:
            st.session_state.replay_toasted_i = i
            fresh = replay_alerts[(replay_alerts["timestamp"] > prev_ts)
                                  & (replay_alerts["timestamp"] <= ts)]
            for ev in fresh.itertuples():
                sev_icon = {"KRITISCH": ":material/error:",
                            "WARNUNG": ":material/warning:",
                            "INFO": ":material/info:"}.get(ev.severity, ":material/info:")
                st.toast(f"{ev.lkw_id}: {ev.message}", icon=sev_icon)

        cur = (replay[replay["timestamp"] <= ts]
               .sort_values("timestamp").groupby("lkw_id").tail(1))
        fdf = cur.merge(fleet[["lkw_id", "driver"]], on="lkw_id", how="left")
        fdf["driver"] = fdf["driver"].fillna("n. n.")
        _overview_kpis(fdf)
        _overview_table(fdf)

        if i < len(steps) - 1:
            st.session_state.replay_idx = i + 1
        else:
            st.info("Ende der Simulation erreicht – »DEMO BEENDEN« oben führt zurück zur aktuellen Ansicht.")

    _frame()


def render_fleet_overview():
    n_total = len(fleet)
    if n_total == 0:
        st.info("Keine Fahrzeugdaten vorhanden.")
        return

    # Live-Demo: gestartet/gestoppt über den Nav-Pill (?demo=1); replay_on
    # bleibt als programmatischer Schalter für Tests erhalten.
    replay_ready = replay is not None and not replay.empty
    if replay_ready and (st.query_params.get("demo") == "1"
                         or st.session_state.get("replay_on")):
        _render_replay()
        return
    st.session_state.replay_idx = 0
    st.session_state.replay_toasted_i = None

    _overview_kpis(fleet)

    _title_text = "Flottenstatus"
    head_l, head_r = st.columns([5, 1], vertical_alignment="bottom")
    with head_l:
        st.markdown(f'<div class="section-title">{_title_text}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="section-sub">SORTIERT NACH PRIORITÄT · BATCH-PIPELINE · '
            f'STAND {datetime.now().strftime("%H:%M")} UHR</div>',
            unsafe_allow_html=True,
        )

    fleet_sorted = _overview_table(fleet)

    with head_r:
        st.download_button(
            "CSV-Export",
            icon=":material/download:",
            data=fleet_sorted.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"prema_flotte_{datetime.now():%Y%m%d_%H%M}.csv",
            mime="text/csv",
            width="stretch",
        )

    render_ml_metrics()


# ============================================================================
# SCREEN 2: TRUCK DETAIL
# ============================================================================
def render_truck_detail():
    truck_id = st.session_state.selected_truck
    truck_rows = fleet[fleet["lkw_id"] == truck_id]
    if truck_rows.empty:
        st.warning(f"Fahrzeug {truck_id} nicht gefunden.")
        st.markdown(f'<a href="{fleet_href()}" target="_self">← Zurück zur Übersicht</a>',
                    unsafe_allow_html=True)
        return
    truck = truck_rows.iloc[0]

    st.markdown(
        f'<div class="section-title" style="font-size:1.35rem; margin-top:0.5rem;">'
        f'{truck_id} &nbsp;·&nbsp; {truck["driver"]}</div>',
        unsafe_allow_html=True,
    )

    # Recommendation banner
    if truck["status"] == "KRITISCH":
        banner_cls = ""
        title = "SOFORTIGE WARTUNG EMPFOHLEN"
        issues = []
        if truck["brake_fluid_pct"] < BRAKE_CRIT_PCT:
            issues.append(f"Bremsflüssigkeit {truck['brake_fluid_pct']:.0f} %")
        if truck["motor_temp_c"] > MOTOR_TEMP_CRIT_C:
            issues.append(f"Motortemperatur {truck['motor_temp_c']:.0f} °C")
        if truck["oil_pressure_bar"] < OIL_PRESSURE_WARN_BAR:
            issues.append(f"Öldruck {truck['oil_pressure_bar']:.1f} bar")
        issue_str = " · ".join(issues) if issues else "Verschleißmuster mehrerer Sensoren"
        text = (f"XGBoost-Modell stuft {truck_id} als kritisch ein. {issue_str} – "
                f"Empfehlung: Fahrzeug aus dem Verkehr ziehen, Werkstattauftrag automatisch angelegt.")
    elif truck["status"] == "WARNUNG":
        banner_cls = "warning"
        title = "VERSCHLEISS ERHÖHT · Wartung innerhalb 14 Tagen"
        text = "Verschleißmuster über Schwellwert. Wartung kann planbar in den nächsten 14 Tagen erfolgen."
    else:
        banner_cls = "ok"
        title = "FAHRZEUG IM NORMALBETRIEB"
        text = f"Alle Sensorwerte im erwarteten Bereich. Nächste turnusgemäße Wartung in {fmt_de(truck['rul_hours'])} h."

    st.markdown(f"""<div class="reco-banner {banner_cls}">
        <div class="reco-title">{status_badge(truck["status"])} &nbsp; {title}</div>
        <div class="reco-text">{text}</div>
    </div>
    """, unsafe_allow_html=True)

    # KPI-Strip: Fahrzeugdaten im selben Kartenstil wie die Flottenübersicht.
    # Die RUL-Karte übernimmt die Statusfärbung des Fahrzeugs.
    status_cls = {"KRITISCH": "critical", "WARNUNG": "warning", "OK": "ok"}.get(truck["status"], "")

    # Wartungstermin: gleiche Logik wie der Wartungsplan (PLANNER_SAFETY * RUL)
    slack_h = float(truck["rul_hours"]) * PLANNER_SAFETY
    deadline = datetime.now() + timedelta(hours=slack_h)
    if slack_h <= 24:
        termin_cls, termin_val = "critical", "SOFORT"
        termin_sub = "Fahrzeug aus dem Verkehr ziehen"
    else:
        termin_cls = ""
        termin_val = f"{WEEKDAYS_DE[deadline.weekday()][:2]} {deadline:%d.%m.}"
        termin_sub = f"bei {PLANNER_SAFETY * 100:.0f} % der Restlaufzeit"

    st.markdown(
        f'<div class="kpi-grid">'
        f'<div class="kpi-card {status_cls}">'
        f'<div class="kpi-card-header"><span class="kpi-card-icon">{icon("wrench")}</span>'
        f'<div class="kpi-label">RUL-Prognose</div></div>'
        f'<div class="kpi-value">{fmt_de(truck["rul_hours"])} h</div>'
        f'<div class="kpi-sub">Restlaufzeit · Random Forest</div></div>'
        f'<div class="kpi-card">'
        f'<div class="kpi-card-header"><span class="kpi-card-icon">{icon("activity")}</span>'
        f'<div class="kpi-label">Kilometerstand</div></div>'
        f'<div class="kpi-value">{fmt_de(truck["km_total"])} km</div>'
        f'<div class="kpi-sub">Gesamtlaufleistung</div></div>'
        f'<div class="kpi-card">'
        f'<div class="kpi-card-header"><span class="kpi-card-icon">{icon("package")}</span>'
        f'<div class="kpi-label">Beladung</div></div>'
        f'<div class="kpi-value">{truck["load_pct"]:.0f} %</div>'
        f'<div class="kpi-sub">aktuelle Auslastung</div></div>'
        f'<div class="kpi-card {termin_cls}">'
        f'<div class="kpi-card-header"><span class="kpi-card-icon">{icon("calendar")}</span>'
        f'<div class="kpi-label">Wartung bis</div></div>'
        f'<div class="kpi-value">{termin_val}</div>'
        f'<div class="kpi-sub">{termin_sub}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Two columns: sensors left, chart right
    line_color = severity_color(truck["status"])
    left, right = st.columns([1, 1.3], gap="large")

    with left:
        st.markdown('<div class="section-title">Sensordaten</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">LIVE-WERTE AUS LETZTER BATCH-INFERENZ</div>', unsafe_allow_html=True)

        # Je Sensor: Schwellenmarker (Position in %, Stufe) für die Warn-/
        # Kritisch-Grenze auf dem Balken sowie das Zahlenformat des Werts.
        sensors = [
            ("Motortemperatur", truck["motor_temp_c"], 110, "°C",
                COLOR_CRIT if truck["motor_temp_c"] > MOTOR_TEMP_CRIT_C else COLOR_OK,
                [(MOTOR_TEMP_CRIT_C / 110 * 100, "crit")], ".1f"),
            ("Öldruck", truck["oil_pressure_bar"], 5.0, "bar",
                COLOR_WARN if truck["oil_pressure_bar"] < OIL_PRESSURE_WARN_BAR else COLOR_OK,
                [(OIL_PRESSURE_WARN_BAR / 5.0 * 100, "warn")], ".1f"),
            ("Bremsflüssigkeit", truck["brake_fluid_pct"], 100, "%",
                COLOR_CRIT if truck["brake_fluid_pct"] < BRAKE_CRIT_PCT
                else (COLOR_WARN if truck["brake_fluid_pct"] < BRAKE_WARN_PCT else COLOR_OK),
                [(BRAKE_WARN_PCT, "warn"), (BRAKE_CRIT_PCT, "crit")], ".1f"),
        ]
        # Weitere Pipeline-Sensoren (ohne definierte Schwellwerte). Guards,
        # falls eine ältere fleet.csv die Spalten noch nicht enthält.
        if "engine_rpm" in truck.index and pd.notna(truck["engine_rpm"]):
            sensors.append(("Motordrehzahl", truck["engine_rpm"], 2000, "U/min", COLOR_OK, [], ".0f"))
        if "fuel_pressure_bar" in truck.index and pd.notna(truck["fuel_pressure_bar"]):
            sensors.append(("Kraftstoffdruck", truck["fuel_pressure_bar"], 20, "bar", COLOR_OK, [], ".1f"))
        if "oil_temp_c" in truck.index and pd.notna(truck["oil_temp_c"]):
            sensors.append(("Öltemperatur", truck["oil_temp_c"], 100, "°C", COLOR_OK, [], ".1f"))
        sensors += [
            ("Reifendruck VL", truck["tire_fl_bar"], 10, "bar", COLOR_OK, [], ".1f"),
            ("Reifendruck VR", truck["tire_fr_bar"], 10, "bar", COLOR_OK, [], ".1f"),
        ]
        for label, val, max_val, unit, color, marks, fmt in sensors:
            pct = max(0, min(100, val / max_val * 100))
            marks_html = "".join(
                f'<span class="sensor-mark sensor-mark-{cls}" style="left:{p:.0f}%"></span>'
                for p, cls in marks
            )
            # Eine Zeile ohne Einrückung: Leerzeilen/eingerückte Zeilen würden
            # den HTML-Block in Markdown beenden (Rest erschiene als Code).
            st.markdown(
                f'<div class="sensor-row">'
                f'<div class="sensor-label">{label}</div>'
                f'<div class="sensor-bar-bg">'
                f'<div class="sensor-bar-fill" style="width: {pct}%; background: {color};"></div>'
                f'{marks_html}</div>'
                f'<div class="sensor-value">{val:{fmt}} {unit}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # RUL-Prognose-Verlauf (30 Tage, Tagesmedian aus rul_history.csv)
        rul_truck = (rul_history[rul_history["lkw_id"] == truck_id]
                     if rul_history is not None and not rul_history.empty
                     else pd.DataFrame())
        if not rul_truck.empty:
            st.markdown('<div class="section-title">RUL-Prognose · letzte 30 Tage</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">TAGESMEDIAN DER RESTLAUFZEIT-VORHERSAGE · RANDOM FOREST</div>', unsafe_allow_html=True)
            x_rul = alt.X("timestamp:T", title=None, axis=alt.Axis(format="%d.%m.", labelFontSize=10))
            y_rul = alt.Y("rul_hours:Q", title="RUL (Stunden)",
                          axis=alt.Axis(labelFontSize=10, titleFontSize=11))
            rul_line = alt.Chart(rul_truck).mark_line(
                color=line_color, strokeWidth=2
            ).encode(
                x=x_rul, y=y_rul,
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Tag", format="%d.%m."),
                    alt.Tooltip("rul_hours:Q", title="RUL (h)", format=".0f"),
                ]
            ).properties(height=180)
            rul_area = gradient_area(rul_truck, x_rul, y_rul, line_color)
            st.altair_chart(styled_chart(alt.layer(rul_area, rul_line)), width="stretch")

        # RUL-Balkendiagramm (FA-6): Flottenvergleich, aktueller LKW markiert
        st.markdown('<div class="section-title">RUL im Flottenvergleich</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">RESTLAUFZEIT JE FAHRZEUG · RANDOM-FOREST-REGRESSION</div>', unsafe_allow_html=True)
        rul_df = fleet[["lkw_id", "rul_hours", "status"]].copy()
        rul_df["ist_auswahl"] = rul_df["lkw_id"] == truck_id
        rul_chart = alt.Chart(rul_df).mark_bar(cornerRadiusEnd=2).encode(
            y=alt.Y("lkw_id:N", sort=alt.EncodingSortField("rul_hours", order="ascending"),
                    title=None, axis=alt.Axis(labelFontSize=10)),
            x=alt.X("rul_hours:Q", title="RUL (Stunden)",
                    axis=alt.Axis(labelFontSize=10, titleFontSize=11)),
            color=alt.Color("status:N", legend=None, scale=alt.Scale(
                domain=["KRITISCH", "WARNUNG", "OK"],
                range=[COLOR_CRIT, COLOR_WARN, COLOR_OK])),
            opacity=alt.condition(alt.datum.ist_auswahl, alt.value(1.0), alt.value(0.35)),
            tooltip=[
                alt.Tooltip("lkw_id:N", title="LKW"),
                alt.Tooltip("rul_hours:Q", title="RUL (h)", format=".0f"),
                alt.Tooltip("status:N", title="Status"),
            ],
        ).properties(height=220)
        st.altair_chart(styled_chart(rul_chart), width="stretch")

    with right:
        ts_truck = timeseries[timeseries["lkw_id"] == truck_id].copy()

        if ts_truck.empty:
            st.info("Keine Zeitreihendaten für dieses Fahrzeug vorhanden.")
        else:
            # Health-Score (FA-6): der ML-Index, auf dem die Statusampel basiert
            if "health_score" in ts_truck.columns and ts_truck["health_score"].notna().any():
                st.markdown('<div class="section-title">Health-Score · letzte 72 h</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-sub">ML-GESUNDHEITSINDEX (0–1) · GRUNDLAGE DER STATUSAMPEL</div>', unsafe_allow_html=True)
                x_hs = alt.X("timestamp:T", title=None, axis=alt.Axis(format="%d.%m %H:%M", labelFontSize=10))
                y_hs = alt.Y("health_score:Q", title="Health-Score",
                             scale=alt.Scale(domain=[0, 1]),
                             axis=alt.Axis(labelFontSize=10, titleFontSize=11))
                hs_line = alt.Chart(ts_truck).mark_line(
                    color=line_color, strokeWidth=2.5
                ).encode(
                    x=x_hs, y=y_hs,
                    tooltip=[
                        alt.Tooltip("timestamp:T", title="Zeit", format="%d.%m %H:%M"),
                        alt.Tooltip("health_score:Q", title="Health-Score", format=".2f"),
                    ]
                ).properties(height=200)
                hs_warn = alt.Chart(pd.DataFrame({"y": [HEALTH_WARN]})).mark_rule(
                    color=COLOR_WARN, strokeDash=[4, 4], strokeWidth=1.5
                ).encode(y="y:Q")
                hs_crit = alt.Chart(pd.DataFrame({"y": [HEALTH_CRIT]})).mark_rule(
                    color=COLOR_CRIT, strokeDash=[4, 4], strokeWidth=1.5
                ).encode(y="y:Q")
                hs_area = gradient_area(ts_truck, x_hs, y_hs, line_color)
                st.altair_chart(styled_chart(alt.layer(hs_area, hs_line, hs_warn, hs_crit)),
                                width="stretch")
                st.caption(f"⎯⎯ Warnschwelle {str(HEALTH_WARN).replace('.', ',')} · "
                           f"⎯⎯ Kritische Schwelle {str(HEALTH_CRIT).replace('.', ',')}")

            # Bremsflüssigkeits-Chart: wird hier nur aufgebaut und erst nach
            # Motortemp/Öldruck gerendert (Chart-Reihenfolge der Spalte).
            x_enc = alt.X("timestamp:T", title=None, axis=alt.Axis(format="%d.%m %H:%M", labelFontSize=10))
            y_enc = alt.Y("brake_fluid_pct:Q", title="Bremsflüssigkeit (%)",
                          scale=alt.Scale(domain=[0, 100]),
                          axis=alt.Axis(labelFontSize=10, titleFontSize=11))

            # Sanfte Verlaufsfläche unter der Linie
            area = gradient_area(ts_truck, x_enc, y_enc, line_color)

            chart = alt.Chart(ts_truck).mark_line(
                color=line_color,
                strokeWidth=2.5
            ).encode(
                x=x_enc,
                y=y_enc,
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Zeit", format="%d.%m %H:%M"),
                    alt.Tooltip("brake_fluid_pct:Q", title="Bremse %", format=".1f"),
                ]
            ).properties(height=280)

            # Threshold rules
            warn_line = alt.Chart(pd.DataFrame({"y": [BRAKE_WARN_PCT]})).mark_rule(
                color=COLOR_WARN, strokeDash=[4, 4], strokeWidth=1.5
            ).encode(y="y:Q")
            crit_line = alt.Chart(pd.DataFrame({"y": [BRAKE_CRIT_PCT]})).mark_rule(
                color=COLOR_CRIT, strokeDash=[4, 4], strokeWidth=1.5
            ).encode(y="y:Q")

            # Alert annotations as vertical rules
            ts_min, ts_max = ts_truck["timestamp"].min(), ts_truck["timestamp"].max()
            alert_markers = truck_alerts[
                (truck_alerts["lkw_id"] == truck_id) &
                (truck_alerts["timestamp"] >= ts_min) &
                (truck_alerts["timestamp"] <= ts_max)
            ].copy()
            alert_layers = [
                alt.Chart(alert_markers[alert_markers["severity"] == sev]).mark_rule(
                    color=severity_color(sev), strokeWidth=1.5, strokeDash=[3, 3]
                ).encode(
                    x="timestamp:T",
                    tooltip=[
                        alt.Tooltip("timestamp:T", title="Alert", format="%d.%m %H:%M"),
                        alt.Tooltip("severity:N", title="Schweregrad"),
                        alt.Tooltip("message:N", title="Meldung"),
                    ],
                )
                for sev in SEVERITIES
                if not alert_markers[alert_markers["severity"] == sev].empty
            ]
            # RUL-Prognose (FA-6): gestrichelte Linie vom letzten Messwert bis
            # zur kritischen Schwelle am prognostizierten RUL-Zeitpunkt. Die
            # Skalierung passt exakt: brake = health*90+5, RUL = Zeit bis
            # health 0.5 = Zeit bis Bremse BRAKE_CRIT_PCT.
            forecast_layers = []
            forecast_note = None
            rul_h = float(truck["rul_hours"])
            if truck["status"] != "KRITISCH" and rul_h > 0:
                t_last = ts_truck["timestamp"].max()
                b_last = float(ts_truck.loc[ts_truck["timestamp"].idxmax(), "brake_fluid_pct"])
                horizon_h = min(rul_h, FORECAST_HORIZON_H)
                b_end = b_last + (BRAKE_CRIT_PCT - b_last) * horizon_h / rul_h
                fc = pd.DataFrame({
                    "timestamp": [t_last, t_last + pd.Timedelta(hours=horizon_h)],
                    "brake_fluid_pct": [b_last, b_end],
                })
                forecast_layers.append(
                    alt.Chart(fc).mark_line(
                        strokeDash=[6, 4], strokeWidth=2, color=line_color, opacity=0.7
                    ).encode(x="timestamp:T", y="brake_fluid_pct:Q")
                )
                if rul_h <= FORECAST_HORIZON_H:
                    t_crit = t_last + pd.Timedelta(hours=rul_h)
                    forecast_layers.append(
                        alt.Chart(fc.tail(1)).mark_point(
                            shape="diamond", size=80, filled=True, color=COLOR_CRIT
                        ).encode(x="timestamp:T", y="brake_fluid_pct:Q")
                    )
                    forecast_note = f"voraussichtlich kritisch: {t_crit:%d.%m. %H:%M} Uhr"
                else:
                    forecast_note = "kritische Schwelle außerhalb des 14-Tage-Horizonts"

            brake_chart = alt.layer(area, chart, warn_line, crit_line,
                                    *alert_layers, *forecast_layers)

            # Motor temperature chart
            st.markdown('<div class="section-title">Motortemperatur · letzte 72 h</div>', unsafe_allow_html=True)
            temp_min = float(min(60, ts_truck["motor_temp_c"].min() - 5))
            temp_max = float(max(110, ts_truck["motor_temp_c"].max() + 5))
            x_enc2 = alt.X("timestamp:T", title=None, axis=alt.Axis(format="%d.%m %H:%M", labelFontSize=10))
            y_enc2 = alt.Y("motor_temp_c:Q", title="Motortemp. (°C)",
                           scale=alt.Scale(domain=[temp_min, temp_max]),
                           axis=alt.Axis(labelFontSize=10, titleFontSize=11))
            chart2 = alt.Chart(ts_truck).mark_line(
                color=COLOR_INFO, strokeWidth=2
            ).encode(
                x=x_enc2, y=y_enc2,
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Zeit", format="%d.%m %H:%M"),
                    alt.Tooltip("motor_temp_c:Q", title="Motortemp. °C", format=".1f"),
                ]
            ).properties(height=180)

            crit_temp = alt.Chart(pd.DataFrame({"y": [MOTOR_TEMP_CRIT_C]})).mark_rule(
                color=COLOR_CRIT, strokeDash=[4, 4], strokeWidth=1.5
            ).encode(y="y:Q")

            temp_area = gradient_area(ts_truck, x_enc2, y_enc2, COLOR_INFO, baseline=temp_min)
            st.altair_chart(styled_chart(alt.layer(temp_area, chart2, crit_temp)),
                            width="stretch")

            # Öldruck-Zeitreihe (Spalte erst seit der aktuellen Adapter-Version)
            if "oil_pressure_bar" in ts_truck.columns:
                st.markdown('<div class="section-title">Öldruck · letzte 72 h</div>', unsafe_allow_html=True)
                oil_max = float(max(5.5, ts_truck["oil_pressure_bar"].max() + 0.5))
                x_enc3 = alt.X("timestamp:T", title=None, axis=alt.Axis(format="%d.%m %H:%M", labelFontSize=10))
                y_enc3 = alt.Y("oil_pressure_bar:Q", title="Öldruck (bar)",
                               scale=alt.Scale(domain=[0, oil_max]),
                               axis=alt.Axis(labelFontSize=10, titleFontSize=11))
                chart3 = alt.Chart(ts_truck).mark_line(
                    color="#8B5CF6", strokeWidth=2
                ).encode(
                    x=x_enc3, y=y_enc3,
                    tooltip=[
                        alt.Tooltip("timestamp:T", title="Zeit", format="%d.%m %H:%M"),
                        alt.Tooltip("oil_pressure_bar:Q", title="Öldruck (bar)", format=".2f"),
                    ]
                ).properties(height=180)

                warn_oil = alt.Chart(pd.DataFrame({"y": [OIL_PRESSURE_WARN_BAR]})).mark_rule(
                    color=COLOR_WARN, strokeDash=[4, 4], strokeWidth=1.5
                ).encode(y="y:Q")

                oil_area = gradient_area(ts_truck, x_enc3, y_enc3, "#8B5CF6")
                st.altair_chart(styled_chart(alt.layer(oil_area, chart3, warn_oil)),
                                width="stretch")

            st.markdown('<div class="section-title">Bremsflüssigkeit · letzte 72 h</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">ZEITREIHEN-DEGRADATION · ISOLATION FOREST + XGBOOST</div>', unsafe_allow_html=True)
            st.altair_chart(styled_chart(brake_chart), width="stretch")
            cap = f"⎯⎯ Warnschwelle {BRAKE_WARN_PCT} % · ⎯⎯ Kritische Schwelle {BRAKE_CRIT_PCT} %"
            if not alert_markers.empty:
                cap += " · ╌╌ Alert-Zeitpunkte"
            if forecast_note:
                cap += f" · ┄┄ RUL-Prognose (Random Forest), {forecast_note}"
            st.caption(cap)

    # Alert history for this truck
    st.markdown(f'<div class="section-title">Alert-Verlauf · {truck_id}</div>', unsafe_allow_html=True)
    truck_history = truck_alerts[truck_alerts["lkw_id"] == truck_id].sort_values("timestamp", ascending=False)

    if truck_history.empty:
        st.info("Keine Alerts für dieses Fahrzeug in den letzten 30 Tagen.")
    else:
        rows = []
        for _, alert in truck_history.iterrows():
            dtc = alert.get("dtc_code")
            dtc_str = f" · DTC {dtc}" if isinstance(dtc, str) and dtc.strip() else ""
            reco = alert.get("recommendation")
            reco_html = (
                f'<div class="alert-reco"><b>Empfehlung:</b> {reco}</div>'
                if isinstance(reco, str) and reco.strip() else ""
            )
            rows.append(
                f'<div class="alert-row history">'
                f'<div class="alert-time">{alert["timestamp"].strftime("%d.%m. %H:%M")}</div>'
                f'<div>{status_badge(alert["severity"])}</div>'
                f'<div class="alert-message">{alert["message"]}{dtc_str}{reco_html}</div>'
                f'</div>'
            )
        st.markdown('<div class="table-card">' + "".join(rows) + '</div>',
                    unsafe_allow_html=True)

    if st.session_state.get("role") == "wl":
        st.markdown('<div class="section-title" style="margin-top:1.5rem;">Wartungsfeedback</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">NACH ABSCHLUSS DER WARTUNG BITTE BESTÄTIGEN · FLIESST INS NÄCHTLICHE RETRAINING EIN</div>', unsafe_allow_html=True)
        fb = last_feedback(truck_id)
        if fb is not None:
            verdict_label = ("Wartung bestätigt" if fb["verdict"] == "bestätigt"
                             else "Als Fehlalarm gemeldet")
            when = f" am {fb['timestamp']:%d.%m.%Y, %H:%M} Uhr" if pd.notna(fb["timestamp"]) else ""
            st.caption(f"Zuletzt erfasst: {verdict_label}{when}")
        col_a, col_b, _sp = st.columns([1.4, 1.4, 4])
        with col_a:
            if st.button("Wartung bestätigt", key=f"fb_ok_{truck_id}", type="primary"):
                save_feedback(truck_id, "bestätigt", truck["status"])
                st.success(f"Feedback für {truck_id} gespeichert: Wartung bestätigt.")
        with col_b:
            if st.button("Fehlalarm", key=f"fb_fa_{truck_id}"):
                save_feedback(truck_id, "Fehlalarm", truck["status"])
                st.success(f"Feedback für {truck_id} gespeichert: Fehlalarm.")


# ============================================================================
# SCREEN 3: ALERT FEED
# ============================================================================
def render_alert_feed():
    n_crit = int((alerts["severity"] == "KRITISCH").sum())
    n_warn = int((alerts["severity"] == "WARNUNG").sum())
    n_info = int((alerts["severity"] == "INFO").sum())
    # savings_eur = EUR pro Stillstandsstunde; eine vermiedene Panne kostet
    # HOURS_PER_BREAKDOWN Stunden Standzeit.
    total_savings = int(alerts["savings_eur"].sum() * HOURS_PER_BREAKDOWN)

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card critical">
            <div class="kpi-card-header"><span class="kpi-card-icon">{icon("alert-octagon")}</span>
            <div class="kpi-label">Kritisch (7 Tage)</div></div>
            <div class="kpi-value">{n_crit}</div>
            <div class="kpi-sub">Sofortmaßnahme</div>
        </div>
        <div class="kpi-card warning">
            <div class="kpi-card-header"><span class="kpi-card-icon">{icon("alert-triangle")}</span>
            <div class="kpi-label">Warnung (7 Tage)</div></div>
            <div class="kpi-value">{n_warn}</div>
            <div class="kpi-sub">Wartung &lt; 14 Tage</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-card-header"><span class="kpi-card-icon">{icon("info")}</span>
            <div class="kpi-label">Info (7 Tage)</div></div>
            <div class="kpi-value">{n_info}</div>
            <div class="kpi-sub">Anomalie erkannt · Isolation Forest</div>
        </div>
        <div class="kpi-card ok">
            <div class="kpi-card-header"><span class="kpi-card-icon">{icon("shield")}</span>
            <div class="kpi-label">Vermiedene Kosten</div></div>
            <div class="kpi-value">{fmt_de(total_savings)} €</div>
            <div class="kpi-sub">letzte 7 Tage · geschätzt</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    feed_l, feed_r = st.columns([5, 1], vertical_alignment="bottom")
    with feed_l:
        st.markdown('<div class="section-title">Alert-Feed</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">CHRONOLOGISCH · XGBOOST-KLASSIFIKATION + ISOLATION FOREST + DTC-MAPPING</div>', unsafe_allow_html=True)

    # Filter links use the same navigation path as the rest of the app.
    filter_markup = []
    for label in ALERT_FILTERS:
        active_cls = " active" if st.session_state.alert_filter == label else ""
        filter_markup.append(
            f'<a class="filter-link{active_cls}" href="{alerts_href(label)}" target="_self">{label}</a>'
        )
    st.markdown(
        '<div class="filter-buttons">' + "".join(filter_markup) + "</div>",
        unsafe_allow_html=True,
    )

    # Filtered alerts
    if st.session_state.alert_filter == "ALLE":
        filtered = alerts.copy()
    else:
        filtered = alerts[alerts["severity"] == st.session_state.alert_filter].copy()
    filtered = filtered.sort_values("timestamp", ascending=False)

    with feed_r:
        st.download_button(
            "CSV-Export",
            icon=":material/download:",
            data=filtered.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"prema_alerts_{datetime.now():%Y%m%d_%H%M}.csv",
            mime="text/csv",
            width="stretch",
        )

    if filtered.empty:
        st.info("Keine Alerts für diesen Filter in den letzten 7 Tagen.")
        return

    MAX_ROWS = 100
    shown = filtered.head(MAX_ROWS)

    today = datetime.now().date()
    current_day = None
    parts = ['<div class="table-card">']
    for _, alert in shown.iterrows():
        # Tages-Trenner: Feed bleibt chronologisch, wird aber gruppiert lesbar
        alert_day = alert["timestamp"].date()
        if alert_day != current_day:
            current_day = alert_day
            delta_days = (today - alert_day).days
            day_label = ("Heute" if delta_days == 0 else
                         "Gestern" if delta_days == 1 else
                         f"{WEEKDAYS_DE[alert_day.weekday()]}, {alert['timestamp']:%d.%m.}")
            parts.append(f'<div class="feed-day">{day_label}</div>')
        # Kontext aus Pipeline-Anreicherung (FA-2): DTC, Wetter, Beladung
        meta_parts = [f"Quelle: {alert['source']}"]
        dtc = alert.get("dtc_code")
        if isinstance(dtc, str) and dtc.strip():
            meta_parts.append(f"DTC {dtc}")
        weather = alert.get("weather")
        temp = alert.get("temperature_c")
        if isinstance(weather, str) and weather.strip() and pd.notna(temp):
            meta_parts.append(f"{weather}, {temp:.0f} °C")
        load = alert.get("load_pct")
        if pd.notna(load):
            meta_parts.append(f"Beladung {load:.0f} %")
        route = alert.get("route_type")
        if isinstance(route, str) and route.strip():
            meta_parts.append(route)

        reco = alert.get("recommendation")
        reco_html = (
            f'<div class="alert-reco"><b>Empfehlung:</b> {reco}</div>'
            if isinstance(reco, str) and reco.strip() else ""
        )
        savings = alert["savings_eur"] * HOURS_PER_BREAKDOWN
        if savings > 0:
            savings_cls, savings_html = "alert-savings", f"~ {fmt_de(savings)} €"
        else:
            savings_cls, savings_html = "alert-savings dash", "–"

        parts.append(
            f'<a class="row-link" href="{detail_href(alert["lkw_id"])}" target="_self">'
            f'<div class="alert-row">'
            f'<div class="alert-time">{alert["timestamp"].strftime("%d.%m. %H:%M")}</div>'
            f'<div>{status_badge(alert["severity"])}</div>'
            f'<div class="alert-truck">{alert["lkw_id"]}</div>'
            f'<div class="alert-message">{alert["message"]}{reco_html}'
            f'<div class="alert-meta">{" · ".join(meta_parts)}</div></div>'
            f'<div class="{savings_cls}">{savings_html}</div>'
            f'</div></a>'
        )
    parts.append('</div>')
    st.markdown("".join(parts), unsafe_allow_html=True)

    if len(filtered) > MAX_ROWS:
        st.caption(f"{MAX_ROWS} von {len(filtered)} Alerts angezeigt · vollständige Liste im CSV-Export.")


# ============================================================================
# SCREEN 0: ROLE SELECTION & LOGIN
# ============================================================================
def render_role_selection():
    check = icon("check")
    arrow = icon("arrow-right")
    # Hinweiszeile passt sich an: mit konfigurierten Secrets ist der Zugang
    # passwortgeschützt, ohne läuft die App im offenen Demo-Modus.
    if _password_for("fm") or _password_for("wl"):
        note = "Passwortgeschützter Zugang · Anmeldung erfolgt nach Rollenwahl"
    else:
        note = "Demo-Modus ohne Passwort · in Produktion rollenbasiert geschützt (st.secrets)"
    st.markdown(f"""
<div class="role-screen">
    <div class="role-screen-brand">PRE<span class="role-screen-brand-accent">MA</span></div>
    <div class="role-screen-tagline">Predictive Maintenance · Spedition Müller GmbH</div>
    <div class="role-screen-heading">Anmeldung</div>
    <div class="role-screen-sub">Wählen Sie Ihre Rolle – die Ansicht passt sich Ihren Aufgaben an.</div>
    <div class="role-grid">
        <a class="role-card" href="?role=fm&view=fleet" target="_self">
            <div class="role-card-head">
                <div class="role-card-avatar">TM</div>
                <div>
                    <div class="role-card-type">Flottenmanager</div>
                    <div class="role-card-name">Thomas Müller</div>
                </div>
            </div>
            <ul class="role-card-features">
                <li>{check} Flottenübersicht &amp; Statusampel</li>
                <li>{check} Alert-Feed aller Fahrzeuge</li>
                <li>{check} Kosteneinsparungs-Kalkulation</li>
                <li>{check} Einzelfahrzeug-Detailansicht</li>
            </ul>
            <div class="role-card-cta">Anmelden {arrow}</div>
        </a>
        <a class="role-card" href="?role=wl&view=fleet" target="_self">
            <div class="role-card-head">
                <div class="role-card-avatar">SB</div>
                <div>
                    <div class="role-card-type">Werkstattleiter</div>
                    <div class="role-card-name">Stefan Berger</div>
                </div>
            </div>
            <ul class="role-card-features">
                <li>{check} Priorisierter Wartungsplan</li>
                <li>{check} RUL-Prognose je Fahrzeug</li>
                <li>{check} Wartungsfeedback erfassen</li>
                <li>{check} Einzelfahrzeug-Detailansicht</li>
            </ul>
            <div class="role-card-cta">Anmelden {arrow}</div>
        </a>
    </div>
    <div class="role-screen-note">{icon("lock")} {note}</div>
</div>
""", unsafe_allow_html=True)


def render_login(role: str):
    """Passwort-Abfrage für eine Rolle mit konfiguriertem Secret (NFR Zugriffsschutz)."""
    label = _role_labels.get(role, role)
    persona, _, role_name = label.partition(" · ")
    initials = "".join(part[0] for part in persona.split()[:2]).upper()
    st.markdown("""
<div style="text-align:center; margin-top:3.5rem; margin-bottom:1.8rem;">
    <div class="role-screen-brand" style="margin-bottom:0.25rem;">PRE<span class="role-screen-brand-accent">MA</span></div>
    <div class="role-screen-tagline" style="margin-bottom:0;">Predictive Maintenance · Spedition Müller GmbH</div>
</div>
""", unsafe_allow_html=True)
    _l, mid, _r = st.columns([1.2, 1, 1.2])
    with mid:
        with st.form(key=f"login_{role}"):
            st.markdown(f"""
<div style="text-align:center; margin-bottom:1.1rem;">
    <div class="login-avatar">{initials}</div>
    <div class="login-name">{persona}</div>
    <div class="role-card-type" style="margin-bottom:0;">{role_name or "Anmeldung"}</div>
</div>
<div class="login-field-label">Passwort</div>
""", unsafe_allow_html=True)
            pw = st.text_input("Passwort", type="password",
                               placeholder="••••••••",
                               label_visibility="collapsed")
            ok = st.form_submit_button("Anmelden", type="primary", width="stretch")
            st.markdown(f"""
<div class="login-divider"></div>
<div class="login-note">{icon("lock")} Zugriff nur für autorisierte Nutzer · Spedition Müller GmbH</div>
""", unsafe_allow_html=True)
        if ok:
            if pw == _password_for(role):
                st.session_state[f"auth_{role}"] = True
                st.session_state.role = role
                st.session_state.pending_role = None
                st.query_params["auth"] = _auth_token(role)  # übersteht F5/Reload
                st.rerun()
            else:
                st.error("Falsches Passwort.")
        st.markdown(
            '<div style="text-align:center; margin-top:0.7rem;">'
            '<a href="?" target="_self" style="font-size:0.76rem; color:var(--text-color); '
            'opacity:0.55; text-decoration:none;">← Andere Rolle wählen</a></div>',
            unsafe_allow_html=True,
        )


# ============================================================================
# Router
# ============================================================================
if not st.session_state.role:
    if st.session_state.get("pending_role"):
        render_login(st.session_state.pending_role)
    else:
        render_role_selection()
elif st.session_state.view == "detail":
    render_truck_detail()
elif st.session_state.view == "alerts" and st.session_state.role == "fm":
    render_alert_feed()
elif st.session_state.view == "plan" and st.session_state.role == "wl":
    render_maintenance_plan()
else:
    render_fleet_overview()

# Footer
st.markdown("""
<div style='text-align:center; padding:1.5rem 0 0.8rem 0; color:var(--text-color); opacity:0.5;
           font-family:"Inter",sans-serif; font-size:0.65rem; font-weight:600;
           letter-spacing:0.1em; border-top:1px solid var(--secondary-background-color); margin-top:2rem;'>
    PREMA MVP · HM BIG DATA SS2026 · TEAM 1 · DATEN SIMULIERT · KEIN PRODUKTIVBETRIEB
</div>
""", unsafe_allow_html=True)
