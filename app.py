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
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import altair as alt
import pandas as pd
import streamlit as st

# ============================================================================
# Konstanten (Schwellenwerte & Kostenmodell, vgl. Pflichtenheft FA-5/FA-7)
# ============================================================================
BRAKE_WARN_PCT = 30      # Warnschwelle Bremsflüssigkeit (%)
BRAKE_CRIT_PCT = 15      # Kritische Schwelle Bremsflüssigkeit (%)
MOTOR_TEMP_CRIT_C = 95   # Kritische Motortemperatur (°C)
OIL_PRESSURE_WARN_BAR = 2.5  # Warnschwelle Öldruck (bar)
COST_PER_HOUR_EUR = 600  # Kosten pro Stillstandsstunde (Pflichtenheft: 500-700 €)
HOURS_PER_BREAKDOWN = 4  # Angenommene Standzeit pro vermiedener Panne

SEVERITIES = ("KRITISCH", "WARNUNG", "INFO")
ALERT_FILTERS = ("ALLE",) + SEVERITIES

COLOR_CRIT = "#FF3D4C"
COLOR_WARN = "#FFA500"
COLOR_OK = "#32C759"
COLOR_INFO = "#2B6CB0"

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
        --p-accent:  #FF3D4C;
        --p-glow:    rgba(255,61,76,0.25);
        --p-ok:      #32C759;
        --p-warn:    #FFA500;
        --p-info:    #2B6CB0;
        /* Rahmen & Schatten: rgba-Basis funktioniert in beiden Modi */
        --p-border:  rgba(120,120,140,0.22);
        --p-shadow:  0 1px 4px rgba(0,0,0,0.1), 0 0 0 1px rgba(120,120,140,0.12);
        --p-shadow-hover: 0 4px 14px rgba(0,0,0,0.15), 0 0 0 1px rgba(120,120,140,0.18);
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
        padding: 0.6rem 1rem;
        background: linear-gradient(90deg, #1A1A1A 0%, #2A2A2A 100%);
        color: #FFF;
        border-radius: 6px;
        margin-bottom: 1rem;
        font-family: 'Plus Jakarta Sans', sans-serif;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .header-brand {
        font-size: 1.25rem; font-weight: 700; letter-spacing: 0.08em;
        display: flex; align-items: center; gap: 0.5rem;
    }
    .header-brand-accent { color: var(--p-accent); }
    .header-user { font-size: 0.8rem; opacity: 0.75; letter-spacing: 0.05em;
        display: flex; align-items: center; gap: 0.4rem; }

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
        background: var(--p-accent); color: #FFF; border-color: var(--p-accent);
        box-shadow: 0 4px 12px var(--p-glow);
    }
    .nav-button.detail { cursor: default; }
    .nav-button.inactive {
        background: rgba(120,120,140,0.1);
        color: var(--text-color);
        border-color: var(--p-border);
    }
    .nav-button.inactive:hover { border-color: var(--p-accent); }
    .nav-count {
        display: inline-flex; align-items: center; justify-content: center;
        min-width: 1.3rem; height: 1.3rem; padding: 0 0.3rem;
        border-radius: 99px; font-size: 0.68rem; font-weight: 700;
        background: var(--p-accent); color: #FFF;
    }
    .nav-button.active .nav-count { background: rgba(255,255,255,0.25); }

    /* ── KPI-Cards ── */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 0.8rem; margin-bottom: 1rem;
    }
    .kpi-card {
        background: var(--background-color);
        border: 1px solid var(--p-border);
        border-left: 5px solid var(--text-color);
        padding: 0.9rem 1rem; border-radius: 6px;
        box-shadow: var(--p-shadow);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover { box-shadow: var(--p-shadow-hover); transform: translateY(-2px); }
    .kpi-card-header { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.3rem; }
    .kpi-card-icon { font-size: 1.2rem; }
    .kpi-card.critical {
        border-left-color: var(--p-accent);
        background: color-mix(in srgb, #FF3D4C 6%, var(--background-color));
    }
    .kpi-card.warning {
        border-left-color: var(--p-warn);
        background: color-mix(in srgb, #FFA500 6%, var(--background-color));
    }
    .kpi-card.ok {
        border-left-color: var(--p-ok);
        background: color-mix(in srgb, #32C759 6%, var(--background-color));
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
    .badge-critical { background: var(--p-accent); color: #FFF; }
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
    .status-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; }
    .status-dot-kritisch { background: var(--p-accent); }
    .status-dot-warnung  { background: var(--p-warn); }
    .status-dot-ok       { background: var(--p-ok); }
    .sensor-bar-bg {
        height: 6px; background: var(--secondary-background-color);
        border-radius: 3px; overflow: hidden;
    }
    .sensor-bar-fill { height: 100%; border-radius: 3px; }
    .sensor-value {
        font-family: 'Inter',monospace; font-size: 0.8rem;
        font-weight: 600; text-align: right; color: var(--text-color);
    }

    /* ── Alert-Feed ── */
    .alert-row {
        display: grid;
        grid-template-columns: 88px 86px 66px minmax(0,1fr) 78px;
        gap: 0.65rem; align-items: center; padding: 0.55rem 0.7rem;
        border-bottom: 1px solid var(--p-border);
        border-radius: 4px; margin-bottom: 0.3rem; font-size: 0.82rem;
        transition: background 0.15s; cursor: pointer; color: var(--text-color);
    }
    .alert-row.history { grid-template-columns: 88px 86px minmax(0,1fr); cursor: default; }
    .alert-row:hover { background: var(--secondary-background-color); }
    .alert-time {
        font-family: 'Inter',monospace; color: var(--text-color);
        opacity: 0.55; font-size: 0.73rem; font-weight: 500;
    }
    .alert-truck  { font-family:'Inter',monospace; font-weight:700; color:var(--text-color); }
    .alert-savings { text-align:right; font-family:'Inter',monospace; color:var(--p-ok); font-weight:700; }
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
    .filter-link:hover { border-color: var(--p-accent); }
    .filter-link.active {
        background: var(--p-accent); border-color: var(--p-accent);
        color: #FFF; box-shadow: 0 3px 10px var(--p-glow);
    }
    .filter-link,.filter-link:visited,.filter-link:hover,.filter-link:active {
        text-decoration: none !important;
    }

    /* ── Section-Titel ── */
    .section-title {
        font-size: 1rem; font-weight: 700; margin: 1rem 0 0.4rem 0;
        color: var(--text-color); letter-spacing: -0.01em;
        font-family: 'Plus Jakarta Sans',sans-serif;
        display: flex; align-items: center; gap: 0.6rem;
    }
    .section-icon { font-size: 1.2rem; }
    .section-sub {
        font-size: 0.72rem; color: var(--text-color); opacity: 0.5;
        margin-bottom: 0.6rem; font-family: 'Inter',monospace;
        letter-spacing: 0.04em; text-transform: uppercase;
    }

    /* ── Empfehlungs-Banner ── */
    .reco-banner {
        background: color-mix(in srgb, #FF3D4C 7%, var(--background-color));
        border-left: 5px solid var(--p-accent);
        padding: 0.9rem 1rem; border-radius: 6px; margin-bottom: 1rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    .reco-banner.warning {
        background: color-mix(in srgb, #FFA500 7%, var(--background-color));
        border-left-color: var(--p-warn);
    }
    .reco-banner.ok {
        background: color-mix(in srgb, #32C759 7%, var(--background-color));
        border-left-color: var(--p-ok);
    }
    .reco-title {
        font-weight: 700; font-size: 0.9rem; margin-bottom: 0.2rem;
        color: var(--text-color); font-family: 'Plus Jakarta Sans',sans-serif;
        display: flex; align-items: center; gap: 0.4rem;
    }
    .reco-text { font-size: 0.8rem; color: var(--text-color); opacity: 0.75; line-height: 1.4; }
    .reco-rul {
        float: right; font-family: 'Inter',monospace;
        font-size: 0.8rem; font-weight: 600; color: var(--text-color);
        background: rgba(120,120,140,0.1);
        padding: 0.2rem 0.5rem; border-radius: 3px;
        border: 1px solid var(--p-border);
    }

    /* ── Flotten-Tabelle ── */
    .truck-table-header {
        display: grid;
        grid-template-columns: 28px minmax(70px,0.8fr) minmax(110px,1.2fr) minmax(85px,0.9fr) minmax(75px,0.8fr) minmax(100px,1fr) 86px;
        gap: 0.7rem; padding: 0.5rem 0.8rem;
        background: linear-gradient(90deg, #1A1A1A 0%, #2A2A2A 100%);
        color: #FFF; font-size: 0.68rem; text-transform: uppercase;
        letter-spacing: 0.08em; font-family: 'Plus Jakarta Sans','Inter',sans-serif;
        border-radius: 4px 4px 0 0; font-weight: 700;
    }
    .truck-table-row {
        display: grid;
        grid-template-columns: 28px minmax(70px,0.8fr) minmax(110px,1.2fr) minmax(85px,0.9fr) minmax(75px,0.8fr) minmax(100px,1fr) 86px;
        gap: 0.7rem; padding: 0.55rem 0.8rem;
        border-bottom: 1px solid var(--p-border);
        background: var(--background-color);
        font-size: 0.85rem; align-items: center;
        transition: background 0.15s; cursor: pointer; color: var(--text-color);
    }
    .truck-icon { font-size: 1.1rem; text-align: center; }
    .truck-table-row:hover { background: var(--secondary-background-color); }
    .truck-table-row.critical {
        background: color-mix(in srgb, #FF3D4C 5%, var(--background-color));
    }
    .truck-table-row.warning {
        background: color-mix(in srgb, #FFA500 5%, var(--background-color));
    }
    .row-link,.row-link *,.row-link:visited,.row-link:hover,.row-link:active {
        color: inherit !important; text-decoration: none !important;
    }
    .row-link { display: block; }
    .truck-id { font-family:'Inter',monospace; font-weight:700; color:var(--text-color); }

    /* ── Dark-Mode: stärkere Rahmen & Schatten ──────────────────── */
    [data-theme="dark"] {
        --p-border:  rgba(255,255,255,0.14);
        --p-shadow:  0 2px 8px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.07);
        --p-shadow-hover: 0 6px 18px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.12);
    }
    [data-theme="dark"] .kpi-card.critical { background: color-mix(in srgb, #FF3D4C 10%, var(--background-color)); }
    [data-theme="dark"] .kpi-card.warning  { background: color-mix(in srgb, #FFA500 10%, var(--background-color)); }
    [data-theme="dark"] .kpi-card.ok       { background: color-mix(in srgb, #32C759 10%, var(--background-color)); }
    [data-theme="dark"] .truck-table-row.critical { background: color-mix(in srgb, #FF3D4C  8%, var(--background-color)); }
    [data-theme="dark"] .truck-table-row.warning  { background: color-mix(in srgb, #FFA500  8%, var(--background-color)); }
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
    .role-screen { max-width: 640px; margin: 3.5rem auto; }
    .role-screen-brand {
        font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.9rem; font-weight: 700;
        letter-spacing: 0.18em; color: var(--text-color); margin-bottom: 0.25rem;
    }
    .role-screen-brand-accent { color: var(--p-accent); }
    .role-screen-tagline {
        font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase;
        color: var(--text-color); opacity: 0.35; margin-bottom: 2.8rem;
    }
    .role-screen-heading {
        font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.4rem; font-weight: 700;
        color: var(--text-color); margin-bottom: 1.4rem;
    }
    .role-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.1rem; }
    .role-card {
        background: var(--background-color); border: 1px solid var(--p-border); border-radius: 8px;
        padding: 1.9rem 1.7rem; cursor: pointer; transition: all 0.18s ease;
        display: block; color: inherit; position: relative; overflow: hidden;
    }
    .role-card::before {
        content: ''; position: absolute; top: 0; left: 0; width: 3px; height: 100%;
        background: var(--p-accent); opacity: 0; transition: opacity 0.18s ease;
    }
    .role-card:hover::before { opacity: 1; }
    .role-card:hover { transform: translateY(-2px); box-shadow: 0 8px 28px rgba(0,0,0,0.09); border-color: rgba(255,61,76,0.22); }
    .role-card,.role-card *,.role-card:visited,.role-card:hover,.role-card:active { text-decoration: none !important; color: inherit; }
    .role-card-type {
        font-family: 'Plus Jakarta Sans', sans-serif; font-size: 0.6rem; font-weight: 700;
        letter-spacing: 0.18em; text-transform: uppercase; color: var(--p-accent); margin-bottom: 0.4rem;
    }
    .role-card-name {
        font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.2rem; font-weight: 700;
        color: var(--text-color); margin-bottom: 1.2rem;
    }
    .role-card-features { list-style: none; padding: 0; margin: 0 0 1.5rem 0; }
    .role-card-features li {
        font-size: 0.77rem; padding: 0.3rem 0; color: var(--text-color); opacity: 0.58;
        border-bottom: 1px solid var(--p-border);
    }
    .role-card-cta {
        font-family: 'Plus Jakarta Sans', sans-serif; font-size: 0.78rem; font-weight: 700;
        color: var(--p-accent); letter-spacing: 0.04em;
    }

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


def status_badge(status):
    cls = {"KRITISCH": "badge-critical", "WARNUNG": "badge-warning",
           "OK": "badge-ok", "INFO": "badge-info"}.get(status, "badge-info")
    return f'<span class="badge {cls}">{status}</span>'


def severity_color(severity):
    return {"KRITISCH": COLOR_CRIT, "WARNUNG": COLOR_WARN,
            "INFO": COLOR_INFO, "OK": COLOR_OK}.get(severity, "#8A8A8F")


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
        return None, None, None, None, None, sync_error, missing

    fleet = pd.read_csv(base / "fleet.csv")
    timeseries = pd.read_csv(base / "timeseries.csv", parse_dates=["timestamp"])
    alerts = pd.read_csv(base / "alerts.csv", parse_dates=["timestamp"])
    truck_alerts = pd.read_csv(base / "truck_alerts.csv", parse_dates=["timestamp"])

    metrics = {}
    metrics_path = base / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            metrics = {}

    return fleet, timeseries, alerts, truck_alerts, metrics, sync_error, []


fleet, timeseries, alerts, truck_alerts, ml_metrics, _sync_error, _missing = load_data()

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
elif query_view == "detail":
    query_truck = st.query_params.get("truck")
    if query_truck in KNOWN_TRUCKS:
        st.session_state.view = "detail"
        st.session_state.selected_truck = query_truck

# Werkstattleiter hat keinen Alert-Feed: auf Wartungsliste umleiten
if st.session_state.role == "wl" and st.session_state.view == "alerts":
    st.session_state.view = "fleet"

# ============================================================================
# Header
# ============================================================================
_role_labels = {
    "fm": "Thomas Müller · Flottenmanager",
    "wl": "Stefan Berger · Werkstattleiter",
}
_h_user = _role_labels.get(st.session_state.role, "")
_switch_link = (
    ' &nbsp;<a href="?" target="_self" style="color:rgba(255,255,255,0.45);'
    'font-size:0.68rem;letter-spacing:0.05em;text-decoration:none!important;">Abmelden</a>'
    if st.session_state.role else ""
)
st.markdown(f"""
<div class="header-bar">
    <div class="header-brand">
        <span>PRE<span class="header-brand-accent">MA</span></span>
    </div>
    <div class="header-user" style="font-size:0.75rem;opacity:0.7;">
        {_h_user}{_switch_link}
    </div>
</div>
""", unsafe_allow_html=True)

if st.session_state.role:
    n_crit_alerts = int((alerts["severity"] == "KRITISCH").sum())
    _fnc = "nav-button active" if st.session_state.view == "fleet" else "nav-button inactive"
    _fleet_label = "FLOTTE" if st.session_state.role == "fm" else "WARTUNG"
    _nav_html = f'<div class="nav-buttons"><a class="{_fnc}" href="{fleet_href()}" target="_self">{_fleet_label}</a>'
    if st.session_state.role == "fm":
        _anc = "nav-button active" if st.session_state.view == "alerts" else "nav-button inactive"
        _alerts_label = (f'ALERTS <span class="nav-count">{n_crit_alerts}</span>'
                         if n_crit_alerts else "ALERTS")
        _nav_html += f'<a class="{_anc}" href="{alerts_href("ALLE")}" target="_self">{_alerts_label}</a>'
    if st.session_state.view == "detail" and st.session_state.selected_truck:
        _nav_html += f'<span class="nav-button detail">{st.session_state.selected_truck}</span>'
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
def render_fleet_overview():
    n_total = len(fleet)
    if n_total == 0:
        st.info("Keine Fahrzeugdaten vorhanden.")
        return
    n_ok = int((fleet["status"] == "OK").sum())
    n_warn = int((fleet["status"] == "WARNUNG").sum())
    n_crit = int((fleet["status"] == "KRITISCH").sum())
    avoided_eur = n_crit * COST_PER_HOUR_EUR * HOURS_PER_BREAKDOWN
    avg_rul = int(fleet["rul_hours"].mean())

    # KPI-Karten 3 und 4 sind rollenspezifisch: nur der Flottenmanager hat
    # einen Alert-Feed, daher verlinken die Karten nur bei ihm dorthin;
    # der Werkstattleiter sieht statt der Kosten-KPI die Ø-Restlaufzeit.
    # Hinweis: keine Zeilen-Einrückung im HTML, sonst interpretiert Markdown
    # die Karten als Code-Block und das Grid verzieht sich.
    def _card(cls, label, value, sub, href=None):
        card = (f'<div class="kpi-card {cls}">'
                f'<div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{value}</div>'
                f'<div class="kpi-sub">{sub}</div></div>')
        if href:
            return f'<a class="kpi-link" href="{href}" target="_self">{card}</a>'
        return card

    is_wl = st.session_state.get("role") == "wl"
    cards = [
        _card("", "Fahrzeuge gesamt", n_total, "aktive Flotte"),
        _card("ok", "Status OK", n_ok, f"{n_ok / n_total * 100:.0f}% der Flotte"),
        _card("warning", "Warnung", n_warn, "Wartung in &lt; 14 Tagen",
              href=None if is_wl else alerts_href("WARNUNG")),
        _card("", "Ø RUL Flotte", f"{fmt_de(avg_rul)} h", "Durchschn. Restlaufzeit")
        if is_wl else
        _card("critical", "Kritisch", n_crit,
              f"~ {fmt_de(avoided_eur)} EUR verhinderte Kosten",
              href=alerts_href("KRITISCH")),
    ]
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)

    _title_text = "Wartungsübersicht" if st.session_state.get("role") == "wl" else "Flottenstatus"
    head_l, head_r = st.columns([5, 1], vertical_alignment="bottom")
    with head_l:
        st.markdown(f'<div class="section-title">{_title_text}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="section-sub">SORTIERT NACH PRIORITÄT · BATCH-PIPELINE · '
            f'STAND {datetime.now().strftime("%H:%M")} UHR</div>',
            unsafe_allow_html=True,
        )

    # Sort: critical first, then warning, then OK; innerhalb gleicher Stufe
    # entscheidet die kürzeste Restlaufzeit.
    status_order = {"KRITISCH": 0, "WARNUNG": 1, "OK": 2}
    fleet_sorted = fleet.copy()
    fleet_sorted["sort_key"] = fleet_sorted["status"].map(status_order)
    fleet_sorted = fleet_sorted.sort_values(["sort_key", "rul_hours"]).drop(columns=["sort_key"])

    with head_r:
        st.download_button(
            "⬇ CSV-Export",
            data=fleet_sorted.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"prema_flotte_{datetime.now():%Y%m%d_%H%M}.csv",
            mime="text/csv",
            width="stretch",
        )

    st.markdown("""
    <div class="truck-table-header">
        <div></div>
        <div>LKW-ID</div>
        <div>Fahrer</div>
        <div>Motortemp.</div>
        <div>Bremse %</div>
        <div>RUL (Stunden)</div>
        <div>Status</div>
    </div>
    """, unsafe_allow_html=True)

    for _, truck in fleet_sorted.iterrows():
        row_class = "truck-table-row"
        if truck["status"] == "KRITISCH":
            row_class += " critical"
        elif truck["status"] == "WARNUNG":
            row_class += " warning"
        dot_cls = str(truck["status"]).lower()
        st.markdown(f"""
        <a class="row-link" href="{detail_href(truck['lkw_id'])}" target="_self">
            <div class="{row_class}">
                <div class="truck-icon"><span class="status-dot status-dot-{dot_cls}"></span></div>
                <div class="truck-id">{truck['lkw_id']}</div>
                <div>{truck['driver']}</div>
                <div>{truck['motor_temp_c']:.0f} °C</div>
                <div>{truck['brake_fluid_pct']:.0f} %</div>
                <div>{fmt_de(truck['rul_hours'])} h</div>
                <div>{status_badge(truck['status'])}</div>
            </div>
        </a>
        """, unsafe_allow_html=True)

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
        <div class="reco-rul">RUL: {fmt_de(truck['rul_hours'])} h</div>
        <div class="reco-title">{status_badge(truck["status"])} &nbsp; {title}</div>
        <div class="reco-text">{text}</div>
    </div>
    """, unsafe_allow_html=True)

    # Two columns: sensors left, chart right
    left, right = st.columns([1, 1.3])

    with left:
        st.markdown('<div class="section-title">Sensordaten</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">LIVE-WERTE AUS LETZTER BATCH-INFERENZ</div>', unsafe_allow_html=True)

        sensors = [
            ("Motortemperatur", truck["motor_temp_c"], 110, "°C",
                COLOR_CRIT if truck["motor_temp_c"] > MOTOR_TEMP_CRIT_C else COLOR_OK),
            ("Öldruck", truck["oil_pressure_bar"], 5.0, "bar",
                COLOR_WARN if truck["oil_pressure_bar"] < OIL_PRESSURE_WARN_BAR else COLOR_OK),
            ("Bremsflüssigkeit", truck["brake_fluid_pct"], 100, "%",
                COLOR_CRIT if truck["brake_fluid_pct"] < BRAKE_CRIT_PCT
                else (COLOR_WARN if truck["brake_fluid_pct"] < BRAKE_WARN_PCT else COLOR_OK)),
            ("Reifendruck VL", truck["tire_fl_bar"], 10, "bar", COLOR_OK),
            ("Reifendruck VR", truck["tire_fr_bar"], 10, "bar", COLOR_OK),
        ]
        for label, val, max_val, unit, color in sensors:
            pct = max(0, min(100, val / max_val * 100))
            st.markdown(f"""
            <div class="sensor-row">
                <div class="sensor-label">{label}</div>
                <div class="sensor-bar-bg">
                    <div class="sensor-bar-fill" style="width: {pct}%; background: {color};"></div>
                </div>
                <div class="sensor-value">{val:.1f} {unit}</div>
            </div>
            """, unsafe_allow_html=True)

        # Vehicle metadata
        st.markdown('<div class="section-title">Fahrzeugdaten</div>', unsafe_allow_html=True)
        meta_col1, meta_col2 = st.columns(2)
        with meta_col1:
            st.metric("Kilometerstand", f"{fmt_de(truck['km_total'])} km")
            st.metric("Fahrer", truck["driver"])
        with meta_col2:
            st.metric("Beladung", f"{truck['load_pct']:.0f} %")
            st.metric("RUL-Prognose", f"{fmt_de(truck['rul_hours'])} h")

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
        st.altair_chart(rul_chart.configure_view(strokeWidth=0), width="stretch")

    with right:
        st.markdown('<div class="section-title">Bremsflüssigkeit · letzte 72 h</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">ZEITREIHEN-DEGRADATION · ISOLATION FOREST + XGBOOST</div>', unsafe_allow_html=True)

        ts_truck = timeseries[timeseries["lkw_id"] == truck_id].copy()

        if ts_truck.empty:
            st.info("Keine Zeitreihendaten für dieses Fahrzeug vorhanden.")
        else:
            chart = alt.Chart(ts_truck).mark_line(
                color=severity_color(truck["status"]),
                strokeWidth=2.5
            ).encode(
                x=alt.X("timestamp:T", title=None, axis=alt.Axis(format="%d.%m %H:%M", labelFontSize=10)),
                y=alt.Y("brake_fluid_pct:Q", title="Bremsflüssigkeit (%)",
                        scale=alt.Scale(domain=[0, 100]),
                        axis=alt.Axis(labelFontSize=10, titleFontSize=11)),
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
            brake_chart = alt.layer(chart, warn_line, crit_line, *alert_layers)
            st.altair_chart(brake_chart.configure_view(strokeWidth=0), width="stretch")

            cap = f"⎯⎯ Warnschwelle {BRAKE_WARN_PCT} % · ⎯⎯ Kritische Schwelle {BRAKE_CRIT_PCT} %"
            if not alert_markers.empty:
                cap += " · ╌╌ Alert-Zeitpunkte"
            st.caption(cap)

            # Motor temperature chart
            st.markdown('<div class="section-title">Motortemperatur · letzte 72 h</div>', unsafe_allow_html=True)
            temp_min = float(min(60, ts_truck["motor_temp_c"].min() - 5))
            temp_max = float(max(110, ts_truck["motor_temp_c"].max() + 5))
            chart2 = alt.Chart(ts_truck).mark_line(
                color=COLOR_INFO, strokeWidth=2
            ).encode(
                x=alt.X("timestamp:T", title=None, axis=alt.Axis(format="%d.%m %H:%M", labelFontSize=10)),
                y=alt.Y("motor_temp_c:Q", title="Motortemp. (°C)",
                        scale=alt.Scale(domain=[temp_min, temp_max]),
                        axis=alt.Axis(labelFontSize=10, titleFontSize=11)),
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Zeit", format="%d.%m %H:%M"),
                    alt.Tooltip("motor_temp_c:Q", title="Motortemp. °C", format=".1f"),
                ]
            ).properties(height=180)

            crit_temp = alt.Chart(pd.DataFrame({"y": [MOTOR_TEMP_CRIT_C]})).mark_rule(
                color=COLOR_CRIT, strokeDash=[4, 4], strokeWidth=1.5
            ).encode(y="y:Q")

            st.altair_chart((chart2 + crit_temp).configure_view(strokeWidth=0), width="stretch")

            # Öldruck-Zeitreihe (Spalte erst seit der aktuellen Adapter-Version)
            if "oil_pressure_bar" in ts_truck.columns:
                st.markdown('<div class="section-title">Öldruck · letzte 72 h</div>', unsafe_allow_html=True)
                oil_max = float(max(5.5, ts_truck["oil_pressure_bar"].max() + 0.5))
                chart3 = alt.Chart(ts_truck).mark_line(
                    color="#8B5CF6", strokeWidth=2
                ).encode(
                    x=alt.X("timestamp:T", title=None, axis=alt.Axis(format="%d.%m %H:%M", labelFontSize=10)),
                    y=alt.Y("oil_pressure_bar:Q", title="Öldruck (bar)",
                            scale=alt.Scale(domain=[0, oil_max]),
                            axis=alt.Axis(labelFontSize=10, titleFontSize=11)),
                    tooltip=[
                        alt.Tooltip("timestamp:T", title="Zeit", format="%d.%m %H:%M"),
                        alt.Tooltip("oil_pressure_bar:Q", title="Öldruck (bar)", format=".2f"),
                    ]
                ).properties(height=180)

                warn_oil = alt.Chart(pd.DataFrame({"y": [OIL_PRESSURE_WARN_BAR]})).mark_rule(
                    color=COLOR_WARN, strokeDash=[4, 4], strokeWidth=1.5
                ).encode(y="y:Q")

                st.altair_chart((chart3 + warn_oil).configure_view(strokeWidth=0), width="stretch")

    # Alert history for this truck
    st.markdown(f'<div class="section-title">Alert-Verlauf · {truck_id}</div>', unsafe_allow_html=True)
    truck_history = truck_alerts[truck_alerts["lkw_id"] == truck_id].sort_values("timestamp", ascending=False)

    if truck_history.empty:
        st.info("ℹ️ Keine Alerts für dieses Fahrzeug in den letzten 30 Tagen.")
    else:
        for _, alert in truck_history.iterrows():
            dtc = alert.get("dtc_code")
            dtc_str = f" · DTC {dtc}" if isinstance(dtc, str) and dtc.strip() else ""
            reco = alert.get("recommendation")
            reco_html = (
                f'<div class="alert-reco"><b>Empfehlung:</b> {reco}</div>'
                if isinstance(reco, str) and reco.strip() else ""
            )
            st.markdown(f"""
            <div class="alert-row history">
                <div class="alert-time">{alert['timestamp'].strftime('%d.%m. %H:%M')}</div>
                <div>{status_badge(alert['severity'])}</div>
                <div class="alert-message">
                    {alert['message']}{dtc_str}
                    {reco_html}
                </div>
            </div>
            """, unsafe_allow_html=True)

    if st.session_state.get("role") == "wl":
        st.markdown('<div class="section-title" style="margin-top:1.5rem;">Wartungsfeedback</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">NACH ABSCHLUSS DER WARTUNG BITTE BESTÄTIGEN · FLIESST INS NÄCHTLICHE RETRAINING EIN</div>', unsafe_allow_html=True)
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
            <div class="kpi-label">Kritisch (7 Tage)</div>
            <div class="kpi-value">{n_crit}</div>
            <div class="kpi-sub">Sofortmaßnahme</div>
        </div>
        <div class="kpi-card warning">
            <div class="kpi-label">Warnung (7 Tage)</div>
            <div class="kpi-value">{n_warn}</div>
            <div class="kpi-sub">Wartung &lt; 14 Tage</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Info (7 Tage)</div>
            <div class="kpi-value">{n_info}</div>
            <div class="kpi-sub">Anomalie erkannt · Isolation Forest</div>
        </div>
        <div class="kpi-card ok">
            <div class="kpi-label">Vermiedene Kosten</div>
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
            "⬇ CSV-Export",
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

    for _, alert in shown.iterrows():
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

        st.markdown(f"""
        <a class="row-link" href="{detail_href(alert['lkw_id'])}" target="_self">
            <div class="alert-row">
                <div class="alert-time">{alert['timestamp'].strftime('%d.%m. %H:%M')}</div>
                <div>{status_badge(alert['severity'])}</div>
                <div class="alert-truck">{alert['lkw_id']}</div>
                <div class="alert-message">
                    {alert['message']}
                    {reco_html}
                    <div class="alert-meta">{' · '.join(meta_parts)}</div>
                </div>
                <div class="{savings_cls}">{savings_html}</div>
            </div>
        </a>
        """, unsafe_allow_html=True)

    if len(filtered) > MAX_ROWS:
        st.caption(f"{MAX_ROWS} von {len(filtered)} Alerts angezeigt · vollständige Liste im CSV-Export.")


# ============================================================================
# SCREEN 0: ROLE SELECTION & LOGIN
# ============================================================================
def render_role_selection():
    st.markdown("""
<div class="role-screen">
    <div class="role-screen-brand">PRE<span class="role-screen-brand-accent">MA</span></div>
    <div class="role-screen-tagline">Predictive Maintenance · Spedition Müller GmbH</div>
    <div class="role-screen-heading">Wer bist du?</div>
    <div class="role-grid">
        <a class="role-card" href="?role=fm&view=fleet" target="_self">
            <div class="role-card-type">Flottenmanager</div>
            <div class="role-card-name">Thomas Müller</div>
            <ul class="role-card-features">
                <li>Flottenübersicht &amp; Statusampel</li>
                <li>Alert-Feed aller Fahrzeuge</li>
                <li>Kosteneinsparungs-Kalkulation</li>
                <li>Einzelfahrzeug-Detailansicht</li>
            </ul>
            <div class="role-card-cta">Weiter →</div>
        </a>
        <a class="role-card" href="?role=wl&view=fleet" target="_self">
            <div class="role-card-type">Werkstattleiter</div>
            <div class="role-card-name">Stefan Berger</div>
            <ul class="role-card-features">
                <li>Priorisierte Wartungsliste</li>
                <li>RUL-Prognose je Fahrzeug</li>
                <li>Wartungsfeedback erfassen</li>
                <li>Einzelfahrzeug-Detailansicht</li>
            </ul>
            <div class="role-card-cta">Weiter →</div>
        </a>
    </div>
</div>
""", unsafe_allow_html=True)


def render_login(role: str):
    """Passwort-Abfrage für eine Rolle mit konfiguriertem Secret (NFR Zugriffsschutz)."""
    label = _role_labels.get(role, role)
    st.markdown(f"""
<div class="role-screen" style="max-width:420px;">
    <div class="role-screen-brand">PRE<span class="role-screen-brand-accent">MA</span></div>
    <div class="role-screen-tagline">Anmeldung · {label}</div>
</div>
""", unsafe_allow_html=True)
    _l, mid, _r = st.columns([1, 1.2, 1])
    with mid:
        with st.form(key=f"login_{role}"):
            pw = st.text_input("Passwort", type="password")
            ok = st.form_submit_button("Anmelden", type="primary", width="stretch")
        if ok:
            if pw == _password_for(role):
                st.session_state[f"auth_{role}"] = True
                st.session_state.role = role
                st.session_state.pending_role = None
                st.query_params["auth"] = _auth_token(role)  # übersteht F5/Reload
                st.rerun()
            else:
                st.error("Falsches Passwort.")
        st.markdown('<a href="?" target="_self" style="font-size:0.78rem;">← Rolle wechseln</a>',
                    unsafe_allow_html=True)


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
else:
    render_fleet_overview()

# Footer
st.markdown("""
<div style='text-align:center; padding:1.5rem 0 0.8rem 0; color:var(--text-color); opacity:0.4;
           font-family:"IBM Plex Mono","Courier New",monospace; font-size:0.65rem;
           letter-spacing:0.1em; border-top:1px solid var(--secondary-background-color); margin-top:2rem;'>
    PREMA MVP · HM BIG DATA SS2026 · TEAM 1 · DATEN SIMULIERT · KEIN PRODUKTIVBETRIEB
</div>
""", unsafe_allow_html=True)
