"""
PREMA - Predictive Maintenance Dashboard (MVP Click-Demo)
Hochschule München | Big Data SS2026 | Team 1 (Predictive)

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    https://share.streamlit.io
"""
import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
from datetime import datetime
from urllib.parse import quote

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
    .kpi-link { display: block; }
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
        grid-template-columns: 28px 88px 86px 66px minmax(0,1fr) 78px;
        gap: 0.65rem; align-items: center; padding: 0.55rem 0.7rem;
        border-bottom: 1px solid var(--p-border);
        border-radius: 4px; margin-bottom: 0.3rem; font-size: 0.82rem;
        transition: background 0.15s; cursor: pointer; color: var(--text-color);
    }
    .alert-icon { font-size: 1.1rem; text-align: center; }
    .alert-row:hover { background: var(--secondary-background-color); }
    .alert-time {
        font-family: 'Inter',monospace; color: var(--text-color);
        opacity: 0.55; font-size: 0.73rem; font-weight: 500;
    }
    .alert-truck  { font-family:'Inter',monospace; font-weight:700; color:var(--text-color); }
    .alert-savings { text-align:right; font-family:'Inter',monospace; color:var(--p-ok); font-weight:700; }
    .alert-meta   { font-size:0.7rem; color:var(--text-color); opacity:0.45; margin-top:0.1rem; }
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
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Data loading
# ============================================================================
def _sync_from_tracking(data_dir: Path) -> None:
    """Generiert die CSVs aus dem tracking-Repo, falls die Quelldaten vorhanden sind."""
    adapter = data_dir / "generate_from_tracking.py"
    tracking_src = data_dir.parent / "pipeline" / "data" / "engine_health" / "engine_data_final.csv"
    if not adapter.exists() or not tracking_src.exists():
        return
    import importlib.util
    spec = importlib.util.spec_from_file_location("generate_from_tracking", adapter)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        mod.main()
    except Exception as e:
        st.warning(f"Tracking-Sync fehlgeschlagen: {e}")


@st.cache_data(ttl=300)
def load_data():
    base = Path(__file__).parent / "data"
    _sync_from_tracking(base)
    fleet = pd.read_csv(base / "fleet.csv")
    timeseries = pd.read_csv(base / "timeseries.csv", parse_dates=["timestamp"])
    alerts = pd.read_csv(base / "alerts.csv", parse_dates=["timestamp"])
    truck_alerts = pd.read_csv(base / "truck_alerts.csv", parse_dates=["timestamp"])
    return fleet, timeseries, alerts, truck_alerts

fleet, timeseries, alerts, truck_alerts = load_data()

# ============================================================================
# Session state for navigation
# ============================================================================
if "role" not in st.session_state:
    st.session_state.role = None
if "view" not in st.session_state:
    st.session_state.view = "fleet"
if "selected_truck" not in st.session_state:
    st.session_state.selected_truck = None
if "alert_filter" not in st.session_state:
    st.session_state.alert_filter = "ALLE"

query_role = st.query_params.get("role")
if query_role in ("fm", "wl"):
    st.session_state.role = query_role

query_view = st.query_params.get("view")
if query_view == "fleet":
    st.session_state.view = "fleet"
    st.session_state.selected_truck = None
elif query_view == "alerts":
    st.session_state.view = "alerts"
    st.session_state.alert_filter = st.query_params.get("filter", "ALLE")
elif query_view == "detail":
    query_truck = st.query_params.get("truck")
    if query_truck in set(fleet["lkw_id"]):
        st.session_state.view = "detail"
        st.session_state.selected_truck = query_truck

# ============================================================================
# Helper functions
# ============================================================================
def status_badge(status):
    cls = {"KRITISCH": "badge-critical", "WARNUNG": "badge-warning",
           "OK": "badge-ok", "INFO": "badge-info"}.get(status, "badge-info")
    return f'<span class="badge {cls}">{status}</span>'

def severity_color(severity):
    return {"KRITISCH": "#FF3D4C", "WARNUNG": "#FFA500",
            "INFO": "#2B6CB0", "OK": "#32C759"}.get(severity, "#8A8A8F")

def compact_number(value):
    return f"{value:,}".replace(",", ".")

def _role_param():
    r = st.session_state.get("role") or ""
    return f"&role={r}" if r else ""

def detail_href(truck_id):
    return f"?view=detail&truck={quote(str(truck_id))}{_role_param()}"

def fleet_href():
    return f"?view=fleet{_role_param()}"

def alerts_href(filter_severity):
    return f"?view=alerts&filter={quote(str(filter_severity))}{_role_param()}"

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
    'font-size:0.68rem;letter-spacing:0.05em;text-decoration:none!important;">Wechseln</a>'
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
    _fnc = "nav-button active" if st.session_state.view == "fleet" else "nav-button inactive"
    _fleet_label = "FLOTTE" if st.session_state.role == "fm" else "WARTUNG"
    _nav_html = f'<div class="nav-buttons"><a class="{_fnc}" href="{fleet_href()}" target="_self">{_fleet_label}</a>'
    if st.session_state.role == "fm":
        _anc = "nav-button active" if st.session_state.view == "alerts" else "nav-button inactive"
        _nav_html += f'<a class="{_anc}" href="{alerts_href("ALLE")}" target="_self">ALERTS</a>'
    if st.session_state.view == "detail" and st.session_state.selected_truck:
        _nav_html += f'<span class="nav-button detail">{st.session_state.selected_truck}</span>'
    _nav_html += '</div>'
    st.markdown(_nav_html, unsafe_allow_html=True)

# ============================================================================
# Feedback helper
# ============================================================================
def save_feedback(truck_id: str, verdict: str) -> None:
    feedback_path = Path(__file__).parent / "data" / "feedback.csv"
    new_row = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(),
        "lkw_id": truck_id,
        "verdict": verdict,
    }])
    if feedback_path.exists():
        existing = pd.read_csv(feedback_path)
        pd.concat([existing, new_row], ignore_index=True).to_csv(feedback_path, index=False)
    else:
        new_row.to_csv(feedback_path, index=False)


# ============================================================================
# SCREEN 1: FLEET OVERVIEW
# ============================================================================
def render_fleet_overview():
    n_total = len(fleet)
    n_ok = (fleet["status"] == "OK").sum()
    n_warn = (fleet["status"] == "WARNUNG").sum()
    n_crit = (fleet["status"] == "KRITISCH").sum()
    avoided_eur = n_crit * 600 * 4  # 4h pro vermiedener Panne
    avg_rul = int(fleet["rul_hours"].mean())

    # KPI cards - clickable via buttons
    st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Fahrzeuge gesamt</div>
            <div class="kpi-value">{n_total}</div>
            <div class="kpi-sub">aktive Flotte</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card ok">
            <div class="kpi-label">Status OK</div>
            <div class="kpi-value">{n_ok}</div>
            <div class="kpi-sub">{n_ok/n_total*100:.0f}% der Flotte</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <a class="kpi-link" href="{alerts_href("WARNUNG")}" target="_self">
            <div class="kpi-card warning">
                <div class="kpi-label">Warnung</div>
                <div class="kpi-value">{n_warn}</div>
                <div class="kpi-sub">Wartung in &lt; 14 Tagen</div>
            </div>
        </a>
        """, unsafe_allow_html=True)

    with col4:
        if st.session_state.get("role") == "wl":
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Ø RUL Flotte</div>
                <div class="kpi-value">{compact_number(avg_rul)} h</div>
                <div class="kpi-sub">Durchschn. Restlaufzeit</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <a class="kpi-link" href="{alerts_href("KRITISCH")}" target="_self">
                <div class="kpi-card critical">
                    <div class="kpi-label">Kritisch</div>
                    <div class="kpi-value">{n_crit}</div>
                    <div class="kpi-sub">~ {compact_number(avoided_eur)} EUR verhinderte Kosten</div>
                </div>
            </a>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    _title_text = "Wartungsübersicht" if st.session_state.get("role") == "wl" else "Flottenstatus"
    st.markdown(f'<div class="section-title">{_title_text}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">SORTIERT NACH PRIORITÄT · LIVE-DATEN AUS BATCH-PIPELINE · STAND {datetime.now().strftime("%H:%M")} UHR</div>', unsafe_allow_html=True)

    # Sort: critical first, then warning, then OK
    status_order = {"KRITISCH": 0, "WARNUNG": 1, "OK": 2}
    fleet_sorted = fleet.copy()
    fleet_sorted["sort_key"] = fleet_sorted["status"].map(status_order)
    fleet_sorted = fleet_sorted.sort_values(["sort_key", "rul_hours"]).drop(columns=["sort_key"])

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
        dot_cls = truck["status"].lower()
        st.markdown(f"""
        <a class="row-link" href="{detail_href(truck['lkw_id'])}" target="_self">
            <div class="{row_class}">
                <div class="truck-icon"><span class="status-dot status-dot-{dot_cls}"></span></div>
                <div class="truck-id">{truck['lkw_id']}</div>
                <div>{truck['driver']}</div>
                <div>{truck['motor_temp_c']:.0f} °C</div>
                <div>{truck['brake_fluid_pct']:.0f} %</div>
                <div>{compact_number(truck['rul_hours'])} h</div>
                <div>{status_badge(truck['status'])}</div>
            </div>
        </a>
        """, unsafe_allow_html=True)

# ============================================================================
# SCREEN 2: TRUCK DETAIL
# ============================================================================
def render_truck_detail():
    truck_id = st.session_state.selected_truck
    truck = fleet[fleet["lkw_id"] == truck_id].iloc[0]

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
        if truck["brake_fluid_pct"] < 15:
            issues.append(f"Bremsflüssigkeit {truck['brake_fluid_pct']:.0f} %")
        if truck["motor_temp_c"] > 95:
            issues.append(f"Motortemperatur {truck['motor_temp_c']:.0f} °C")
        if truck["oil_pressure_bar"] < 2.5:
            issues.append(f"Öldruck {truck['oil_pressure_bar']:.1f} bar")
        issue_str = " · ".join(issues) if issues else "Kritische Sensorwerte"
        text = f"XGBoost-Modell stuft {truck_id} als kritisch ein. {issue_str} – Grenzwerte überschritten. Empfehlung: Fahrzeug aus dem Verkehr ziehen, Werkstattauftrag automatisch angelegt."
    elif truck["status"] == "WARNUNG":
        banner_cls = "warning"
        title = "VERSCHLEISS ERHÖHT · Wartung innerhalb 14 Tagen"
        text = "Verschleißmuster über Schwellwert. Wartung kann planbar in den nächsten 14 Tagen erfolgen."
    else:
        banner_cls = "ok"
        title = "FAHRZEUG IM NORMALBETRIEB"
        text = f"Alle Sensorwerte im erwarteten Bereich. Nächste turnusgemäße Wartung in {truck['rul_hours']:,} h.".replace(",", ".")

    st.markdown(f"""<div class="reco-banner {banner_cls}">
        <div class="reco-rul">RUL: {truck['rul_hours']:,} h</div>
        <div class="reco-title">{status_badge(truck["status"])} &nbsp; {title}</div>
        <div class="reco-text">{text}</div>
    </div>
    """.replace(",", "."), unsafe_allow_html=True)

    # Two columns: sensors left, chart right
    left, right = st.columns([1, 1.3])

    with left:
        st.markdown('<div class="section-title">Sensordaten</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">LIVE-WERTE AUS LETZTER BATCH-INFERENZ</div>', unsafe_allow_html=True)

        sensors = [
            ("Motortemperatur", truck["motor_temp_c"], 110, "°C", "#FF3D4C" if truck["motor_temp_c"] > 95 else "#32C759"),
            ("Öldruck",         truck["oil_pressure_bar"], 5.0, "bar", "#FFA500" if truck["oil_pressure_bar"] < 2.5 else "#32C759"),
            ("Bremsflüssigkeit", truck["brake_fluid_pct"], 100, "%",
                "#FF3D4C" if truck["brake_fluid_pct"] < 15 else ("#FFA500" if truck["brake_fluid_pct"] < 35 else "#32C759")),
            ("Reifendruck VL",  truck["tire_fl_bar"], 10, "bar", "#32C759"),
            ("Reifendruck VR",  truck["tire_fr_bar"], 10, "bar", "#32C759"),
        ]
        for label, val, max_val, unit, color in sensors:
            pct = min(100, val / max_val * 100)
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
            st.metric("Kilometerstand", f"{truck['km_total']:,} km".replace(",", "."))
            st.metric("Fahrer", truck["driver"])
        with meta_col2:
            st.metric("Beladung", f"{truck['load_pct']} %")
            st.metric("RUL-Prognose", f"{truck['rul_hours']:,} h".replace(",", "."))

    with right:
        st.markdown('<div class="section-title">Bremsflüssigkeit · letzte 72 h</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">ZEITREIHEN-DEGRADATION · ISOLATION FOREST + XGBOOST</div>', unsafe_allow_html=True)

        ts_truck = timeseries[timeseries["lkw_id"] == truck_id].copy()

        # Threshold lines
        chart = alt.Chart(ts_truck).mark_line(
            color="#FF3D4C" if truck["status"] == "KRITISCH" else ("#FFA500" if truck["status"] == "WARNUNG" else "#32C759"),
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
        warn_line = alt.Chart(pd.DataFrame({"y": [30]})).mark_rule(
            color="#FFA500", strokeDash=[4, 4], strokeWidth=1.5
        ).encode(y="y:Q")
        crit_line = alt.Chart(pd.DataFrame({"y": [15]})).mark_rule(
            color="#FF3D4C", strokeDash=[4, 4], strokeWidth=1.5
        ).encode(y="y:Q")

        # Alert annotations as vertical rules
        ts_min, ts_max = ts_truck["timestamp"].min(), ts_truck["timestamp"].max()
        alert_markers = truck_alerts[
            (truck_alerts["lkw_id"] == truck_id) &
            (truck_alerts["timestamp"] >= ts_min) &
            (truck_alerts["timestamp"] <= ts_max)
        ].copy()
        sev_colors = {"KRITISCH": "#FF3D4C", "WARNUNG": "#FFA500", "INFO": "#2B6CB0"}
        alert_layers = [
            alt.Chart(alert_markers[alert_markers["severity"] == sev]).mark_rule(
                color=color, strokeWidth=1.5, strokeDash=[3, 3]
            ).encode(
                x="timestamp:T",
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Alert", format="%d.%m %H:%M"),
                    alt.Tooltip("severity:N", title="Schweregrad"),
                    alt.Tooltip("message:N", title="Meldung"),
                ],
            )
            for sev, color in sev_colors.items()
            if not alert_markers[alert_markers["severity"] == sev].empty
        ]
        brake_chart = (
            alt.layer(chart, warn_line, crit_line, *alert_layers)
            if alert_layers else chart + warn_line + crit_line
        )
        st.altair_chart(brake_chart.configure_view(strokeWidth=0), use_container_width=True)

        cap = "⎯⎯ Warnschwelle 30 % · ⎯⎯ Kritische Schwelle 15 %"
        if not alert_markers.empty:
            cap += " · ╌╌ Alert-Zeitpunkte"
        st.caption(cap)

        # Motor temperature chart
        st.markdown('<div class="section-title">Motortemperatur · letzte 72 h</div>', unsafe_allow_html=True)
        chart2 = alt.Chart(ts_truck).mark_line(
            color="#1A1A1A", strokeWidth=2
        ).encode(
            x=alt.X("timestamp:T", title=None, axis=alt.Axis(format="%d.%m %H:%M", labelFontSize=10)),
            y=alt.Y("motor_temp_c:Q", title="Motortemp. (°C)",
                    scale=alt.Scale(domain=[60, 110]),
                    axis=alt.Axis(labelFontSize=10, titleFontSize=11)),
        ).properties(height=180)

        crit_temp = alt.Chart(pd.DataFrame({"y": [95]})).mark_rule(
            color="#FF3D4C", strokeDash=[4, 4], strokeWidth=1.5
        ).encode(y="y:Q")

        st.altair_chart((chart2 + crit_temp).configure_view(strokeWidth=0), use_container_width=True)

    # Alert history for this truck
    st.markdown(f'<div class="section-title">Alert-Verlauf · {truck_id}</div>', unsafe_allow_html=True)
    truck_history = truck_alerts[truck_alerts["lkw_id"] == truck_id].sort_values("timestamp", ascending=False)

    if truck_history.empty:
        st.info("ℹ️ Keine Alerts für dieses Fahrzeug in den letzten 30 Tagen.")
    else:
        for _, alert in truck_history.iterrows():
            st.markdown(f"""
            <div class="alert-row" style="grid-template-columns: 100px 92px minmax(0, 1fr); cursor: default;">
                <div class="alert-time">{alert['timestamp'].strftime('%d.%m. %H:%M')}</div>
                <div>{status_badge(alert['severity'])}</div>
                <div class="alert-message">{alert['message']}</div>
            </div>
            """, unsafe_allow_html=True)

    if st.session_state.get("role") == "wl":
        st.markdown('<div class="section-title" style="margin-top:1.5rem;">Wartungsfeedback</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">NACH ABSCHLUSS DER WARTUNG BITTE BESTÄTIGEN</div>', unsafe_allow_html=True)
        col_a, col_b, _ = st.columns([1.4, 1.4, 4])
        with col_a:
            if st.button("Wartung bestätigt", key=f"fb_ok_{truck_id}", type="primary"):
                save_feedback(truck_id, "bestätigt")
                st.success(f"Feedback für {truck_id} gespeichert: Wartung bestätigt.")
        with col_b:
            if st.button("Fehlalarm", key=f"fb_fa_{truck_id}"):
                save_feedback(truck_id, "Fehlalarm")
                st.success(f"Feedback für {truck_id} gespeichert: Fehlalarm.")

# ============================================================================
# SCREEN 3: ALERT FEED
# ============================================================================
def render_alert_feed():
    n_crit = (alerts["severity"] == "KRITISCH").sum()
    n_warn = (alerts["severity"] == "WARNUNG").sum()
    n_info = (alerts["severity"] == "INFO").sum()
    total_savings = alerts["savings_eur"].sum() * 4  # 4h pro Panne

    # KPIs for alert feed
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
            <div class="kpi-sub">Anomalie erkannt</div>
        </div>
        <div class="kpi-card ok">
            <div class="kpi-label">Vermiedene Kosten</div>
            <div class="kpi-value">{total_savings:,} €</div>
            <div class="kpi-sub">letzte 7 Tage · geschätzt</div>
        </div>
    </div>
    """.replace(",", "."), unsafe_allow_html=True)

    st.markdown('<div class="section-title">Alert-Feed</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">CHRONOLOGISCH · ALLE ML-AUSGABEN UND REGELBASIERTEN WARNUNGEN</div>', unsafe_allow_html=True)

    # Filter links use the same navigation path as the rest of the app.
    filter_markup = []
    for label in ["ALLE", "KRITISCH", "WARNUNG", "INFO"]:
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

    # Alert rows
    for _, alert in filtered.iterrows():
        st.markdown(f"""
        <a class="row-link" href="{detail_href(alert['lkw_id'])}" target="_self">
            <div class="alert-row" style="grid-template-columns: 88px 86px 66px minmax(0,1fr) 78px;">
                <div class="alert-time">{alert['timestamp'].strftime('%d.%m. %H:%M')}</div>
                <div>{status_badge(alert['severity'])}</div>
                <div class="alert-truck">{alert['lkw_id']}</div>
                <div class="alert-message">
                    {alert['message']}
                    <div class="alert-meta">Quelle: {alert['source']} · Einsparung: ~ {alert['savings_eur']} €/h</div>
                </div>
                <div class="alert-savings">~ {alert['savings_eur']*4} €</div>
            </div>
        </a>
        """, unsafe_allow_html=True)

# ============================================================================
# SCREEN 0: ROLE SELECTION
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


# ============================================================================
# Router
# ============================================================================
if not st.session_state.role:
    render_role_selection()
elif st.session_state.view == "fleet":
    render_fleet_overview()
elif st.session_state.view == "detail":
    render_truck_detail()
elif st.session_state.view == "alerts" and st.session_state.role == "fm":
    render_alert_feed()

# Footer
st.markdown("""
<div style='text-align:center; padding:1.5rem 0 0.8rem 0; color:var(--text-color); opacity:0.4;
           font-family:"IBM Plex Mono","Courier New",monospace; font-size:0.65rem;
           letter-spacing:0.1em; border-top:1px solid var(--secondary-background-color); margin-top:2rem;'>
    PREMA MVP · HM BIG DATA SS2026 · TEAM 1 · DATEN SIMULIERT · KEIN PRODUKTIVBETRIEB
</div>
""", unsafe_allow_html=True)
