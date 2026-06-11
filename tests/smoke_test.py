"""Smoke-Tests fuer das PREMA-Dashboard.

Rendert alle Views ueber streamlit.testing.AppTest und prueft den
Login-Flow mit und ohne konfigurierte Secrets.

Aufruf (aus dem Projekt-Root):
    python tests/smoke_test.py
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parent.parent / "app.py")


def run_view(params: dict, label: str) -> bool:
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params.update(params)
    at.run()
    errs = [e.value for e in at.exception]
    print(f'{label:32s} exceptions: {errs if errs else "keine"}')
    return not errs


def test_views() -> bool:
    ok = True
    ok &= run_view({}, "Rollenauswahl")
    ok &= run_view({"role": "fm", "view": "fleet"}, "FM Flotte")
    ok &= run_view({"role": "wl", "view": "fleet"}, "WL Wartung")
    ok &= run_view({"role": "fm", "view": "alerts", "filter": "ALLE"}, "FM Alerts ALLE")
    ok &= run_view({"role": "fm", "view": "alerts", "filter": "INFO"}, "FM Alerts INFO")
    ok &= run_view({"role": "fm", "view": "alerts", "filter": "QUATSCH"}, "FM Alerts ungueltiger Filter")
    ok &= run_view({"role": "fm", "view": "detail", "truck": "LKW-01"}, "FM Detail LKW-01")
    ok &= run_view({"role": "wl", "view": "detail", "truck": "LKW-07"}, "WL Detail LKW-07")
    ok &= run_view({"role": "wl", "view": "alerts"}, "WL Alerts (Umleitung)")
    ok &= run_view({"role": "fm", "view": "detail", "truck": "LKW-99"}, "FM Detail unbekannter LKW")
    return ok


def test_login() -> bool:
    # Rolle mit Passwort -> Login-Formular statt Dashboard
    at = AppTest.from_file(APP, default_timeout=60)
    at.secrets["passwords"] = {"fm": "geheim", "wl": "werkstatt"}
    at.query_params.update({"role": "fm", "view": "fleet"})
    at.run()
    assert not at.exception, at.exception
    assert len(at.text_input) == 1, "Login-Formular erwartet"
    print("Login-Formular erscheint: OK")

    # Falsches Passwort -> Fehlermeldung
    at.text_input[0].set_value("falsch")
    at.button[0].set_value(True).run()
    assert any("Falsches Passwort" in e.value for e in at.error), "Fehlermeldung erwartet"
    print("Falsches Passwort abgelehnt: OK")

    # Richtiges Passwort -> Dashboard
    at.text_input[0].set_value("geheim")
    at.button[0].set_value(True).run()
    assert not at.exception, at.exception
    assert at.session_state["role"] == "fm"
    print("Korrektes Passwort -> Dashboard: OK")

    # Auth-Token in der URL ueberlebt Link-Navigation (= neue Session)
    import hashlib
    token = hashlib.sha256(b"geheim").hexdigest()[:16]
    at3 = AppTest.from_file(APP, default_timeout=60)
    at3.secrets["passwords"] = {"fm": "geheim", "wl": "werkstatt"}
    at3.query_params.update({"role": "fm", "view": "alerts", "auth": token})
    at3.run()
    assert not at3.exception, at3.exception
    assert at3.session_state["role"] == "fm", "Token-Login erwartet"
    assert len(at3.text_input) == 0, "kein Login-Formular bei gueltigem Token"
    print("Auth-Token ueberlebt Navigation: OK")

    # Ungueltiges Token -> Login-Formular
    at4 = AppTest.from_file(APP, default_timeout=60)
    at4.secrets["passwords"] = {"fm": "geheim", "wl": "werkstatt"}
    at4.query_params.update({"role": "fm", "view": "fleet", "auth": "falschtoken"})
    at4.run()
    assert not at4.exception, at4.exception
    assert len(at4.text_input) == 1, "Login-Formular bei ungueltigem Token erwartet"
    print("Ungueltiges Token abgelehnt: OK")

    # Ohne Secrets -> Demo-Modus, direkt Dashboard
    at2 = AppTest.from_file(APP, default_timeout=60)
    at2.query_params.update({"role": "wl", "view": "fleet"})
    at2.run()
    assert not at2.exception
    assert at2.session_state["role"] == "wl"
    print("Demo-Modus ohne Secrets: OK")
    return True


if __name__ == "__main__":
    ok = test_views()
    ok &= test_login()
    print("GESAMT:", "OK" if ok else "FEHLER")
    raise SystemExit(0 if ok else 1)
