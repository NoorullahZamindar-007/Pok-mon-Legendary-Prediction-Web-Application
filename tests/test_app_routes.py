"""Route tests for the Pokémon app's error pages and body-size limit.

test_feature_vector.py stubs Flask entirely, so the error handlers and
the MAX_CONTENT_LENGTH cap had zero coverage. These tests use the real
Flask test client; the model is never loaded (the 404/413 paths and
the stubbed 500 path do not touch it). The /dashboard tests stub the
model so the page renders for real without the pickle.
"""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# test_feature_vector.py installs a minimal flask stub in sys.modules
# when it runs first in the suite; the route tests need the real Flask,
# so drop the stub (a stub module has no __file__) and any app module
# imported against it.
_dropped_stub = False
for _name in ("flask",):
    _mod = sys.modules.get(_name)
    if _mod is not None and getattr(_mod, "__file__", None) is None:
        del sys.modules[_name]
        _dropped_stub = True
if _dropped_stub:
    sys.modules.pop("app", None)

import app  # noqa: E402


def _ensure_real_flask():
    """Restore the real Flask package in sys.modules.

    test_feature_vector.py swaps in a minimal stub at collection time;
    the real app's test_client() lazily imports flask.testing, which
    fails against the stub. The feature-vector tests only use names
    bound into the stub-imported app module, so restoring the real
    package here does not disturb them.
    """
    import importlib

    mod = sys.modules.get("flask")
    if mod is not None and getattr(mod, "__file__", None) is None:
        del sys.modules["flask"]
        importlib.import_module("flask")


@pytest.fixture
def client():
    _ensure_real_flask()
    app.app.testing = False
    return app.app.test_client()


def test_missing_page_renders_error_page(client):
    response = client.get("/no-such-page")

    assert response.status_code == 404
    assert b"404" in response.data
    assert b"Page not found" in response.data
    assert b"Back to the predictor" in response.data


def test_get_predict_returns_branded_405_page(client):
    # /predict only accepts POST: a direct GET (typed URL, crawler)
    # used to show Flask's bare 405 page instead of the app's own
    # error page with a way back to the predictor.
    response = client.get("/predict")

    assert response.status_code == 405
    assert b"405" in response.data
    assert b"Method not allowed" in response.data
    assert b"Back to the predictor" in response.data


def test_oversized_form_returns_413_page(client):
    # MAX_CONTENT_LENGTH is 16 KB: a 20 KB form must hit the 413
    # handler, not the bare Flask default page.
    assert app.app.config["MAX_CONTENT_LENGTH"] == 16 * 1024
    response = client.post("/predict", data={"Total": "9" * 20_000})

    assert response.status_code == 413
    assert b"413" in response.data
    assert b"Form too large" in response.data


def test_oversized_form_returns_413_even_when_model_unavailable(client, monkeypatch):
    # The size cap is about the request, not the model: with a missing
    # or corrupt pickle the route used to answer 503 (the model page)
    # for an oversized form, because get_model() ran before the form
    # was ever parsed and Flask's lazy 413 never fired.
    monkeypatch.setattr(app, "get_model", lambda: None)

    response = client.post("/predict", data={"Total": "9" * 20_000})

    assert response.status_code == 413
    assert b"413" in response.data
    assert b"Form too large" in response.data


def test_health_returns_ok(client):
    # Liveness probe: always 200 while the app is up. The model field
    # is not asserted — the probe must pass whether or not the pickle
    # loads.
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


class _StubModel:
    """Stand-in for the trained model: feature names + importances only."""

    feature_names_in_ = ["Total", "HP", "Attack"]
    feature_importances_ = [0.5, 0.3, 0.2]


def test_dashboard_renders_chart_with_stub_model(client, monkeypatch):
    # /dashboard had no real-Flask coverage: the helper tests call the
    # view against a render stub, so a typo in dashboard.html (or a
    # missing template variable) would ship silently. Render it for
    # real with a stub model and check the chart lands on the page.
    monkeypatch.setattr(app, "get_model", lambda: _StubModel())

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert b"importanceChart" in response.data
    assert b"window.__IMPORTANCE__" in response.data
    assert b"Total" in response.data


def test_dashboard_shows_warning_without_importances(client, monkeypatch):
    # A model with no feature_importances_ must show the dashboard's
    # "not available" warning, not an empty chart card.
    class _NoImportances:
        feature_names_in_ = ["Total", "HP"]

    monkeypatch.setattr(app, "get_model", lambda: _NoImportances())

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert b"Feature importance not available" in response.data
    assert b"importanceChart" not in response.data


def test_index_renders_predictor_form_with_real_model(client):
    # index.html had no real-Flask coverage: the predict tests call
    # the view against a render stub, so a typo in the form template
    # (or a missing variable like model_name) would ship silently.
    # The real pickle loads here, so the form renders for real.
    response = client.get("/")

    assert response.status_code == 200
    assert b"Legendary Prediction" in response.data
    assert b"Generation" in response.data
    assert b"RandomForestClassifier" in response.data


def test_predict_renders_result_with_real_model(client):
    # The predict happy path was only ever exercised against a render
    # stub, so result.html itself (the probability line, the inputs
    # table) was never rendered for real. Post the form's own defaults
    # and check the page that comes back.
    response = client.post(
        "/predict",
        data={
            "Generation": "1",
            "Catch_Rate": "45",
            "Total": "390",
            "HP": "60",
            "Attack": "70",
            "Defense": "60",
            "Sp_Atk": "70",
            "Sp_Def": "60",
            "Speed": "70",
            "Height_m": "1.0",
            "Weight_kg": "10",
            "Pr_Male": "0.5",
            "Body_Style_new": "1",
            "hasGender": "on",
        },
    )

    assert response.status_code == 200
    assert b"Prediction Result" in response.data
    assert b"Probability (Legendary)" in response.data
    assert b"Total" in response.data


def test_index_and_dashboard_render_model_error_page_when_model_unavailable(
    client, monkeypatch
):
    # With a missing or corrupt pickle both pages answer 503 with the
    # branded model_error.html; that page had no real-Flask coverage,
    # so a typo in it (or a renamed template variable) would ship
    # silently. The error text is set explicitly: monkeypatching
    # get_model alone leaves _model_error at whatever a previous
    # load left behind.
    monkeypatch.setattr(app, "get_model", lambda: None)
    monkeypatch.setattr(
        app,
        "_model_error",
        "Model file not found: pokemon_model.pickle. Train the model "
        "with Pokemon.ipynb and place pokemon_model.pickle next to app.py.",
    )

    index_response = client.get("/")
    assert index_response.status_code == 503
    assert b"Prediction model unavailable" in index_response.data
    assert b"Model file not found" in index_response.data
    assert b"Try again" in index_response.data

    dashboard_response = client.get("/dashboard")
    assert dashboard_response.status_code == 503
    assert b"Prediction model unavailable" in dashboard_response.data


def test_server_error_renders_error_page(client, monkeypatch):
    def _boom():
        raise RuntimeError("boom")

    monkeypatch.setitem(app.app.view_functions, "index", _boom)

    response = client.get("/")

    assert response.status_code == 500
    assert b"500" in response.data
    assert b"Something went wrong" in response.data
