"""Route tests for the Pokémon app's error pages and body-size limit.

test_feature_vector.py stubs Flask entirely, so the error handlers and
the MAX_CONTENT_LENGTH cap had zero coverage. These tests use the real
Flask test client; the model is never loaded (the 404/413 paths and
the stubbed 500 path do not touch it).
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


def test_health_returns_ok(client):
    # Liveness probe: always 200 while the app is up. The model field
    # is not asserted — the probe must pass whether or not the pickle
    # loads.
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_server_error_renders_error_page(client, monkeypatch):
    def _boom():
        raise RuntimeError("boom")

    monkeypatch.setitem(app.app.view_functions, "index", _boom)

    response = client.get("/")

    assert response.status_code == 500
    assert b"500" in response.data
    assert b"Something went wrong" in response.data
