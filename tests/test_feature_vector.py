"""Regression tests for form parsing and feature-vector building.

app.py imports Flask at module level, so a minimal stub stands in for it
here; the tests only exercise the pure parsing helpers, never the web
server. Runnable with pytest or directly: ``python3 tests/test_feature_vector.py``.
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if "flask" not in sys.modules:
    flask_stub = types.ModuleType("flask")

    class _Flask:
        def __init__(self, *args, **kwargs):
            self.secret_key = None

        def route(self, *args, **kwargs):
            def deco(func):
                return func

            return deco

    flask_stub.Flask = _Flask
    flask_stub.render_template = lambda *a, **k: ""
    flask_stub.request = types.SimpleNamespace(form={})
    flask_stub.redirect = lambda *a, **k: None
    flask_stub.url_for = lambda *a, **k: ""
    flask_stub.flash = lambda *a, **k: None
    sys.modules["flask"] = flask_stub

import app  # noqa: E402  (import after the flask stub is intentional)


class FakeForm(dict):
    """Dict-like stand-in for Flask's request.form (adds getlist)."""

    def getlist(self, key):
        value = self.get(key)
        if value is None:
            return []
        return value if isinstance(value, list) else [value]


FULL_FEATURES = [
    "Total",
    "HP",
    "Attack",
    "Defense",
    "Sp_Atk",
    "Sp_Def",
    "Speed",
    "Generation",
    "hasGender",
    "hasMegaEvolution",
    "Pr_Male",
    "Height_m",
    "Weight_kg",
    "Catch_Rate",
    "Body_Style_new",
    *app.TYPE_FEATURES,
    *app.COLOR_FEATURES,
]


def test_to_int_clamps_to_range():
    assert app.to_int("99", 1, minimum=1, maximum=8) == 8
    assert app.to_int("-3", 1, minimum=1, maximum=8) == 1
    assert app.to_int("4", 1, minimum=1, maximum=8) == 4
    assert app.to_int("junk", 1, minimum=1, maximum=8) == 1
    assert app.to_int("", 1, minimum=1, maximum=8) == 1
    assert app.to_int(None, 1, minimum=1, maximum=8) == 1


def test_generation_clamped_to_real_generations():
    form = FakeForm({"Generation": "25"})
    _, x = app.build_feature_vector(form, FULL_FEATURES)
    assert x["Generation"] == 8.0

    form = FakeForm({"Generation": "-2"})
    _, x = app.build_feature_vector(form, FULL_FEATURES)
    assert x["Generation"] == 1.0


def test_feature_vector_shape_and_order():
    form = FakeForm(
        {
            "Total": "600",
            "HP": "80",
            "types": ["Fire", "Flying"],
            "color": "Red",
            "hasGender": "on",
        }
    )
    vector, x = app.build_feature_vector(form, FULL_FEATURES)
    assert vector.shape == (1, len(FULL_FEATURES))
    assert x["Total"] == 600.0
    assert x["Fire"] == 1.0 and x["Flying"] == 1.0
    assert x["Water"] == 0.0
    assert x["Red"] == 1.0 and x["Blue"] == 0.0
    assert x["hasGender"] == 1.0 and x["hasMegaEvolution"] == 0.0
    # unknown form fields never crash the builder
    form["nonsense"] = "x"
    vector2, _ = app.build_feature_vector(form, FULL_FEATURES)
    assert vector2.shape == vector.shape


def test_missing_model_features_raise_clear_error():
    form = FakeForm({"Total": "600"})
    features = [f for f in FULL_FEATURES if f != "Total"]
    try:
        app.build_feature_vector(form, features)
    except ValueError as exc:
        assert "Total" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing feature")


def test_optional_missing_features_degrade_gracefully():
    # A retrained model without the color one-hots still predicts.
    form = FakeForm({"Total": "600", "color": "Red"})
    features = [f for f in FULL_FEATURES if f not in app.COLOR_FEATURES]
    vector, x = app.build_feature_vector(form, features)
    assert vector.shape == (1, len(features))
    assert "Red" not in x


if __name__ == "__main__":
    test_to_int_clamps_to_range()
    test_generation_clamped_to_real_generations()
    test_feature_vector_shape_and_order()
    test_missing_model_features_raise_clear_error()
    test_optional_missing_features_degrade_gracefully()
    print("all feature-vector tests pass")
