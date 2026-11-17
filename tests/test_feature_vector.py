"""Regression tests for form parsing and feature-vector building.

app.py imports Flask at module level, so a minimal stub stands in for it
here; the tests only exercise the pure parsing helpers, never the web
server. Runnable with pytest or directly: ``python3 tests/test_feature_vector.py``.
"""

import sys
import os
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


def test_to_float_rejects_non_finite_values():
    # float("nan") / float("inf") parse successfully, so without an
    # isfinite check they would reach the model as NaN/inf features.
    assert app.to_float("nan", 0.0) == 0.0
    assert app.to_float("inf", 0.0) == 0.0
    assert app.to_float("-inf", 5.0) == 5.0
    assert app.to_float("42.5", 0.0) == 42.5


def test_to_int_rejects_fractional_strings():
    # "3.9" used to truncate silently to 3 via int(float(...)); a
    # fractional string is not a valid integer input, so it must fall
    # back to the default like any other unparseable value.
    assert app.to_int("3.9", 1, minimum=1, maximum=9) == 1
    assert app.to_int(" 4 ", 1, minimum=1, maximum=9) == 4
    assert app.to_int("4.0", 1, minimum=1, maximum=9) == 1


def test_to_int_clamps_to_range():
    assert app.to_int("99", 1, minimum=1, maximum=8) == 8
    assert app.to_int("-3", 1, minimum=1, maximum=8) == 1
    assert app.to_int("4", 1, minimum=1, maximum=8) == 4
    assert app.to_int("junk", 1, minimum=1, maximum=8) == 1
    assert app.to_int("", 1, minimum=1, maximum=8) == 1
    assert app.to_int(None, 1, minimum=1, maximum=8) == 1


def test_to_float_clamps_to_maximum():
    assert app.to_float("99999", 45.0, minimum=0, maximum=255) == 255.0
    assert app.to_float("100", 45.0, minimum=0, maximum=255) == 100.0
    assert app.to_float("-5", 45.0, minimum=0, maximum=255) == 0.0


def test_height_and_weight_clamped_to_plausible_ranges():
    # The tallest/heaviest real Pokemon are ~20 m / ~1000 kg; a tampered
    # value above that must not reach the model unclamped.
    form = FakeForm({"Height_m": "1000000000", "Weight_kg": "1e9"})
    _, x = app.build_feature_vector(form, FULL_FEATURES)
    assert x["Height_m"] == 30.0
    assert x["Weight_kg"] == 1000.0

    form = FakeForm({"Height_m": "1.7", "Weight_kg": "90.5"})
    _, x = app.build_feature_vector(form, FULL_FEATURES)
    assert x["Height_m"] == 1.7
    assert x["Weight_kg"] == 90.5


def test_only_first_two_types_are_encoded():
    # Real Pokemon have at most two types; ticking more must not produce
    # an 11-hot vector the model never saw in training.
    form = FakeForm({"types": ["Fire", "Water", "Grass", "Electric"]})
    _, x = app.build_feature_vector(form, FULL_FEATURES)
    # First two in TYPE_FEATURES order, not form order.
    assert x["Electric"] == 1.0 and x["Fire"] == 1.0
    assert x["Grass"] == 0.0 and x["Water"] == 0.0


def test_catch_rate_clamped_to_game_range():
    # Catch rates in the games run 0..255; a tampered value above that
    # must not reach the model unclamped.
    form = FakeForm({"Catch_Rate": "99999"})
    _, x = app.build_feature_vector(form, FULL_FEATURES)
    assert x["Catch_Rate"] == 255.0

    form = FakeForm({"Catch_Rate": "45"})
    _, x = app.build_feature_vector(form, FULL_FEATURES)
    assert x["Catch_Rate"] == 45.0


def test_body_style_clamped_to_dataset_size():
    # Body_Style_new is a frequency count (Pokemon sharing that body
    # style), so real values run ~1-800 and the dataset holds ~1000
    # Pokemon; a tampered value above that must not reach the model.
    form = FakeForm({"Body_Style_new": "99999"})
    _, x = app.build_feature_vector(form, FULL_FEATURES)
    assert x["Body_Style_new"] == 1000.0

    form = FakeForm({"Body_Style_new": "12"})
    _, x = app.build_feature_vector(form, FULL_FEATURES)
    assert x["Body_Style_new"] == 12.0


def test_generation_clamped_to_real_generations():
    form = FakeForm({"Generation": "25"})
    _, x = app.build_feature_vector(form, FULL_FEATURES)
    assert x["Generation"] == 9.0

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


def test_display_inputs_hides_unset_one_hots():
    # The result table used to list every one-hot column, so ~20 rows
    # of 0.0 (every unpicked type and color) buried the values the
    # user actually entered. Only the set one-hots are shown now.
    form = FakeForm(
        {
            "Total": "600",
            "HP": "80",
            "types": ["Fire", "Flying"],
            "color": "Red",
        }
    )
    _, x = app.build_feature_vector(form, FULL_FEATURES)
    shown = app.display_inputs(x)
    assert shown["Total"] == 600.0 and shown["HP"] == 80.0
    assert shown["Fire"] == 1.0 and shown["Flying"] == 1.0
    assert shown["Red"] == 1.0
    assert "Water" not in shown and "Grass" not in shown
    assert "Blue" not in shown and "Green" not in shown
    # Numeric features stay visible even when zero, and the order of
    # the model's feature list is preserved.
    assert "Attack" in shown
    assert list(shown)[:2] == ["Total", "HP"]


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


def test_env_int_falls_back_on_bad_values():
    # A malformed env var must not crash the app at startup.
    os.environ["POKEMON_TEST_PORT"] = "not-a-number"
    try:
        assert app._env_int("POKEMON_TEST_PORT", 5000) == 5000
        os.environ["POKEMON_TEST_PORT"] = "8080"
        assert app._env_int("POKEMON_TEST_PORT", 5000) == 8080
        del os.environ["POKEMON_TEST_PORT"]
        assert app._env_int("POKEMON_TEST_PORT", 5000) == 5000
    finally:
        os.environ.pop("POKEMON_TEST_PORT", None)


def test_valid_port_rejects_out_of_range_values():
    # A PORT typo used to crash app.run at startup instead of starting
    # on the default port; 0 would bind a random port the user cannot find.
    assert app._valid_port(5000) == 5000
    assert app._valid_port(8080) == 8080
    assert app._valid_port(1) == 1
    assert app._valid_port(65535) == 65535
    assert app._valid_port(0) == 5000
    assert app._valid_port(-1) == 5000
    assert app._valid_port(65536) == 5000
    assert app._valid_port(70000) == 5000
    assert app._valid_port("abc") == 5000
    assert app._valid_port(None) == 5000


def test_positive_class_proba_handles_single_class_models():
    # Normal two-class model: positive class is True, wherever it sits.
    assert app.positive_class_proba([0.25, 0.75], [False, True]) == 0.75
    assert app.positive_class_proba([0.9, 0.1], [True, False]) == 0.9
    # Single-class models used to crash on a fallback index of 1.
    assert app.positive_class_proba([1.0], [True]) == 1.0
    assert app.positive_class_proba([1.0], [False]) == 0.0
    # No positive class and no single class to infer from: unknown.
    assert app.positive_class_proba([0.5, 0.5], [0, 2]) is None
    # A non-finite probability is unknown, not a renderable percentage.
    # (The NaN/inf must sit in the positive class's own column.)
    assert app.positive_class_proba([float("nan"), 0.5], [True, False]) is None
    assert app.positive_class_proba([0.5, float("inf")], [False, True]) is None
    assert app.positive_class_proba([float("nan"), 0.5], [False, True]) == 0.5


class _RaisingProbaModel:
    classes_ = [False, True]

    def predict_proba(self, X):
        raise RuntimeError("not fitted")


class _WorkingProbaModel:
    classes_ = [False, True]

    def predict_proba(self, X):
        return [[0.2, 0.8]]


def test_safe_positive_proba_returns_none_when_predict_proba_raises():
    # A predict_proba failure must not lose the prediction: the route
    # renders it with an unknown probability instead of redirecting.
    assert app.safe_positive_proba(_RaisingProbaModel(), [[0.0]]) is None
    assert app.safe_positive_proba(_WorkingProbaModel(), [[0.0]]) == 0.8
    assert app.safe_positive_proba(object(), [[0.0]]) is None


class _OneDimensionalProbaModel:
    classes_ = [False, True]

    def predict_proba(self, X):
        # A 1-D row instead of a (1, 2) row: [0] is a scalar float, and
        # positive_class_proba used to raise TypeError on len() outside
        # the try, losing a prediction predict() could have rendered.
        return [0.2, 0.8]


def test_safe_positive_proba_returns_none_for_1d_proba_row():
    assert app.safe_positive_proba(_OneDimensionalProbaModel(), [[0.0]]) is None


def test_interpret_prediction_handles_string_labels():
    # bool("False") is True, so string labels must be read explicitly.
    assert app.interpret_prediction(True) is True
    assert app.interpret_prediction(False) is False
    assert app.interpret_prediction(1) is True
    assert app.interpret_prediction(0) is False
    assert app.interpret_prediction("True") is True
    assert app.interpret_prediction("False") is False
    assert app.interpret_prediction("false") is False
    assert app.interpret_prediction("1") is True
    assert app.interpret_prediction("0") is False
    assert app.interpret_prediction("maybe") is False


def test_positive_class_proba_handles_string_labels():
    # A model retrained with string labels used to report the
    # probability as unknown: `True in ["False", "True"]` is False,
    # so no positive column was ever found even though predict_proba
    # returned a usable row.
    assert app.positive_class_proba([0.2, 0.8], ["False", "True"]) == 0.8
    assert app.positive_class_proba([0.8, 0.2], ["True", "False"]) == 0.8
    assert app.positive_class_proba([0.2, 0.8], ["0", "1"]) == 0.8
    # bool and numeric labels keep working as before
    assert app.positive_class_proba([0.2, 0.8], [False, True]) == 0.8
    assert app.positive_class_proba([0.2, 0.8], [0, 1]) == 0.8
    # single-class string models: ["False"] is the negative class, so
    # its Legendary probability is 0.0, not 1.0 (bool("False") is True).
    assert app.positive_class_proba([1.0], ["False"]) == 0.0
    assert app.positive_class_proba([1.0], ["True"]) == 1.0
    assert app.positive_class_proba([1.0], [False]) == 0.0
    assert app.positive_class_proba([1.0], [True]) == 1.0


class _ScalarImportanceModel:
    feature_names_in_ = ["Total", "HP"]
    feature_importances_ = 0.5


class _RankedImportanceModel:
    feature_names_in_ = ["Total", "HP", "Attack"]
    feature_importances_ = [0.1, float("nan"), 0.9]


def test_top_feature_importances_handles_scalar_importances():
    # A retrained estimator can expose feature_importances_ as a scalar;
    # len() on it used to raise TypeError inside the dashboard route and
    # crash /dashboard with a 500 instead of the "not available" warning.
    assert app.top_feature_importances(_ScalarImportanceModel(), ["Total", "HP"]) == ([], [])


def test_top_feature_importances_ranks_and_drops_non_finite():
    labels, values = app.top_feature_importances(
        _RankedImportanceModel(), ["Total", "HP", "Attack"]
    )
    assert labels == ["Attack", "Total"]
    assert values == [0.9, 0.1]
    # Length mismatch: no usable ranking.
    assert app.top_feature_importances(_RankedImportanceModel(), ["Total"]) == ([], [])
    # No importances attribute at all.
    assert app.top_feature_importances(object(), ["Total"]) == ([], [])


def test_dashboard_degrades_gracefully_for_scalar_importances():
    # End-to-end through the route: a scalar-importance model must render
    # (stubbed render_template returns ""), not raise.
    previous = app._model
    app._model = _ScalarImportanceModel()
    try:
        assert app.dashboard() == ""
    finally:
        app._model = previous


def test_predict_hides_model_valueerror_from_user():
    # A retrained pickle whose predict() raises ValueError (e.g. a
    # feature-count mismatch) used to flash the raw sklearn message to
    # the user; only build_feature_vector errors are user-friendly.
    # The stubbed Flask harness: capture flash, feed an empty form
    # (defaults fill the vector), and restore module state afterwards.
    import types as _types

    class _MismatchModel:
        feature_names_in_ = FULL_FEATURES
        classes_ = [False, True]

        def predict_proba(self, X):
            return [[0.2, 0.8]]

        def predict(self, X):
            raise ValueError(
                "X has 37 features, but RandomForestClassifier is expecting 42"
            )

    messages = []
    previous_model = app._model
    previous_form = app.request.form
    previous_flash = app.flash
    app._model = _MismatchModel()
    app.request.form = FakeForm({})
    app.flash = lambda message, *a, **k: messages.append(message)
    try:
        app.predict()
    finally:
        app._model = previous_model
        app.request.form = previous_form
        app.flash = previous_flash
    assert len(messages) == 1
    assert messages[0] == "Prediction failed — please try again."
    assert "RandomForestClassifier" not in messages[0]
    assert "37 features" not in messages[0]


def test_predict_still_shows_friendly_build_errors():
    # The split must not swallow the friendly message: a model missing
    # a required feature still reports it by name.
    class _MissingFeatureModel:
        feature_names_in_ = [f for f in FULL_FEATURES if f != "Total"]
        classes_ = [False, True]

        def predict_proba(self, X):
            return [[0.2, 0.8]]

        def predict(self, X):
            return [True]

    messages = []
    previous_model = app._model
    previous_form = app.request.form
    previous_flash = app.flash
    app._model = _MissingFeatureModel()
    app.request.form = FakeForm({})
    app.flash = lambda message, *a, **k: messages.append(message)
    try:
        app.predict()
    finally:
        app._model = previous_model
        app.request.form = previous_form
        app.flash = previous_flash
    assert len(messages) == 1
    assert "Total" in messages[0]


def test_first_prediction_tolerates_scalar_returns():
    # sklearn predict() normally returns a 1-element array, but a
    # retrained/custom model can return a 0-d array or a bare scalar:
    # [0] on those raises IndexError/TypeError and used to lose the
    # prediction to the generic error page even though the model
    # answered.
    import numpy as _np

    class _ScalarModel:
        def __init__(self, value):
            self._value = value

        def predict(self, X):
            return self._value

    assert app.first_prediction(_ScalarModel(_np.array(True)), [[0.0]]) == True
    assert app.first_prediction(_ScalarModel(True), [[0.0]]) == True
    assert app.first_prediction(_ScalarModel([False]), [[0.0]]) == False
    assert app.first_prediction(_ScalarModel(_np.array([True])), [[0.0]]) == True
    # An empty result (or None) is still an error: there is no
    # prediction to render.
    for bad in ([], _np.array([]), None):
        try:
            app.first_prediction(_ScalarModel(bad), [[0.0]])
        except ValueError as exc:
            assert "no prediction" in str(exc)
        else:
            raise AssertionError("expected ValueError for %r" % (bad,))


def test_predict_renders_a_scalar_prediction():
    # End-to-end through the route: a model whose predict() returns a
    # 0-d array must render (stubbed render_template returns ""), not
    # flash the generic error.
    import numpy as _np

    class _ScalarPredictModel:
        feature_names_in_ = FULL_FEATURES
        classes_ = [False, True]

        def predict_proba(self, X):
            return [[0.2, 0.8]]

        def predict(self, X):
            return _np.array(True)

    messages = []
    previous_model = app._model
    previous_form = app.request.form
    previous_flash = app.flash
    app._model = _ScalarPredictModel()
    app.request.form = FakeForm({})
    app.flash = lambda message, *a, **k: messages.append(message)
    try:
        assert app.predict() == ""
    finally:
        app._model = previous_model
        app.request.form = previous_form
        app.flash = previous_flash
    assert messages == []


def test_predict_errors_on_empty_prediction():
    # A model returning no predictions has nothing to render: the
    # route must flash the generic error, not crash on [0].
    class _EmptyPredictModel:
        feature_names_in_ = FULL_FEATURES
        classes_ = [False, True]

        def predict_proba(self, X):
            return [[0.2, 0.8]]

        def predict(self, X):
            return []

    messages = []
    previous_model = app._model
    previous_form = app.request.form
    previous_flash = app.flash
    app._model = _EmptyPredictModel()
    app.request.form = FakeForm({})
    app.flash = lambda message, *a, **k: messages.append(message)
    try:
        app.predict()
    finally:
        app._model = previous_model
        app.request.form = previous_form
        app.flash = previous_flash
    assert len(messages) == 1
    assert messages[0] == "Prediction failed — please try again."


if __name__ == "__main__":
    # Kept at the end of the file: Python executes top to bottom, so a
    # block placed earlier runs before later tests are defined — the old
    # hand-listed block crashed on a not-yet-defined test and silently
    # skipped six others on direct runs (pytest still collected them).
    # Auto-discover every test_* function so a new test can never be
    # missed from the direct run again.
    for _name, _func in sorted(globals().items()):
        if _name.startswith("test_") and callable(_func):
            _func()
    print("all feature-vector tests pass")
