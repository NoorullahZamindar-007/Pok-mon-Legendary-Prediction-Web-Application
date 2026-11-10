import math
import os
import sys
import pickle
import threading
import numpy as np 
from flask import Flask, render_template, request, redirect, url_for, flash 
 
# ---- Fix for pickles created with newer numpy paths (numpy._core) ---- 
# If your pickle throws: ModuleNotFoundError: No module named 'numpy._core'
# this alias usually fixes it.
try:
    import numpy as _np
    sys.modules["numpy._core"] = _np.core
    sys.modules["numpy._core._multiarray_umath"] = _np.core._multiarray_umath 
    sys.modules["numpy._core.multiarray"] = _np.core.multiarray 
    sys.modules["numpy._core.umath"] = _np.core.umath    
except Exception:
    pass

app = Flask(__name__)
# Read the session-signing key from the environment so every deployment can
# use its own secret. The fallback keeps local development working; set
# SECRET_KEY in production because anyone with the default key can forge
# session cookies.
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret-key")
if app.secret_key == "change-this-secret-key":
    print("WARNING: using the default Flask secret key; set the SECRET_KEY env var.") 

MODEL_PATH = os.path.join(os.path.dirname(__file__), "pokemon_model.pickle")


def _env_int(name, default):
    """Read an integer environment variable, falling back to *default* when
    unset or malformed instead of crashing the app at startup."""
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _valid_port(port, default=5000):
    """Return a usable TCP port, falling back to *default* when the
    configured value is out of range.

    A PORT of 0 asks the OS for a random port the user cannot find, and
    a negative or >65535 value crashes app.run at startup; a typo like
    70000 used to kill the app instead of starting on the default port.
    """
    try:
        port = int(port)
    except (TypeError, ValueError):
        return default
    if 1 <= port <= 65535:
        return port
    return default

# Load the model lazily on first use instead of at import time, so a
# missing or corrupt pickle no longer crashes the whole app on startup.
# Callers check get_model() and render a friendly 503 page when it fails.
_model = None
_model_error = None
# Guards the lazy load: without it, two concurrent first requests could
# both unpickle the model and race on the globals above.
_model_lock = threading.Lock()


def get_model():
    """Return the trained model, loading it on first call.

    Returns None when the model file is missing or cannot be unpickled;
    the reason is stored in _model_error for the error page.
    """
    global _model, _model_error
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        # Not cached yet: retry the load on every call so that dropping the
        # model file into place is enough to recover (no app restart needed).
        _model_error = None
        try:
            with open(MODEL_PATH, "rb") as f:
                _model = pickle.load(f)
        except FileNotFoundError:
            _model_error = (
                f"Model file not found: {os.path.basename(MODEL_PATH)}. "
                "Train the model with Pokemon.ipynb and place "
                "pokemon_model.pickle next to app.py."
            )
        except Exception as exc:
            _model_error = f"Could not load the model from {os.path.basename(MODEL_PATH)}: {exc}"
    return _model


def get_features():
    """Feature names in the order the model expects (empty if no model)."""
    model = get_model()
    if model is None:
        return []
    return list(getattr(model, "feature_names_in_", []))


def model_unavailable():
    """Friendly 503 response used when the model could not be loaded."""
    return render_template("model_error.html", error=_model_error), 503

TYPE_FEATURES = [
    "Dark", "Dragon", "Electric", "Fighting", "Fire", "Flying",
    "Grass", "Normal", "Poison", "Rock", "Water"
]

COLOR_FEATURES = [
    "Black", "Blue", "Brown", "Green", "Grey", "Pink",
    "Purple", "Red", "White", "Yellow"
]


def to_float(val, default=0.0, minimum=None, maximum=None):
    """Parse a numeric form field, falling back to default on bad input.

    minimum/maximum clamp the result; game stats and measurements can
    never be negative, so a tampered form value cannot poison the model.
    Non-finite values (nan/inf) also fall back: float("nan") parses fine
    but would otherwise flow straight into the model as a NaN feature.
    """
    try:
        if val is None or str(val).strip() == "":
            result = float(default)
        else:
            result = float(val)
    except Exception:
        result = float(default)
    if not math.isfinite(result):
        result = float(default)
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def to_int(val, default=0, minimum=None, maximum=None):
    """Parse an integer form field, falling back to default on bad input.

    minimum/maximum clamp the result, so a tampered form value cannot
    smuggle out-of-range numbers into the model. A fractional string
    such as "3.9" is not a valid integer: the old int(float(val))
    silently truncated it to 3, so it now falls back to default
    instead of quietly changing the user's input.
    """
    try:
        if val is None or str(val).strip() == "":
            result = int(default)
        else:
            result = int(str(val).strip(), 10)
    except Exception:
        result = int(default)
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


# Battle stats every trained model must provide; a model missing one of
# these cannot produce a meaningful prediction.
_REQUIRED_FEATURES = {
    "Total",
    "HP",
    "Attack",
    "Defense",
    "Sp_Atk",
    "Sp_Def",
    "Speed",
}


def build_feature_vector(form, features):
    """
    Build a single-row feature vector in the exact order model expects.

    Assignments are guarded by the model's actual feature list: if the
    pickle was retrained with different features, a missing column raises
    a clear error instead of a bare KeyError, and extra form fields are
    ignored instead of crashing the prediction.
    """
    known = set(features)

    # The core battle stats must exist in the trained model; anything else
    # degrades gracefully to its default below.
    missing = sorted(n for n in _REQUIRED_FEATURES if n not in known)
    if missing:
        raise ValueError(
            "The trained model is missing expected feature(s): "
            + ", ".join(missing)
            + ". Retrain it with Pokemon.ipynb."
        )

    values = {}

    def put(name, value):
        if name in known:
            values[name] = value

    # numeric inputs; base stats are capped at 255 in the games, so
    # Total cannot exceed 6 x 255 — a tampered value must not reach
    # the model as a wildly out-of-distribution feature.
    put("Total", to_float(form.get("Total"), 0, minimum=0, maximum=1530))
    put("HP", to_float(form.get("HP"), 0, minimum=0, maximum=255))
    put("Attack", to_float(form.get("Attack"), 0, minimum=0, maximum=255))
    put("Defense", to_float(form.get("Defense"), 0, minimum=0, maximum=255))
    put("Sp_Atk", to_float(form.get("Sp_Atk"), 0, minimum=0, maximum=255))
    put("Sp_Def", to_float(form.get("Sp_Def"), 0, minimum=0, maximum=255))
    put("Speed", to_float(form.get("Speed"), 0, minimum=0, maximum=255))

    # Generation is 1-based; nine generations exist since Scarlet/Violet
    # (Gen 9, 2022). A tampered form value outside that range is clamped
    # instead of reaching the model.
    put("Generation", to_int(form.get("Generation"), 1, minimum=1, maximum=9))

    # bool-ish
    put("hasGender", 1.0 if form.get("hasGender") == "on" else 0.0)
    put("hasMegaEvolution", 1.0 if form.get("hasMegaEvolution") == "on" else 0.0)

    # proportion male (0..1)
    pr_male = max(0.0, min(1.0, to_float(form.get("Pr_Male"), 0.5)))
    put("Pr_Male", pr_male)

    # size + catch: the tallest/heaviest real Pokemon are ~20 m / ~1000 kg,
    # so generous caps keep tampered values from reaching the model as
    # wildly out-of-distribution features, like the other numeric fields.
    put("Height_m", to_float(form.get("Height_m"), 1.0, minimum=0, maximum=30))
    put("Weight_kg", to_float(form.get("Weight_kg"), 10.0, minimum=0, maximum=1000))
    # Catch rates in the games run 0..255; a tampered value above that is
    # outside anything the model was trained on, so clamp it like
    # Generation instead of feeding it straight to the model.
    put("Catch_Rate", to_float(form.get("Catch_Rate"), 45.0, minimum=0, maximum=255))

    # Body_Style_new was frequency-encoded in your notebook
    # We accept it as a number. If user doesn't know, keep default.
    put("Body_Style_new", to_float(form.get("Body_Style_new"), 1.0, minimum=0))

    # types (multi-select): real Pokemon have at most two types, so an
    # 11-hot vector is input the model never saw in training. Keep only
    # the first two selected types, in TYPE_FEATURES order.
    selected_types = [t for t in TYPE_FEATURES if t in form.getlist("types")][:2]
    for t in TYPE_FEATURES:
        put(t, 1.0 if t in selected_types else 0.0)

    # colors (single-select)
    selected_color = form.get("color")
    if selected_color in COLOR_FEATURES:
        put(selected_color, 1.0)

    # Return np array in correct column order
    x = {name: values.get(name, 0.0) for name in features}
    return np.array([[x[name] for name in features]], dtype=float), x


def _is_positive_label(value):
    """True when a class label means the positive (Legendary) class.

    String labels are read explicitly (bool("False") is True), and a
    numeric label counts only when it equals 1: a two-class model with
    labels [0, 2] has no identifiable positive class, so its
    probability stays unknown instead of picking column 1.
    """
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "legendary"}
    try:
        return bool(value == 1)
    except Exception:
        return False


def positive_class_proba(proba_row, classes):
    """Probability of the positive (True) class from a predict_proba row.

    Returns None when the positive class cannot be determined. A model
    trained on a single class has no positive-class column to index:
    its probability is 1.0 if that class is the positive one, else 0.0.
    The old inline code fell back to index 1 here, crashed with
    IndexError, and lost the whole prediction to a redirect.
    """
    classes = list(classes)
    # String labels need the same explicit reading as predictions:
    # `True in ["False", "True"]` is False, so a retrained model with
    # string labels used to report its probability as unknown even
    # though predict_proba returned a usable row.
    positive_idx = next(
        (i for i, label in enumerate(classes) if _is_positive_label(label)),
        None,
    )
    if positive_idx is not None:
        if positive_idx >= len(proba_row):
            return None
        # A NaN/inf probability is not a usable percentage: report it
        # as unknown instead of rendering "nan%" on the result page.
        value = float(proba_row[positive_idx])
        return value if math.isfinite(value) else None
    if len(classes) == 1:
        # No positive label among a single class means that class is
        # the negative one (e.g. ["False"]): its Legendary probability
        # is 0.0. The old `1.0 if classes[0] else 0.0` read
        # bool("False") as True and reported 1.0 instead.
        return 0.0
    return None


def safe_positive_proba(model, X):
    """Probability of the positive class, or None when it cannot be computed.

    A model with predict_proba but no classes_ already degrades to
    unknown; a model whose predict_proba itself raises (unfitted
    estimator, mismatched features) used to lose the whole prediction
    to the error redirect even though predict() still works. The
    probability is decorative — render the prediction without it.
    """
    if not hasattr(model, "predict_proba"):
        return None
    classes = getattr(model, "classes_", None)
    if classes is None:
        return None
    try:
        proba_row = model.predict_proba(X)[0]
        # The row conversion itself can fail too: a model returning a
        # 1-D array makes [0] a scalar, and positive_class_proba then
        # raises TypeError on len() — the same lost-prediction failure
        # as a raising predict_proba, so it belongs inside this try.
        return positive_class_proba(proba_row, classes)
    except Exception:
        return None


def interpret_prediction(prediction) -> bool:
    """Interpret a raw model prediction as a boolean.

    bool("False") is True, so a model retrained with string labels
    would render every "Not Legendary" prediction as Legendary.
    """
    if isinstance(prediction, str):
        return prediction.strip().lower() in {"true", "1", "yes", "legendary"}
    return bool(prediction)


@app.route("/", methods=["GET"])
def index():
    if get_model() is None:
        return model_unavailable()
    return render_template(
        "index.html",
        features=get_features(),
        type_features=TYPE_FEATURES,
        color_features=COLOR_FEATURES
    )


@app.route("/predict", methods=["POST"])
def predict():
    model = get_model()
    if model is None:
        return model_unavailable()
    try:
        X, x_dict = build_feature_vector(request.form, get_features())

        proba = safe_positive_proba(model, X)

        # Interpret the raw label explicitly: bool("False") is True,
        # so a model retrained with string labels would otherwise
        # render every "Not Legendary" prediction as Legendary.
        pred = interpret_prediction(model.predict(X)[0])

        return render_template(
            "result.html",
            pred=pred,
            proba=proba,
            input_data=x_dict
        )
    except Exception as e:
        flash(f"Prediction error: {e}")
        return redirect(url_for("index"))


@app.route("/dashboard", methods=["GET"])
def dashboard():
    model = get_model()
    if model is None:
        return model_unavailable()
    labels, values = top_feature_importances(model, get_features())

    return render_template(
        "dashboard.html",
        top_labels=labels,
        top_values=values
    )


def top_feature_importances(model, features, limit=12):
    """Top *limit* (feature, importance) pairs, most important first.

    Returns ([], []) when the model has no usable importances, so the
    dashboard shows its "not available" warning. A retrained estimator
    can expose feature_importances_ as a scalar (or another unexpected
    shape): len() on it used to raise TypeError inside the route and
    crash /dashboard with a 500. Non-finite importances (NaN/inf from
    a degenerate model) are dropped: they sort unpredictably and render
    as "nan" on the chart, while a finite-only list still ranks the
    real features.
    """
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return [], []
    try:
        if len(features) != len(importances):
            return [], []
        pairs = []
        for name, importance in zip(features, importances):
            try:
                value = float(importance)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                pairs.append((name, value))
    except TypeError:
        # feature_importances_ is a scalar or otherwise unsized: there
        # is no per-feature list to rank, so report "not available".
        return [], []
    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[:limit]
    return [p[0] for p in top], [float(p[1]) for p in top]


if __name__ == "__main__":
    # Debug mode only when explicitly enabled; the reloader and debugger
    # must never run in production because they expose the Werkzeug console.
    # Host/port are overridable so container and PaaS hosts that assign $PORT
    # (Render, Railway, ...) can start the app.
    app.run(
        host=os.environ.get("FLASK_HOST", "127.0.0.1"),
        # A malformed PORT (e.g. "abc") used to crash the app at startup;
        # fall back to 5000 instead. _valid_port also rejects out-of-range
        # values (0, negative, >65535) that would crash app.run.
        port=_valid_port(_env_int("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )
