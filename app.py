import os
import sys
import pickle 
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

# Load the model lazily on first use instead of at import time, so a
# missing or corrupt pickle no longer crashes the whole app on startup.
# Callers check get_model() and render a friendly 503 page when it fails.
_model = None
_model_error = None


def get_model():
    """Return the trained model, loading it on first call.

    Returns None when the model file is missing or cannot be unpickled;
    the reason is stored in _model_error for the error page.
    """
    global _model, _model_error
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
            f"Model file not found at {MODEL_PATH}. "
            "Train the model with Pokemon.ipynb and place "
            "pokemon_model.pickle next to app.py."
        )
    except Exception as exc:
        _model_error = f"Could not load the model from {MODEL_PATH}: {exc}"
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


def to_float(val, default=0.0):
    try:
        if val is None or str(val).strip() == "":
            return float(default)
        return float(val)
    except Exception:
        return float(default)


def to_int(val, default=0):
    try:
        if val is None or str(val).strip() == "":
            return int(default)
        return int(float(val))
    except Exception:
        return int(default)


def build_feature_vector(form, features):
    """
    Build a single-row feature vector in the exact order model expects.
    """
    x = {name: 0.0 for name in features}

    # numeric inputs
    x["Total"] = to_float(form.get("Total"), 0)
    x["HP"] = to_float(form.get("HP"), 0)
    x["Attack"] = to_float(form.get("Attack"), 0)
    x["Defense"] = to_float(form.get("Defense"), 0)
    x["Sp_Atk"] = to_float(form.get("Sp_Atk"), 0)
    x["Sp_Def"] = to_float(form.get("Sp_Def"), 0)
    x["Speed"] = to_float(form.get("Speed"), 0)

    x["Generation"] = to_int(form.get("Generation"), 1)

    # bool-ish
    x["hasGender"] = 1.0 if form.get("hasGender") == "on" else 0.0
    x["hasMegaEvolution"] = 1.0 if form.get("hasMegaEvolution") == "on" else 0.0

    # proportion male (0..1)
    x["Pr_Male"] = to_float(form.get("Pr_Male"), 0.5)
    x["Pr_Male"] = max(0.0, min(1.0, x["Pr_Male"]))

    # size + catch
    x["Height_m"] = to_float(form.get("Height_m"), 1.0)
    x["Weight_kg"] = to_float(form.get("Weight_kg"), 10.0)
    x["Catch_Rate"] = to_float(form.get("Catch_Rate"), 45.0)

    # Body_Style_new was frequency-encoded in your notebook
    # We accept it as a number. If user doesn't know, keep default.
    x["Body_Style_new"] = to_float(form.get("Body_Style_new"), 1.0)

    # types (multi-select)
    selected_types = form.getlist("types")
    for t in TYPE_FEATURES:
        x[t] = 1.0 if t in selected_types else 0.0

    # colors (single-select)
    selected_color = form.get("color")
    if selected_color in COLOR_FEATURES:
        x[selected_color] = 1.0

    # Return np array in correct column order
    return np.array([[x[name] for name in features]], dtype=float), x


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

        proba = None
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)[0]
            # classes_ = [False, True]
            idx_true = list(model.classes_).index(True) if True in list(model.classes_) else 1
            proba = float(p[idx_true])

        pred = bool(model.predict(X)[0])

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
    features = get_features()
    # Feature importances (if available)
    labels = []
    values = []
    if hasattr(model, "feature_importances_") and len(features) == len(model.feature_importances_):
        pairs = list(zip(features, model.feature_importances_))
        pairs.sort(key=lambda x: x[1], reverse=True)
        top = pairs[:12]
        labels = [p[0] for p in top]
        values = [float(p[1]) for p in top]

    return render_template(
        "dashboard.html",
        top_labels=labels,
        top_values=values
    )


if __name__ == "__main__":
    app.run(debug=True)
