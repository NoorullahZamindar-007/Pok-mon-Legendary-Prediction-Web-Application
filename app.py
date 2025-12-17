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
app.secret_key = "change-this-secret-key"

MODEL_PATH = os.path.join(os.path.dirname(__file__), "pokemon_model.pickle")

# Load model once on startup
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURES = list(getattr(model, "feature_names_in_", []))

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


def build_feature_vector(form):
    """
    Build a single-row feature vector in the exact order model expects.
    """
    x = {name: 0.0 for name in FEATURES}

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
    return np.array([[x[name] for name in FEATURES]], dtype=float), x


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        features=FEATURES,
        type_features=TYPE_FEATURES,
        color_features=COLOR_FEATURES
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        X, x_dict = build_feature_vector(request.form)

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
    # Feature importances (if available)
    labels = []
    values = []
    if hasattr(model, "feature_importances_") and len(FEATURES) == len(model.feature_importances_):
        pairs = list(zip(FEATURES, model.feature_importances_))
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
