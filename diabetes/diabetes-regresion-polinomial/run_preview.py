"""
Servidor ligero Flask de prueba para el "Caso 3: Predicción de Diabetes".

Conecta la vista HTML/CSS (`templates/vista_diabetes.html`) con el pipeline
REAL de Regresión Polinomial generado en la Fase B (`models/train_models.py`).

Artefactos cargados tal cual fueron exportados por el entrenamiento:

    - `data/diabetes_scaler.joblib`      -> StandardScaler ajustado en X_train
    - `models/best_diabetes_model.joblib`-> Pipeline completo:
        PolynomialFeatures(grado d=2) + Ridge/Lasso (ya ajustado)

Pipeline de inferencia aplicado en `/predict` (mismo orden que el entrenamiento):

    Entrada cruda (10 variables, unidades reales)
        -> 1. Estandarización (StandardScaler del Módulo A)
        -> 2. Características polinomiales (PolynomialFeatures d=2, interno)
        -> 3. Predicción (Ridge regularizado)

Además del valor predicho, la respuesta JSON incluye los metadatos reales
del modelo (grado polinomial, regresor y métricas R2/RMSE del resumen).

Ejecución:
    python run_preview.py
    # o con auto-reload (útil en desarrollo):
    flask --app run_preview run --debug

Después abre http://127.0.0.1:5000/ en tu navegador.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FOLDER = PROJECT_ROOT / "data"
MODEL_FOLDER = PROJECT_ROOT / "models"

# Orden EXACTO de columnas que exige el dataset (y el PolynomialFeatures).
FEATURE_COLUMNS = ["AGE", "SEX", "BMI", "BP", "S1", "S2", "S3", "S4", "S5", "S6"]

SCALER_PATH = DATA_FOLDER / "diabetes_scaler.joblib"
MODEL_PATH = MODEL_FOLDER / "best_diabetes_model.joblib"
MODEL_INFO_PATH = MODEL_FOLDER / "best_model_info.joblib"
MODEL_SUMMARY_PATH = MODEL_FOLDER / "model_summary.csv"

# Rangos reales del dataset (para validación básica de entrada).
FEATURE_RANGES = {
    "AGE": (19.0, 79.0),
    "SEX": (1.0, 2.0),
    "BMI": (18.0, 42.2),
    "BP": (60.0, 140.0),
    "S1": (97.0, 301.0),
    "S2": (41.6, 242.4),
    "S3": (22.0, 99.0),
    "S4": (2.0, 9.09),
    "S5": (3.258, 6.107),
    "S6": (58.0, 124.0),
}

app = Flask(__name__)

# --------------------------------------------------------------------------- #
# Carga de artefactos reales (una sola vez al arrancar Flask)
# --------------------------------------------------------------------------- #
_artifacts: dict | None = None


def load_artifacts() -> dict:
    """Carga (y cachea) escalador, pipeline polinomial y metadatos del modelo.

    Lanza FileNotFoundError si algún artefacto de la Fase B no existe.
    """
    global _artifacts  # noqa: PLW0603
    if _artifacts is not None:
        return _artifacts

    missing = [p for p in (SCALER_PATH, MODEL_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Faltan artefactos de entrenamiento: "
            + ", ".join(str(p) for p in missing)
            + ". Ejecuta primero:\n"
            "  python data/process_data.py\n"
            "  python models/train_models.py"
        )

    _artifacts = {
        "scaler": joblib.load(SCALER_PATH),
        "model": joblib.load(MODEL_PATH),
        "metrics": _load_model_metrics(),
        "info": _extract_pipeline_info(joblib.load(MODEL_PATH)),
    }
    return _artifacts


def _load_model_metrics() -> dict:
    """Lee métricas reales (R2/RMSE/MAE, train/test) del resumen de la Fase B."""
    try:
        if MODEL_INFO_PATH.exists():
            row = joblib.load(MODEL_INFO_PATH)
            metrics = {k: float(v) for k, v in row.items() if isinstance(v, (int, float))}
            if metrics:
                return metrics
        if MODEL_SUMMARY_PATH.exists():
            best = pd.read_csv(MODEL_SUMMARY_PATH).sort_values(
                "r2_test", ascending=False
            ).iloc[0]
            cols = ["r2_train", "r2_test", "rmse_train", "rmse_test",
                    "mae_train", "mae_test", "cv_r2_mean", "cv_rmse_mean"]
            return {c: float(best[c]) for c in cols if c in best.index}
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] No se pudieron leer las métricas del modelo: {exc}")
    return {}


def _extract_pipeline_info(pipeline) -> dict:
    """Extrae el grado, el regresor y el alpha reales del Pipeline guardado."""
    info = {"regressor": type(pipeline).__name__, "degree": None, "alpha": None}

    if not hasattr(pipeline, "named_steps"):
        return info

    poly = pipeline.named_steps.get("poly")
    model_step = pipeline.named_steps.get("model")

    if poly is not None:
        info["degree"] = int(getattr(poly, "degree", 0) or 0)
        info["pipeline"] = (
            f"PolynomialFeatures(d={info['degree']})"
        )
    if model_step is not None:
        info["regressor"] = type(model_step).__name__
        info["alpha"] = float(getattr(model_step, "alpha", 0)) or None

    info["pipeline"] = (
        f"PolynomialFeatures(d={info['degree']}) + {info['regressor']}"
        + (f" (alpha={info['alpha']:g})" if info["alpha"] else "")
    )
    info["label"] = (
        f"Predicción calculada mediante Regresión Polinomial "
        f"[Grado {info['degree']} + {info['regressor']}]"
    )
    return info


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def severity(prediction: float) -> str:
    """Clasifica la severidad según los rangos observados del dataset."""
    if prediction <= 100:
        return "BAJA"
    if prediction <= 200:
        return "MODERADA"
    return "ALTA"


def normalise_payload(data: dict) -> dict:
    """Normaliza claves del JSON a mayúsculas (acepta minúsculas de los sliders)."""
    return {str(k).strip().upper(): v for k, v in data.items()}


def transform_and_predict(artifacts: dict, row: dict) -> float:
    """Aplica la cadena exacta: Estandarización -> Polinomial -> Ridge.

    El modelo guardado es un Pipeline `PolynomialFeatures + Ridge` que recibe
    las características YA Estandarizadas (el escalador vive fuera del pipeline,
    en `data/diabetes_scaler.joblib`, tal como se exportó en la Fase A).
    """
    raw = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    scaled = pd.DataFrame(
        artifacts["scaler"].transform(raw[FEATURE_COLUMNS]),
        columns=FEATURE_COLUMNS,
    )
    prediction = float(artifacts["model"].predict(scaled)[0])
    return prediction


# --------------------------------------------------------------------------- #
# CORS: cabeceras permitidas para llamadas desde el navegador
# --------------------------------------------------------------------------- #
@app.after_request
def add_cors_headers(response):
    """Añade cabeceras CORS para permitir peticiones desde el front-end."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept"
    response.headers["Access-Control-Max-Age"] = "86400"
    return response


# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    """Renderiza el panel clínico HTML."""
    try:
        return render_template("vista_diabetes.html")
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"No se pudo renderizar la vista: {exc}"}), 500


@app.route("/health", methods=["GET"])
def health():
    """Healthcheck: confirma que todos los artefactos están disponibles."""
    try:
        artifacts = load_artifacts()
        return jsonify(
            {
                "status": "ok",
                "model": artifacts["info"]["pipeline"],
                "metrics": artifacts["metrics"],
            }
        )
    except FileNotFoundError as exc:
        return jsonify({"status": str(exc)}), 503


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    """Recibe las 10 variables basales y devuelve la predicción polinomial."""
    if request.method == "OPTIONS":
        return jsonify({}), 204

    try:
        artifacts = load_artifacts()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(
            {
                "error": "Se esperaba un objeto JSON con las 10 variables "
                "(AGE, SEX, BMI, BP, S1..S6)."
            }
        ), 400

    data = normalise_payload(payload)

    # Validación de presencia y coherencia de cada variable.
    errors = []
    row = {}
    for feature in FEATURE_COLUMNS:
        if feature not in data:
            errors.append(f"Falta la variable '{feature}'.")
            continue
        value = data[feature]
        try:
            value = float(value)
        except (TypeError, ValueError):
            errors.append(f"La variable '{feature}' debe ser numérica.")
            continue

        lo, hi = FEATURE_RANGES[feature]
        if not lo <= value <= hi:
            errors.append(
                f"'{feature}'={value} fuera del rango válido [{lo}, {hi}]."
            )
        row[feature] = value

    if errors:
        return jsonify({"error": " ".join(errors)}), 400

    # Predicción con la cadena real del pipeline.
    try:
        prediction = transform_and_predict(artifacts, row)
        level = severity(prediction)
    except Exception as exc:  # noqa: BLE001
        return jsonify(
            {"error": f"Error al ejecutar la predicción: {exc}"}
        ), 500

    metrics = artifacts["metrics"]
    return jsonify(
        {
            "prediction": round(float(prediction), 2),
            "severity": level,
            "model_info": artifacts["info"],
            "metrics": {
                "r2_test": round(metrics.get("r2_test", float("nan")), 4),
                "rmse_test": round(metrics.get("rmse_test", float("nan")), 3),
                "r2_train": round(metrics.get("r2_train", float("nan")), 4),
                "rmse_train": round(metrics.get("rmse_train", float("nan")), 3),
            },
            "message": "Progresión estimada para el perfil clínico ingresado.",
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)