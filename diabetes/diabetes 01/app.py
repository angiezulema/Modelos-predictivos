# -*- coding: utf-8 -*-
"""
app.py
======
Aplicación Web (Flask) - Predicción de Progresión de Diabetes
Regresión Lineal Múltiple (Scikit-learn)

Rutas:
    GET  /              -> Página principal con el formulario de predicción
    POST /predict       -> Recibe las 10 variables y devuelve la predicción (JSON)
    GET  /api/predict   -> Endpoint JSON para consultar el modelo directamente
    GET  /eda           -> Página con el análisis exploratorio de datos

Ejecutar:
    python app.py
    http://localhost:5000
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

# ----------------------------------------------------------------------------
# Configuración de la aplicación
# ----------------------------------------------------------------------------
app = Flask(__name__)

MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

FEATURE_COLS = ["AGE", "SEX", "BMI", "BP", "S1", "S2", "S3", "S4", "S5", "S6"]

# Etiquetas en español para mostrar en la interfaz
FEATURE_LABELS = {
    "AGE": "Edad",
    "SEX": "Sexo",
    "BMI": "Índice de Masa Corporal (IMC)",
    "BP": "Presión Arterial Media",
    "S1": "S1 - Nivel sanguíneo",
    "S2": "S2 - Nivel sanguíneo",
    "S3": "S3 - Nivel sanguíneo",
    "S4": "S4 - Nivel sanguíneo",
    "S5": "S5 - Nivel sanguíneo",
    "S6": "S6 - Nivel sanguíneo",
}

FEATURE_UNITS = {
    "AGE": "años",
    "SEX": "",
    "BMI": "kg/m²",
    "BP": "mmHg",
    "S1": "",
    "S2": "",
    "S3": "",
    "S4": "",
    "S5": "",
    "S6": "",
}

# ----------------------------------------------------------------------------
# Carga de artefactos (modelo, scaler, rangos y métricas)
# ----------------------------------------------------------------------------
model = joblib.load(os.path.join(MODEL_DIR, "modelo_regresion.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

with open(os.path.join(MODEL_DIR, "feature_ranges.json"), "r", encoding="utf-8") as f:
    FEATURE_RANGES = json.load(f)

with open(os.path.join(MODEL_DIR, "metrics.json"), "r", encoding="utf-8") as f:
    METRICS = json.load(f)


def _validate_features(values):
    """Valida que las 10 variables estén presentes y sean numéricas."""
    errors = []
    for col in FEATURE_COLS:
        value = values.get(col)
        if value is None or value == "":
            errors.append(f"Falta el valor de la variable {col}")
            continue
        try:
            values[col] = float(value)
        except (TypeError, ValueError):
            errors.append(f"Valor inválido para la variable {col}")
    return values, errors


def make_prediction(values):
    """
    Aplica el scaler a las variables ingresadas, ejecuta el modelo y
    devuelve la predicción de la progresión de la enfermedad.
    """
    X = pd.DataFrame([values], columns=FEATURE_COLS)
    X_scaled = scaler.transform(X)
    prediction = float(model.predict(X_scaled)[0])
    return prediction


# ----------------------------------------------------------------------------
# Servir las imágenes del análisis exploratorio
# ----------------------------------------------------------------------------
@app.route("/outputs/<path:filename>")
def outputs(filename):
    """Sirve las imágenes generadas durante el entrenamiento."""
    return send_from_directory(os.path.join(app.root_path, OUTPUT_DIR), filename)


# ----------------------------------------------------------------------------
# Ruta principal: formulario interactivo
# ----------------------------------------------------------------------------
@app.route("/")
def index():
    """Página principal con los sliders y los resultados del modelo."""
    return render_template(
        "index.html",
        features=FEATURE_COLS,
        labels=FEATURE_LABELS,
        units=FEATURE_UNITS,
        ranges=FEATURE_RANGES,
        metrics=METRICS,
    )


# ----------------------------------------------------------------------------
# Ruta de predicción (usada por el JavaScript de la interfaz)
# ----------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """Recibe las variables por POST (JSON) y responde con la predicción."""
    data = request.get_json(silent=True) or request.form
    values, errors = _validate_features({k: data.get(k) for k in FEATURE_COLS})

    if errors:
        return jsonify({"success": False, "errors": errors}), 400

    prediction = make_prediction(values)
    return jsonify(
        {
            "success": True,
            "prediccion": round(prediction, 2),
            "variables": {k: values[k] for k in FEATURE_COLS},
            "interpretacion": interpret_prediction(prediction),
        }
    )


# ----------------------------------------------------------------------------
# Endpoint API (GET) para consumir el modelo desde cualquier cliente
# ----------------------------------------------------------------------------
@app.route("/api/predict", methods=["GET"])
def api_predict():
    """Endpoint REST: /api/predict?AGE=50&SEX=1&BMI=25.7&BP=93&...
       Devuelve la predicción en formato JSON.
    """
    values, errors = _validate_features({k: request.args.get(k) for k in FEATURE_COLS})

    if errors:
        return jsonify({"success": False, "errors": errors}), 400

    prediction = make_prediction(values)
    return jsonify(
        {
            "success": True,
            "prediccion": round(prediction, 2),
            "interpretacion": interpret_prediction(prediction),
            "modelo": "Regresión Lineal Múltiple",
            "metricas": METRICS,
        }
    )


# ----------------------------------------------------------------------------
# Página de análisis exploratorio
# ----------------------------------------------------------------------------
@app.route("/eda")
def eda():
    """Página con todas las visualizaciones del EDA y los análisis."""
    outputs = sorted(os.listdir(OUTPUT_DIR))
    images = [
        f
        for f in outputs
        if f.endswith((".png", ".jpg", ".jpeg"))
    ]
    vif = pd.read_csv(os.path.join(OUTPUT_DIR, "vif.csv"))
    outliers = pd.read_csv(os.path.join(OUTPUT_DIR, "outliers.csv"))
    return render_template(
        "eda.html",
        images=images,
        vif=vif.to_dict("records"),
        outliers=outliers.to_dict("records"),
    )


# ----------------------------------------------------------------------------
# Utilidades auxiliares
# ----------------------------------------------------------------------------
def interpret_prediction(prediction):
    """Clasifica el valor predicho para dar una interpretación sencilla."""
    if prediction < 80:
        return {"nivel": "Bajo", "clase": "bajo", "descripcion": "Riesgo bajo de progresión."}
    elif prediction < 160:
        return {"nivel": "Moderado", "clase": "moderado", "descripcion": "Riesgo moderado de progresión."}
    elif prediction < 240:
        return {"nivel": "Alto", "clase": "alto", "descripcion": "Riesgo alto de progresión."}
    else:
        return {"nivel": "Muy alto", "clase": "muy-alto", "descripcion": "Riesgo muy alto de progresión."}


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)