import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import pandas as pd

from wine_predictor import cargar_modelo

BASE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE, "winequality-red.csv")
TARGET = "quality"

LINEAL = "Regresión Lineal Múltiple"
POLINOMIAL = "Regresión Polinomial (grado 2)"
OPCIONES = [LINEAL, POLINOMIAL]

LABELS = {
    "fixed acidity": "Acidez fija",
    "volatile acidity": "Acidez volátil",
    "citric acid": "Ácido cítrico",
    "residual sugar": "Azúcar residual",
    "chlorides": "Cloruros",
    "free sulfur dioxide": "Dióxido de azufre libre",
    "total sulfur dioxide": "Dióxido de azufre total",
    "density": "Densidad",
    "pH": "pH",
    "sulphates": "Sulfatos",
    "alcohol": "Alcohol (vol %)",
}

df_datos = pd.read_csv(CSV_PATH, sep=";")
FEATURES = [c for c in df_datos.columns if c != TARGET]

RANGES = {c: (float(df_datos[c].min()), float(df_datos[c].max())) for c in FEATURES}
DEFAULTS = {c: float(df_datos[c].median()) for c in FEATURES}

pipeline_lineal = cargar_modelo(polinomial=False)
pipeline_polinomial = cargar_modelo(polinomial=True)


def _paso(lo, hi):
    delta = hi - lo
    if delta <= 0:
        return 0.01
    return round(delta / 100, 5)


def predecir(modelo, *valores):
    datos = {c: v for c, v in zip(FEATURES, valores)}
    fila = pd.DataFrame([datos], columns=FEATURES)
    pipe = pipeline_lineal if modelo == LINEAL else pipeline_polinomial
    pred = float(pipe.predict(fila)[0])
    return f"Calidad del vino predicha ({modelo}): {pred:.2f} / 10 (0 = peor, 10 = mejor)"


def crear_tab():
    gr.Markdown(
        "### 🍷 Wine Quality — Regresión Lineal Múltiple / Polinomial\n"
        "Predice la **calidad del vino tinto** (0-10) a partir de 11 propiedades físico-químicas. "
        "Elige el modelo y ajusta los valores."
    )
    modelo = gr.Radio(OPCIONES, value=LINEAL, label="Modelo de regresión")
    entrada = {c: None for c in FEATURES}
    for c in FEATURES:
        lo, hi = RANGES[c]
        entrada[c] = gr.Slider(
            lo, hi, value=DEFAULTS[c], label=LABELS[c], step=_paso(lo, hi)
        )
    boton = gr.Button("Predecir", variant="primary")
    salida = gr.Textbox(label="Resultado", lines=2)

    boton.click(
        fn=predecir,
        inputs=[modelo] + [entrada[c] for c in FEATURES],
        outputs=salida,
    )