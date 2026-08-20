import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import pandas as pd

from diabetes_predictor import FEATURES, cargar_modelo

BASE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE, "diabetes_raw.csv")

LINEAL = "Regresión Lineal Múltiple"
POLINOMIAL = "Regresión Polinomial (grado 2)"
OPCIONES = [LINEAL, POLINOMIAL]

LABELS = {
    "age": "Edad (años)",
    "sex": "Sexo (1 = mujer, 2 = hombre)",
    "bmi": "Índice de Masa Corporal / IMC (kg/m²)",
    "bp": "Presión arterial media (mmHg)",
    "s1": "S1 — Nivel sanguíneo (tcf)",
    "s2": "S2 — Nivel sanguíneo (tcf)",
    "s3": "S3 — Nivel sanguíneo (tcf)",
    "s4": "S4 — Nivel sanguíneo (tcf)",
    "s5": "S5 — Nivel sanguíneo (tcf)",
    "s6": "S6 — Nivel sanguíneo (tcf)",
}

df_datos = pd.read_csv(CSV_PATH)
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
    return f"Progresión de la enfermedad predicha ({modelo}): {pred:.2f} (medida cuantitativa a 1 año)"


def crear_tab():
    gr.Markdown(
        "### 🩸 Diabetes — Regresión Lineal Múltiple / Polinomial\n"
        "Predice la **progresión de la enfermedad** en pacientes diabéticos a partir de 10 variables "
        "clínicas (dataset clásico de scikit-learn). Elige el modelo y ajusta los valores."
    )
    modelo = gr.Radio(OPCIONES, value=LINEAL, label="Modelo de regresión")
    entrada = {c: None for c in FEATURES}
    for c in FEATURES:
        lo, hi = RANGES[c]
        entrada[c] = gr.Slider(lo, hi, value=DEFAULTS[c], label=LABELS[c], step=_paso(lo, hi))
    boton = gr.Button("Predecir", variant="primary")
    salida = gr.Textbox(label="Resultado", lines=2)

    boton.click(
        fn=predecir,
        inputs=[modelo] + [entrada[c] for c in FEATURES],
        outputs=salida,
    )