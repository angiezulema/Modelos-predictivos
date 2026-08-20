import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import pandas as pd

from housing_predictor import (
    CATEGORIES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    SUGGESTED,
    cargar_modelo,
)

LINEAL = "Regresión Lineal Múltiple"
POLINOMIAL = "Regresión Polinomial (grado 2)"
OPCIONES = [LINEAL, POLINOMIAL]

LABELS = {
    "longitude": "Longitud",
    "latitude": "Latitud",
    "housing_median_age": "Edad mediana de la vivienda",
    "total_rooms": "Total de habitaciones",
    "total_bedrooms": "Total de dormitorios",
    "population": "Población",
    "households": "Hogares",
    "median_income": "Ingreso mediano",
}

INT_COLS = {
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
}

pipeline_lineal = cargar_modelo(polinomial=False)
pipeline_polinomial = cargar_modelo(polinomial=True)

RANGES = {c: SUGGESTED[c][:2] for c in NUMERIC_FEATURES}
DEFAULTS = {c: SUGGESTED[c][2] for c in NUMERIC_FEATURES}


def predecir(modelo, longitude, latitude, housing_median_age, total_rooms,
             total_bedrooms, population, households, median_income,
             ocean_proximity):
    datos = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity,
    }
    fila = pd.DataFrame([datos], columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    pipe = pipeline_lineal if modelo == LINEAL else pipeline_polinomial
    prediccion = float(pipe.predict(fila)[0])
    return f"Precio mediano predicho ({modelo}): ${prediccion:,.2f}"


def crear_tab():
    gr.Markdown(
        "### 🏠 California Housing — Regresión Lineal Múltiple / Polinomial\n"
        "Predice el **precio mediano de viviendas** en California a partir de 9 variables. "
        "Elige el modelo y ajusta los valores."
    )
    modelo = gr.Radio(OPCIONES, value=LINEAL, label="Modelo de regresión")
    entradas = {}
    for c in NUMERIC_FEATURES:
        lo, hi = RANGES[c]
        es_int = c in INT_COLS
        entradas[c] = gr.Slider(
            lo, hi, value=DEFAULTS[c], step=1 if es_int else 0.01, label=LABELS[c]
        )
    categoria = gr.Dropdown(CATEGORIES, value="INLAND", label="Proximidad al océano")
    boton = gr.Button("Predecir", variant="primary")
    salida = gr.Textbox(label="Resultado", lines=2)

    boton.click(
        fn=predecir,
        inputs=[modelo] + [entradas[c] for c in NUMERIC_FEATURES] + [categoria],
        outputs=salida,
    )