"""
Módulo C: Interfaz de usuario Streamlit para el "Caso 3: Predicción de Diabetes".

Este módulo construye un formulario dinámico con las 10 variables basales del
dataset de diabetes. Los límites mínimos/máximos y los valores por defecto de
cada campo se calculan a partir de las estadísticas reales del dataset cargado
desde "Taller_1/" (con fallback a sklearn).

Al pulsar "Predecir Progresión", la entrada se:
    1. Estandariza con el StandardScaler persistido en `data/`.
    2. Predice con el mejor modelo guardado en `models/best_diabetes_model.joblib`.
    3. Muestra la predicción cuantitativa con un indicador de severidad.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.process_data import FEATURE_COLUMNS, get_data_stats, load_diabetes_dataset  # noqa: E402

FEATURE_LABELS = {
    "AGE": "Edad (años)",
    "SEX": "Sexo (1=Hombre, 2=Mujer)",
    "BMI": "Índice de Masa Corporal (kg/m²)",
    "BP": "Presión Arterial Media (mm Hg)",
    "S1": "Suero S1 (Células rojas, tcells)",
    "S2": "Suero S2 (Células blancas, ldl)",
    "S3": "Suero S3 (Colesterol total, hdl)",
    "S4": "Suero S4 (tch/ldl ratio)",
    "S5": "Suero S5 (Glucemia log) [ltg]",
    "S6": "Suero S6 (Insulina) [glu]",
}


@st.cache_data(show_spinner=False)
def get_feature_stats() -> dict:
    """Carga las estadísticas reales del dataset para configurar la UI."""
    df = load_diabetes_dataset()
    df.columns = [str(c).strip().upper().replace(" ", "_") for c in df.columns]
    stats = get_data_stats(df)
    for col in FEATURE_COLUMNS:
        if col not in stats:
            stats[col] = {"min": 0.0, "max": 1.0, "mean": 0.0, "std": 1.0}
    return stats


def _load_artifacts() -> tuple:
    """Carga escalador y modelo persistidos por los módulos A y B."""
    from data.process_data import OUTPUT_FOLDER
    from models.train_models import MODEL_FOLDER

    scaler_path = OUTPUT_FOLDER / "diabetes_scaler.joblib"
    model_path = MODEL_FOLDER / "best_diabetes_model.joblib"

    if not scaler_path.exists() or not model_path.exists():
        st.error(
            "No se encontraron los artefactos de entrenamiento.\n\n"
            "Ejecuta primero los pipelines:\n"
            "```bash\n"
            f"python {PROJECT_ROOT / 'data' / 'process_data.py'}\n"
            f"python {PROJECT_ROOT / 'models' / 'train_models.py'}\n"
            "```"
        )
        st.stop()

    import joblib

    return joblib.load(scaler_path), joblib.load(model_path)


def build_form(stats: dict) -> dict:
    """Construye el formulario dinámico con las 10 variables basales.

    Uses st.number_input for continuous variables and steady custom handling
    for the categorical SEX (1/2). Returns a dict with float values.
    """
    st.markdown("### 📋 Formulario de Variables Basales")
    inputs = {}

    col1, col2 = st.columns(2)
    left_fields = ["AGE", "SEX", "BMI", "BP", "S1"]
    right_fields = ["S2", "S3", "S4", "S5", "S6"]

    for field in left_fields:
        with col1:
            inputs[field] = _input_for_feature(field, stats)
    for field in right_fields:
        with col2:
            inputs[field] = _input_for_feature(field, stats)

    return inputs


def _input_for_feature(field: str, stats: dict) -> float:
    """Crea el widget de entrada adecuado para cada variable basal."""
    label = FEATURE_LABELS.get(field, field)
    s = stats[field]

    if field == "SEX":
        sex_labels = {
            "Hombre (1)": 1.0,
            "Mujer (2)": 2.0,
        }
        chosen = st.selectbox(label, options=list(sex_labels.keys()))
        return sex_labels[chosen]

    value = float(np.clip(s["mean"], s["min"], s["max"]))
    return st.number_input(
        label,
        min_value=float(s["min"]),
        max_value=float(s["max"]),
        value=value,
        step=round(float((s["max"] - s["min"]) / 100), 2) or 0.01,
    )


def make_prediction(inputs: dict, scaler, model) -> float:
    """Estandariza las entradas y devuelve la predicción de progresión."""
    raw = pd.DataFrame([inputs]).astype(float)
    scaled = pd.DataFrame(
        scaler.transform(raw[FEATURE_COLUMNS]),
        columns=FEATURE_COLUMNS,
    )
    return float(model.predict(scaled)[0])


def _severity(pred: float) -> tuple[str, str]:
    """Clasifica la severidad de la progresión según rangos del dataset."""
    if pred <= 100:
        return "BAJA", "success"
    if pred <= 200:
        return "MODERADA", "warning"
    return "ALTA", "danger"


def render_case3_ui() -> None:
    """Renderiza la interfaz completa del Caso 3 (Diabetes)."""
    st.title("🩺 Caso 3 — Predicción de Diabetes")
    st.markdown(
        """
        Predicción cuantitativa de la **progresión de la enfermedad un año
        después del diagnóstico** (variable `Y`), basada en **regresión
        polinomial con regularización** sobre las 10 variables basales.
        """
    )

    stats = get_feature_stats()
    inputs = build_form(stats)

    st.markdown("---")

    if st.button("Predecir Progresión", type="primary", use_container_width=True):
        scaler, model = _load_artifacts()
        with st.spinner("Prediciendo..."):
            prediction = make_prediction(inputs, scaler, model)

        level, tone = _severity(prediction)

        st.markdown("#### Resultado de la predicción")
        st.metric(
            label="Progresión estimada de la enfermedad (Y)",
            value=f"{prediction:,.1f}",
            delta=f"Severidad {level.lower()}",
        )

        if tone == "success":
            st.success("Progresión baja: valor dentro del rango de bajo riesgo.")
        elif tone == "warning":
            st.warning("Progresión moderada: se recomienda seguimiento clínico.")
        else:
            st.error("Progresión alta: se recomienda atención médica prioritaria.")

    with st.expander("ℹ️ Sobre el modelo y los datos"):
        st.markdown(
            """
            - **Caso 3** usa el dataset clásico *diabetes* (442 pacientes).
            - **Pipeline:** StandardScaler -> PolynomialFeatures (d ∈ [2, 3])
              -> Ridge/Lasso regularizado si hay sobreajuste.
            - **Validación:** Cross-Validation k=5; métricas R², RMSE y MAE
              en train/test (ver `models/model_summary.csv`).
            - El formulario recibe **valores crudos** (unidades reales) y los
              estandariza internamente antes de predecir.
            """
        )

    st.markdown("---")
    st.caption(
        f"Dataset con {int(stats['AGE']['mean']):.0f}+ años de medio en edad de los pacientes "
        f"| 10 variables basales | Salida: progresión 1 año después."
    )


if __name__ == "__main__":
    render_case3_ui()