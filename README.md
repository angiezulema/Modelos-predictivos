# Modelos Predictivos

Repositorio con **3 casos de estudio** de Machine Learning (regresión) usando Python, scikit-learn y **Gradio**.
La interfaz principal (`app.py`) tiene 3 pestañas interactivas para probar cada predicción.

## 🚀 Interfaz pública interactiva

`app.py` (raíz) construye una interfaz Gradio con 3 pestañas:

| Pestaña | Caso | Modelos disponibles |
|---|---|---|
| 🏠 California Housing | Precio mediano de vivienda | Regresión Lineal Múltiple + Regresión Polinomial (grado 2) |
| 🍷 Wine Quality | Calidad del vino tinto (0-10) | Regresión Lineal Múltiple + Regresión Polinomial (grado 2) |
| 🩸 Diabetes | Progresión de la enfermedad | Regresión Lineal Múltiple + Regresión Polinomial (grado 2) |

```bash
pip install -r requirements.txt
python app.py
```

Abrir http://localhost:7860

## 📁 Estructura

```
Modelos-predictivos/
├── app.py                     # Interfaz Gradio principal (3 pestañas)
├── requirements.txt
├── render.yaml
├── california-housing/
│   ├── housing_app.py         # Pestaña California (selector Lineal/Polinomial)
│   ├── housing_predictor.py   # Entrenamiento lineal y polinomial (grado 2)
│   ├── housing_model.joblib           # Modelo Lineal Múltiple
│   ├── housing_model_poly.joblib      # Modelo Polinomial (grado 2)
│   └── archive/housing.csv
├── wine-quality/
│   ├── wine_app.py            # Pestaña Wine Quality (selector Lineal/Polinomial)
│   ├── wine_predictor.py      # Entrenamiento lineal y polinomial
│   ├── wine_model.joblib             # Modelo Lineal Múltiple
│   ├── wine_model_poly.joblib        # Modelo Polinomial (grado 2)
│   └── winequality-red.csv    # Dataset UCI (vino tinto)
└── diabetes/
    ├── diabetes_app.py        # Pestaña Diabetes (selector Lineal/Polinomial)
    ├── diabetes_predictor.py  # Entrenamiento lineal y polinomial
    ├── diabetes_model.joblib         # Modelo Lineal Múltiple
    ├── diabetes_model_poly.joblib    # Modelo Polinomial (grado 2)
    ├── diabetes_raw.csv       # Dataset con valores reales (sklearn load_diabetes)
    ├── diabetes 01/                     # Proyecto extra Flask (+ EDA)
    └── diabetes-regresion-polinomial/   # Proyecto extra Streamlit (Polinomial + Ridge)
```

## 🌐 Desplegar en Render (gratis)

1. Subir este repositorio a GitHub.
2. En [render.com](https://render.com) → **New** → **Web Service**, conectar el repo.
3. **Root directory:** dejar en blanco (raíz del repo).
4. Build: `pip install -r requirements.txt`
5. Start: `python app.py`
6. Plan **Free** → **Create Web Service**.

También puedes usar `render.yaml` (Blueprint).

> Nota: en el plan gratis Render se duerme a los 15 min. La primera visita tarda ~40 s en despertar.