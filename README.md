# Modelos Predictivos

Repositorio con proyectos de modelos predictivos de Machine Learning (regresión) usando Python, scikit-learn y Gradio.

## Estructura

```
Modelos-predictivos/
├── california-housing/        # Predicción de precios de vivienda (California Housing)
│   ├── app.py                 # Interfaz Gradio
│   ├── housing_predictor.py   # Entrenamiento + consola interactiva
│   ├── housing_model.joblib   # Modelo entrenado
│   ├── requirements.txt
│   └── archive/housing.csv    # Dataset
├── wine-quality/              # (pendiente) Calidad del vino
└── diabetes/                  # (pendiente) Diabetes
|____diabetes-regresion-polinomial/         # Caso 3: Diabetes (Regresión Polinomial + Ridge)
│   ├── Taller 1/                      # Dataset fuente original
│   ├── data/                          # Módulo de preprocesamiento
│   │   ├── process_data.py            # EDA, tratamiento de outliers, VIF y escalado
│   │   └── scaler.joblib              # Escalador guardado (StandardScaler)
│   ├── models/                        # Módulo de entrenamiento estadístico
│   │   ├── train_models.py            # Entrenamiento de Regresión Polinomial (Grado 2 + Ridge)
│   │   └── best_diabetes_model.joblib # Modelo polinomial final optimizado y guardado
│   ├── app/                           # Aplicativo multi-caso en Streamlit
│   │   ├── app.py                     # Interfaz principal y menú de navegación
│   │   └── components/
│   │       └── case3_ui.py            # Vista específica en Streamlit para Diabetes
│   ├── static/
│   │   └── estilos_diabetes.css       # Hoja de estilos CSS (diseño de sliders y badges)
│   ├── templates/
│   │   └── vista_diabetes.html        # Dashboard clínico HTML5 con controles interactivos
│   ├── run_preview.py                 # Servidor Flask para conectar el HT

## California Housing

Regresión lineal múltiple para predecir el **precio mediano de viviendas** en California a partir de 9 variables (ubicación, edad, cuartos, ingreso, proximidad al océano, etc.).

### Ejecutar localmente

```bash
cd california-housing
pip install -r requirements.txt
python app.py
```

Abrir http://localhost:7860

## Diabetes
###Ejecutar localmente (Regresion Polimonial)
python data/process_data.py
python models/train_models.py
python run_preview.py

Abrir: http://127.0.0.1:5000

### Desplegar en Render (gratis)

1. Subir este repositorio a GitHub.
2. En [render.com](https://render.com) → **New** → **Web Service**, conectar el repo.
3. Root directory: `california-housing`.
4. Build: `pip install -r requirements.txt`
5. Start: `python app.py`
6. Plan **Free** → **Create Web Service**.

También se incluye `render.yaml` para desplegar con un Blueprint.
