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
```

## California Housing

Regresión lineal múltiple para predecir el **precio mediano de viviendas** en California a partir de 9 variables (ubicación, edad, cuartos, ingreso, proximidad al océano, etc.).

### Ejecutar localmente

```bash
cd california-housing
pip install -r requirements.txt
python app.py
```

Abrir http://localhost:7860

### Desplegar en Render (gratis)

1. Subir este repositorio a GitHub.
2. En [render.com](https://render.com) → **New** → **Web Service**, conectar el repo.
3. Root directory: `california-housing`.
4. Build: `pip install -r requirements.txt`
5. Start: `python app.py`
6. Plan **Free** → **Create Web Service**.

También se incluye `render.yaml` para desplegar con un Blueprint.