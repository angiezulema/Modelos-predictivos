import joblib
import pandas as pd
import gradio as gr
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

CSV_PATH = "archive/housing.csv"
MODEL_PATH = "housing_model.joblib"

NUMERIC_FEATURES = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]
CATEGORICAL_FEATURES = ["ocean_proximity"]
TARGET = "median_house_value"

RANGES = {
    "longitude": (-124.35, -114.31),
    "latitude": (32.54, 41.95),
    "housing_median_age": (1, 52),
    "total_rooms": (2, 39320),
    "total_bedrooms": (1, 6445),
    "population": (3, 35682),
    "households": (1, 6082),
    "median_income": (0.5, 15.02),
}

DEFAULTS = {
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 29,
    "total_rooms": 2636,
    "total_bedrooms": 537,
    "population": 1425,
    "households": 499,
    "median_income": 3.87,
}

CATEGORIES = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]


def obtener_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regresion", LinearRegression()),
        ]
    )


def entrenar():
    df = pd.read_csv(CSV_PATH).dropna(subset=["total_bedrooms"])
    X = df.drop(columns=[TARGET]).reset_index(drop=True)
    y = df[TARGET].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipe = obtener_pipeline()
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, MODEL_PATH)
    r2 = r2_score(y_test, pipe.predict(X_test))
    rmse = root_mean_squared_error(y_test, pipe.predict(X_test))
    print(f"Modelo re-entrenado. R² test = {r2:.4f}, RMSE = {rmse:,.0f}")
    return pipe


def cargar_modelo():
    try:
        return joblib.load(MODEL_PATH)
    except (FileNotFoundError, EOFError, ValueError):
        return entrenar()


pipeline = cargar_modelo()


def predecir(
    longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
    population, households, median_income, ocean_proximity,
):
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
    fila = pd.DataFrame([datos])
    prediccion = float(pipeline.predict(fila)[0])
    return f"Precio mediano predicho: ${prediccion:,.2f}"


interface = gr.Interface(
    fn=predecir,
    inputs=[
        gr.Slider(*RANGES["longitude"], value=DEFAULTS["longitude"], label="Longitud"),
        gr.Slider(*RANGES["latitude"], value=DEFAULTS["latitude"], label="Latitud"),
        gr.Slider(*RANGES["housing_median_age"], step=1, value=DEFAULTS["housing_median_age"], label="Edad mediana de la vivienda"),
        gr.Slider(*RANGES["total_rooms"], step=1, value=DEFAULTS["total_rooms"], label="Total de habitaciones"),
        gr.Slider(*RANGES["total_bedrooms"], step=1, value=DEFAULTS["total_bedrooms"], label="Total de dormitorios"),
        gr.Slider(*RANGES["population"], step=1, value=DEFAULTS["population"], label="Población"),
        gr.Slider(*RANGES["households"], step=1, value=DEFAULTS["households"], label="Hogares"),
        gr.Slider(*RANGES["median_income"], value=DEFAULTS["median_income"], label="Ingreso mediano"),
        gr.Dropdown(CATEGORIES, value="INLAND", label="Proximidad al océano"),
    ],
    outputs=gr.Textbox(label="Resultado"),
    title="Predicción de precios de vivienda (California)",
    description="Regresión lineal múltiple entrenada con el dataset California Housing. Ajusta los valores y presiona 'Submit' para simular una predicción.",
)

if __name__ == "__main__":
    interface.launch()