import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE, "diabetes_raw.csv")
MODEL_PATH = os.path.join(BASE, "diabetes_model.joblib")
POLY_PATH = os.path.join(BASE, "diabetes_model_poly.joblib")
POLY_DEGREE = 2
TARGET = "progression"

FEATURES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]


def cargar_datos():
    return pd.read_csv(CSV_PATH)


def obtener_pipeline(polinomial=False, grado=POLY_DEGREE):
    pasos = [("scaler", StandardScaler())]
    if polinomial:
        pasos.append(("poly", PolynomialFeatures(degree=grado, include_bias=False)))
    pasos.append(("regresion", LinearRegression()))
    return Pipeline(steps=pasos)


def _descripcion(polinomial):
    return "REGRESIÓN POLINOMIAL (GRADO 2)" if polinomial else "REGRESIÓN LINEAL MÚLTIPLE"


def entrenar(polinomial=False, grado=POLY_DEGREE):
    ruta = POLY_PATH if polinomial else MODEL_PATH
    df = cargar_datos()
    X = df[FEATURES]
    y = df[TARGET].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipe = obtener_pipeline(polinomial=polinomial, grado=grado)
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, ruta)
    r2 = r2_score(y_test, pipe.predict(X_test))
    rmse = root_mean_squared_error(y_test, pipe.predict(X_test))
    print(f"Diabetes ({_descripcion(polinomial)}): R² test = {r2:.4f}, RMSE = {rmse:.2f}")
    print(f"Modelo guardado en {ruta}")
    return pipe


def cargar_modelo(polinomial=False):
    ruta = POLY_PATH if polinomial else MODEL_PATH
    try:
        return joblib.load(ruta)
    except (FileNotFoundError, EOFError, ValueError):
        print(f"No se encontró el modelo {ruta}. Entrenando de nuevo...")
        return entrenar(polinomial=polinomial)


if __name__ == "__main__":
    if "--entrenar" in sys.argv:
        entrenar(polinomial="--polinomial" in sys.argv)
    else:
        cargar_modelo(polinomial="--polinomial" in sys.argv)