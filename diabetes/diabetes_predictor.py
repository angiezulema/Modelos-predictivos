import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE, "diabetes_raw.csv")
MODEL_PATH = os.path.join(BASE, "diabetes_model.joblib")
TARGET = "progression"

FEATURES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]


def cargar_datos():
    return pd.read_csv(CSV_PATH)


def obtener_pipeline():
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regresion", LinearRegression()),
        ]
    )


def entrenar():
    df = cargar_datos()
    X = df[FEATURES]
    y = df[TARGET].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipe = obtener_pipeline()
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, MODEL_PATH)
    r2 = r2_score(y_test, pipe.predict(X_test))
    rmse = root_mean_squared_error(y_test, pipe.predict(X_test))
    print(f"Diabetes (Regresión Lineal Múltiple): R² test = {r2:.4f}, RMSE = {rmse:.2f}")
    return pipe


def cargar_modelo():
    try:
        return joblib.load(MODEL_PATH)
    except (FileNotFoundError, EOFError, ValueError):
        return entrenar()


if __name__ == "__main__":
    if "--entrenar" in sys.argv:
        entrenar()
    else:
        cargar_modelo()