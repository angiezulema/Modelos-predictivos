import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error

BASE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE, "archive", "housing.csv")
MODEL_PATH = os.path.join(BASE, "housing_model.joblib")
POLY_PATH = os.path.join(BASE, "housing_model_poly.joblib")
POLY_DEGREE = 2

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

SUGGESTED = {
    "longitude": (-124.35, -114.31, -122.23),
    "latitude": (32.54, 41.95, 37.88),
    "housing_median_age": (1, 52, 29),
    "total_rooms": (2, 39320, 2636),
    "total_bedrooms": (1, 6445, 537),
    "population": (3, 35682, 1425),
    "households": (1, 6082, 499),
    "median_income": (0.50, 15.02, 3.87),
}

CATEGORIES = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]


def cargar_datos():
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["total_bedrooms"])
    return df


def obtener_pipeline(polinomial=False, grado=POLY_DEGREE):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",
    )
    pasos = [("preprocess", preprocessor)]
    if polinomial:
        pasos.append(("poly", PolynomialFeatures(degree=grado, include_bias=False)))
    pasos.append(("regresion", LinearRegression()))
    return Pipeline(steps=pasos)


def _descripcion(polinomial):
    return "REGRESIÓN POLINOMIAL (GRADO 2)" if polinomial else "REGRESIÓN LINEAL MÚLTIPLE"


def entrenar(polinomial=False, grado=POLY_DEGREE):
    ruta = POLY_PATH if polinomial else MODEL_PATH
    print(f"Cargando datos desde {CSV_PATH} ...")
    df = cargar_datos()
    print(f"Registros cargados: {len(df)}")

    X = df.drop(columns=[TARGET]).reset_index(drop=True)
    y = df[TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = obtener_pipeline(polinomial=polinomial, grado=grado)
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, ruta)

    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_train = root_mean_squared_error(y_train, y_pred_train)
    rmse_test = root_mean_squared_error(y_test, y_pred_test)

    print("\n" + "=" * 56)
    print(_descripcion(polinomial) + " - RESULTADOS")
    print("=" * 56)
    print(f"Train   | R² = {r2_train:.4f} | RMSE = ${rmse_train:,.2f}")
    print(f"Test    | R² = {r2_test:.4f} | RMSE = ${rmse_test:,.2f}")
    print("=" * 56)

    reg = pipeline.named_steps["regresion"]
    print(f"Intercepto (b0): ${reg.intercept_:,.2f}")
    if not polinomial:
        print("\nCoeficientes por variable:")
        cat_names = pipeline.named_steps["preprocess"].named_transformers_[
            "cat"
        ].get_feature_names_out()
        num_names = [c for c in X.columns if c not in CATEGORICAL_FEATURES]
        all_names = list(cat_names) + num_names
        for nombre, coef in zip(all_names, reg.coef_):
            print(f"  {nombre:20s} → {coef:+,.2f}")
    else:
        n_coef = len(reg.coef_)
        print(f"Características polinomiales (grado {grado}): {n_coef}")

    print(f"\nModelo guardado en {ruta}")
    return pipeline


def cargar_modelo(polinomial=False):
    ruta = POLY_PATH if polinomial else MODEL_PATH
    try:
        return joblib.load(ruta)
    except (FileNotFoundError, EOFError, ValueError):
        print(f"No se encontró el modelo {ruta}. Entrenando de nuevo...")
        return entrenar(polinomial=polinomial)


def leer_float(mensaje, minimo, maximo, sugerido):
    while True:
        entrada = input(f"{mensaje} (sugerido {sugerido}): ").strip()
        if entrada.lower() == "exit":
            print("Hasta luego.")
            sys.exit(0)
        try:
            entrada = entrada or str(sugerido)
            valor = float(entrada)
            if minimo <= valor <= maximo:
                return valor
            print(f"  ❌ Valor fuera de rango [{minimo} - {maximo}]. Intenta de nuevo.")
        except ValueError:
            print("  ❌ Entrada no válida. Debe ser un número.")


def leer_categoria():
    print("\nOpciones de ocean_proximity:")
    for i, c in enumerate(CATEGORIES, 1):
        print(f"  {i}. {c}")
    while True:
        entrada = input("Selecciona un número (1-5): ").strip()
        if not entrada:
            return "INLAND"
        try:
            idx = int(entrada)
            if 1 <= idx <= len(CATEGORIES):
                return CATEGORIES[idx - 1]
        except ValueError:
            pass
        print("  ❌ Selección no válida.")


def predecir(pipeline):
    print("\n" + "=" * 56)
    print("INGRESA LAS CARACTERÍSTICAS DE LA CASA")
    print("=" * 56)

    datos = {}
    for col in NUMERIC_FEATURES:
        minimo, maximo, sugerido = SUGGESTED[col]
        datos[col] = leer_float(f"  {col}", minimo, maximo, sugerido)

    datos["ocean_proximity"] = leer_categoria()

    fila = pd.DataFrame([datos], columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    prediccion = float(pipeline.predict(fila)[0])

    print("\n" + "=" * 56)
    print(f"🏠 PRECIO MEDIANO PREDICHO: ${prediccion:,.2f}")
    print("=" * 56)


def main():
    polinomial = "--polinomial" in sys.argv
    if "--entrenar" in sys.argv:
        entrenar(polinomial=polinomial)
        print("\nEntrenamiento completado.")
        return

    pipeline = cargar_modelo(polinomial=polinomial)

    print("\n💡 Escribe 'exit' en cualquier campo numérico para salir.")
    while True:
        predecir(pipeline)
        print("\n(Enter para predecir otra casa, 'exit' para salir)")
        if input().strip().lower() == "exit":
            break
    print("Hasta luego.")


if __name__ == "__main__":
    main()