"""
Módulo B: Entrenamiento y Evaluación de modelos para el "Caso 3: Predicción de Diabetes".

Responsabilidades:
    1. Entrenar un modelo base de Regresión Lineal Múltiple (OLS).
    2. Implementar Regresión Polinomial (PolynomialFeatures) con grados d in [2, 3].
    3. Si hay sobreajuste, integrar automáticamente Ridge/Lasso con búsqueda de
       hiperparámetros (GridSearchCV / RidgeCV).
    4. Validar robustez con Cross-Validation (k=5).
    5. Reportar y persistir métricas: R2, RMSE y MAE (train y test).
    6. Exportar el mejor modelo a `models/best_diabetes_model.joblib`.

Ejecución:
    python models/train_models.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --------------------------------------------------------------------------- #
# Configuración global
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_ROOT / "data"
MODEL_FOLDER = Path(__file__).resolve().parent  # models/

SEED = 42
N_CV = 5  # k-fold para cross-validation

FEATURE_COLUMNS = ["AGE", "SEX", "BMI", "BP", "S1", "S2", "S3", "S4", "S5", "S6"]


# --------------------------------------------------------------------------- #
# Carga de datos
# --------------------------------------------------------------------------- #
def load_preprocessed_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga los datasets preprocesados (salida del Módulo A).

    Supports both scaled and raw partitions depending on file availability.

    Returns:
        (X_train, X_test, y_train, y_test).
    """
    X_train = joblib.load(DATA_FOLDER / "X_train.joblib")
    X_test = joblib.load(DATA_FOLDER / "X_test.joblib")
    y_train = joblib.load(DATA_FOLDER / "y_train.joblib")
    y_test = joblib.load(DATA_FOLDER / "y_test.joblib")
    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #
def evaluate_model(
    model, X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
) -> dict:
    """Calcula y devuelve métricas R2, RMSE y MAE para train y test."""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        "model_name": type(model).__name__,
        "r2_train": float(r2_score(y_train, y_train_pred)),
        "r2_test": float(r2_score(y_test, y_test_pred)),
        "rmse_train": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
        "rmse_test": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
        "mae_train": float(mean_absolute_error(y_train, y_train_pred)),
        "mae_test": float(mean_absolute_error(y_test, y_test_pred)),
    }
    return metrics


def cv_scores(model, X_train: pd.DataFrame, y_train: pd.Series, n_cv: int = N_CV) -> dict:
    """Evalúa R2 y RMSE con Cross-Validation (k-fold)."""
    try:
        r2_cv = cross_val_score(model, X_train, y_train, cv=n_cv, scoring="r2")
        rmse_cv = -cross_val_score(model, X_train, y_train, cv=n_cv,
                                   scoring="neg_root_mean_squared_error")
        return {
            "cv_r2_mean": float(r2_cv.mean()),
            "cv_r2_std": float(r2_cv.std()),
            "cv_rmse_mean": float(rmse_cv.mean()),
            "cv_rmse_std": float(rmse_cv.std()),
        }
    except Exception:  # noqa: BLE001
        return {"cv_r2_mean": float("nan"), "cv_r2_std": float("nan"),
                "cv_rmse_mean": float("nan"), "cv_rmse_std": float("nan")}


# --------------------------------------------------------------------------- #
# Modelos
# --------------------------------------------------------------------------- #
def build_linear_model() -> Pipeline:
    """Modelo base: Regresión Lineal Múltiple (sin características polinómicas)."""
    return Pipeline([("model", LinearRegression())])


def build_polynomial_model(degree: int, regularized: bool = False) -> Pipeline:
    """Construye un pipeline de Regresión Polinomial con grado dado.

    Args:
        degree: Grado del polinomio (d in [2, 3]).
        regularized: Si True, usa Ridge/Lasso en lugar de regresión lineal simple.

    Returns:
        Pipeline (PolynomialFeatures -> regresor).
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    if regularized:
        regressor = Ridge()  # Lasso también se explora vía GridSearchCV
    else:
        regressor = LinearRegression()
    return Pipeline([("poly", poly), ("model", regressor)])


def tune_regularized_polynomial(degree: int, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Busca los mejores hiperparámetros Ridge/Lasso para un grado polinomial.

    Si el polinomio sobreajusta, esta función encuentra la regularización óptima.

    Returns:
        Pipeline con la mejor combinación encontrada por GridSearchCV.
    """
    param_grid = {
        "model": [
            Ridge(),
            Lasso(max_iter=200_000, tol=1e-4),
        ],
        "model__alpha": np.logspace(-3, 3, 7),
    }

    pipeline = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("model", Ridge()),
    ])

    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=N_CV,
        scoring="r2",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    print(f"[TUNE] Grado={degree}: mejor C={grid.best_params_} "
          f"| CV R2={grid.best_score_:.4f}")
    return grid.best_estimator_


def is_overfitting(train_r2: float, test_r2: float, tolerance: float = 0.05) -> bool:
    """Define si existe sobreajuste comparando R2 train vs test."""
    return (train_r2 - test_r2) > tolerance


# --------------------------------------------------------------------------- #
# Orquestación del entrenamiento
# --------------------------------------------------------------------------- #
def train_and_evaluate() -> dict:
    """Entrena y evalúa todos los modelos, eligiendo el mejor y guardándolo.

    Returns:
        Dict con el resumen (DataFrame de métricas), el mejor modelo (objeto
        ajustado) y su nombre.
    """
    print("=" * 60)
    print("Entrenamiento de modelos — Caso 3: Diabetes")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_preprocessed_data()
    print(f"[DATA] Train: {X_train.shape} | Test: {X_test.shape}")

    # 1) Modelo base + 2) Polinomial d in [2, 3]
    fitted = {"Linear Regression": build_linear_model()}
    for degree in (2, 3):
        fitted[f"Polynomial d={degree}"] = build_polynomial_model(degree)

    # Entrenar todos
    for name, model in fitted.items():
        model.fit(X_train, y_train)

    results = []

    def _record(name, model, degree=None):
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
        metrics.update(cv_scores(model, X_train, y_train))
        metrics.update({"name": name, "degree": degree})
        results.append(metrics)
        print(f"\n[{name}]")
        _print_metrics(metrics)

    _record("Linear Regression", fitted["Linear Regression"])
    _record("Polynomial d=2", fitted["Polynomial d=2"], degree=2)
    _record("Polynomial d=3", fitted["Polynomial d=3"], degree=3)

    # 3) Regularización automática si hay sobreajuste
    for degree in (2, 3):
        name = f"Polynomial d={degree}"
        metrics = next(r for r in results if r["name"] == name)
        if is_overfitting(metrics["r2_train"], metrics["r2_test"]):
            print(f"\n[REGULARIZE] Sobreajuste detectado en grado {degree}: "
                  f"train R2={metrics['r2_train']:.4f} vs test R2={metrics['r2_test']:.4f}")
            reg_model = tune_regularized_polynomial(degree, X_train, y_train)
            reg_name = f"Ridge/Lasso Poly d={degree}"
            fitted[reg_name] = reg_model
            _record(reg_name, reg_model, degree=degree)

    summary = pd.DataFrame(results)

    # 4) Selección del mejor modelo por R2 en test (generalización)
    best_row = summary.sort_values(
        by=["r2_test", "r2_train"], ascending=[False, False]
    ).iloc[0]
    best_model = fitted[best_row["name"]]

    print("\n" + "=" * 60)
    print("MEJOR MODELO:")
    print(best_row.to_string())
    print("=" * 60)

    # 5) Persistir artefactos
    save_artifacts(summary, best_model, best_row)

    return {"summary": summary, "model": best_model, "best_name": best_row["name"]}


def _print_metrics(metrics: dict) -> None:
    """Imprime de forma legible las métricas de un modelo."""
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<12}: {v:.4f}")


def save_artifacts(summary: pd.DataFrame, best_model, best_row) -> None:
    """Guarda resumen de métricas y el mejor modelo entrenado."""
    MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

    summary.to_csv(MODEL_FOLDER / "model_summary.csv", index=False)
    joblib.dump(best_model, MODEL_FOLDER / "best_diabetes_model.joblib")
    joblib.dump(best_row, MODEL_FOLDER / "best_model_info.joblib")

    print(f"\n[SAVE] Mejor modelo exportado a: {MODEL_FOLDER / 'best_diabetes_model.joblib'}")
    print("[SAVE] Resumen de métricas exportado a: models/model_summary.csv")


def load_best_model() -> Pipeline:
    """Carga el mejor modelo guardado para usarlo en la app de predicción."""
    path = MODEL_FOLDER / "best_diabetes_model.joblib"
    if not path.exists():
        raise FileNotFoundError(
            "No existe 'best_diabetes_model.joblib'. Ejecuta primero el pipeline de modelos."
        )
    return joblib.load(path)


if __name__ == "__main__":
    result = train_and_evaluate()

    print("\n[FIN] Entrenamiento completado.")
    print(result["summary"].round(4).to_string(index=False))