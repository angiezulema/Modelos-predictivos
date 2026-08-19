# -*- coding: utf-8 -*-
"""
train_model.py
==============
Caso de estudio: Diabetes (Scikit-learn) con Regresión Lineal Múltiple.

Este script realiza todo el pipeline de Machine Learning:

    1. Carga y exploración del dataset (diabetes.tab.txt)
    2. Preprocesamiento (nulos, outliers, correlación, VIF)
    3. Visualizaciones del EDA (histogramas, boxplots, pairplot, heatmap)
    4. Estandarización (StandardScaler) y división 80/20 (random_state=42)
    5. Entrenamiento de Regresión Lineal Múltiple
    6. Evaluación del modelo (R², MSE, RMSE, MAE) y gráficos
    7. Guardado de artefactos (modelo, scaler, rangos) para la app Flask

Ejecutar una sola vez:
    python train_model.py
"""

import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # sin ventana gráfica
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Configuración global
# ----------------------------------------------------------------------------
DATA_FILE = "diabetes.tab.txt"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

FEATURE_COLS = ["AGE", "SEX", "BMI", "BP", "S1", "S2", "S3", "S4", "S5", "S6"]
TARGET_COL = "Y"

FEATURE_LABELS = {
    "AGE": "Edad",
    "SEX": "Sexo",
    "BMI": "IMC",
    "BP": "Presión arterial",
    "S1": "S1",
    "S2": "S2",
    "S3": "S3",
    "S4": "S4",
    "S5": "S5",
    "S6": "S6",
}

RANDOM_STATE = 42

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 100


def ensure_dirs():
    """Crea las carpetas de salida si no existen."""
    for d in (MODEL_DIR, OUTPUT_DIR):
        os.makedirs(d, exist_ok=True)


# ----------------------------------------------------------------------------
# 1. Carga y exploración del dataset
# ----------------------------------------------------------------------------
def load_data():
    """Carga el dataset desde el archivo tabulado diabetes.tab.txt."""
    df = pd.read_csv(DATA_FILE, sep="\t")
    return df


def explore_data(df):
    """
    Muestra información general del dataset:
    descripción, dimensiones, primeras filas y estadísticas descriptivas.
    """
    print("=" * 70)
    print("INFORMACIÓN DEL DATASET")
    print("=" * 70)
    print(f"Dimensiones (filas, columnas): {df.shape}")
    print(f"\nVariables independientes (X): {', '.join(FEATURE_COLS)}")
    print(f"Variable objetivo (Y): {TARGET_COL}")
    print("\nPrimeras 5 filas:")
    print(df.head().to_string(index=False))
    print("\nEstadísticas descriptivas:")
    print(df.describe().round(3).to_string())


# ----------------------------------------------------------------------------
# 2. Preprocesamiento
# ----------------------------------------------------------------------------
def check_missing_values(df):
    """Detecta y reporta valores nulos en el dataset."""
    nulls = df.isnull().sum()
    print("=" * 70)
    print("VALORES NULOS")
    print("=" * 70)
    if nulls.sum() == 0:
        print("No se detectaron valores nulos en ninguna columna.")
    else:
        print(nulls[nulls > 0])
        # Si existieran, se imputan con la mediana para robustez
        for col in nulls[nulls > 0].index:
            df[col].fillna(df[col].median(), inplace=True)
        print("Valores nulos imputados con la mediana.")
    return nulls.sum()


def detect_outliers_iqr(df):
    """
    Detección de outliers con el método IQR (Rango Intercuartílico).
    Se reportan y exportan a CSV, pero NO se eliminan: para regresión
    lineal básica se conservan para no perder información del caso de estudio.
    """
    print("=" * 70)
    print("DETECCIÓN DE OUTLIERS (método IQR)")
    print("=" * 70)
    summary = {}
    all_outliers = {}
    for col in FEATURE_COLS:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        summary[col] = int(mask.sum())
        if mask.sum() > 0:
            all_outliers[col] = df.loc[mask, col].tolist()
        print(
            f"  {col:>4} | límites [{lower:8.2f}, {upper:8.2f}] | "
            f"outliers: {mask.sum()}"
        )

    pd.DataFrame(
        [{"Variable": k, "Outliers": v} for k, v in summary.items()]
    ).to_csv(os.path.join(OUTPUT_DIR, "outliers.csv"), index=False)

    total = sum(summary.values())
    print(f"\nTotal de outliers detectados: {total}")
    print("Decisión: se conservan (no se eliminan) para el modelo.")
    return summary, all_outliers


# ----------------------------------------------------------------------------
# 3. Análisis de correlación y multicolinealidad
# ----------------------------------------------------------------------------
def correlation_heatmap(df):
    """Matriz de correlación de Pearson + heatmap."""
    corr = df[FEATURE_COLS + [TARGET_COL]].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Matriz de Correlación - Dataset de Diabetes", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "heatmap_correlacion.png"))
    plt.close(fig)

    print("=" * 70)
    print("CORRELACIÓN CON LA VARIABLE OBJETIVO (Y)")
    print("=" * 70)
    corr_y = corr[TARGET_COL].drop(TARGET_COL).sort_values(ascending=False)
    print(corr_y.round(3).to_string())
    return corr


def compute_vif(df):
    """
    Cálculo del Factor de Inflación de la Varianza (VIF) sin statsmodels.
    VIF = 1 / (1 - R²) donde R² sale de regresar cada variable contra las demás.
    """
    from sklearn.linear_model import LinearRegression

    print("=" * 70)
    print("ANÁLISIS DE MULTICOLINEALIDAD (VIF)")
    print("=" * 70)
    X = df[FEATURE_COLS].values
    vif = {}
    for i, col in enumerate(FEATURE_COLS):
        y = X[:, i]
        X_others = np.delete(X, i, axis=1)
        model = LinearRegression().fit(X_others, y)
        r2 = model.score(X_others, y)
        vif[col] = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")

    vif_df = pd.DataFrame(
        {"Variable": list(vif.keys()), "VIF": [round(v, 3) for v in vif.values()]}
    )
    vif_df.to_csv(os.path.join(OUTPUT_DIR, "vif.csv"), index=False)
    print(vif_df.to_string(index=False))
    print(
        "\nInterpretación: VIF > 10 indica multicolinealidad alta. "
        "S1 y S2 suelen tenerla en este dataset."
    )
    return vif_df


# ----------------------------------------------------------------------------
# 4. Visualizaciones del EDA
# ----------------------------------------------------------------------------
def plot_histograms(df):
    """Histogramas de cada variable independiente."""
    n = len(FEATURE_COLS)
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_COLS):
        sns.histplot(df[col], kde=True, ax=axes[i], color="steelblue", edgecolor="white")
        axes[i].set_title(FEATURE_LABELS[col])
    # Ocupamos los 2 huecos vacíos (12 subplots - 10 variables)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Histogramas de las Variables Independientes", fontsize=15, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "histogramas.png"))
    plt.close(fig)


def plot_boxplots(df):
    """Boxplots por variable para detectar outliers visualmente."""
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_COLS):
        sns.boxplot(y=df[col], ax=axes[i], color="lightcoral")
        axes[i].set_title(FEATURE_LABELS[col])
    for j in range(len(FEATURE_COLS), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Boxplots de las Variables Independientes", fontsize=15, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "boxplots.png"))
    plt.close(fig)


def plot_pairplot(df):
    """Scatter matrix (pairplot) de las variables independientes."""
    g = sns.pairplot(
        df[FEATURE_COLS],
        diag_kind="kde",
        corner=True,
        plot_kws={"s": 12, "alpha": 0.5, "color": "steelblue"},
    )
    g.figure.suptitle("Scatter Matrix de las Variables Independientes", y=1.02, fontsize=15)
    g.savefig(os.path.join(OUTPUT_DIR, "pairplot.png"))
    plt.close(g.figure)


def plot_target_distribution(df):
    """Distribución de la variable objetivo Y."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(df[TARGET_COL], kde=True, ax=axes[0], color="mediumseagreen", edgecolor="white")
    axes[0].set_title("Histograma de Y")
    sns.boxplot(y=df[TARGET_COL], ax=axes[1], color="mediumseagreen")
    axes[1].set_title("Boxplot de Y")
    fig.suptitle("Distribución de la Variable Objetivo (Progresión de Diabetes)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "distribucion_target.png"))
    plt.close(fig)


# ----------------------------------------------------------------------------
# 5. Entrenamiento y evaluación
# ----------------------------------------------------------------------------
def train_and_evaluate(df):
    """
    Estandariza, divide 80/20 (random_state=42), entrena la regresión lineal
    múltiple, evalúa con R²/MSE/RMSE/MAE y genera los gráficos del modelo.
    """
    print("=" * 70)
    print("ENTRENAMIENTO DEL MODELO")
    print("=" * 70)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # Estandarización de las variables independientes
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # División entrenamiento / prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=RANDOM_STATE
    )
    print(f"Entrenamiento: {X_train.shape[0]} muestras | Prueba: {X_test.shape[0]} muestras")

    # Modelo de Regresión Lineal Múltiple
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predicciones
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Métricas en el conjunto de prueba
    r2 = r2_score(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred_test)

    metrics = {
        "r2": r2,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2_entrenamiento": r2_score(y_train, y_pred_train),
    }

    print("\n--- MÉTRICAS DE EVALUACIÓN (conjunto de prueba) ---")
    print(f"  R²   = {r2:.4f}")
    print(f"  MSE  = {mse:.2f}")
    print(f"  RMSE = {rmse:.2f}")
    print(f"  MAE  = {mae:.2f}")
    print(f"  R² (entrenamiento) = {metrics['r2_entrenamiento']:.4f}")

    with open(os.path.join(OUTPUT_DIR, "metricas.txt"), "w", encoding="utf-8") as f:
        f.write("MÉTRICAS DE EVALUACIÓN - REGRESIÓN LINEAL MÚLTIPLE\n")
        f.write("=" * 55 + "\n")
        f.write(f"R²  (coef. determinación): {r2:.4f}\n")
        f.write(f"MSE (error cuadrático medio): {mse:.2f}\n")
        f.write(f"RMSE (raíz del error cuadrático medio): {rmse:.2f}\n")
        f.write(f"MAE (error absoluto medio): {mae:.2f}\n")
        f.write(f"R² entrenamiento: {metrics['r2_entrenamiento']:.4f}\n")

    # --- Coeficientes del modelo (importancia de cada variable) ---
    coeffs = pd.DataFrame(
        {"Variable": FEATURE_COLS, "Coeficiente": model.coef_}
    ).sort_values("Coeficiente", ascending=False)

    print("\n--- COEFICIENTES DEL MODELO ---")
    for _, row in coeffs.iterrows():
        print(f"  {row['Variable']:>4}: {row['Coeficiente']:+.4f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["mediumseagreen" if c > 0 else "lightcoral" for c in coeffs["Coeficiente"]]
    bars = ax.bar(coeffs["Variable"], coeffs["Coeficiente"], color=colors)
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_title("Importancia de las Variables (Coeficientes de la Regresión)")
    ax.set_xlabel("Variable")
    ax.set_ylabel("Coeficiente")
    for bar, v in zip(bars, coeffs["Coeficiente"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v:+.2f}",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "importancia_variables.png"))
    plt.close(fig)

    # --- Gráfico real vs predicho ---
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred_test, alpha=0.6, color="steelblue", edgecolor="white")
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Línea ideal")
    ax.set_xlabel("Valores Reales (Y)")
    ax.set_ylabel("Valores Predichos (Ŷ)")
    ax.set_title(f"Valores Reales vs Predichos  (R² = {r2:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "real_vs_predicho.png"))
    plt.close(fig)

    # --- Gráfico de residuos ---
    residuals = y_test - y_pred_test
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y_pred_test, residuals, alpha=0.6, color="steelblue", edgecolor="white")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Valores Predichos (Ŷ)")
    axes[0].set_ylabel("Residuos (Y - Ŷ)")
    axes[0].set_title("Residuos vs Predichos")
    sns.histplot(residuals, kde=True, ax=axes[1], color="mediumseagreen", edgecolor="white")
    axes[1].set_xlabel("Residuos")
    axes[1].set_title("Distribución de los Residuos")
    fig.suptitle("Análisis de Residuos", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "residuos.png"))
    plt.close(fig)

    return model, scaler, metrics, coeffs


# ----------------------------------------------------------------------------
# 6. Guardado de artefactos
# ----------------------------------------------------------------------------
def save_artifacts(df, model, scaler, metrics):
    """Guarda modelo, scaler y rangos de variables para la app Flask."""
    model_path = os.path.join(MODEL_DIR, "modelo_regresion.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print("=" * 70)
    print("ARTEFACTOS GUARDADOS")
    print("=" * 70)
    print(f"  Modelo : {model_path}")
    print(f"  Scaler : {scaler_path}")

    # Rangos reales por variable (para los sliders de la interfaz)
    ranges = {}
    for col in FEATURE_COLS:
        ranges[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "default": float(round(df[col].median(), 2)),
            "step": 0.01,
        }
    ranges["AGE"]["step"] = 1.0
    ranges["SEX"]["step"] = 1.0

    ranges_path = os.path.join(MODEL_DIR, "feature_ranges.json")
    with open(ranges_path, "w", encoding="utf-8") as f:
        json.dump(ranges, f, ensure_ascii=False, indent=2)
    print(f"  Rangos : {ranges_path}")

    # Métricas también en JSON para la interfaz
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    print(f"  Métricas: {os.path.join(MODEL_DIR, 'metrics.json')}")


# ----------------------------------------------------------------------------
# Ejecución principal
# ----------------------------------------------------------------------------
def main():
    ensure_dirs()
    df = load_data()
    explore_data(df)
    check_missing_values(df)
    detect_outliers_iqr(df)
    corr = correlation_heatmap(df)
    compute_vif(df)

    plot_histograms(df)
    plot_boxplots(df)
    plot_pairplot(df)
    plot_target_distribution(df)

    model, scaler, metrics, coeffs = train_and_evaluate(df)
    save_artifacts(df, model, scaler, metrics)

    print("\nPipeline completado correctamente.")
    print("Archivos de salida:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  outputs/{f}")


if __name__ == "__main__":
    main()