"""
Módulo A: Análisis y Preprocesamiento de datos para el "Caso 3: Predicción de Diabetes".

Responsabilidades:
    1. Carga del dataset desde la carpeta local "Taller 1/" (fallback a sklearn).
    2. Análisis exploratorio (EDA) y resumen estadístico.
    3. Detección y tratamiento de outliers (método IQR + Z-score).
    4. Matriz de correlación de Pearson y cálculo de VIF (multicolinealidad).
    5. Estandarización con StandardScaler (fit solo sobre X_train para evitar data leakage).
    6. Exportación de datasets procesados y del escalador a objetos `.joblib`.

Ejecución:
    python data/process_data.py
"""

from __future__ import annotations

import warnings
import zipfile
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:  # noqa: C901  (imports opcionales para VIF)
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_STATSMODELS = True
except ImportError:  # pragma: no cover
    HAS_STATSMODELS = False

# --------------------------------------------------------------------------- #
# Configuración global
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_ROOT / "Taller 1"
if not DATA_FOLDER.exists():
    DATA_FOLDER = PROJECT_ROOT / "Taller_1"  # variante sin espacio
OUTPUT_FOLDER = Path(__file__).resolve().parent  # data/ (exporta aquí los .joblib)

FEATURE_COLUMNS = ["AGE", "SEX", "BMI", "BP", "S1", "S2", "S3", "S4", "S5", "S6"]
TARGET_COLUMN = "Y"

Z_THRESHOLD = 3.0  # Umbral Z-score para considerar un punto como outlier
SEED = 42
TEST_SIZE = 0.2

def load_diabetes_dataset() -> pd.DataFrame:
    """Carga el dataset de diabetes.

    Busca el archivo de datos dentro de "Taller_1/". Soporta archivos planos
    (.txt/.csv/.tsv) y archivos comprimidos (.zip/csv.gz). Si la carga falla,
    usa el fallback de sklearn.datasets.load_diabetes(as_frame=True).

    Returns:
        pd.DataFrame con las 10 variables basales y la variable objetivo `Y`.
    """
    if not DATA_FOLDER.exists():
        warnings.warn(f"Carpeta de datos no encontrada: {DATA_FOLDER}")
        return _load_from_sklearn()

    # Candidatos: cualquier archivo cuyo nombre contenga "diabetes".
    candidates = sorted(
        p for p in DATA_FOLDER.iterdir()
        if p.name.lower().count("diabetes") > 0
    )

    # Prioridad: archivos planos antes que comprimidos.
    flat = [p for p in candidates if p.suffix.lower() in {".txt", ".csv", ".tsv", ".tab", ".data"}]
    zipped = [p for p in candidates if p.suffix.lower() in {".zip", ".gz"}]

    for source in [*flat, *zipped]:
        try:
            df = _read_candidate(source)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Fallo al leer {source.name}: {exc}")
            continue

        if _looks_like_diabetes(df, source):
            print(f"[OK] Datos cargados desde: {source}")
            return df

    warnings.warn("No se encontró un dataset de diabetes válido en Taller_1/.")
    return _load_from_sklearn()


def _read_candidate(source: Path) -> pd.DataFrame:
    """Lee un archivo plano o comprimido y devuelve un DataFrame."""
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            inner = zf.namelist()[0]
            with zf.open(inner) as fh:
                sep = _guess_separator(fh.read(2048).decode(errors="ignore"))
                fh.seek(0)
                return pd.read_csv(fh, sep=sep)
    if source.suffix.lower() == ".gz":
        return pd.read_csv(source, sep="\t", compression="gzip")
    sep = _guess_separator(source.read_text(errors="ignore")[:2048])
    return pd.read_csv(source, sep=sep)


def _guess_separator(sample: str) -> str:
    """Infiere el separador de un archivo de texto (tab, coma o punto y coma)."""
    for sep in ("\t", ",", ";"):
        first_line = sample.splitlines()[0]
        if sep in first_line:
            return sep
    return "\t"


def _looks_like_diabetes(df: pd.DataFrame, source: Path) -> bool:
    """Valida que el DataFrame tenga la firma del dataset de diabetes."""
    expected = {"age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"}
    cols = {str(c).lower() for c in df.columns}
    return expected.issubset(cols)


def _load_from_sklearn() -> pd.DataFrame:
    """Fallback: carga el dataset de diabetes directamente de scikit-learn."""
    from sklearn.datasets import load_diabetes

    data = load_diabetes(as_frame=True)
    frame = data.data.copy()
    frame.columns = [c.upper() for c in frame.columns]
    frame["Y"] = data.target
    print("[OK] Datos cargados desde sklearn.datasets.load_diabetes (fallback).")
    return frame


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas a mayúsculas sin espacios."""
    df = df.copy()
    df.columns = [str(c).strip().upper().replace(" ", "_") for c in df.columns]
    if "AGE" in df.columns and "Y" in df.columns:
        pass
    return df


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve el resumen estadístico del DataFrame (describe + nulos)."""
    return df[FEATURE_COLUMNS + [TARGET_COLUMN]].describe().T


def detect_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """Detecta outliers por el método del Rango Intercuartílico (IQR).

    Returns:
        DataFrame booleano (True = outlier) con las mismas columnas.
    """
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (df < lower) | (df > upper)
    return mask


def detect_outliers_zscore(df: pd.DataFrame, threshold: float = Z_THRESHOLD) -> pd.DataFrame:
    """Detecta outliers mediante Z-score (|z| > threshold)."""
    mean = df.mean()
    std = df.std(ddof=0)
    z = (df - mean) / std.replace(0, np.nan)
    mask = z.abs() > threshold
    return mask.fillna(False)


def treat_outliers(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Trata los outliers combinando IQR y Z-score.

    Estrategia:
        - Se marcan puntos que sean outliers por IQR Y por Z-score.
        - Estos valores se recortan (winsorización) a los límites IQR
          [Q1 - 1.5*IQR, Q3 + 1.5*IQR], preservando el tamaño de la muestra.

    Returns:
        DataFrame con los outliers tratados (sin filas eliminadas).
    """
    iqr_mask = detect_outliers_iqr(df[features])
    z_mask = detect_outliers_zscore(df[features])

    combined = iqr_mask & z_mask
    n_outliers = int(combined.sum().sum())
    print(f"[EDAI] Outliers detectados por IQR & Z-score: {n_outliers} celdas.")

    q1 = df[features].quantile(0.25)
    q3 = df[features].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    df_clean = df.copy()
    for col in features:
        mask_col = combined[col]
        df_clean.loc[mask_col, col] = df_clean.loc[mask_col, col].clip(
            lower[col], upper[col]
        )
    return df_clean


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula la matriz de correlación de Pearson de las variables basales."""
    return df[FEATURE_COLUMNS].corr(method="pearson")


def compute_vif(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Calcula el Variance Inflation Factor (VIF) por variable.

    Un VIF > 10 indica multicolinealidad severa; 5-10 indica moderada.

    Returns:
        DataFrame con columnas ['feature', 'vif'].
    """
    vif_records = []
    if HAS_STATSMODELS:
        for i, col in enumerate(features):
            vif = variance_inflation_factor(df[features].values, i)
            vif_records.append({"feature": col, "vif": round(float(vif), 3)})
    else:
        # Fallback: implementación manual con R² de regresión de cada variable.
        X = df[features].values
        n = X.shape[0]
        for i, col in enumerate(features):
            y = X[:, i]
            others = np.delete(X, i, axis=1)
            design = np.column_stack([np.ones(n), others])
            beta, *_ = np.linalg.lstsq(design, y, rcond=None)
            pred = design @ beta
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            vif = 1.0 / (1.0 - r2) if r2 < 0.9999 else float("inf")
            vif_records.append({"feature": col, "vif": round(vif, 3)})
    return pd.DataFrame(vif_records)


def plot_eda(df: pd.DataFrame, output_dir: Path | None = None) -> Path | None:
    """Genera y guarda una figura de diagnóstico (histogramas + correlaciones).

    Returns:
        Ruta del archivo PNG generado, o None si falla matplotlib.
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "eda_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 3, figsize=(14, 12))
    axes = axes.flatten()
    for ax, col in zip(axes, FEATURE_COLUMNS + [TARGET_COLUMN]):
        ax.hist(df[col].dropna(), bins=30, edgecolor="black", alpha=0.7)
        ax.set_title(col)
    axes[-1].axis("off")

    plt.tight_layout()
    out_path = output_dir / "diabetes_eda.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def split_and_scale(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide en train/test (80/20) y estandariza usando SOLO el train.

    Importante: el StandardScaler se ajusta únicamente sobre X_train para no
    causar data leakage con X_test.

    Returns:
        Tupla (X_train, X_test, y_train, y_test, scaler) con escalador ajustado.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = df[FEATURE_COLUMNS].astype(float)
    y = df[TARGET_COLUMN].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=FEATURE_COLUMNS,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=FEATURE_COLUMNS,
        index=X_test.index,
    )
    return X_train_scaled, X_test_scaled, y_train.copy(), y_test.copy(), scaler


def save_artifacts(
    X_train, X_test, y_train, y_test, scaler, df: pd.DataFrame, report: dict
) -> None:
    """Persiste los datasets procesados, el escalador y el reporte en `data/`."""
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    joblib.dump(X_train, OUTPUT_FOLDER / "X_train.joblib")
    joblib.dump(X_test, OUTPUT_FOLDER / "X_test.joblib")
    joblib.dump(y_train, OUTPUT_FOLDER / "y_train.joblib")
    joblib.dump(y_test, OUTPUT_FOLDER / "y_test.joblib")
    joblib.dump(scaler, OUTPUT_FOLDER / "diabetes_scaler.joblib")

    df.to_csv(OUTPUT_FOLDER / "diabetes_clean.csv", index=False)
    pd.DataFrame([report]).to_csv(OUTPUT_FOLDER / "data_report.csv", index=False)

    print("[SAVE] Artefactos guardados en:", OUTPUT_FOLDER)


def run_pipeline() -> dict:
    """Orquesta el pipeline completo de preprocesamiento.

    Returns:
        Un diccionario con los artefactos clave (df limpio, splits, scaler y reporte).
    """
    print("=" * 60)
    print("Pipeline de preprocesamiento — Caso 3: Diabetes")
    print("=" * 60)

    # 1) Carga
    df_raw = load_diabetes_dataset()
    df_raw = clean_column_names(df_raw)

    # 2) EDA
    features = [c for c in FEATURE_COLUMNS if c in df_raw.columns]
    target = TARGET_COLUMN if TARGET_COLUMN in df_raw.columns else None
    summary = summary_statistics(df_raw)
    print("\n[EDA] Resumen estadístico:")
    print(summary.round(3).to_string())

    if target is None:
        raise ValueError("El dataset no contiene la columna objetivo 'Y'.")

    # 3) Tratamiento de outliers
    df_clean = treat_outliers(df_raw, features)

    # 4) Correlación y VIF
    corr = correlation_matrix(df_clean)
    corr.to_csv(OUTPUT_FOLDER / "diabetes_correlation.csv", index=True)
    print("\n[EDA] Correlación de Pearson (abs > 0.5):")
    mask = corr.abs() > 0.5
    strong = corr.where(mask).stack()
    print(strong.round(3).to_string() if len(strong) else "  No hay correlaciones fuertes entre variables basales.")

    vif = compute_vif(df_clean, features)
    print("\n[EDA] VIF (multicolinealidad):")
    print(vif.to_string(index=False))

    # 5) Split y estandarización
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df_clean)
    print(f"\n[SPLIT] Train: {X_train.shape[0]} filas | Test: {X_test.shape[0]} filas")

    # 6) Exportación
    report = {
        "source_file": "Taller_1/diabetes",
        "n_rows": int(df_clean.shape[0]),
        "n_features": len(features),
        "test_size": TEST_SIZE,
        "seed": SEED,
        "outliers_treated": True,
    }
    save_artifacts(X_train, X_test, y_train, y_test, scaler, df_clean, report)

    # Figura EDA opcional (solo si matplotlib funciona)
    try:
        plot_path = plot_eda(df_clean)
        print("[EDA] Figura guardada en:", plot_path)
    except Exception as exc:  # noqa: BLE001
        print("[WARN] No se pudo generar la figura EDA:", exc)

    return {
        "df_clean": df_clean,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
    }


def get_data_stats(df: pd.DataFrame) -> dict:
    """Devuelve estadísticas reales por variable para configurar la UI.

    Returns:
        Dict con 'min', 'max', 'mean' y 'std' por cada variable basal.
    """
    stats = {}
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            continue
        series = df[col].astype(float)
        stats[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "std": float(series.std(ddof=1)),
        }
    return stats


if __name__ == "__main__":
    artifacts = run_pipeline()

    # Demo opcional: estadísticas para la UI
    stats = get_data_stats(artifacts["df_clean"])
    print("\n[UI] Estadísticas por variable (para límites de la interfaz):")
    for col, s in stats.items():
        print(f"  {col}: min={s['min']:.2f} max={s['max']:.2f} mean={s['mean']:.2f} std={s['std']:.2f}")