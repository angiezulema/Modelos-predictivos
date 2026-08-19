# 🤖 Proyecto Multi-Caso IA — Caso 3: Predicción de Diabetes

Caso de estudio de **Machine Learning supervisado (regresión)** que predice la
**progresión de la diabetes un año después del diagnóstico** a partir de
10 variables basales, siguiendo una arquitectura modular y reproducible con
Scikit-learn y una aplicación web interactiva con Streamlit.

---

## 📁 Estructura del Proyecto

```
project_diabetes/
│
├── Taller 1/                      # Dataset original (diabetes.zip → diabetes.tab.txt)
├── data/
│   ├── __init__.py
│   ├── process_data.py            # Carga, EDA, outliers, VIF y estandarización
│   ├── X_train.joblib             # (generados) splits preprocesados
│   ├── X_test.joblib
│   ├── y_train.joblib
│   ├── y_test.joblib
│   ├── diabetes_scaler.joblib     # StandardScaler ajustado en train
│   ├── diabetes_clean.csv
│   ├── diabetes_correlation.csv
│   └── eda_plots/diabetes_eda.png # Histogramas y diagnóstico
├── models/
│   ├── __init__.py
│   ├── train_models.py            # OLS, Polinomial (d∈[2,3]), Ridge/Lasso + CV
│   ├── best_diabetes_model.joblib # Mejor modelo entrenado
│   ├── best_model_info.joblib     # Metadatos del mejor modelo
│   └── model_summary.csv          # Métricas R², RMSE, MAE (train/test)
├── app/
│   ├── __init__.py
│   ├── app.py                     # App Streamlit con navegación multi-caso
│   └── components/
│       ├── __init__.py
│       └── case3_ui.py            # Formulario y predicción del Caso 3
├── requirements.txt               # Dependencias del proyecto
└── README.md
```

---

## 🚀 Clonar el Repositorio

```bash
git clone <url-del-repositorio>
cd project_diabetes
```

## 🐍 Crear el Entorno Virtual (Recomendado)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

## 📦 Instalar Dependencias

```bash
pip install -r requirements.txt
```

---

## ⚙️ Ejecución del Pipeline

### 1. Preprocesamiento de datos

```bash
python data/process_data.py
```

Esto:
- Carga el dataset desde `Taller 1/` (fallback a `load_diabetes` de sklearn).
- Genera el resumen estadístico, trata outliers (IQR + Z-score).
- Calcula correlación de Pearson y VIF (multicolinealidad).
- Aplica `StandardScaler` solo a `X_train` (evita data leakage).
- Exporta splits y escalador a `data/*.joblib`.

### 2. Entrenamiento y evaluación

```bash
python models/train_models.py
```

Esto:
- Entrena Regresión Lineal Múltiple como base.
- Evalúa Regresión Polinomial d = 2 y 3.
- Si detecta sobreajuste, usa **GridSearchCV** sobre **Ridge/Lasso**.
- Valida con **Cross-Validation (k=5)**.
- Guarda las métricas en `models/model_summary.csv` y el mejor modelo en
  `models/best_diabetes_model.joblib`.

---

## 🖥️ Aplicación Web Interactiva (Streamlit)

```bash
streamlit run app/app.py
```

La app se abre en `http://localhost:8501`. En el **menú lateral** puedes
navegar entre los tres casos. El **Caso 3 — Diabetes** ofrece:

1. Formulario dinámico con las 10 variables basales (límites y valores por
   defecto obtenidos de las estadísticas reales del dataset).
2. Botón **"Predecir Progresión"**.
3. La predicción cuantitativa se presenta como tarjeta de métrica junto a un
   indicador de severidad (baja / moderada / alta).

---

## 📊 Variables del Dataset

| Variable | Descripción                               | Rango (min–max) | Promedio |
|----------|-------------------------------------------|-----------------|----------|
| AGE      | Edad (años)                               | 19 – 79          | 48.5     |
| SEX      | Sexo (1=hombre, 2=mujer)                  | 1 – 2            | 1.5      |
| BMI      | Índice de Masa Corporal (kg/m²)           | 18.0 – 42.2      | 26.4     |
| BP       | Presión arterial media (mm Hg)            | 62 – 133         | 94.6     |
| S1       | Suero S1 (tcells)                         | 97 – 301         | 189.1    |
| S2       | Suero S2 (ldl)                            | 41.6 – 242.4     | 115.4    |
| S3       | Suero S3 (hdl)                            | 22 – 99          | 49.8     |
| S4       | Suero S4 (tch/ldl ratio)                  | 2.0 – 9.1        | 4.1      |
| S5       | Suero S5 (ltg)                            | 3.3 – 6.1        | 4.6      |
| S6       | Suero S6 (glu)                            | 58 – 124         | 91.3     |
| Y        | **Objetivo:** progresión 1 año después     | 25 – 346         | 152.1    |

> Los valores exactos se derivan de las estadísticas reales del dataset de
> `Taller 1/` en tiempo de ejecución (función `get_data_stats`).

---

## 🔍 Regresión del Modelo

- **Modelo base:** `LinearRegression` sobre las 10 variables.
- **No linealidad:** `PolynomialFeatures` con grado d ∈ {2, 3}.
- **Regularización:** si el polinomio sobreajusta (R²_train − R²_test > 0.05),
  se aplica `GridSearchCV` sobre `Ridge` y `Lasso` con α ∈ {10⁻³, …, 10³}.
- **Validación:** 5-fold cross-validation.
- **Métricas:** R², RMSE y MAE en train/test, guardadas de forma tabular.

---

## 🧪 Métricas de Referencia (reproducible)

Valores obtenidos con el dataset de `Taller 1/`, split 80/20 (seed 42):

| Modelo                   | R²_test | RMSE_test | MAE_test |
|--------------------------|---------|-----------|----------|
| Linear Regression        | 0.4519  | 53.886    | 42.665   |
| Polynomial d=2           | 0.4210  | 55.384    | 43.153   |
| Polynomial d=3 (sin reg.) | −17.00  | 308.82    | 186.95   |
| **Ridge/Lasso Poly d=2** | **0.4942** | **51.769** | **41.374** |
| Ridge/Lasso Poly d=3     | 0.4178  | 55.538    | 45.842   |

> El **mejor modelo** (Ridge · PolynomialFeatures d=2 · α=100) es el seleccionado
> automáticamente y exportado a `models/best_diabetes_model.joblib`. El grado 3
> sin regularización muestra un sobreajuste severo (R²_test = −17), por lo que
> el pipeline lo regulariza automáticamente con Lasso.
>
> Los valores aparecen nuevamente tras ejecutar `python models/train_models.py`.

---

## 🛠️ Despliegue (Opcional)

### Streamlit Community Cloud

1. Sube el repositorio a GitHub.
2. Ve a [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Define el *Main file path* como `app/app.py` y el entorno en
   `requirements.txt`.

### Docker

```bash
docker build -t project_diabetes .
docker run -p 8501:8501 project_diabetes
```

---

## ✅ Buenas Prácticas Implementadas

- **Sin Data Leakage:** el escalador solo se ajusta con `X_train`.
- **Modularidad:** cada fase de la ingeniería reside en su propio módulo.
- **Reproducibilidad:** `random_state=42` en splits y validación.
- **Robustez:** fallback de datos y de `statsmodels` (VIF manual opcional).
- **PEP 8:** código documentado con dos cadenas de esquema y nombres
  descriptivos.

---

## 📚 Referencias

- Efron, B. et al. *Least Angle Regression*, Annals of Statistics (2004).
- Documentación: [Scikit-learn Diabetes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html).