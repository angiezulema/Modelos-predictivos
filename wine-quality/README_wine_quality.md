# 🍷 Wine Quality — Regresión Lineal Múltiple y Polinomial

> **Caso 2 de 3** · Repositorio [Modelos-predictivos](https://github.com/angiezulema/Modelos-predictivos) · `wine-quality/wine_predictor.py` + `wine-quality/wine_app.py` · scikit-learn + Gradio

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-OLS-F7931E?logo=scikitlearn&logoColor=white)
![Gradio](https://img.shields.io/badge/Interfaz-Gradio-orange)
![Dataset](https://img.shields.io/badge/Dataset-UCI%20Vino%20Tinto-7c2d12)
![Modelo](https://img.shields.io/badge/R2_score-m%C3%A9trica%20clave-success)

---

## 📌 Introducción: ¿qué hace la app y cuál es su finalidad?

La **app de Wine Quality** es una herramienta interactiva construida con **Gradio** cuyo objetivo es
**predecir la calidad del vino tinto en una escala continua (0–10)** a partir de sus propiedades
fisicoquímicas medidas en laboratorio. El usuario mueve los deslizadores con las características del vino,
elige entre dos modelos de regresión y obtiene al instante la nota estimada.

**Finalidad doble:**

- ✦ Demostrar el flujo completo de ML: *datos → preprocesamiento → entrenamiento → evaluación → serialización → inferencia*.
- ✦ Comparar en vivo dos familias de regresión sobre el mismo problema: **Lineal Múltiple** vs **Polinomial grado 2**.

Es el caso más simple de los tres: **todas las variables son numéricas**, sin categóricas (a diferencia
de housing) ni valores codificados especiales, por lo que el pipeline se centra en la **estandarización**.

### 🖥️ Cómo funciona la interfaz

```
Radio: elegir modelo  →  11 Sliders (propiedades fisicoquímicas)  →  Botón "Predecir"
      →  pipeline.predict()  →  Resultado en Textbox
```

```python
def predecir(modelo, *valores):
    datos = {c: v for c, v in zip(FEATURES, valores)}
    fila = pd.DataFrame([datos], columns=FEATURES)
    pipe = pipeline_lineal if modelo == LINEAL else pipeline_polinomial
    pred = float(pipe.predict(fila)[0])
    return f"Calidad del vino predicha ({modelo}): {pred:.2f}"
```

💡 Misma mecánica que la pestaña de Diabetes: si los archivos `.joblib` no existen,
`cargar_modelo()` reentrena automáticamente antes de servir la interfaz.

---

## 🗺️ ¿Cómo se planeó hacer?

1. Obtener el dataset público UCI **Vinho Verde tinto** (`winequality-red.csv`, 1.599 muestras, separador `;`).
2. Definir las 11 propiedades fisicoquímicas como X y la nota `quality` como target.
3. Construir **dos pipelines paralelos** (lineal y polinomial g2), ambos con `StandardScaler`.
4. Entrenar con split **80/20** y semilla fija `random_state=42` (reproducibilidad).
5. Evaluar con **R² y RMSE** en test para decidir qué modelo generaliza mejor.
6. Persistir con `joblib` → `wine_model.joblib` / `wine_model_poly.joblib`.
7. Exponer en Gradio e integrarlo al app raíz `app.py` (pestaña 🍷).

---

## 📋 Variables que usa el modelo

| Variable | Descripción | Tipo |
|---|---|---|
| `fixed acidity` | Acidez fija (ácido tartárico, g/L) | Numérica |
| `volatile acidity` | Acidez volátil (ácido acético, g/L) | Numérica |
| `citric acid` | Ácido cítrico (g/L) | Numérica |
| `residual sugar` | Azúcar residual (g/L) | Numérica |
| `chlorides` | Cloruros / sal (g/L) | Numérica |
| `free sulfur dioxide` | SO₂ libre (mg/L) | Numérica |
| `total sulfur dioxide` | SO₂ total (mg/L) | Numérica |
| `density` | Densidad (g/cm³) | Numérica |
| `pH` | pH del vino | Numérica |
| `sulphates` | Sulfatos (g/L) | Numérica |
| `alcohol` | Grado de alcohol (% vol) | Numérica |
| **`quality`** | 🎯 **TARGET** — nota de calidad sensorial (0–10, mediana por panel de catadores) | Entero |

```python
TARGET = "quality"
# En este caso X = todas las columnas menos el target:
X = df.drop(columns=[TARGET])
```

---

## 📈 Tipos de regresión usados y sus diferencias

Ambas son **regresión lineal por mínimos cuadrados (OLS)**: cambia *cómo se transforman las x antes de ajustar*.
La polinomial sigue siendo **lineal en sus coeficientes β**, pero no lineal en las variables originales.

| Criterio | ➖ Lineal Múltiple | ∫ Polinomial (g2) |
|---|---|---|
| Forma geométrica | Hiperplano recto en 11D | Superficie curva |
| Fórmula | ŷ = β₀ + Σβᵢxᵢ′ | ŷ = β₀ + Σβᵢzᵢ + Σβᵢᵢzᵢ² + Σβᵢⱼzᵢzⱼ |
| Parámetros estimados | 12 (β₀ + 11) | 78 (β₀ + 77) |
| Features al OLS | 11 | 77 (11 + 11 cuadrados + 55 interacciones) |
| Interpretabilidad | Alta (ej: ↑acidez volátil ⇒ ↓calidad) | Baja (coeficientes cruzados) |
| Riesgo principal | Subajuste | Sobreajuste (memoriza el train) |
| Archivo generado | `wine_model.joblib` | `wine_model_poly.joblib` |

---

## ➖ Regresión Lineal Múltiple

$$\hat{y} = \beta_0 + \beta_1 \cdot acidez\_fija' + \beta_2 \cdot acidez\_volatil' + \cdots + \beta_{10} \cdot sulfatos' + \beta_{11} \cdot alcohol'$$

donde $x' = \dfrac{x-\mu}{\sigma}$ (estandarización por `StandardScaler`).

**¿Qué librería/fórmula lo hace posible?** `LinearRegression` resuelve los **mínimos cuadrados ordinarios**:

$$\hat{\beta} = (X^TX)^{-1}X^Ty \quad \text{(implementada numéricamente con SVD)}$$

```python
# Pipeline LINEAL: escalar → OLS
def obtener_pipeline(polinomial=False, grado=POLY_DEGREE):
    pasos = [("scaler", StandardScaler())]           # ← x' = (x − μ)/σ
    pasos.append(("regresion", LinearRegression()))  # ← β̂ = (XᵀX)⁻¹Xᵀy
    return Pipeline(steps=pasos)
```

## ∫ Regresión Polinomial (grado 2)

$$\hat{y} = \beta_0 + \sum_{i} \beta_i z_i + \sum_{i} \beta_{ii} z_i^2 + \sum_{i<j} \beta_{ij}\, z_i z_j$$

Ejemplo con 2 variables: $\hat{y} = \beta_0 + \beta_1 alcohol' + \beta_2 sulphates' + \beta_3 alcohol'^2 + \beta_4 sulphates'^2 + \beta_5\, alcohol'\, sulphates'$

**El truco:** no existe una "clase polinomial". `PolynomialFeatures` solo **inventa columnas nuevas**
(cuadrados y productos cruzados) y después aplica el mismo OLS:

```
[z₁ … z₁₁]  →  [z₁ … z₁₁, z₁²…z₁₁², z₁z₂, z₁z₃, … z₁₀z₁₁]   →  11 + 11 + 55 = 77
```

```python
# Pipeline POLINOMIAL: escalar → expandir a grado 2 → OLS
def obtener_pipeline(polinomial=False, grado=POLY_DEGREE):
    pasos = [("scaler", StandardScaler())]
    if polinomial:
        pasos.append(("poly", PolynomialFeatures(degree=grado, include_bias=False)))
    pasos.append(("regresion", LinearRegression()))   # ← mismo OLS que el lineal
    return Pipeline(steps=pasos)
```

> ⚠️ `include_bias=False`: porque `LinearRegression` ya añade su propio intercepto β₀; una columna extra de unos duplicaría ese término.
>
> 💡 Aquí el `StandardScaler` es **crucial antes del `PolynomialFeatures`**: sin escalar, elevar al cuadrado
> variables como `total sulfur dioxide` (valores hasta ~289) generaría números gigantes que dominarían el ajuste.

---

## 📚 Librerías y fórmulas que hacen posible el modelo

| Librería / Clase | Rol en el código | Fórmula / concepto |
|---|---|---|
| `pandas` | `read_csv(sep=";")` carga el CSV de UCI | Matriz X tabular |
| `StandardScaler` | Paso `"scaler"` del pipeline | $z=\frac{x-\mu}{\sigma}$ |
| `PolynomialFeatures` | Paso `"poly"` (modo polinomial) | Genera $z_i^2$ y $z_i z_j$ |
| `LinearRegression` | Paso `"regresion"` — el modelo | OLS: $\hat{\beta}=(X^TX)^{-1}X^Ty$ |
| `Pipeline` | Encadena scaler→poly→regresión | Evita fuga de datos (μ,σ solo del train) |
| `train_test_split` | Divide 80% / 20% | Muestreo reproducible (seed 42) |
| `r2_score` | Métrica de bondad | Ver sección R² ⬇️ |
| `root_mean_squared_error` | Métrica de error | $RMSE=\sqrt{\overline{(y-\hat{y})^2}}$ |
| `joblib` | `dump`/`load` del pipeline entrenado | Serialización binaria |
| `gradio` | Radio, Slider, Button, Textbox | UI web interactiva |

---

## 🏋️ Cómo entrena este modelo

```
pd.read_csv(sep=";") → X,y → split 80/20 → fit(μ,σ → poly → β̂) → joblib.dump → R²/RMSE test
```

```python
def entrenar(polinomial=False, grado=POLY_DEGREE):
    ruta = POLY_PATH if polinomial else MODEL_PATH
    df = pd.read_csv(CSV_PATH, sep=";")               # 1. winequality-red.csv (1.599 filas)
    X = df.drop(columns=[TARGET])                     # 2. las 11 columnas fisicoquímicas
    y = df[TARGET].astype(float)                      #    quality = target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)         # 3. 80% train / 20% test
    pipe = obtener_pipeline(polinomial, grado)
    pipe.fit(X_train, y_train)                        # 4. scaler aprende μ,σ;
                                                      #    (poly genera 77 columnas);
                                                      #    OLS calcula β̂ = (XᵀX)⁻¹Xᵀy
    joblib.dump(pipe, ruta)                           # 5. persiste el pipeline completo
    r2   = r2_score(y_test, pipe.predict(X_test))     # 6. evalúa en el 20% nunca visto
    rmse = root_mean_squared_error(y_test, pipe.predict(X_test))
    print(f"Wine Quality (...): R² test = {r2:.4f}, RMSE = {rmse:.4f}")
```

---

## 📐 La métrica R²: por qué se eligió y dónde vive en el código

### ¿Qué es?

$$R^2 = 1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2} = 1 - \frac{SS_{res}}{SS_{tot}}$$

Es el **porcentaje de varianza de la calidad del vino que el modelo explica**:
`R² = 1.0` → predicción perfecta · `R² = 0` → no mejora a predecir el promedio · `< 0` → peor que el promedio.

### ¿Por qué se eligió para este caso?

| Razón | Explicación |
|---|---|
| ✅ Es adimensional | Permite comparar los 3 casos del repositorio aunque midan cosas distintas ($, puntos, unidades clínicas) |
| ✅ Compara Lineal vs Polinomial justamente | Ambos se evalúan sobre el mismo `X_test`; gana quien explique más varianza de `quality` |
| ✅ Detecta sobreajuste junto a RMSE | Si la polinomial sube R² en train pero cae en test, memorizó en vez de aprender |
| ✅ Fácil de comunicar | "El modelo explica el X% de la variabilidad en la calidad sensorial" |
| ✅ Estándar de facto | Métrica por defecto para seleccionar modelos de regresión en scikit-learn |

⚠️ **Matiz importante:** aquí R² **no selecciona el modelo automáticamente**. Es métrica *informativa*
que fundamenta tu elección; la selección final la hace el usuario con el Radio de Gradio:

```
entrenar(polinomial=False) → R² test lineal      ┐
entrenar(polinomial=True)  → R² test polinomial  ┘→ comparas los prints
                                                   ↓
                          Gradio Radio → eliges el pipeline ganador
```

### ¿Dónde está en el código?

| Ubicación | Archivo | Línea | Código |
|---|---|---|---|
| Import | `wine_predictor.py` | **8** | `from sklearn.metrics import r2_score, root_mean_squared_error` |
| Cálculo | `wine_predictor.py` | **48** | `r2 = r2_score(y_test, pipe.predict(X_test))` |
| Reporte | `wine_predictor.py` | **50** | `print(f"... R² test = {r2:.4f}, RMSE = ...")` |
| Selección manual | `wine_app.py` | `predecir()` | `pipe = pipeline_lineal if modelo == LINEAL else pipeline_polinomial` |

### Interpretación práctica

```text
R² test ≈ 0.30–0.40  → típico en vino tinto con OLS lineal (la calidad es subjetiva y ruidosa)
RMSE ≈ 0.5–0.6       → el error medio anda alrededor de medio punto de calidad
R² train alto + test bajo → polinomial sobreajustando → regularizar (Ridge) o reducir grado
Quality es discreta (3–8 real): tratarla como regresión aproxima, no clasifica
```

---

## 🔗 Navegación

- 🏠 [California Housing](../california-housing/) · 🍷 **Wine Quality (este caso)** · 🩸 [Diabetes](../diabetes/)
- App unificada: [`app.py`](../app.py) — 3 pestañas Gradio · Despliegue: `render.yaml` (Render free)

---

*Fuente: [angiezulema/Modelos-predictivos](https://github.com/angiezulema/Modelos-predictivos) · Dataset: UCI Machine Learning Repository — Wine Quality (Cortez et al., 2009)*
