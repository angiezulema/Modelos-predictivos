# 🏠 California Housing — Regresión Lineal Múltiple y Polinomial

> **Caso 1 de 3** · Repositorio [Modelos-predictivos](https://github.com/angiezulema/Modelos-predictivos) · `california-housing/housing_predictor.py` + `california-housing/housing_app.py` · scikit-learn + Gradio

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-OLS-F7931E?logo=scikitlearn&logoColor=white)
![Gradio](https://img.shields.io/badge/Interfaz-Gradio-orange)
![Dataset](https://img.shields.io/badge/Dataset-California%20Housing-065f46)
![Modelo](https://img.shields.io/badge/R2_score-m%C3%A9trica%20clave-success)

---

## 📌 Introducción: ¿qué hace la app y cuál es su finalidad?

La **app de California Housing** es una herramienta interactiva construida con **Gradio** cuyo objetivo es
**predecir el precio mediano de una vivienda (`median_house_value`) en un distrito de California**
a partir de su ubicación geográfica, características físicas y proximidad al océano.
El usuario mueve los deslizadores con los datos del distrito, elige entre dos modelos de regresión
y obtiene al instante la estimación en dólares.

**Finalidad doble:**

- ✦ Demostrar el flujo completo de ML: *datos → limpieza → preprocesamiento mixto (numérico + categórico) → entrenamiento → evaluación → serialización → inferencia*.
- ✦ Comparar en vivo dos familias de regresión sobre el mismo problema: **Lineal Múltiple** vs **Polinomial grado 2**.

Es el caso **más completo de los tres**: es el único que mezcla **variables numéricas con una categórica**
(`ocean_proximity`), lo que exige un `ColumnTransformer` con `OneHotEncoder`. También es el único que
evalúa R² en **train Y test**, para vigilar el sobreajuste de la versión polinomial (~104 features).

### 🖥️ Cómo funciona la interfaz

```
Radio: elegir modelo  →  Sliders numéricos + selector ocean_proximity  →  Botón "Predecir"
      →  pipeline.predict()  →  Resultado en Textbox ($ USD)
```

```python
def predecir(modelo, longitude, latitude, ..., median_income, ocean_proximity):
    fila = pd.DataFrame([[...]], columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    pipe = pipeline_lineal if modelo == LINEAL else pipeline_polinomial
    pred = float(pipe.predict(fila)[0])
    return f"🏠 Precio mediano predicho: ${pred:,.2f}"
```

💡 Misma mecánica que las otras pestañas: si los `.joblib` no existen,
`cargar_modelo()` reentrena automáticamente antes de servir la interfaz.

---

## 🗺️ ¿Cómo se planeó hacer?

1. Obtener el dataset clásico **California Housing** (`archive/housing.csv`, ~20.640 distritos del censo de 1990).
2. Definir 8 variables numéricas + 1 categórica como X y `median_house_value` como target.
3. Limpiar: eliminar filas con `total_bedrooms` nulo (`dropna`).
4. Resolver el reto técnico: codificar `ocean_proximity` con **OneHotEncoder dentro de un `ColumnTransformer`**.
5. Construir **dos pipelines paralelos** (lineal y polinomial g2) sobre ese preprocesador común.
6. Entrenar con split **80/20** y semilla fija `random_state=42` (reproducibilidad).
7. Evaluar con **R² y RMSE en train Y test** (doble chequeo anti-sobreajuste) e imprimir coeficientes interpretables.
8. Persistir con `joblib` → `housing_model.joblib` / `housing_model_poly.joblib`.
9. Exponer en Gradio (pestaña 🏠 del app raíz) y desplegar en Render.

---

## 📋 Variables que usa el modelo

| Variable | Descripción | Tipo |
|---|---|---|
| `longitude` | Longitud del distrito (−124 a −114) | Numérica |
| `latitude` | Latitud del distrito (32 a 42) | Numérica |
| `housing_median_age` | Antigüedad mediana de las casas (años) | Numérica |
| `total_rooms` | Total de cuartos por distrito | Numérica |
| `total_bedrooms` | Total de dormitorios por distrito | Numérica |
| `population` | Población del distrito | Numérica |
| `households` | Número de hogares | Numérica |
| `median_income` | Ingreso mediano (en decenas de miles USD) | Numérica |
| `ocean_proximity` | Proximidad al océano: `<1H OCEAN, INLAND, ISLAND, NEAR BAY, NEAR OCEAN` | **Categórica** → one-hot |
| **`median_house_value`** | 🎯 **TARGET** — precio mediano de la vivienda (USD, tope en $500.001) | Numérica |

```python
NUMERIC_FEATURES = ["longitude", "latitude", "housing_median_age", "total_rooms",
                    "total_bedrooms", "population", "households", "median_income"]
CATEGORICAL_FEATURES = ["ocean_proximity"]
TARGET = "median_house_value"
```

---

## 📈 Tipos de regresión usados y sus diferencias

Ambas son **regresión lineal por mínimos cuadrados (OLS)**: cambia *cómo se transforman las x antes de ajustar*.
La polinomial sigue siendo **lineal en sus coeficientes β**, pero no lineal en las variables originales.

| Criterio | ➖ Lineal Múltiple | ∫ Polinomial (g2) |
|---|---|---|
| Forma geométrica | Hiperplano recto en 13D (tras one-hot) | Superficie curva |
| Fórmula | ŷ = β₀ + Σβᵢxᵢ | ŷ = β₀ + Σβᵢxᵢ + Σβᵢᵢxᵢ² + Σβᵢⱼxᵢxⱼ |
| Parámetros estimados | 14 (β₀ + 13) | 105 (β₀ + 104) |
| Features al OLS | 13 (8 numéricas + 5 one-hot) | 104 (13 + 13 cuadrados + 78 interacciones) |
| Interpretabilidad | Alta — imprime cada β (ej: +median_income ⇒ sube el precio) | Baja (coeficientes cruzados) |
| Evaluación especial | R² train y test | R² train vs test ⇒ detector de sobreajuste |
| Archivo generado | `housing_model.joblib` | `housing_model_poly.joblib` |

---

## ➖ Regresión Lineal Múltiple

$$\hat{y} = \beta_0 + \beta_1 \cdot longitude + \beta_2 \cdot latitude + \cdots + \beta_9 \cdot median\_income + \sum_{c} \gamma_c \cdot \mathbb{1}[ocean\_proximity = c]$$

donde $\mathbb{1}[\cdot]$ son las columnas **one-hot** generadas por `OneHotEncoder`
(una columna 0/1 por categoría: `<1H OCEAN`, `INLAND`, `ISLAND`, `NEAR BAY`, `NEAR OCEAN`).

**¿Qué librería/fórmula lo hace posible?** `LinearRegression` resuelve los **mínimos cuadrados ordinarios**:

$$\hat{\beta} = (X^TX)^{-1}X^Ty \quad \text{(implementada numéricamente con SVD)}$$

```python
# Pipeline LINEAL: one-hot → OLS   (¡sin StandardScaler en este caso!)
def obtener_pipeline(polinomial=False, grado=POLY_DEGREE):
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)],
        remainder="passthrough")                     # ← las numéricas pasan tal cual
    pasos = [("preprocess", preprocessor)]
    pasos.append(("regresion", LinearRegression()))  # ← β̂ = (XᵀX)⁻¹Xᵀy
    return Pipeline(steps=pasos)
```

💡 A diferencia de wine/diabetes, aquí **no hay `StandardScaler`**: las variables entran crudas
(`remainder="passthrough"`). El OLS no necesita escalado para ser correcto; solo afecta la legibilidad
de los coeficientes.

## ∫ Regresión Polinomial (grado 2)

$$\hat{y} = \beta_0 + \sum_{i} \beta_i x_i + \sum_{i} \beta_{ii} x_i^2 + \sum_{i<j} \beta_{ij}\, x_i x_j$$

Ejemplo: $\hat{y}$ puede capturar efectos como *"el ingreso importa más cerca de la bahía"*
(término $median\_income \times near\_bay$) o rendimientos decrecientes ($income^2$).

**El truco:** no existe una "clase polinomial". `PolynomialFeatures` solo **inventa columnas nuevas**
(cuadrados y productos cruzados) y después aplica el mismo OLS:

```
[8 numéricas + 5 one-hot]  →  [13 originales + 13 cuadradas + 78 cruces]  →  104 features
```

```python
# Pipeline POLINOMIAL: one-hot → expandir a grado 2 → OLS
def obtener_pipeline(polinomial=False, grado=POLY_DEGREE):
    pasos = [("preprocess", preprocessor)]
    if polinomial:
        pasos.append(("poly", PolynomialFeatures(degree=grado, include_bias=False)))
    pasos.append(("regresion", LinearRegression()))   # ← mismo OLS que el lineal
    return Pipeline(steps=pasos)
```

> ⚠️ `include_bias=False`: porque `LinearRegression` ya añade su propio intercepto β₀.
>
> 💡 El orden del pipeline importa: primero **one-hot** y luego **poly** — así las categorías también
> participan en las interacciones (ej: `INLAND × median_income`), y `handle_unknown="ignore"`
> evita errores en producción si llega una categoría nueva.

---

## 📚 Librerías y fórmulas que hacen posible el modelo

| Librería / Clase | Rol en el código | Fórmula / concepto |
|---|---|---|
| `pandas` | `read_csv` + `dropna(subset=["total_bedrooms"])` | Matriz X tabular + limpieza |
| `ColumnTransformer` | Aplica transformaciones por tipo de columna | X = [X_cat \| X_num] en bloque |
| `OneHotEncoder` | Codifica `ocean_proximity` (5 columnas 0/1) | $\mathbb{1}[x=c]$ — variables dummy |
| `PolynomialFeatures` | Paso `"poly"` (modo polinomial) | Genera $x_i^2$ y $x_i x_j$ |
| `LinearRegression` | Paso `"regresion"` — el modelo | OLS: $\hat{\beta}=(X^TX)^{-1}X^Ty$ |
| `Pipeline` | Encadena preprocess→poly→regresión | Evita fuga de datos en train/test |
| `train_test_split` | Divide 80% / 20% | Muestreo reproducible (seed 42) |
| `r2_score` | Métrica de bondad (train Y test aquí) | Ver sección R² ⬇️ |
| `root_mean_squared_error` | Métrica de error en dólares | $RMSE=\sqrt{\overline{(y-\hat{y})^2}}$ |
| `joblib` | `dump`/`load` del pipeline entrenado | Serialización binaria |
| `gradio` | Radio, Slider, Dropdown, Textbox | UI web interactiva |

---

## 🏋️ Cómo entrena este modelo

```
read_csv → dropna → split 80/20 → fit(one-hot → poly → β̂) → joblib.dump → R²/RMSE train y test
```

```python
def entrenar(polinomial=False, grado=POLY_DEGREE):
    ruta = POLY_PATH if polinomial else MODEL_PATH
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["total_bedrooms"])         # 1. limpieza de nulos
    X = df.drop(columns=[TARGET])                     # 2. 8 numéricas + 1 categórica
    y = df[TARGET].astype(float)                      #    median_house_value = target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)         # 3. 80% train / 20% test
    pipeline = obtener_pipeline(polinomial, grado)
    pipeline.fit(X_train, y_train)                    # 4. one-hot aprende categorías;
                                                      #    (poly genera 104 columnas);
                                                      #    OLS calcula β̂ = (XᵀX)⁻¹Xᵀy
    joblib.dump(pipeline, ruta)                       # 5. persiste el pipeline completo
    y_pred_train = pipeline.predict(X_train)          # 6. evalúa AMBOS conjuntos:
    y_pred_test  = pipeline.predict(X_test)           #    R² train vs R² test
    r2_train = r2_score(y_train, y_pred_train)        #    ⇒ detector de sobreajuste
    r2_test  = r2_score(y_test, y_pred_test)
    reg = pipeline.named_steps["regresion"]           # 7. bonus: imprime intercepto
    print(f"Intercepto (b0): ${reg.intercept_:,.2f}") #    y coeficientes por variable
```

---

## 📐 La métrica R²: por qué se eligió y dónde vive en el código

### ¿Qué es?

$$R^2 = 1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2} = 1 - \frac{SS_{res}}{SS_{tot}}$$

Es el **porcentaje de varianza del precio que el modelo explica**:
`R² = 1.0` → predicción perfecta · `R² = 0` → no mejora a predecir el promedio · `< 0` → peor que el promedio.

### ¿Por qué se eligió para este caso?

| Razón | Explicación |
|---|---|
| ✅ Es adimensional | Permite comparar los 3 casos del repositorio aunque midan cosas distintas ($, puntos, unidades clínicas) |
| ✅ Compara Lineal vs Polinomial justamente | Ambos se evalúan sobre el mismo `X_test`; gana quien explique más varianza del precio |
| ✅ Aquí se usa en DOBLE papel: train + test | Con 104 features polinomiales, comparar `R² train` vs `R² test` es el detector de sobreajuste |
| ✅ Se complementa con RMSE en dólares | R² dice "cuánto explica"; RMSE dice "me equivoco en ±$X" — ambos se imprimen juntos |
| ✅ Estándar de facto | Métrica por defecto para seleccionar modelos de regresión en scikit-learn |

⚠️ **Matiz importante:** aquí R² **no selecciona el modelo automáticamente**. Es métrica *informativa*
que fundamenta tu elección; la selección final la hace el usuario con el Radio de Gradio:

```
entrenar(polinomial=False) → R² train + R² test lineal     ┐
entrenar(polinomial=True)  → R² train + R² test polinomial ┘→ comparas los prints
                                                             ↓
                          Gradio Radio → eliges el pipeline ganador
```

### ¿Dónde está en el código?

| Ubicación | Archivo | Línea | Código |
|---|---|---|---|
| Import | `housing_predictor.py` | **12** | `from sklearn.metrics import r2_score, root_mean_squared_error` |
| Cálculo train | `housing_predictor.py` | **92** | `r2_train = r2_score(y_train, y_pred_train)` |
| Cálculo test | `housing_predictor.py` | **93** | `r2_test = r2_score(y_test, y_pred_test)` |
| Reporte doble | `housing_predictor.py` | **100–101** | `print(f"Train \| R² = ...")` / `print(f"Test \| R² = ...")` |
| Selección manual | `housing_app.py` | `predecir()` | `pipeline_lineal if modelo == LINEAL else pipeline_polinomial` |

### Interpretación práctica

```text
R² test ≈ 0.60–0.66 (lineal)   → típico: el ingreso mediano es la variable dominante
R² test ≈ 0.65–0.70 (poli g2)  → mejora modesta, pero cuidado...
R² train ≈ 0.75+ vs test ≈ 0.65 → brecha = sobreajuste de las 104 features
RMSE ≈ $50,000–$70,000          → error típico en dólares; compáralo con el tope de $500k
Ojo: target censurado en $500,001 (los distritos carísimos quedan "apilados" ahí) → techo estructural del R²
```

---

## 🔗 Navegación

- 🏠 **California Housing (este caso)** · 🍷 [Wine Quality](../wine-quality/) · 🩸 [Diabetes](../diabetes/)
- App unificada: [`app.py`](../app.py) — 3 pestañas Gradio · Despliegue: `render.yaml` (Render free)

---

*Fuente: [angiezulema/Modelos-predictivos](https://github.com/angiezulema/Modelos-predictivos) · Dataset: California Housing Prices (StatLib, censo de 1990 — popularizado por Aurélien Géron)*
