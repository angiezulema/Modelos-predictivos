# 🩸 Diabetes — Regresión Lineal Múltiple y Polinomial

> **Caso 3 de 3** · Repositorio [Modelos-predictivos](https://github.com/angiezulema/Modelos-predictivos) · `diabetes/diabetes_predictor.py` + `diabetes/diabetes_app.py` · scikit-learn + Gradio

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-OLS-F7931E?logo=scikitlearn&logoColor=white)
![Gradio](https://img.shields.io/badge/Interfaz-Gradio-orange)
![Modelo](https://img.shields.io/badge/R2_score-m%C3%A9trica%20clave-success)

---

## 📌 Introducción: ¿qué hace la app y cuál es su finalidad?

La **app de Diabetes** es una herramienta interactiva construida con **Gradio** cuyo objetivo es
**predecir la progresión de la enfermedad diabética un año después del examen inicial (baseline)**,
como valor cuantitativo continuo (no clasificación "sí/no"). El usuario mueve 10 *sliders* con datos
clínicos del paciente, elige entre dos modelos de regresión y obtiene al instante la estimación.

**Finalidad doble:**

- ✦ Demostrar el flujo completo de ML: *datos → preprocesamiento → entrenamiento → evaluación → serialización → inferencia*.
- ✦ Comparar en vivo dos familias de regresión sobre el mismo problema: **Lineal Múltiple** vs **Polinomial grado 2**.

### 🖥️ Cómo funciona la interfaz

```
Radio: elegir modelo  →  10 Sliders (rango min–max real del CSV)  →  Botón "Predecir"
      →  pipeline.predict()  →  Resultado en Textbox
```

```python
def predecir(modelo, *valores):
    datos = {c: v for c, v in zip(FEATURES, valores)}
    fila = pd.DataFrame([datos], columns=FEATURES)
    pipe = pipeline_lineal if modelo == LINEAL else pipeline_polinomial
    pred = float(pipe.predict(fila)[0])
    return f"Progresión de la enfermedad predicha ({modelo}): {pred:.2f}"
```

💡 Los rangos mín/máx de cada slider se calculan desde el CSV (`RANGES`), el default es la **mediana**
y el paso es `(max-min)/100`. Si los `.joblib` no existen, la app reentrena sola vía `cargar_modelo()`.

---

## 🗺️ ¿Cómo se planeó hacer?

1. Obtener el dataset clásico `sklearn.datasets.load_diabetes` guardándolo como `diabetes_raw.csv` con **valores reales**.
2. Definir las 10 variables clínicas y el target `progression`.
3. Construir **dos pipelines paralelos** (lineal y polinomial g2), ambos estandarizados.
4. Entrenar con split **80/20** y semilla fija `random_state=42` (reproducibilidad).
5. Evaluar con **R² y RMSE** en test para decidir qué modelo generaliza mejor.
6. Persistir con `joblib` → `diabetes_model.joblib` / `diabetes_model_poly.joblib`.
7. Exponer en Gradio y desplegar en Render.

> 🔁 **Iteraciones previas:** `"diabetes 01/"` versión Flask (+EDA) y `"diabetes-regresion-polinomial/"` versión Streamlit (Polinomial + **Ridge**, respuesta al sobreajuste). Ambas consolidaron la pestaña Gradio final.

---

## 📋 Variables que usa el modelo

| Variable | Etiqueta en el app | Descripción |
|---|---|---|
| `age` | Edad (años) | Edad del paciente |
| `sex` | Sexo (1=mujer, 2=hombre) | Sexo biológico |
| `bmi` | Índice de Masa Corporal | IMC en kg/m² |
| `bp` | Presión arterial media | mmHg |
| `s1` | Nivel sanguíneo (tc) | Colesterol total sérico |
| `s2` | Nivel sanguíneo (ldl) | LDL |
| `s3` | Nivel sanguíneo (hdl) | HDL |
| `s4` | Nivel sanguíneo (tch) | Colesterol total / HDL |
| `s5` | Nivel sanguíneo (ltg) | Log de triglicéridos |
| `s6` | Nivel sanguíneo (glu) | Glucosa en sangre |
| **`progression`** | 🎯 **TARGET** | Progresión de la diabetes a 1 año |

```python
FEATURES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
TARGET   = "progression"
```

---

## 📈 Tipos de regresión usados y sus diferencias

Ambas son **regresión lineal por mínimos cuadrados (OLS)**: cambia *cómo se transforman las x antes de ajustar*.
La polinomial sigue siendo **lineal en sus coeficientes β**, pero no lineal en las variables originales.

| Criterio | ➖ Lineal Múltiple | ∫ Polinomial (g2) |
|---|---|---|
| Forma geométrica | Hiperplano recto | Superficie curva |
| Fórmula | ŷ = β₀ + Σβᵢxᵢ′ | ŷ = β₀ + Σβᵢzᵢ + Σβᵢᵢzᵢ² + Σβᵢⱼzᵢzⱼ |
| Parámetros estimados | 11 | 66 (β₀ + 65) |
| Features al OLS | 10 | 65 (10 + 10 cuadrados + 45 interacciones) |
| Interpretabilidad | Alta (cada β = efecto unitario) | Baja (coeficientes cruzados) |
| Riesgo principal | Subajuste | Sobreajuste → por eso existe la variante Ridge |
| Archivo generado | `diabetes_model.joblib` | `diabetes_model_poly.joblib` |

---

## ➖ Regresión Lineal Múltiple

$$\hat{y} = \beta_0 + \beta_1 \cdot age' + \beta_2 \cdot sex' + \beta_3 \cdot bmi' + \beta_4 \cdot bp' + \sum_{i=1}^{6} \beta_{i+4} \cdot s_i'$$

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

Ejemplo con 2 variables: $\hat{y} = \beta_0 + \beta_1 bmi' + \beta_2 s5' + \beta_3 bmi'^2 + \beta_4 s5'^2 + \beta_5\, bmi' s5'$

**El truco:** no existe una "clase polinomial". `PolynomialFeatures` solo **inventa columnas nuevas**
(cuadrados y productos cruzados) y después aplica el mismo OLS:

```
[z₁ … z₁₀]  →  [z₁ … z₁₀, z₁²…z₁₀², z₁z₂, z₁z₃, … z₉z₁₀]   →  10 + 10 + 45 = 65
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

---

## 📚 Librerías y fórmulas que hacen posible el modelo

| Librería / Clase | Rol en el código | Fórmula / concepto |
|---|---|---|
| `pandas` | `read_csv`, arma el DataFrame de entrada | Matriz X tabular |
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
pd.read_csv → X,y → split 80/20 → fit(μ,σ → poly → β̂) → joblib.dump → R²/RMSE test
```

```python
def entrenar(polinomial=False, grado=POLY_DEGREE):
    ruta = POLY_PATH if polinomial else MODEL_PATH
    df = pd.read_csv(CSV_PATH)                        # 1. diabetes_raw.csv (442 filas)
    X = df[FEATURES]                                  # 2. las 10 columnas clínicas
    y = df[TARGET].astype(float)                      #    progression = target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)         # 3. 80% train / 20% test
    pipe = obtener_pipeline(polinomial, grado)
    pipe.fit(X_train, y_train)                        # 4. scaler aprende μ,σ;
                                                      #    (poly genera 65 columnas);
                                                      #    OLS calcula β̂ = (XᵀX)⁻¹Xᵀy
    joblib.dump(pipe, ruta)                           # 5. persiste el pipeline completo
    r2   = r2_score(y_test, pipe.predict(X_test))     # 6. evalúa en el 20% nunca visto
    rmse = root_mean_squared_error(y_test, pipe.predict(X_test))
    print(f"Diabetes (...): R² test = {r2:.4f}, RMSE = {rmse:.2f}")
```

---

## 📐 La métrica R²: por qué se eligió y dónde vive en el código

### ¿Qué es?

$$R^2 = 1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2} = 1 - \frac{SS_{res}}{SS_{tot}}$$

Es el **porcentaje de varianza del target que el modelo explica**:
`R² = 1.0` → predicción perfecta · `R² = 0` → no mejora a predecir el promedio · `< 0` → peor que el promedio.

### ¿Por qué se eligió para este caso?

| Razón | Explicación |
|---|---|
| ✅ Es adimensional | Permite comparar targets de escalas distintas ($ en housing, puntos en wine, unidades clínicas en diabetes) |
| ✅ Compara Lineal vs Polinomial justamente | Ambos se evalúan sobre el mismo `X_test` nunca visto; gana quien explique más varianza |
| ✅ Detecta sobreajuste junto a RMSE | R² alto en train pero bajo en test ⇒ la polinomial memorizó |
| ✅ Fácil de comunicar | "El modelo explica el X% de la variabilidad de la progresión" |
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
| Import | `diabetes_predictor.py` | **8** | `from sklearn.metrics import r2_score, root_mean_squared_error` |
| Cálculo | `diabetes_predictor.py` | **50** | `r2 = r2_score(y_test, pipe.predict(X_test))` |
| Reporte | `diabetes_predictor.py` | **52** | `print(f"... R² test = {r2:.4f}, RMSE = ...")` |
| Train+test (anti-overfitting) | `housing_predictor.py` | **92–101** | calcula e imprime R² de train Y test |
| Selección manual | `diabetes_app.py` | `predecir()` | `pipe = pipeline_lineal if modelo == LINEAL else pipeline_polinomial` |

### Interpretación práctica

```text
R² test = 0.45–0.60  → típico en este dataset (la progresión tiene mucho ruido biológico)
R² test >> R² train  → sospechar fuga de datos (no ocurre aquí: Pipeline lo evita)
R² train ≈ 0.9 pero test ≈ 0.4 → polinomial sobreajustando → usar Ridge
```

---

## 🔗 Navegación

- 🏠 [California Housing](../california-housing/) · 🍷 [Wine Quality](../wine-quality/) · 🩸 **Diabetes (este caso)**
- App unificada: [`app.py`](../app.py) — 3 pestañas Gradio · Despliegue: `render.yaml` (Render free)

---

*Fuente: [angiezulema/Modelos-predictivos](https://github.com/angiezulema/Modelos-predictivos)*
