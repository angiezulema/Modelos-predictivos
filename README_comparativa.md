# 🔎 Comparativa de los 3 Casos — Modelos Predictivos

> Análisis cruzado de los **3 casos de regresión** del repositorio [Modelos-predictivos](https://github.com/angiezulema/Modelos-predictivos) · scikit-learn + Gradio

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-OLS-F7931E?logo=scikitlearn&logoColor=white)
![Gradio](https://img.shields.io/badge/UI-Gradio%203%20pesta%C3%B1as-orange)
![Casos](https://img.shields.io/badge/Casos-Housing%20·%20Wine%20·%20Diabetes-informational)

---

## 🗂️ Los tres casos de un vistazo

| | 🏠 California Housing | 🍷 Wine Quality | 🩸 Diabetes |
|---|---|---|---|
| **Caso** | 1 | 2 | 3 |
| **Carpeta** | `california-housing/` | `wine-quality/` | `diabetes/` |
| **Predice** | Precio mediano de vivienda (USD) | Calidad del vino tinto (0–10) | Progresión de la enfermedad a 1 año |
| **Dificultad didáctica** | ⭐⭐⭐ La más completa | ⭐ La más simple | ⭐⭐ Intermedia |

---

## 📊 1. Datos: dataset, origen y tamaño

| Característica | 🏠 Housing | 🍷 Wine | 🩸 Diabetes |
|---|---|---|---|
| Archivo local | `archive/housing.csv` | `winequality-red.csv` | `diabetes_raw.csv` |
| Origen | StatLib — censo de California 1990 (versión Géron/Kaggle) | UCI — Vinho Verde tinto (Cortez et al., 2009) | `sklearn.datasets.load_diabetes` guardado con valores reales |
| Registros | ~20.640 (→ ~20.433 tras limpieza) | 1.599 | 442 |
| Unidad de análisis | Distrito censal | Vino (muestra física) | Paciente |
| Limpieza necesaria | ✅ `dropna(subset=["total_bedrooms"])` | ❌ Ninguna | ❌ Ninguna |
| Separador CSV | `,` estándar | `;` (`pd.read_csv(sep=";")`) | `,` estándar |
| Particularidad | ⚠️ Target **censurado** en $500.001 | Nota discreta tratada como continua | Ya viene semi-normalizado en versiones sklearn; aquí se usan valores reales |

---

## 🎯 2. Target y variables

| Característica | 🏠 Housing | 🍷 Wine | 🩸 Diabetes |
|---|---|---|---|
| Columna target | `median_house_value` | `quality` | `progression` |
| Tipo de target | Continuo (USD) | Entero 3–8 (tratado como continuo) | Continuo (25–346) |
| Variables predictoras | **8 numéricas + 1 categórica** | **11 numéricas** | **10 numéricas** |
| Nombres clave | longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity | fixed/volatile acidity, citric acid, residual sugar, chlorides, free/total SO₂, density, pH, sulphates, alcohol | age, sex, bmi, bp, s1–s6 |
| Variable categórica | ✅ `ocean_proximity` (5 categorías) | ❌ | ❌ |
| Variable dominante | `median_income` (correlación fuerte con el precio) | `alcohol` y `volatile acidity` | `bmi` y `s5` (glucosa/log triglicéridos) |
| Selección de X | `df.drop(columns=[TARGET])` | `df.drop(columns=[TARGET])` | `df[FEATURES]` (lista explícita) |

---

## ⚙️ 3. Pipelines de preprocesamiento (la gran diferencia)

```
🏠 HOUSING (mixto):
   ColumnTransformer ──┬─ cat: OneHotEncoder(handle_unknown="ignore") → 5 columnas 0/1
                       └─ remainder="passthrough" (8 numéricas SIN escalar)
   [→ (poly g2)] → LinearRegression          ← ¡sin StandardScaler!

🍷 WINE (numérico simple):
   StandardScaler → [→ (poly g2)] → LinearRegression

🩸 DIABETES (numérico simple):
   StandardScaler → [→ (poly g2)] → LinearRegression     ← espejo del caso wine
```

| Característica | 🏠 Housing | 🍷 Wine | 🩸 Diabetes |
|---|---|---|---|
| Codificación categórica | ✅ `OneHotEncoder` en `ColumnTransformer` | — | — |
| Estandarización | ❌ No usa `StandardScaler` | ✅ | ✅ |
| ¿Por qué ese preproceso? | OLS no exige escala; one-hot es lo imprescindible para la categórica | Escalas muy distintas (SO₂ hasta 289 vs densidad 0.99) exigen normalizar | Mismo criterio que wine |
| Orden crítico del pipeline | one-hot **antes** de poly (las categorías participan en interacciones) | scaler **antes** de poly (evita explosión numérica al elevar al cuadrado) | ídem wine |

---

## ➖ ∫ 4. Fórmulas y expansión polinomial

**Fórmula base común (OLS):**

$$\hat{y} = \beta_0 + \sum_i \beta_i x_i \qquad \xrightarrow{\text{grado 2}} \qquad \hat{y} = \beta_0 + \sum_i \beta_i z_i + \sum_i \beta_{ii} z_i^2 + \sum_{i<j} \beta_{ij} z_i z_j$$

$$\hat{\beta} = (X^TX)^{-1}X^Ty \quad \text{(idéntico en los 3 casos)}$$

| Característica | 🏠 Housing | 🍷 Wine | 🩸 Diabetes |
|---|---|---|---|
| Features tras one-hot (lineal) | 13 | 11 | 10 |
| Parámetros lineal (β₀ + coef) | 14 | 12 | 11 |
| Features polinomiales g2 | **104** = 13 + 13² + C(13,2)−13 | **77** = 11 + 11 + 55 | **65** = 10 + 10 + 45 |
| Parámetros polinomial | 105 | 78 | 66 |
| Fórmula general expansión | $n + n + \frac{n(n-1)}{2}$ con n = features de entrada | ídem | ídem |
| Ejemplo de interacción capturada | `INLAND × median_income` (el ingreso pesa distinto tierra adentro) | `alcohol × sulphates` | `bmi × s5` (IMC y glucosa sinergian) |
| `include_bias=False` | ✅ (β₀ ya lo aporta LinearRegression) | ✅ | ✅ |

---

## 🏋️ 5. Entrenamiento: igualdades y diferencias

### Flujo común (los 3 casos)

```
cargar CSV → definir X,y → train_test_split(0.2, random_state=42)
→ obtener_pipeline(polinomial?) → pipe.fit(X_train, y_train)
→ joblib.dump(pipe, *.joblib) → r2_score + root_mean_squared_error → print
```

| Característica | 🏠 Housing | 🍷 Wine | 🩸 Diabetes |
|---|---|---|---|
| Split train/test | 80 / 20 | 80 / 20 | 80 / 20 |
| Semilla | `random_state=42` | `random_state=42` | `random_state=42` |
| Limpieza previa al split | ✅ dropna | — | — |
| Métricas evaluadas | **R² y RMSE en train Y test** (doble) | R² y RMSE solo test | R² y RMSE solo test |
| Imprime interpretabilidad | ✅ Intercepto + coeficiente por variable (solo modo lineal) | ❌ | ❌ |
| Líneas del cálculo R² | 92–93 (train/test) | 48 | 50 |
| Persistencia | `housing_model.joblib` / `_poly.joblib` | `wine_model.joblib` / `_poly.joblib` | `diabetes_model.joblib` / `_poly.joblib` |
| Reentrenamiento automático | ✅ `cargar_modelo()` si falta el `.joblib` (patrón idéntico en los 3) | ✅ | ✅ |
| Sub-proyectos extra | — | — | ✅ Flask (+EDA) y Streamlit (**Poly + Ridge**) |

---

## 📐 6. La métrica R² en los tres casos

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y-\hat{y})^2}{\sum(y-\bar{y})^2}$$

**Métrica elegida en los 3 por las mismas razones:** adimensional (permite comparar $, puntos y unidades clínicas), compara Lineal vs Polinomial sobre el mismo test set, detecta sobreajuste junto a RMSE, y es el estándar scikit-learn. En **ningún caso selecciona el modelo automáticamente**: el usuario decide en el Radio de Gradio viendo los prints.

### Dónde vive R² a nivel código

| | 🏠 Housing | 🍷 Wine | 🩸 Diabetes |
|---|---|---|---|
| Import | L12 `from sklearn.metrics import r2_score, ...` | L8 | L8 |
| Cálculo | L92–93 (train y test) | L48 | L50 |
| Reporte | L100–101 (`Train \| R²` / `Test \| R²`) | L50 | L52 |
| Selección manual | `housing_app.py → predecir()` | `wine_app.py → predecir()` | `diabetes_app.py → predecir()` |

### Rendimiento típico esperado (OLS)

| Métrica | 🏠 Housing | 🍷 Wine | 🩸 Diabetes |
|---|---|---|---|
| R² test lineal | ≈ 0.60–0.66 | ≈ 0.30–0.40 | ≈ 0.45–0.55 |
| R² test polinomial | ≈ 0.65–0.70 | similar/slightly ↑ | similar/slightly ↑ |
| RMSE típico | \$50k–\$70k | 0.5–0.6 puntos | 53–58 unidades |
| Techo estructural | Target cortado en $500k | Calidad subjetiva/ruidosa | Ruido biológico intrínseco |
| Riesgo principal | Sobreajuste poli (104 feats) | Underfitting lineal | Sobreajuste poli (mitigado con Ridge en Streamlit) |

---

## 🖥️ 7. Interfaz Gradio y despliegue

| Característica | Común a los 3 |
|---|---|
| Selector de modelo | `gr.Radio(["Regresión Lineal Múltiple", "Regresión Polinomial (grado 2)"])` |
| Entradas | Sliders con rangos min–max reales del dataset, default en mediana/sugerido |
| Botón + salida | `gr.Button("Predecir")` → `gr.Textbox` con resultado formateado |
| Integración | Cada `*_app.py` se monta como pestaña en el `app.py` raíz (Gradio Tabs) |
| Despliegue | Render plan Free vía `render.yaml` (duerme a los 15 min, ~40 s despertar) |
| Extras CLI | Los `*_predictor.py` también funcionan por consola (`--entrenar`, `--polinomial`) |

**Diferencia UI:** housing añade un selector de categoría (`ocean_proximity`, 5 opciones tipo dropdown/radio) además de los sliders numéricos; wine y diabetes son 100% sliders.

---

## 🧬 8. ADN compartido (lo idéntico en los 3)

```python
# Este patrón se repite tal cual en housing_predictor.py, wine_predictor.py y diabetes_predictor.py:
train_test_split(X, y, test_size=0.2, random_state=42)
PolynomialFeatures(degree=2, include_bias=False)
LinearRegression()                      # OLS: β̂ = (XᵀX)⁻¹Xᵀy (SVD)
Pipeline(steps=pasos)                   # anti data-leakage
joblib.dump(pipe, ruta)                 # persistencia
r2_score(...) + root_mean_squared_error # evaluación
cargar_modelo()                         # carga o reentrena si no existe el .joblib
```

---

## 🎓 9. Qué enseña cada caso

| Lección | ¿Dónde se aprende? |
|---|---|
| Manejar variables **categóricas** en regresión (one-hot dentro de pipelines) | 🏠 Housing |
| Vigilar **sobreajuste** comparando R² train vs test | 🏠 Housing (único que imprime ambos) |
| Importancia de **estandarizar** antes de expandir a polinomio | 🍷 Wine y 🩸 Diabetes |
| El caso **minimalista**: todo-numérico, pipeline mínimo | 🍷 Wine |
| Interpretación de **coeficientes clínicos** e iteración hacia regularización (Ridge) | 🩸 Diabetes |
| Patrón profesional: **Pipeline + joblib + auto-reload** en producción | Los 3 |

---

## 🧭 Conclusión

Los tres casos son **el mismo algoritmo (OLS)** con **tres decisiones de ingeniería distintas**:

1. **Housing** resuelve el problema de *datos mixtos* → `ColumnTransformer + OneHotEncoder`.
2. **Wine** resuelve el problema de *escalas heterogéneas* → `StandardScaler`.
3. **Diabetes** replica wine y abre la puerta a la *regularización* (Ridge) cuando la polinomial sobreajusta.

Mismo split, misma semilla, misma métrica (R²), misma estrategia de despliegue: una base común bien diseñada donde cada dataset aporta UNA lección nueva.

---

## 🔗 Navegación

- 🏠 [California Housing](california-housing/) · 🍷 [Wine Quality](wine-quality/) · 🩸 [Diabetes](diabetes/)
- App unificada: [`app.py`](app.py) · Despliegue: [`render.yaml`](render.yaml)

---

*Fuente: [angiezulema/Modelos-predictivos](https://github.com/angiezulema/Modelos-predictivos)*
