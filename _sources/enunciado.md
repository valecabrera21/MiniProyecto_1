# Enunciado
## 7.12.1. Objetivo
Construir un modelo de clasificación supervisada usando **MLP (Multilayer Perceptron / Red Neuronal Multicapa)** para predecir si un préstamo emitido por la plataforma **Lending Club** resultará en **default (1)** o será pagado completamente **(0)**.  
Se comparará el desempeño de los modelos construidos con **scikit-learn** y **PySpark**. Además, se aplicará **LIME** para interpretar predicciones.

## 7.12.2. Dataset
- **Nombre:** Lending Club Loan Data (2007–2020)  
- **Tamaño:** más de 1,3 millones de registros  
- **Características:** variables socioeconómicas, financieras, de crédito, empleo y propósito del préstamo

## 7.12.3. Target
- **Variable binaria:** `loan_status`  
  - `0` = Fully Paid  
  - `1` = Charged Off (default)  

Generar la variable `default` basada en la columna `loan_status`:

```python
df["default"] = df["loan_status"].apply(lambda x: 1 if x == "Charged Off" else 0)
```

## 7.12.4. Exploración de datos (EDA)
- Cargar el dataset CSV
- Analizar la distribución de la variable `default`
- Verificar valores faltantes y tipos de datos

**Visualizaciones requeridas:**
- Histogramas de variables numéricas
- Boxplots por clase (`default`)
- Matriz de correlación o mapa de calor

## 7.12.5. scikit-learn
- Seleccionar variables relevantes: `loan_amnt`, `int_rate`, `fico_range_high`, `emp_length`, `annual_inc`, `purpose`, `home_ownership`, `dti`, `addr_state`, etc.
- Codificación de variables categóricas (`OneHotEncoder` o `LabelEncoder`)
- Escalado de variables numéricas con `StandardScaler`
- División train/test (`80/20`), estratificada por clase

## 7.12.6. PySpark
- Leer el CSV utilizando `SparkSession`
- Uso de `StringIndexer` y `OneHotEncoder` para variables categóricas
- Combinar columnas de entrada en una sola columna `features` usando `VectorAssembler`
- Escalar las variables con `StandardScaler` (opcional)
- Dividir en train y test usando `randomSplit`, estratificada si es posible

## 7.12.7. Modelado con scikit-learn
- Usar `MLPClassifier`
- Realizar búsqueda de hiperparámetros (`GridSearchCV`) para:
  - `hidden_layer_sizes` (por ejemplo: `[(10,), (50,), (100,)]`)
  - `alpha` (por ejemplo: `[0.0001, 0.001, 0.01]`)
  - `max_iter` (por ejemplo: `[100, 200]`)

**Métricas a evaluar:**
- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC
- Matriz de confusión

**Medir:** tiempo de entrenamiento y predicción

## 7.12.8. Modelado con PySpark
- Usar `pyspark.ml.classification.MultilayerPerceptronClassifier`
- Probar combinaciones de hiperparámetros:
  - Estructura de capas (`layers`): `[num_features, 10, 2]`, `[num_features, 50, 2]`, `[num_features, 100, 2]`
  - `stepSize`: `0.1`, `0.01`, `0.001`
  - `maxIter`: `100`, `200`
- Evaluar con `BinaryClassificationEvaluator`
- Calcular precisión, F1-score y matriz de confusión
- Medir el tiempo total de entrenamiento y predicción

## 7.12.9. Interpretabilidad con LIME
- Instalar LIME: `pip install lime`
- Seleccionar una o dos instancias clasificadas erróneamente
- Aplicar `lime.lime_tabular.LimeTabularExplainer` para analizar la(s) predicción(es)
- Visualizar variables más influyentes en la decisión del modelo
- Incluir gráficos de explicación local

## 7.12.10. Comparación de Resultados
- **Tabla comparativa con:**
  - Métricas para scikit-learn vs PySpark
  - Tiempo de cómputo (entrenamiento y predicción)

- **Gráficos:**
  - Curvas ROC
  - Gráficos comparativos de tiempo

## 7.12.11. Entregable
Un **Jupyter Book** bien documentado que incluya:
- Análisis exploratorio
- Preprocesamiento en ambos entornos
- Modelado y tuning de hiperparámetros
- Evaluación de métricas
- Interpretabilidad con LIME (con visualizaciones)
- Reflexión crítica sobre:
  - ¿Qué entorno fue más rápido?
  - ¿Cuál más preciso?
  - ¿Cuándo es útil PySpark?
  - ¿Qué aporta LIME?


