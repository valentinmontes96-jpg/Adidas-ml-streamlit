# Sistema de Predicción de Demanda - Adidas US Sales

Trabajo Práctico de Entrenamiento y Despliegue de Modelos de Machine Learning desarrollado en Python. El proyecto aborda el análisis exploratorio, preprocesamiento, entrenamiento comparativo y puesta en producción mediante **Streamlit** de un modelo predictivo basado en el conjunto de datos público **Adidas US Sales Dataset** (Kaggle).

---

## 1. Objetivo y Justificación del Proyecto

El objetivo principal consiste en desarrollar un modelo de **Aprendizaje Supervisado de Regresión** capaz de estimar la cantidad de **unidades a vender (`Units Sold`)** de un producto antes de que se efectúe la operación comercial, permitiendo proyectar de forma derivada la **facturación bruta estimada (`Total Sales`)**.

### Prevención de Fuga de Datos (Data Leakage)
El dataset original cuenta con 9.648 registros y 13 columnas que incluyen métricas financieras posventa como `Total Sales`, `Operating Profit` y `Operating Margin`. Dado que `Total Sales` es el resultado directo de multiplicar `Price per Unit * Units Sold`, incluir dichas variables como predictoras generaría *Data Leakage*, haciendo que el algoritmo aprenda una fórmula matemática trivial sobre datos que no existen antes de concretar la venta. 

Por este motivo, el modelo fue entrenado exclusivamente con variables conocidas *a priori* al momento de planificar un escenario comercial.

---

## 2. Limpieza y Preprocesamiento de Datos

El flujo de preparación de datos implementado en `entrenar_modelo.py` abarca las siguientes etapas:

1. **Ajuste de lectura del archivo fuente:** El archivo `Adidas US Sales Datasets.xlsx` presenta un encabezado visual en las primeras cuatro filas y una primera columna vacía. Se configuró la lectura con `skiprows=4` y se eliminó la columna residual `Unnamed: 0`.
2. **Normalización y tipos de datos:** Se verificó la ausencia de valores nulos mediante `.dropna()`, se limpiaron caracteres especiales (`$`, `,`) en los campos monetarios asegurando su conversión a formato numérico y se eliminaron espacios marginales (`.str.strip()`) en las columnas de texto.
3. **Ingeniería de Características (Feature Engineering):** A partir de la fecha de facturación (`Invoice Date`), se extrajeron las variables numéricas `Month` (mes del año, 1 a 12) y `Quarter` (trimestre fiscal, 1 a 4) para capturar la estacionalidad de las ventas. Se descartaron `Retailer ID` e `Invoice Date`.
4. **Transformación mediante Pipeline:** Se dividió el dataset en un **80% para entrenamiento (7.718 filas)** y un **20% para prueba (1.930 filas)** con `random_state=42`. Se implementó un `ColumnTransformer` integrado en un `Pipeline` de *Scikit-Learn* con:
   * `StandardScaler`: Para estandarizar las variables numéricas (`Price per Unit`, `Month`, `Quarter`).
   * `OneHotEncoder`: Para codificar las variables categóricas nominales (`Retailer`, `Region`, `State`, `City`, `Product`, `Sales Method`) con `handle_unknown="ignore"`.

---

## 3. Modelos Evaluados y Métricas de Desempeño

Se entrenaron y compararon dos algoritmos de regresión sobre el mismo conjunto de validación (20% de prueba):

* **Modelo Base (Alternativa): Regresión Lineal Regularizada (Ridge)**
  * Aplica penalización L2 (`alpha=1.0`) para mitigar la multicolinealidad tras la codificación *One-Hot*. Su limitación principal radica en asumir relaciones estrictamente lineales y aditivas entre las variables.
* **Modelo Principal Seleccionado: Random Forest Regressor**
  * Modelo de ensamble basado en árboles de decisión (`n_estimators=150`, `max_depth=20`). Captura interacciones no lineales complejas entre la ubicación geográfica, la categoría de producto, el canal de venta y el precio unitario.

### Resultados en el Conjunto de Prueba

| Modelo | MAE (Error Absoluto Medio) | RMSE (Raíz del Error Cuadrático Medio) | R2 (Coeficiente de Determinación) |
| :--- | :---: | :---: | :---: |
| **Regresión Lineal (Ridge)** | ~108.45 | ~143.12 | ~0.5510 |
| **Random Forest Regressor** | **~44.18** | **~69.85** | **~0.8935** |

El modelo **Random Forest Regressor** logra explicar cerca del **89.3% de la variabilidad** de las unidades vendidas y reduce el error absoluto medio a menos de la mitad respecto a la alternativa lineal, por lo que fue seleccionado y exportado en el archivo `modelo_adidas.joblib`.

---

## 4. Estructura del Repositorio

```text
├── .devcontainer/
├── AdidasDataset/
│   ├── Adidas US Sales Datasets.xlsx   # Conjunto de datos original (Kaggle)
│   ├── entrenar_modelo.py              # Limpieza, entrenamiento, evaluación y exportación
│   ├── modelo_adidas.joblib            # Pipeline entrenado y metadatos serializados
│   ├── app.py                          # Interfaz de usuario en Streamlit
│   └── requirements.txt                # Librerías necesarias para la ejecución
└── README.md                           # Documentación general del proyecto
```

---

## 5. Interfaz de Usuario en Streamlit

La aplicación web desarrollada en `app.py` se organiza en tres pestañas:

1. **Predicción Individual:** Permite simular un escenario comercial seleccionando Ubicación (Ciudad y Estado), Minorista, Método de venta, Producto, Precio por unidad y Mes. El sistema deduce automáticamente la Región geográfica y el Trimestre (`Quarter`), evitando el ingreso de datos redundantes, y devuelve las **Unidades estimadas a vender** junto con las **Ventas totales estimadas (USD)**.
2. **Carga por Archivo (CSV):** Permite descargar una plantilla CSV de referencia y cargar un archivo con múltiples combinaciones comerciales para obtener las predicciones por lote y descargar la tabla resultante.
3. **Métricas de los Modelos:** Expone la tabla comparativa de desempeño (`MAE`, `RMSE` y `R2`) obtenida durante el entrenamiento junto con la interpretación técnica de cada métrica.

---

## 6. Instalación y Ejecución Local

1. Clonar el repositorio e ingresar al directorio del proyecto:
   ```bash
   git clone [https://github.com/valentinmontes96-jpg/adidas-ml-streamlit.git](https://github.com/valentinmontes96-jpg/adidas-ml-streamlit.git)
   cd adidas-ml-streamlit/AdidasDataset
   ```

2. Instalar las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
   ```

3. (Opcional) Ejecutar el script de entrenamiento para regenerar el archivo `.joblib` y ver las métricas por consola:
   ```bash
   python entrenar_modelo.py
   ```

4. Iniciar la interfaz gráfica en Streamlit:
   ```bash
   python -m streamlit run app.py
   ```

---

## 7. Despliegue en la Nube

El proyecto se encuentra desplegado y accesible públicamente en **Streamlit Community Cloud**:
* **Enlace a la aplicación:** [Pegar acá la URL de tu app en Streamlit Cloud]
