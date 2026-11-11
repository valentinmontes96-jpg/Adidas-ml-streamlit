import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main():
  print("=" * 60)
  print("1. CARGA Y EXPLORACIÓN INICIAL DEL DATASET")
  print("=" * 60)

  archivo_excel = "Adidas US Sales Datasets.xlsx"

  # En el Excel original de Kaggle, las primeras 4 filas son decorativas
  df = pd.read_excel(archivo_excel, skiprows=4)

  # Eliminar la primera columna si está vacía (suele ser 'Unnamed: 0')
  if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

  print(f"Dimensiones iniciales del dataset: {df.shape}")
  print("\nValores nulos por columna antes de limpiar:")
  print(df.isnull().sum())

  # Eliminamos filas completamente vacías o con nulos si existieran
  df = df.dropna().copy()

  print("\n" + "=" * 60)
  print("2. LIMPIEZA Y PREPROCESAMIENTO DE DATOS")
  print("=" * 60)

  # Asegurar limpieza de tipos numéricos (por si alguna celda vino como texto con '$' o ',')
  cols_numericas_raw = [
      "Price per Unit",
      "Units Sold",
      "Total Sales",
      "Operating Profit",
  ]
  for col in cols_numericas_raw:
    if df[col].dtype == "object":
      df[col] = (
          df[col].astype(str).str.replace("$", "", regex=False)
          .str.replace(",", "", regex=False).str.strip()
      )
      df[col] = pd.to_numeric(df[col], errors="coerce")

  # Ingeniería de características (Feature Engineering) sobre la fecha
  df["Invoice Date"] = pd.to_datetime(df["Invoice Date"])
  df["Month"] = df["Invoice Date"].dt.month
  df["Quarter"] = df["Invoice Date"].dt.quarter

  # Limpieza de espacios en blanco en columnas categóricas
  cols_categoricas = [
      "Retailer",
      "Region",
      "State",
      "City",
      "Product",
      "Sales Method",
  ]
  for col in cols_categoricas:
    df[col] = df[col].astype(str).str.strip()

  print("Tipos de datos luego de la limpieza:")
  print(
      df[
          cols_categoricas + ["Price per Unit", "Month", "Quarter", "Units Sold"]
      ].dtypes
  )
  print("\nEstadísticas descriptivas de las variables numéricas clave:")
  print(df[["Price per Unit", "Units Sold", "Total Sales"]].describe().round(2))

  # Definición de variables predictoras (X) y variable objetivo (y)
  # Se excluyen 'Total Sales', 'Operating Profit' y 'Operating Margin' para evitar Data Leakage
  features_cat = [
      "Retailer",
      "Region",
      "State",
      "City",
      "Product",
      "Sales Method",
  ]
  features_num = ["Price per Unit", "Month", "Quarter"]

  X = df[features_cat + features_num]
  y = df["Units Sold"]

  # División en entrenamiento (80%) y prueba (20%)
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.20, random_state=42
  )
  print(f"\nDatos de entrenamiento: {X_train.shape[0]} registros")
  print(f"Datos de prueba: {X_test.shape[0]} registros")

  # Preprocesador con ColumnTransformer (Estandarización + One-Hot Encoding)
  preprocesador = ColumnTransformer(
      transformers=[
          ("num", StandardScaler(), features_num),
          (
              "cat",
              OneHotEncoder(handle_unknown="ignore", sparse_output=False),
              features_cat,
          ),
      ]
  )

  print("\n" + "=" * 60)
  print("3. ENTRENAMIENTO Y COMPARACIÓN DE MODELOS")
  print("=" * 60)

  # Modelo 1 (Alternativa Base): Regresión Lineal Regularizada (Ridge)
  pipeline_ridge = Pipeline(
      steps=[
          ("preprocessor", preprocesador),
          ("model", Ridge(alpha=1.0, random_state=42)),
      ]
  )

  # Modelo 2 (Modelo Principal): Random Forest Regressor
  pipeline_rf = Pipeline(
      steps=[
          ("preprocessor", preprocesador),
          (
              "model",
              RandomForestRegressor(
                  n_estimators=150,
                  max_depth=20,
                  random_state=42,
                  n_jobs=-1,
              ),
          ),
      ]
  )

  modelos = {
      "Regresión Lineal (Ridge)": pipeline_ridge,
      "Random Forest Regressor": pipeline_rf,
  }

  resultados = {}

  for nombre, pipeline in modelos.items():
    print(f"\nEntrenando {nombre}...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    resultados[nombre] = {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
    }

  print("\n" + "=" * 60)
  print("4. MÉTRICAS DE DESEMPEÑO OBTENIDAS (PARA EL INFORME)")
  print("=" * 60)
  df_resultados = pd.DataFrame(resultados).T
  print(df_resultados)

  print("\n" + "=" * 60)
  print("5. GUARDADO DEL MODELO Y METADATOS PARA STREAMLIT")
  print("=" * 60)

  # Guardamos jerarquías geográficas y opciones únicas para que los desplegables
  # de Streamlit sean dinámicos (ej: al elegir una Región, filtre sus Estados y Ciudades)
  jerarquia_geo = {}
  for region in sorted(df["Region"].unique()):
    df_reg = df[df["Region"] == region]
    jerarquia_geo[region] = {}
    for state in sorted(df_reg["State"].unique()):
      ciudades = sorted(df_reg[df_reg["State"] == state]["City"].unique().tolist())
      jerarquia_geo[region][state] = ciudades

  artefacto_streamlit = {
      "pipeline": pipeline_rf,  # Guardamos el Random Forest que es el superador
      "metricas_comparativas": resultados,
      "retailers": sorted(df["Retailer"].unique().tolist()),
      "productos": sorted(df["Product"].unique().tolist()),
      "metodos_venta": sorted(df["Sales Method"].unique().tolist()),
      "jerarquia_geo": jerarquia_geo,
      "precio_min": float(df["Price per Unit"].min()),
      "precio_max": float(df["Price per Unit"].max()),
      "precio_medio": float(df["Price per Unit"].median()),
  }

  nombre_salida = "modelo_adidas.joblib"
  joblib.dump(artefacto_streamlit, nombre_salida, compress=3)
  print(f"Archivo '{nombre_salida}' generado exitosamente.")


if __name__ == "__main__":
  main()