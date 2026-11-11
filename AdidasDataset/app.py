import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Ventas Adidas | ML",
    page_icon="👟",
    layout="wide",
)


# Carga del modelo y metadatos en caché para que la app sea rápida
@st.cache_resource
def cargar_artefacto():
  return joblib.load("modelo_adidas.joblib")


try:
  datos_modelo = cargar_artefacto()
except FileNotFoundError:
  st.error(
      "⚠️ No se encontró el archivo 'modelo_adidas.joblib'. Ejecutá primero"
      " `python entrenar_modelo.py`."
  )
  st.stop()

pipeline = datos_modelo["pipeline"]
metricas = datos_modelo["metricas_comparativas"]
retailers = datos_modelo["retailers"]
productos = datos_modelo["productos"]
metodos_venta = datos_modelo["metodos_venta"]
jerarquia_geo = datos_modelo["jerarquia_geo"]
precio_min = datos_modelo["precio_min"]
precio_max = datos_modelo["precio_max"]
precio_medio = datos_modelo["precio_medio"]

# Encabezado principal
st.title("👟 Sistema de Predicción de Demanda - Adidas US Sales")
st.markdown("""
Esta aplicación utiliza un modelo de **Machine Learning (Random Forest Regressor)** entrenado sobre el conjunto de datos históricos de ventas de Adidas en EE. UU. para estimar las **unidades a vender (`Units Sold`)** y la **facturación bruta proyectada (`Total Sales`)**.
""")

# Pestañas para organizar la interfaz y cumplir con la exposición
tab_prediccion, tab_lote, tab_metricas = st.tabs([
    "🎯 Simulador Individual",
    "📂 Carga de Nuevos Ejemplos (CSV)",
    "📊 Desempeño y Comparación de Modelos",
])

# ==========================================
# PESTAÑA 1: SIMULADOR INDIVIDUAL
# ==========================================
with tab_prediccion:
  st.subheader("Configuración del Escenario Comercial")
  st.caption(
      "Seleccioná las características del producto, ubicación geográfica y"
      " canal de venta para predecir su demanda."
  )

  col1, col2, col3 = st.columns(3)

  with col1:
    st.markdown("**📍 Ubicación Geográfica**")
    region_sel = st.selectbox("Región (Region)", options=list(jerarquia_geo.keys()))
    estados_disponibles = list(jerarquia_geo[region_sel].keys())
    state_sel = st.selectbox("Estado (State)", options=estados_disponibles)
    ciudades_disponibles = jerarquia_geo[region_sel][state_sel]
    city_sel = st.selectbox("Ciudad (City)", options=ciudades_disponibles)

  with col2:
    st.markdown("**🏪 Producto y Canal Comercial**")
    retailer_sel = st.selectbox("Minorista (Retailer)", options=retailers)
    product_sel = st.selectbox("Categoría de Producto (Product)", options=productos)
    method_sel = st.selectbox("Método de Venta (Sales Method)", options=metodos_venta)

  with col3:
    st.markdown("**💲 Precio y Estacionalidad**")
    price_sel = st.slider(
        "Precio por Unidad (USD)",
        min_value=float(max(5.0, precio_min)),
        max_value=float(precio_max * 1.2),
        value=float(precio_medio),
        step=1.0,
    )

    meses_nombres = {
        1: "1 - Enero",
        2: "2 - Febrero",
        3: "3 - Marzo",
        4: "4 - Abril",
        5: "5 - Mayo",
        6: "6 - Junio",
        7: "7 - Julio",
        8: "8 - Agosto",
        9: "9 - Septiembre",
        10: "10 - Octubre",
        11: "11 - Noviembre",
        12: "12 - Diciembre",
    }
    month_sel = st.selectbox(
        "Mes proyectado de venta",
        options=list(meses_nombres.keys()),
        format_func=lambda x: meses_nombres[x],
        index=5,
    )

    # Cálculo automático del trimestre según el mes elegido
    quarter_sel = (month_sel - 1) // 3 + 1
    st.info(f"📅 Trimestre calculado automáticamente: **Q{quarter_sel}**")

  st.divider()

  if st.button("🚀 Predecir Demanda y Ventas", type="primary", use_container_width=True):
    # Armar el DataFrame con los mismos nombres de columnas usados en el entrenamiento
    nuevo_ejemplo = pd.DataFrame([{
        "Retailer": retailer_sel,
        "Region": region_sel,
        "State": state_sel,
        "City": city_sel,
        "Product": product_sel,
        "Sales Method": method_sel,
        "Price per Unit": price_sel,
        "Month": month_sel,
        "Quarter": quarter_sel,
    }])

    # Predicción
    unidades_pred = pipeline.predict(nuevo_ejemplo)[0]
    unidades_redondeadas = int(round(max(0, unidades_pred)))
    ventas_estimadas = unidades_redondeadas * price_sel

    st.subheader("📈 Resultados de la Predicción")
    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
      st.metric(
          label="Unidades Estimadas a Vender (Units Sold)",
          value=f"{unidades_redondeadas:,} u.",
      )
    with res_col2:
      st.metric(
          label="Facturación Bruta Estimada (Total Sales)",
          value=f"USD ${ventas_estimadas:,.2f}",
      )
    with res_col3:
      st.metric(
          label="Precio Unitario Aplicado",
          value=f"USD ${price_sel:,.2f}",
      )

    with st.expander("Ver vector de entrada procesado por el Pipeline"):
      st.dataframe(nuevo_ejemplo, use_container_width=True)

# ==========================================
# PESTAÑA 2: CARGA DE LOTE POR CSV
# ==========================================
with tab_lote:
  st.subheader("Procesamiento de Múltiples Ejemplos mediante archivo CSV")
  st.markdown("""
  Podés descargar la plantilla de ejemplo, agregarle nuevas filas con los escenarios que quieras evaluar y subirla acá para que el modelo las procese en bloque.
  """)

  df_plantilla = pd.DataFrame([
      {
          "Retailer": "Foot Locker",
          "Region": "Northeast",
          "State": "New York",
          "City": "New York",
          "Product": "Men's Street Footwear",
          "Sales Method": "In-store",
          "Price per Unit": 50.0,
          "Month": 12,
          "Quarter": 4,
      },
      {
          "Retailer": "Amazon",
          "Region": "West",
          "State": "California",
          "City": "Los Angeles",
          "Product": "Women's Apparel",
          "Sales Method": "Online",
          "Price per Unit": 65.0,
          "Month": 7,
          "Quarter": 3,
      },
  ])

  csv_plantilla = df_plantilla.to_csv(index=False).encode("utf-8")
  st.download_button(
      label="⬇️ Descargar plantilla CSV de ejemplo",
      data=csv_plantilla,
      file_name="ejemplos_adidas_input.csv",
      mime="text/csv",
  )

  archivo_subido = st.file_uploader(
      "Subí tu archivo CSV con nuevos ejemplos", type=["csv"]
  )

  if archivo_subido is not None:
    try:
      df_nuevos = pd.read_csv(archivo_subido)
      columnas_requeridas = [
          "Retailer",
          "Region",
          "State",
          "City",
          "Product",
          "Sales Method",
          "Price per Unit",
          "Month",
          "Quarter",
      ]

      faltantes = [c for c in columnas_requeridas if c not in df_nuevos.columns]
      if faltantes:
        st.error(f"Al archivo CSV le faltan las siguientes columnas: {faltantes}")
      else:
        predicciones = np.round(pipeline.predict(df_nuevos[columnas_requeridas])).astype(int)
        df_nuevos["Unidades_Predichas"] = np.maximum(0, predicciones)
        df_nuevos["Ventas_Estimadas_USD"] = (
            df_nuevos["Unidades_Predichas"] * df_nuevos["Price per Unit"]
        ).round(2)

        st.success("✅ ¡Predicciones generadas correctamente!")
        st.dataframe(df_nuevos, use_container_width=True)

        csv_resultado = df_nuevos.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Descargar resultados en CSV",
            data=csv_resultado,
            file_name="predicciones_adidas_resultado.csv",
            mime="text/csv",
        )
    except Exception as e:
      st.error(f"Error al procesar el archivo: {e}")

# ==========================================
# PESTAÑA 3: MÉTRICAS Y COMPARACIÓN DE MODELOS
# ==========================================
with tab_metricas:
  st.subheader("Evaluación de Desempeño (Set de Prueba - 20%)")
  st.markdown("""
  Para cumplir con el diseño experimental, se comparó un modelo lineal regularizado (**Regresión Lineal Ridge**) frente a un modelo de ensamble basado en árboles de decisión (**Random Forest Regressor**).
  """)

  df_metricas = pd.DataFrame(metricas).T
  df_metricas.columns = [
      "MAE (Error Absoluto Medio)",
      "RMSE (Error Cuadrático Medio)",
      "R² (Coeficiente de Determinación)",
  ]

  st.table(df_metricas)

  st.info("""
  **Interpretación técnica:**
  * **MAE:** Indica en promedio por cuántas unidades difiere la predicción del valor real.
  * **RMSE:** Penaliza con mayor peso los desvíos grandes en volúmenes de venta.
  * **R²:** Representa la proporción de la varianza de las unidades vendidas que logra explicar el modelo (mientras más cercano a 1.0, mejor ajuste). El **Random Forest Regressor** supera ampliamente a la alternativa lineal al capturar relaciones no lineales entre regiones, precios y canales de venta.
  """)