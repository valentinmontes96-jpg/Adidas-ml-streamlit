import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Predicción de Ventas - Adidas",
    layout="wide",
)


@st.cache_resource
def cargar_artefacto():
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_modelo = os.path.join(ruta_actual, "modelo_adidas.joblib")
    return joblib.load(ruta_modelo)


try:
    datos_modelo = cargar_artefacto()
except FileNotFoundError:
    st.error(
        "No se encontró el archivo 'modelo_adidas.joblib'. Ejecutá primero"
        " python entrenar_modelo.py."
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

st.title("Predicción de Demanda - Adidas US Sales")
st.write(
    "Aplicación para estimar las unidades vendidas y el total de ventas"
    " a partir del modelo Random Forest entrenado."
)

tab_prediccion, tab_lote, tab_metricas = st.tabs([
    "Predicción Individual",
    "Carga por Archivo (CSV)",
    "Métricas de los Modelos",
])

# ==========================================
# PESTAÑA 1: PREDICCIÓN INDIVIDUAL
# ==========================================
with tab_prediccion:
    st.subheader("Parámetros de entrada")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Ubicación**")
        region_sel = st.selectbox("Región", options=list(jerarquia_geo.keys()))
        estados_disponibles = list(jerarquia_geo[region_sel].keys())
        state_sel = st.selectbox("Estado", options=estados_disponibles)
        ciudades_disponibles = jerarquia_geo[region_sel][state_sel]
        city_sel = st.selectbox("Ciudad", options=ciudades_disponibles)

    with col2:
        st.markdown("**Producto y Canal**")
        retailer_sel = st.selectbox("Minorista (Retailer)", options=retailers)
        product_sel = st.selectbox("Producto", options=productos)
        method_sel = st.selectbox("Método de venta", options=metodos_venta)

    with col3:
        st.markdown("**Precio y Fecha**")
        price_sel = st.number_input(
            "Precio por unidad (USD)",
            min_value=float(max(5.0, precio_min)),
            max_value=float(precio_max * 1.5),
            value=float(precio_medio),
            step=1.0,
        )

        meses_nombres = {
            1: "Enero",
            2: "Febrero",
            3: "Marzo",
            4: "Abril",
            5: "Mayo",
            6: "Junio",
            7: "Julio",
            8: "Agosto",
            9: "Septiembre",
            10: "Octubre",
            11: "Noviembre",
            12: "Diciembre",
        }
        month_sel = st.selectbox(
            "Mes",
            options=list(meses_nombres.keys()),
            format_func=lambda x: meses_nombres[x],
            index=5,
        )

        quarter_sel = (month_sel - 1) // 3 + 1
        st.caption(f"Trimestre correspondiente: Q{quarter_sel}")

    st.divider()

    if st.button("Calcular predicción", type="primary"):
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

        unidades_pred = pipeline.predict(nuevo_ejemplo)[0]
        unidades_redondeadas = int(round(max(0, unidades_pred)))
        ventas_estimadas = unidades_redondeadas * price_sel

        st.subheader("Resultados")
        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.metric(
                label="Unidades estimadas (Units Sold)",
                value=f"{unidades_redondeadas}",
            )
        with res_col2:
            st.metric(
                label="Ventas totales estimadas (Total Sales)",
                value=f"USD {ventas_estimadas:,.2f}",
            )
        with res_col3:
            st.metric(
                label="Precio unitario",
                value=f"USD {price_sel:,.2f}",
            )

        with st.expander("Ver datos de entrada enviados al modelo"):
            st.dataframe(nuevo_ejemplo, use_container_width=True)

# ==========================================
# PESTAÑA 2: CARGA POR LOTE (CSV)
# ==========================================
with tab_lote:
    st.subheader("Cargar nuevos ejemplos desde un archivo CSV")
    st.write(
        "El archivo debe contener las columnas: Retailer, Region, State, City,"
        " Product, Sales Method, Price per Unit, Month, Quarter."
    )

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
        label="Descargar CSV de ejemplo",
        data=csv_plantilla,
        file_name="ejemplo_entrada.csv",
        mime="text/csv",
    )

    archivo_subido = st.file_uploader("Seleccionar archivo CSV", type=["csv"])

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

            faltantes = [
                c for c in columnas_requeridas if c not in df_nuevos.columns
            ]
            if faltantes:
                st.error(f"Faltan las siguientes columnas en el CSV: {faltantes}")
            else:
                predicciones = np.round(
                    pipeline.predict(df_nuevos[columnas_requeridas])
                ).astype(int)
                df_nuevos["Unidades_Predichas"] = np.maximum(0, predicciones)
                df_nuevos["Ventas_Estimadas_USD"] = (
                    df_nuevos["Unidades_Predichas"] * df_nuevos["Price per Unit"]
                ).round(2)

                st.write("Resultados procesados:")
                st.dataframe(df_nuevos, use_container_width=True)

                csv_resultado = df_nuevos.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Descargar resultados en CSV",
                    data=csv_resultado,
                    file_name="predicciones_resultado.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")

# ==========================================
# PESTAÑA 3: MÉTRICAS
# ==========================================
with tab_metricas:
    st.subheader("Comparación de modelos entrenados")
    st.write(
        "Resultados obtenidos sobre el conjunto de prueba (20% de los datos)"
        " comparando Regresión Lineal (Ridge) contra Random Forest Regressor."
    )

    df_metricas = pd.DataFrame(metricas).T
    df_metricas.columns = ["MAE", "RMSE", "R2"]

    st.table(df_metricas)

    st.info("""
    **Interpretación de las métricas:**
    * **MAE (Error Absoluto Medio):** Indica en promedio por cuántas unidades difiere la predicción del valor real.
    * **RMSE (Raíz del Error Cuadrático Medio):** Penaliza con mayor peso los errores grandes en las unidades vendidas.
    * **R2 (Coeficiente de Determinación):** Representa qué proporción de la variación de las ventas explica el modelo (mientras más cercano a 1.0, mejor ajuste).
    """)
