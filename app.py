import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- IMPORTACIONES DE TUS MÓDULOS ---
from calculos.tiempo import calcular_iniciacion
from calculos.CONTEVECT import ejecutar_simulacion_completa
from calculos.ModelCode import simulacion_total

# CONFIGURACIÓN DE PÁGINA
st.set_page_config(page_title="Visor de Corrosión Estructural", layout="wide")

st.title("🏗️ Modelos de Corrosión de Estructuras")

# =========================================================
# BARRA LATERAL: INPUTS ORGANIZADOS
# =========================================================
st.sidebar.header("⚙️ Parámetros de Entrada")

tipo_ataque = st.sidebar.selectbox("Tipo de Análisis", ["Carbonatación", "Cloruros"])

with st.sidebar.expander("📐 Geometría y Materiales", expanded=True):
    recubrimiento = st.sidebar.number_input("Recubrimiento (mm)", value=30.0)
    t_analisis = st.sidebar.slider("Tiempo total de estudio (años)", 50, 500, 100)
    ancho_b = st.sidebar.number_input("Ancho sección b (mm)", value=150)
    canto_d = st.sidebar.number_input("Canto útil d (mm)", value=300)
    phi_base = st.sidebar.number_input("Diámetro barras inf. (mm)", value=20)
    n_barras = st.sidebar.number_input("Número de barras inferiores", value=2)
    fck = st.sidebar.number_input("Resistencia fck (MPa)", value=25)
    fy = st.sidebar.number_input("Límite elástico fy (MPa)", value=500)
    r2 = st.sidebar.number_input("Recubrimiento superior r2 (mm)", value=20)

# Diccionario base de inputs
inputs_calculo = {
    't_analisis': t_analisis, 'recubrimiento': recubrimiento, 
    'ancho_b': ancho_b, 'canto_d': canto_d, 'phi_base': phi_base,
    'n_barras': n_barras, 'fck': fck, 'fy': fy, 'r2': r2
}

# Inputs específicos según el ataque
if tipo_ataque == "Carbonatación":
    st.sidebar.subheader("☁️ Parámetros Carbonatación")
    inputs_calculo['c_cemento'] = st.sidebar.number_input("Contenido cemento (kg/m3)", value=450.0)
    inputs_calculo['cs_co2'] = st.sidebar.number_input("Concentración Cs CO2 (mg/m3)", value=800.0)
    inputs_calculo['i_corr'] = st.sidebar.number_input("Intensidad i_corr (μA/cm2)", value=0.1)
    i_corr_val = inputs_calculo['i_corr']
else:
    st.sidebar.subheader("🌊 Parámetros Cloruros")
    inputs_calculo['i_corr'] = st.sidebar.number_input("Intensidad i_corr (μA/cm2)", value=2.58)
    i_corr_val = inputs_calculo['i_corr']



# =========================================================
# PROCESAMIENTO Y PESTAÑAS
# =========================================================
try:
    # 1. Obtenemos el ti y los datos de Px desde tiempo.py
    ti, tiempos_px, px_plot = calcular_iniciacion(tipo_ataque, inputs_calculo)

    # 2. Ejecutamos los modelos de resistencia
    df_cv, t_v_cv, lim_cv = ejecutar_simulacion_completa(tipo_ataque, inputs_calculo, ti)
    df_mc, t_v_mc = simulacion_total(tipo_ataque, inputs_calculo, ti)

    tab1, tab2, tab3 = st.tabs(["📊 CONTEVECT", "🏗️ Model Code", "📚 Teoría"])

    with tab1:
        st.header(f"Análisis CONTEVECT ({tipo_ataque})")
        st.write(f"**Tiempo de iniciación:** {ti:.2f} años")
        
        col1, col2 = st.columns(2)
        with col1:
            # Gráfica Px (Sin zoom)
            fig_px, ax_px = plt.subplots(figsize=(6, 4))
            ax_px.plot(tiempos_px, px_plot, color='blue', lw=2, label='$P_x$')
            ax_px.axvline(x=ti, color='red', linestyle='--', label=f'$t_i$ = {ti:.2f} y')
            ax_px.axvline(x=t_v_cv, color='black', linestyle=':', label='Límite')
            ax_px.set_title("Penetración de Corrosión"); ax_px.set_xlabel("Años"); ax_px.legend(); ax_px.grid(True, alpha=0.3)
            st.pyplot(fig_px)

        with col2:
            # Gráfica Mrd CONTEVECT (Sin zoom)
            fig_cv, ax_cv = plt.subplots(figsize=(6, 4))
            ax_cv.plot(df_cv["Tiempo (y)"], df_cv["Mu (kNm)"], color='navy', lw=2)
            ax_cv.set_title("Mrd vs Tiempo (CONTEVECT)"); ax_cv.set_xlabel("Años"); ax_cv.grid(True, alpha=0.3)
            st.pyplot(fig_cv)

    with tab2:
        st.header(f"Análisis Model Code 2023")
        st.write(f"**Tiempo de iniciación ($t_i$):** {ti:.2f} años")

        # Fila 1: Px y Área
        col_a, col_b = st.columns(2)
        
        with col_a:
            # 1. Gráfica Px vs Tiempo
            fig_mc_px, ax_mc_px = plt.subplots(figsize=(6, 4))
            ax_mc_px.plot(df_mc["Time"], df_mc["Px"], color="blue", lw=2)
            ax_mc_px.axvline(x=ti, color="red", ls="--", label="ti")
            ax_mc_px.axvline(x=t_v_mc, color="black", ls=":", label="Límite")
            ax_mc_px.set_title("Penetración Px [mm]")
            ax_mc_px.set_xlabel("Años")
            ax_mc_px.grid(True, alpha=0.3)
            st.pyplot(fig_mc_px)

        with col_b:
            # 3. Gráfica Área vs Tiempo
            fig_mc_area, ax_mc_area = plt.subplots(figsize=(6, 4))
            ax_mc_area.plot(df_mc["Time"], df_mc["A_corr"], color="darkgreen", lw=2)
            ax_mc_area.axvline(x=ti, color="red", ls="--")
            ax_mc_area.set_title("Área de Armadura [mm²]")
            ax_mc_area.set_xlabel("Años")
            ax_mc_area.grid(True, alpha=0.3)
            st.pyplot(fig_mc_area)

        # Fila 2: Capacidad Mrd (El que ya tenías)
        st.subheader("Capacidad Resistente")
        fig_mc, ax_mc = plt.subplots(figsize=(10, 4))
        ax_mc.plot(df_mc["Time"], df_mc["Mu (kNm)"], label="Estándar", color="navy", lw=2)
        ax_mc.plot(df_mc["Time"], df_mc["Mu Cons (kNm)"], label="Conservador", color="orange", ls="--", lw=2)
        ax_mc.axvline(x=ti, color="red", ls="--", label=f"ti={ti:.1f}")
        ax_mc.axvline(x=t_v_mc, color="black", ls=":", label="Límite")
        ax_mc.set_title("Resistencia Residual Mrd [kNm]")
        ax_mc.set_xlabel("Años")
        ax_mc.legend()
        ax_mc.grid(True, alpha=0.3)
        st.pyplot(fig_mc)

    with tab3:
        st.subheader("Detalles del cálculo")
        st.write(f"Intensidad de corrosión considerada: **{i_corr_val} μA/cm²**")
        st.write(f"Límite vertical para {tipo_ataque}: **{lim_cv*1000:.0f} μm**")
        st.dataframe(df_cv[["Tiempo (y)", "Px (mm)", "Mu (kNm)"]].tail(10))

except Exception as e:
    st.error(f"Error en la conexión de módulos: {e}")
    st.info("Revisa que los archivos en /calculos devuelvan los valores correctos.")
