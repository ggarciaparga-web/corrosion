import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from calculos.tiempo import calcular_iniciacion
from calculos.CONTEVECT import ejecutar_simulacion_completa
from calculos.ModelCode import simulacion_total

# CONFIGURACIÓN DE PÁGINA
st.set_page_config(page_title="Visor de Corrosión Estructural", layout="wide")

st.title("🏗️ Modelos de Corrosión de Estructuras")

# ==========================================
# BARRA LATERAL: INPUTS
# ==========================================
st.sidebar.header("⚙️ Parámetros de Entrada")

tipo_ataque = st.sidebar.selectbox("Tipo de Análisis", ["Carbonatación", "Ataque por Cloruros"])

with st.sidebar.expander("📐 Geometría y Materiales", expanded=True):
    recubrimiento = st.sidebar.number_input("Recubrimiento (mm)", value=30.0)
    t_analisis = st.sidebar.slider("Tiempo total (años)", 50, 700, 250)
    ancho_b = st.sidebar.number_input("Ancho sección b (mm)", value=150)
    canto_d = st.sidebar.number_input("Canto útil d (mm)", value=300)
    phi_base = st.sidebar.number_input("Diámetro barras inf. (mm)", value=20)
    n_barras = st.sidebar.number_input("Número de barras inferiores", value=2)
    fck = st.sidebar.number_input("Resistencia fck (MPa)", value=25)
    fy = st.sidebar.number_input("Límite elástico fy (MPa)", value=500)
    r2 = st.sidebar.number_input("Distancia armadura superior r2 (mm)", value=20)

inputs = {
    't_analisis': t_analisis, 'recubrimiento': recubrimiento, 
    'ancho_b': ancho_b, 'canto_d': canto_d, 'phi_base': phi_base,
    'n_barras': n_barras, 'fck': fck, 'fy': fy, 'r2': r2
}

if tipo_ataque == "Carbonatación":
    inputs['c_cemento'] = st.sidebar.number_input("Contenido cemento (kg/m3)", value=450.0)
    inputs['cs_co2'] = st.sidebar.number_input("Concentración Cs CO2 (mg/m3)", value=800.0)
    inputs['i_corr'] = st.sidebar.number_input("Intensidad i_corr (μA/cm2)", value=0.1)
else:
    inputs['i_corr'] = st.sidebar.number_input("Intensidad i_corr (μA/cm2)", value=2.58)

# ==========================================
# PROCESAMIENTO Y GRÁFICAS (ESTÁTICAS)
# ==========================================
try:
    ti = calcular_iniciacion(tipo_ataque, inputs)
    df_cv, t_v_cv, lim_cv = ejecutar_simulacion_completa(tipo_ataque, inputs, ti)
    df_mc, t_v_mc = simulacion_total(tipo_ataque, inputs, ti)

    tab1, tab2 = st.tabs(["📊 CONTEVECT", "💻 Model Code"])

    with tab1:
        st.header(f"Lógica CONTEVECT - ti: {ti:.2f} años")
        c1, c2 = st.columns(2)
        
        with c1:
            fig1, ax1 = plt.subplots(figsize=(6,4))
            ax1.plot(df_cv["Tiempo (y)"], df_cv["Px (mm)"], color='blue', label='$P_x$')
            ax1.axvline(x=ti, color='red', linestyle='--', label='$t_i$')
            ax1.axvline(x=t_v_cv, color='black', linestyle=':', label='Límite')
            ax1.set_title("Penetración Px"); ax1.grid(True, alpha=0.3); ax1.legend()
            st.pyplot(fig1)

        with c2:
            fig2, ax2 = plt.subplots(figsize=(6,4))
            ax2.plot(df_cv["Tiempo (y)"], df_cv["Mu (kNm)"], color='navy')
            ax2.set_title("Capacidad Mrd"); ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)

    with tab2:
        st.header(f"Model Code 2023 - ti: {ti:.2f} años")
        fig3, ax3 = plt.subplots(figsize=(10,4))
        ax3.plot(df_mc["Time"], df_mc["Mu (kNm)"], label="Standard", color="navy")
        ax3.plot(df_mc["Time"], df_mc["Mu Cons (kNm)"], label="Cons.", color="orange", ls="--")
        ax3.axvline(x=ti, color="red", ls="--")
        ax3.axvline(x=t_v_mc, color="black", ls=":")
        ax3.set_title("Resistencia Residual"); ax3.grid(True, alpha=0.3); ax3.legend()
        st.pyplot(fig3)

except Exception as e:
    st.error(f"Error al conectar archivos: {e}")
