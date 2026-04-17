import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# IMPORTANTE: Asegúrate de que los archivos en /calculos terminen en .py
from calculos.tiempo_iniciacion import calcular_tiempo-iniciacion
from calculos.CONTEVECT import ejecutar_CONTEVECT
from calculos.ModelCode import ejecutar_ModelCode

# Configuración de página
st.set_page_config(page_title="Visor de Corrosión Estructural", layout="wide")

# --- BARRA LATERAL (INPUTS) ---
st.sidebar.header("🛠️ Configuración Global")

with st.sidebar.expander("🏗️ Sección y Materiales", expanded=True):
    ancho_b = st.number_input("Ancho b (mm)", value=150)
    canto_d = st.number_input("Canto d (mm)", value=300)
    recubrimiento = st.number_input("Recubrimiento (mm)", value=20)
    fck = st.slider("fck (MPa)", 20, 50, 25)
    fy = st.number_input("fy (MPa)", value=500)

with st.sidebar.expander("🔢 Armaduras", expanded=True):
    n_barras = st.number_input("Nº barras inferiores", value=2, min_value=1)
    phi_base = st.number_input("Diámetro barras (mm)", value=20)
    r2 = st.number_input("Distancia armadura superior r2 (mm)", value=20)

with st.sidebar.expander("🧬 Parámetros de Corrosión", expanded=True):
    tipo_ataque = st.radio("Tipo de ataque", ["Carbonatación", "Cloruros"])
    i_corr = st.number_input("Intensidad i_corr (μA/cm2)", value=2.0)
    t_analisis = st.slider("Tiempo de análisis (años)", 10, 150, 100)
    
    inputs_calculo = {
        'ancho_b': ancho_b, 'canto_d': canto_d, 'recubrimiento': recubrimiento,
        'fck': fck, 'fy': fy, 'n_barras': n_barras, 'phi_base': phi_base,
        'r2': r2, 'i_corr': i_corr, 't_analisis': t_analisis
    }
    
    if tipo_ataque == "Carbonatación":
        inputs_calculo['c_cemento'] = st.number_input("Cemento (kg/m3)", value=350)
        inputs_calculo['cs_co2'] = st.selectbox("Ambiente (Cs)", [800, 600], format_func=lambda x: "Interior (800)" if x==800 else "Exterior (600)")

# --- DIBUJO DE LA SECCIÓN ---
def dibujar_seccion():
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=ancho_b, y1=canto_d, line=dict(color="Black"), fillcolor="LightGrey")
    for i in range(n_barras):
        x_pos = (ancho_b / (n_barras + 1)) * (i + 1)
        fig.add_shape(type="circle", x0=x_pos-phi_base/2, y0=recubrimiento, x1=x_pos+phi_base/2, y1=recubrimiento+phi_base, fillcolor="Red")
    fig.update_layout(width=200, height=300, margin=dict(l=10, r=10, t=10, b=10), xaxis_visible=False, yaxis_visible=False)
    return fig

st.sidebar.write("### Esquema de Sección")
st.sidebar.plotly_chart(dibujar_seccion())

# --- PROCESAMIENTO ---
# 1. Calculamos TI
ti = calcular_iniciacion(tipo_ataque, inputs_calculo)

# 2. Calculamos los dos modelos
df_contevect = ejecutar_contevect(tipo_ataque, inputs_calculo, ti)
df_modelcode = ejecutar_model_code(tipo_ataque, inputs_calculo, ti)

# 3. Calculamos t_vertical (50 o 500 μm)
limite_um = 0.05 if tipo_ataque == "Carbonatación" else 0.5
t_vert = ti + (limite_um / (0.0116 * i_corr))

# --- CUERPO PRINCIPAL ---
st.title("🛡️ Simulación de Corrosión")

tab_teoria, tab_contevect, tab_modelcode = st.tabs(["📚 Teoría", "📊 CONTEVECT", "🏗️ Model Code 2023"])

with tab_teoria:
    st.markdown(f"""
    ## Fundamentos Teóricos
    - **Tiempo de Iniciación ($t_i$):** {ti:.2f} años.
    - **Fase de Propagación:** Comienza tras el año $t_i$.
    - **Límite de Servicio:** Recta vertical calculada para una pérdida de sección de {limite_um*1000:.0f} μm.
    """)

with tab_contevect:
    col1, col2 = st.columns(2)
    # Px vs Time
    fig_px = go.Figure()
    fig_px.add_trace(go.Scatter(x=df_contevect["Tiempo (y)"], y=df_contevect["Px (mm)"], name="Px"))
    fig_px.add_vline(x=ti, line_dash="dot", annotation_text="ti")
    fig_px.add_vline(x=t_vert, line_color="red", annotation_text="Límite")
    col1.plotly_chart(fig_px)
    
    # Area vs Time
    fig_area = go.Figure()
    fig_area.add_trace(go.Scatter(x=df_contevect["Tiempo (y)"], y=df_contevect["A1 (mm2)"], name="Área"))
    col2.plotly_chart(fig_area)

with tab_modelcode:
    fig_mu = go.Figure()
    fig_mu.add_trace(go.Scatter(x=df_modelcode["Time"], y=df_modelcode["Mu"], name="Mu Estándar"))
    fig_mu.add_trace(go.Scatter(x=df_modelcode["Time"], y=df_modelcode["Mu_cons"], name="Mu Conservador", line=dict(dash='dash')))
    fig_mu.add_vline(x=t_vert, line_color="red")
    st.plotly_chart(fig_mu, use_container_width=True)
