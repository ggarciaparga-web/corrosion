import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from calculos.motor_final import simulacion_total # Asegúrate de que el nombre coincida

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
    
    # Inputs dinámicos
    inputs_calculo = {
        'ancho_b': ancho_b, 'canto_d': canto_d, 'recubrimiento': recubrimiento,
        'fck': fck, 'fy': fy, 'n_barras': n_barras, 'phi_base': phi_base,
        'r2': r2, 'i_corr': i_corr, 't_analisis': t_analisis
    }
    
    if tipo_ataque == "Carbonatación":
        inputs_calculo['c_cemento'] = st.number_input("Cemento (kg/m3)", value=350)
        inputs_calculo['cs_co2'] = st.selectbox("Ambiente (Cs)", [800, 600], format_func=lambda x: "Interior (800)" if x==800 else "Exterior (600)")
    else:
        st.info("Usando parámetros estándar para Cloruros (Fick)")

# --- DIBUJO DE LA SECCIÓN (Sidebar) ---
def dibujar_seccion():
    fig = go.Figure()
    # Hormigón
    fig.add_shape(type="rect", x0=0, y0=0, x1=ancho_b, y1=canto_d, line=dict(color="Black"), fillcolor="LightGrey")
    # Barras (esquemático)
    for i in range(n_barras):
        x_pos = (ancho_b / (n_barras + 1)) * (i + 1)
        fig.add_shape(type="circle", x0=x_pos-phi_base/2, y0=recubrimiento, x1=x_pos+phi_base/2, y1=recubrimiento+phi_base, fillcolor="Red")
    fig.update_layout(width=200, height=300, margin=dict(l=10, r=10, t=10, b=10), xaxis_visible=False, yaxis_visible=False)
    return fig

st.sidebar.write("### Esquema de Sección")
st.sidebar.plotly_chart(dibujar_seccion())

# --- PROCESAMIENTO ---
df_res, ti, t_vert = simulacion_total(tipo_ataque, inputs_calculo)

# --- CUERPO PRINCIPAL (PESTAÑAS) ---
st.title("🛡️ Simulación de Degradación por Corrosión")

tab_teoria, tab_contevect, tab_modelcode = st.tabs(["📚 Teoría", "📊 CONTEVECT", "🏗️ Model Code 2023"])

with tab_teoria:
    st.markdown("""
    ## Fundamentos Teóricos
    Este visor calcula el **Tiempo de Iniciación ($t_i$)** y la fase de **Propagación**.
    - **Carbonatación:** Basado en la raíz cuadrada del tiempo $x = v \cdot \sqrt{t}$.
    - **Cloruros:** Segunda ley de Fick con coeficiente de difusión dependiente del tiempo.
    - **Degradación:** Reducción de sección de acero según $ \phi(t) = \phi_0 - \alpha \cdot P_x $.
    """)

with tab_contevect:
    st.subheader("Análisis según CONTEVECT")
    col1, col2 = st.columns(2)
    
    # Gráfica Px
    fig_px = go.Figure()
    fig_px.add_trace(go.Scatter(x=df_res["Time"], y=df_res["Px"], name="Px (mm)"))
    fig_px.add_vline(x=ti, line_dash="dot", annotation_text="Iniciación (ti)")
    fig_px.add_vline(x=t_vert, line_color="red", annotation_text="Límite 50/500μm")
    fig_px.update_layout(title="Penetración vs Tiempo", xaxis_title="Años", yaxis_title="mm")
    col1.plotly_chart(fig_px)
    
    # Gráfica Áreas
    fig_area = go.Figure()
    area_t = (np.pi * df_res["phi"]**2 / 4) * n_bottom
    fig_area.add_trace(go.Scatter(x=df_res["Time"], y=area_t, name="Área Acero (mm2)"))
    fig_area.update_layout(title="Pérdida de Área vs Tiempo", xaxis_title="Años", yaxis_title="mm2")
    col2.plotly_chart(fig_area)

with tab_modelcode:
    st.subheader("Capacidad Flexurial (fib Model Code 2023)")
    fig_mu = go.Figure()
    fig_mu.add_trace(go.Scatter(x=df_res["Time"], y=df_res["Mu (kNm)"], name="Mu Estándar"))
    fig_mu.add_trace(go.Scatter(x=df_res["Time"], y=df_res["Mu Cons (kNm)"], name="Mu Conservador", line=dict(dash='dash')))
    fig_mu.add_vline(x=t_vert, line_color="red", annotation_text="Aviso Crítico")
    fig_mu.update_layout(title="Mrd vs Tiempo", xaxis_title="Años", yaxis_title="kNm")
    st.plotly_chart(fig_mu, use_container_width=True)
    
    st.write(f"**Tiempo de Iniciación detectado:** {ti:.2f} años")
    st.write(f"**Tiempo hasta límite de servicio (vertical):** {t_vert:.2f} años")
