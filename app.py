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
    df_cv, t_v_cv, lim_cv, pts_criticos = ejecutar_simulacion_completa(tipo_ataque, inputs_calculo, ti)
    df_mc, t_v_mc = simulacion_total(tipo_ataque, inputs_calculo, ti)

    tab1, tab2, tab3 = st.tabs(["📊 CONTEVECT", "🏗️ Model Code", "📚 Teoría"])

    with tab1:
        st.header(f"Análisis CONTEVECT ({tipo_ataque})")
        st.write(f"**Tiempo de iniciación:** {ti:.2f} años")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfica Px
            fig_px, ax_px = plt.subplots(figsize=(6, 4))
            ax_px.plot(tiempos_px, px_plot, color='blue', lw=2, label='$P_x$')
            ax_px.axvline(x=ti, color='red', linestyle='--', label=f'$t_i$ = {ti:.2f} y')
            ax_px.axvline(x=t_v_cv, color='black', linestyle=':', label='Límite')
            ax_px.set_title("Penetración de Corrosión"); ax_px.set_xlabel("Años")
            ax_px.legend(); ax_px.grid(True, alpha=0.3)
            st.pyplot(fig_px)

        with col2:
            # Gráfica Mrd CONTEVECT (Con Hitos)
            fig_cv, ax_cv = plt.subplots(figsize=(6, 4))
            
            # 1. Línea de evolución continua
            ax_cv.plot(df_cv["Tiempo (y)"], df_cv["Mu (kNm)"], color='navy', lw=2, zorder=1, label="Mrd")
            
            # 2. Dibujar los Puntos Críticos (Hitos)
            # Extraemos valores de la tabla pts_criticos que devuelve CONTEVECT.py
            t_pts = pts_criticos["Tiempo (y)"].values
            m_pts = pts_criticos["Mu (kNm)"].values
            
            ax_cv.scatter(t_pts, m_pts, color='red', s=100, edgecolor='white', linewidth=1.5, zorder=3, label='Hitos')

            # 3. Etiquetas de los hitos
            labels_hitos = ["Inicio", "Fisuración", "Ev. 3", "Ev. 4"]
            for i, (x, y) in enumerate(zip(t_pts, m_pts)):
                if i < len(labels_hitos):
                    ax_cv.annotate(labels_hitos[i], (x, y), xytext=(5, 5), 
                                   textcoords='offset points', fontsize=8, 
                                   fontweight='bold', color='darkred')

            ax_cv.set_title("Resistencia Residual y Puntos Críticos", fontweight='bold')
            ax_cv.set_xlabel("Años")
            ax_cv.set_ylabel("Mrd [kNm]")
            ax_cv.grid(True, alpha=0.3)
            ax_cv.legend(loc='upper right', fontsize=8)
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
        st.subheader("Detalles de la sección")
        def dibujar_inspeccion_2d(inputs, df_simulacion, año, px0_valor):
    # 1. Extraer datos del año seleccionado
    fila = df_simulacion[df_simulacion["Tiempo (y)"] == año].iloc[0]
    px_actual = fila["Px (mm)"]
    b_actual = fila["b"]
    d_actual = fila["d"]
    phi_actual = fila["phi1 (mm)"]
    
    # Inputs originales
    b_0 = inputs['ancho_b']
    d_0 = inputs['canto_d']
    recu = inputs['recubrimiento']
    n_b = int(inputs['n_barras'])
    phi_0 = inputs['phi_base']

    fig, ax = plt.subplots(figsize=(5, 7))
    
    # 2. Contorno original (Referencia fantasma)
    rect_fantom = plt.Rectangle((0, 0), b_0, d_0, linewidth=1, 
                                 edgecolor='gray', facecolor='none', ls=':')
    ax.add_patch(rect_fantom)

    # 3. Hormigón actual (Se encoge si hay desprendimiento)
    off_x = (b_0 - b_actual) / 2
    rect_h = plt.Rectangle((off_x, 0), b_actual, d_actual, 
                            linewidth=2, edgecolor='black', facecolor='lightgrey', alpha=0.8)
    ax.add_patch(rect_h)

    # 4. FISURACIÓN: Si Px > Px0 y aún no se ha desprendido el hormigón
    if px_actual >= px0_valor and d_actual == d_0:
        for i in range(n_b):
            x_f = (b_0 / (n_b + 1)) * (i + 1)
            # Dibujamos fisuras verticales desde la barra hacia abajo
            ax.plot([x_f, x_f], [0, recu], color='black', lw=1.5, alpha=0.7)
            # Pequeñas ramificaciones
            ax.plot([x_f-3, x_f+3], [recu/2, recu/2], color='black', lw=1)

    # 5. Armaduras Inferiores (Reduciendo diámetro y cambiando color)
    for i in range(n_b):
        x_pos = (b_0 / (n_b + 1)) * (i + 1)
        color_acero = 'red' if px_actual == 0 else '#8B4513' # Marrón óxido
        
        circ = plt.Circle((x_pos, recu), phi_actual/2, 
                          facecolor=color_acero, edgecolor='black', lw=1, zorder=5)
        ax.add_patch(circ)

    ax.set_xlim(-b_0*0.1, b_0*1.1)
    ax.set_ylim(-d_0*0.1, d_0*1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    estado = "SANO"
    if px_actual >= px0_valor: estado = "FISURADO"
    if d_actual < d_0: estado = "DESPRENDIDO"
    
    ax.set_title(f"AÑO {año} - ESTADO: {estado}\n$P_x$: {px_actual:.3f} mm", 
                 fontsize=12, fontweight='bold', pad=20)
    
    return fig

except Exception as e:
    st.error(f"Error en la conexión de módulos: {e}")
    st.info("Revisa que los archivos en /calculos devuelvan los valores correctos.")
    st.error(f"El error exacto es: {e}")
