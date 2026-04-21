import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erf
import math

# --- IMPORTACIÓN DE MÓDULOS EXTERNOS ---
from calculos.tiempo import calcular_iniciacion
from calculos.CONTEVECT import ejecutar_simulacion_completa as ejecutar_cv_base
from calculos.ModelCode import simulacion_total
from calculos.opciones_corrosion import ejecutar_simulacion_corrosion_zona as ejecutar_cv_opciones
from calculos.pretensado import ejecutar_simulacion_pretensado

# CONFIGURACIÓN DE PÁGINA
st.set_page_config(page_title="Structural Corrosion Viewer", layout="wide")

# --- FUNCIÓN AUXILIAR PARA SECCIÓN 2D ---
def draw_section_2d(inputs, df_sim, year, px0_value):
    row = df_sim[df_sim["Tiempo (y)"] == year].iloc[0]
    px_actual = row["Px (mm)"]
    b_current = row["b"]
    d_current = row["d"]
    phi_current = row["phi1 (mm)"]
    b_0 = inputs["ancho_b"]
    d_0 = inputs["canto_d"]
    cover = inputs["recubrimiento"]
    n_bars = int(inputs["n_barras"])

    fig, ax = plt.subplots(figsize=(5, 7))
    phantom = plt.Rectangle((0, 0), b_0, d_0, linewidth=1, edgecolor="gray", facecolor="none", linestyle=":")
    ax.add_patch(phantom)

    off_x = (b_0 - b_current) / 2
    rect_h = plt.Rectangle((off_x, 0), b_current, d_current, linewidth=2, edgecolor="black", facecolor="lightgrey", alpha=0.8)
    ax.add_patch(rect_h)

    if px_actual >= px0_value and d_current == d_0:
        for i in range(n_bars):
            x_f = (b_0 / (n_bars + 1)) * (i + 1)
            ax.plot([x_f, x_f], [0, cover], color="black", lw=1.5, alpha=0.7)
            ax.plot([x_f - 3, x_f + 3], [cover / 2, cover / 2], color="black", lw=1)

    for i in range(n_bars):
        x_pos = (b_0 / (n_bars + 1)) * (i + 1)
        steel_color = "red" if px_actual == 0 else "#8B4513"
        circ = plt.Circle((x_pos, cover), phi_current / 2, facecolor=steel_color, edgecolor="black", lw=1, zorder=5)
        ax.add_patch(circ)

    ax.set_xlim(-b_0 * 0.1, b_0 * 1.1); ax.set_ylim(-d_0 * 0.1, d_0 * 1.1)
    ax.set_aspect("equal"); ax.axis("off")
    return fig

# =========================================================
# BARRA LATERAL: INPUTS DINÁMICOS
# =========================================================
st.sidebar.header("⚙️ Input Parameters")
analysis_type = st.sidebar.selectbox("Analysis Type", ["Carbonatación", "Cloruros"])

with st.sidebar.expander("📐 Geometry and Materials", expanded=True):
    cover_mm = st.sidebar.number_input("Concrete cover (mm)", value=30.0)
    t_analysis = st.sidebar.slider("Total analysis time (years)", 50, 700, 250)
    width_b = st.sidebar.number_input("Section width b (mm)", value=150)
    depth_d = st.sidebar.number_input("Effective depth d (mm)", value=300)
    phi_base = st.sidebar.number_input("Bottom bar diameter (mm)", value=20)
    n_bars = st.sidebar.number_input("Number of bottom bars", value=2)
    fck = st.sidebar.number_input("Concrete strength fck (MPa)", value=25)
    fy = st.sidebar.number_input("Steel yield fy (MPa)", value=500)
    r2 = st.sidebar.number_input("Top cover r2 (mm)", value=20)

with st.sidebar.expander("🏗️ Prestressing Parameters"):
    fpu_pres = st.number_input("Prestressing fpu (MPa)", value=1896.0)
    d_prima_pres = st.number_input("Prestress depth d' (mm)", value=240.0)
    n_pres = st.number_input("Nº elements", value=2)
    phi0_pres = st.number_input("Diameter (mm)", value=20.0)

# Construcción de diccionario de inputs
inputs_calc = {
    "t_analisis": t_analysis,
    "recubrimiento": cover_mm,
    "ancho_b": width_b,
    "canto_d": depth_d,
    "phi_base": phi_base,
    "n_barras": n_bars,
    "fck": fck,
    "fy": fy,
    "r2": r2,
    "fpu_prestress": fpu_pres,
    "d_prima_prestress": d_prima_pres,
    "n_prestress": n_pres,
    "phi0_prestress": phi0_pres
}

if analysis_type == "Carbonatación":
    st.sidebar.subheader("☁️ Carbonation Parameters")
    inputs_calc["c_cemento"] = st.sidebar.number_input("Cement content (kg/m3)", value=450.0)
    inputs_calc["cs_co2"] = st.sidebar.number_input("CO2 concentration Cs (mg/m3)", value=800.0)
    inputs_calc["i_corr"] = st.sidebar.number_input("i_corr (μA/cm²)", value=1.0)
    alpha_pit = 2.0
else:
    st.sidebar.subheader("🌊 Chlorides Parameters")
    inputs_calc["i_corr"] = st.sidebar.number_input("i_corr (μA/cm²)", value=2.58)
    alpha_pit = 10.0

st.title("🏗️ Structural Corrosion Models")

# --- CACHE DE FUNCIONES ---
@st.cache_data(show_spinner=False)
def run_cv_base(atype, inc, ti_in): return ejecutar_cv_base(atype, inc, ti_in)
@st.cache_data(show_spinner=False)
def run_cv_options(atype, inc, ti_in, zone): return ejecutar_cv_opciones(atype, inc, ti_in, corrosion_zone=zone)
@st.cache_data(show_spinner=False)
def run_model_code(atype, inc, ti_in): return simulacion_total(atype, inc, ti_in)

try:
    # 1) Iniciación
    ti, times_px, px_plot = calcular_iniciacion(analysis_type, inputs_calc)

    # 2) Modelos
    df_cv, t_v_cv, lim_cv, pts_crit = run_cv_base(analysis_type, inputs_calc, ti)
    df_mc, t_v_mc = run_model_code(analysis_type, inputs_calc, ti)

    # 3) Pestañas
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 CONTEVECT", "🏗️ Model Code", "🔍 Section Details", "🧪 Corrosion Zone", "⚙️ Prestress Stresses"
    ])

    # Función común para Px
    def plot_px_global(ax, t_ax, px_ax, ti_ax):
        ax.plot(t_ax, px_ax, color="blue", lw=2, label="$P_x$")
        ax.axvline(x=ti_ax, color="red", linestyle="--", label=f"$t_i$ = {ti_ax:.2f} y")
        ax.set_title("Corrosion penetration depth")
        ax.set_xlabel("Years"); ax.set_ylabel("Px [mm]"); ax.legend(); ax.grid(True, alpha=0.3)

    with tab1:
        st.header(f"CONTEVECT Analysis ({analysis_type})")
        col1, col2 = st.columns(2)
        with col1:
            fig_px, ax_px = plt.subplots(figsize=(6, 4))
            plot_px_global(ax_px, times_px, px_plot, ti)
            st.pyplot(fig_px)
        with col2:
            fig_cv, ax_cv = plt.subplots(figsize=(6, 4))
            ax_cv.plot(df_cv["Tiempo (y)"], df_cv["Mu (kNm)"], color="navy", lw=2, label="Mrd")
            ax_cv.scatter(pts_crit["Tiempo (y)"], pts_crit["Mu (kNm)"], color="red", s=100, edgecolor="white", zorder=3)
            ax_cv.set_title("Residual resistance and critical points")
            ax_cv.set_xlabel("Years"); ax_cv.set_ylabel("Mrd [kNm]"); ax_cv.grid(True, alpha=0.3)
            st.pyplot(fig_cv)

    with tab2:
        st.header("Model Code 2023 Analysis")
        col_a, col_b = st.columns(2)
        with col_a:
            fig_mc_px, ax_mc_px = plt.subplots(figsize=(6, 4))
            plot_px_global(ax_mc_px, df_mc["Time"], df_mc["Px"], ti)
            st.pyplot(fig_mc_px)
        with col_b:
            fig_mc_area, ax_mc_area = plt.subplots(figsize=(6, 4))
            ax_mc_area.plot(df_mc["Time"], df_mc["A_corr"], color="darkgreen", lw=2)
            ax_mc_area.set_title("Steel area evolution [mm²]"); ax_mc_area.grid(True, alpha=0.3)
            st.pyplot(fig_mc_area)
        fig_mc, ax_mc = plt.subplots(figsize=(10, 4))
        ax_mc.plot(df_mc["Time"], df_mc["Mu (kNm)"], label="Standard", color="navy", lw=2)
        ax_mc.plot(df_mc["Time"], df_mc["Mu Cons (kNm)"], label="Conservative", color="orange", ls="--", lw=2)
        ax_mc.set_title("Residual resistance Mrd [kNm]"); ax_mc.legend(); ax_mc.grid(True, alpha=0.3)
        st.pyplot(fig_mc)

    with tab3:
        st.header("🔍 Visual section inspection")
        fci_val = 0.333 * inputs_calc["fck"] ** (2 / 3)
        px0_val = max(0.0, (83.8 + 7.4 * (inputs_calc["recubrimiento"] / inputs_calc["phi_base"]) - 22.6 * fci_val) * 1e-3)
        col_ui, col_render = st.columns([1, 2])
        with col_ui:
            year_sel = st.select_slider("Select year", options=list(df_cv["Tiempo (y)"]), value=0)
            f_sel = df_cv[df_cv["Tiempo (y)"] == year_sel].iloc[0]
            st.metric("Penetration Px", f"{f_sel['Px (mm)']:.3f} mm")
            st.metric("Residual moment", f"{f_sel['Mu (kNm)']:.2f} kNm")
        
        with col_render:
            st.pyplot(draw_section_2d(inputs_calc, df_cv, year_sel, px0_val))
            st.image("https://github.com/user-attachments/assets/36960bd8-5f2d-4faf-961e-e340d4b7f6a8", 
            caption="Referencia de la sección visual",
                     use_container_width=True
                    )

    with tab4:
        st.header("🧪 Corrosion zone sensitivity")
        zone = st.radio("Corrosion zone", options=["tension", "compression", "both"], index=2)
        df_opt, _, _, _ = run_cv_options(analysis_type, inputs_calc, ti, zone)
        fig_opt, ax_opt = plt.subplots(figsize=(10, 4))
        ax_opt.plot(df_opt["Tiempo (y)"], df_opt["Mu (kNm)"], color="navy", lw=2)
        ax_opt.set_title(f"Mu (zone = {zone})"); ax_opt.grid(True, alpha=0.3)
        st.pyplot(fig_opt)

    with tab5:
        st.header("⚙️ Prestress Concrete Stresses")
        # Llamada al módulo externo
        df_pres = ejecutar_simulacion_pretensado(inputs_calc, ti, inputs_calc["i_corr"], alpha_pit)
    
        
        # Gráfica de Tensiones solicitada
        fig_pres, ax_pres = plt.subplots(figsize=(10, 5))
        ax_pres.plot(df_pres["time"], df_pres["sigma_inferior"], marker="o", markersize=3, label="Bottom stress", color="blue")
        ax_pres.plot(df_pres["time"], df_pres["sigma_superior"], marker="s", markersize=3, label="Top stress", color="orange")
        ax_pres.axvline(x=ti, color="red", linestyle="--", label="Initiation")
        
        ax_pres.set_xlabel("Time [years]", fontsize=12, fontweight="bold")
        ax_pres.set_ylabel("Stress [MPa]", fontsize=12, fontweight="bold")
        ax_pres.set_title("Final prestress concrete stress vs time", fontsize=14, fontweight="bold")
        ax_pres.grid(True, alpha=0.3); ax_pres.legend()
        plt.tight_layout()
        st.pyplot(fig_pres)

except Exception as exc:
    st.error(f"Error detected: {exc}")
