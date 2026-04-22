import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from calculos.tiempo import calcular_iniciacion
from calculos.CONTEVECT import ejecutar_simulacion_completa_actualizada_en_tiempo as ejecutar_cv_base
from calculos.ModelCode import simulacion_total
from calculos.opciones_corrosion import ejecutar_simulacion_corrosion_zona as ejecutar_cv_opciones
from calculos.pretensado import ejecutar_simulacion_pretensado


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(page_title="Structural Corrosion Viewer", layout="wide")


# =========================================================
# AUXILIARY FUNCTION – RECTANGULAR SECTION DRAWING
# =========================================================
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

    phantom = plt.Rectangle(
        (0, 0),
        b_0,
        d_0,
        linewidth=1,
        edgecolor="gray",
        facecolor="none",
        linestyle=":",
    )
    ax.add_patch(phantom)

    off_x = (b_0 - b_current) / 2.0
    rect_h = plt.Rectangle(
        (off_x, 0),
        b_current,
        d_current,
        linewidth=2,
        edgecolor="black",
        facecolor="lightgrey",
        alpha=0.8,
    )
    ax.add_patch(rect_h)

    if px_actual >= px0_value and d_current == d_0:
        for i in range(n_bars):
            x_f = (b_0 / (n_bars + 1)) * (i + 1)
            ax.plot([x_f, x_f], [0, cover], color="black", lw=1.5)

    for i in range(n_bars):
        x_pos = (b_0 / (n_bars + 1)) * (i + 1)
        steel_color = "red" if px_actual == 0 else "#8B4513"
        circ = plt.Circle(
            (x_pos, cover),
            phi_current / 2.0,
            facecolor=steel_color,
            edgecolor="black",
            lw=1,
        )
        ax.add_patch(circ)

    ax.set_xlim(-b_0 * 0.1, b_0 * 1.1)
    ax.set_ylim(-d_0 * 0.1, d_0 * 1.1)
    ax.set_aspect("equal")
    ax.axis("off")

    return fig


# =========================================================
# SIDEBAR – INPUT PARAMETERS
# =========================================================
st.sidebar.header("⚙️ Input Parameters")

analysis_type = st.sidebar.selectbox(
    "Analysis Type", ["Carbonatación", "Cloruros"]
)

with st.sidebar.expander("📐 Geometry and Materials", expanded=True):
    geometry_type = st.selectbox(
        "Section geometry",
        ["rect", "i"],
        format_func=lambda x: "Rectangular" if x == "rect" else "I / Double-T",
    )

    cover_mm = st.number_input("Concrete cover (mm)", value=30.0)
    t_analysis = st.slider("Total analysis time (years)", 50, 700, 250)

    if geometry_type == "rect":
        width_b = st.number_input("Section width b (mm)", value=150.0)
        depth_d = st.number_input("Effective depth d (mm)", value=300.0)
    else:
        st.markdown("**I / Double‑T geometry**")
        b_top = st.number_input("Top flange width b_top (mm)", value=1200.0)
        t_top = st.number_input("Top flange thickness t_top (mm)", value=120.0)
        b_web = st.number_input("Web width b_web (mm)", value=300.0)
        b_bot = st.number_input("Bottom flange width b_bot (mm)", value=800.0)
        t_bot = st.number_input("Bottom flange thickness t_bot (mm)", value=120.0)
        h_tot = st.number_input("Total height h (mm)", value=700.0)
        depth_d = st.number_input("Tension steel depth d (mm)", value=620.0)

    phi_base = st.number_input("Bottom bar diameter (mm)", value=20.0)
    n_bars = st.number_input("Number of bottom bars", value=2)
    fck = st.number_input("Concrete strength fck (MPa)", value=25.0)
    fy = st.number_input("Steel yield fy (MPa)", value=500.0)
    r2 = st.number_input("Top cover r2 (mm)", value=20.0)

with st.sidebar.expander("🏗️ Prestressing Parameters"):
    fpu_pres = st.number_input("Prestressing fpu (MPa)", value=1896.0)
    d_prima_pres = st.number_input("Prestress depth d' (mm)", value=240.0)
    n_pres = st.number_input("Number of elements", value=2)
    phi0_pres = st.number_input("Prestress diameter (mm)", value=20.0)


# =========================================================
# INPUT DICTIONARY
# =========================================================
inputs_calc = {
    "geometry_type": geometry_type,
    "t_analisis": t_analysis,
    "recubrimiento": cover_mm,
    "phi_base": phi_base,
    "n_barras": n_bars,
    "fck": fck,
    "fy": fy,
    "r2": r2,
    "fpu_prestress": fpu_pres,
    "d_prima_prestress": d_prima_pres,
    "n_prestress": n_pres,
    "phi0_prestress": phi0_pres,
}

if geometry_type == "rect":
    inputs_calc.update(
        {
            "ancho_b": width_b,
            "canto_d": depth_d,
        }
    )
else:
    inputs_calc.update(
        {
            "i_b_top": b_top,
            "i_t_top": t_top,
            "i_b_web": b_web,
            "i_b_bot": b_bot,
            "i_t_bot": t_bot,
            "i_h": h_tot,
            "canto_d": depth_d,
        }
    )

if analysis_type == "Carbonatación":
    inputs_calc["c_cemento"] = st.sidebar.number_input(
        "Cement content (kg/m3)", value=450.0
    )
    inputs_calc["cs_co2"] = st.sidebar.number_input(
        "CO2 concentration Cs (mg/m3)", value=800.0
    )
    inputs_calc["i_corr"] = st.sidebar.number_input(
        "i_corr (μA/cm²)", value=1.0
    )
    alpha_pit = 2.0
else:
    inputs_calc["i_corr"] = st.sidebar.number_input(
        "i_corr (μA/cm²)", value=2.58
    )
    alpha_pit = 10.0


# =========================================================
# MAIN APPLICATION
# =========================================================
st.title("🏗️ Structural Corrosion Models")

@st.cache_data(show_spinner=False)
def run_cv_base_cached(atype, inc, ti):
    return ejecutar_cv_base(atype, inc, ti)

@st.cache_data(show_spinner=False)
def run_model_code_cached(atype, inc, ti):
    return simulacion_total(atype, inc, ti)


try:
    ti, times_px, px_plot = calcular_iniciacion(analysis_type, inputs_calc)

    df_cv, t_v_cv, lim_cv, pts_crit = run_cv_base_cached(
        analysis_type, inputs_calc, ti
    )
    df_mc, t_v_mc = run_model_code_cached(
        analysis_type, inputs_calc, ti
    )

    tab1, tab2, tab3 = st.tabs(
        ["📊 CONTEVECT", "🏗️ Model Code", "🔍 Section"]
    )

    with tab1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df_cv["Tiempo (y)"], df_cv["Mu (kNm)"])
        ax.set_xlabel("Years")
        ax.set_ylabel("Mrd [kNm]")
        ax.grid(True)
        st.pyplot(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df_mc["Time (y)"], df_mc["Mu (kNm)"], label="Standard")
        ax.plot(
            df_mc["Time (y)"],
            df_mc["Mu Cons (kNm)"],
            linestyle="--",
            label="Conservative",
        )
        ax.set_xlabel("Years")
        ax.set_ylabel("Mrd [kNm]")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    with tab3:
        if geometry_type != "rect":
            st.info(
                "Section visualization is available only for rectangular sections."
            )
        else:
            fci_val = 0.333 * fck ** (2.0 / 3.0)
            px0_val = max(
                0.0,
                (
                    83.8
                    + 7.4 * (cover_mm / phi_base)
                    - 22.6 * fci_val
                )
                * 1e-3,
            )

            year_sel = st.select_slider(
                "Select year",
                options=list(df_cv["Tiempo (y)"]),
                value=0,
            )

            fig = draw_section_2d(
                inputs_calc, df_cv, year_sel, px0_val
            )
            st.pyplot(fig)

except Exception as exc:
    st.error(f"Error detected: {exc}")
