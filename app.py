import copy
import streamlit as st
import matplotlib.pyplot as plt

from calculos.CONTEVECT import (
    ejecutar_simulacion_completa_actualizada_en_tiempo,
)
from calculos.tiempo import calcular_iniciacion


st.set_page_config(page_title="Structural Corrosion Viewer", layout="wide")
st.title("Structural Corrosion Models")

# =========================================================
# SIDEBAR INPUTS
# =========================================================
analysis_type = st.sidebar.selectbox(
    "Analysis Type", ["Carbonatación", "Cloruros"]
)

geometry_type = st.sidebar.selectbox(
    "Section geometry",
    ["rect", "i"],
    format_func=lambda x: "Rectangular" if x == "rect" else "I / Double‑T",
)

cover_mm = st.sidebar.number_input("Concrete cover (mm)", value=30.0)
t_analysis = st.sidebar.slider("Analysis time (years)", 50, 700, 250)

inputs_calc = {
    "geometry_type": geometry_type,
    "t_analisis": t_analysis,
    "recubrimiento": cover_mm,
    "phi_base": st.sidebar.number_input("Bar diameter (mm)", value=20.0),
    "n_barras": st.sidebar.number_input("Number of bars", value=2),
    "fck": st.sidebar.number_input("fck (MPa)", value=25.0),
    "fy": st.sidebar.number_input("fy (MPa)", value=500.0),
    "r2": st.sidebar.number_input("Top cover r2 (mm)", value=20.0),
}

if geometry_type == "rect":
    inputs_calc.update(
        {
            "ancho_b": st.sidebar.number_input("Width b (mm)", 150.0),
            "canto_d": st.sidebar.number_input("Depth d (mm)", 300.0),
        }
    )
else:
    inputs_calc.update(
        {
            "i_b_top": st.sidebar.number_input("b_top (mm)", 1200.0),
            "i_t_top": st.sidebar.number_input("t_top (mm)", 120.0),
            "i_b_web": st.sidebar.number_input("b_web (mm)", 300.0),
            "i_b_bot": st.sidebar.number_input("b_bot (mm)", 800.0),
            "i_t_bot": st.sidebar.number_input("t_bot (mm)", 120.0),
            "i_h": st.sidebar.number_input("Total height h (mm)", 700.0),
            "canto_d": st.sidebar.number_input("Effective d (mm)", 620.0),
        }
    )

inputs_calc["i_corr"] = st.sidebar.number_input(
    "i_corr (µA/cm²)", value=1.0
)

# =========================================================
# CACHED RUN (FIXED)
# =========================================================
@st.cache_data(show_spinner=False)
def run_contevect_cached(atype, geom_type, inputs, ti):
    return ejecutar_simulacion_completa_actualizada_en_tiempo(
        atype, inputs, ti
    )


ti, _, _ = calcular_iniciacion(analysis_type, inputs_calc)

df_cv, t_v, lim_px, df_events = run_contevect_cached(
    analysis_type,
    geometry_type,
    copy.deepcopy(inputs_calc),  # ✅ CRITICAL FIX
    ti,
)

# =========================================================
# PLOT
# =========================================================
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(df_cv["Tiempo (y)"], df_cv["Mu (kNm)"])
ax.set_xlabel("Years")
ax.set_ylabel("Moment capacity Mu (kNm)")
ax.grid(True)
st.pyplot(fig)
