from __future__ import annotations

import copy
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from calculos.tiempo import calcular_iniciacion
from calculos.CONTEVECT import ejecutar_simulacion_completa_actualizada_en_tiempo
from calculos.ModelCode import simulacion_total


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Structural Corrosion Viewer", layout="wide")
st.title("🏗️ Structural Corrosion Models")


# =========================================================
# HELPERS
# =========================================================
def freeze_inputs(inputs: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    frozen = []
    for k, v in inputs.items():
        if isinstance(v, (int, float)):
            frozen.append((k, float(v)))
        else:
            frozen.append((k, v))
    return tuple(sorted(frozen))


def unfreeze_inputs(frozen: Tuple[Tuple[str, Any], ...]) -> Dict[str, Any]:
    return {k: v for k, v in frozen}


def normalize_time_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Time (y)" not in df.columns and "Tiempo (y)" in df.columns:
        df.rename(columns={"Tiempo (y)": "Time (y)"}, inplace=True)
    return df


# =========================================================
# SIDEBAR – INPUTS
# =========================================================
st.sidebar.header("⚙️ Input Parameters")

analysis_type = st.sidebar.selectbox(
    "Analysis Type", ["Carbonatación", "Cloruros"]
)

with st.sidebar.expander("📐 Geometry and Materials", expanded=True):
    geometry_type = st.sidebar.selectbox(
        "Section geometry",
        ["rect", "i"],
        format_func=lambda x: "Rectangular" if x == "rect" else "I / Double‑T",
    )

    cover_mm = st.sidebar.number_input("Concrete cover (mm)", value=30.0)
    t_analysis = st.sidebar.slider("Total analysis time (years)", 50, 700, 250)

    if geometry_type == "rect":
        width_b = st.sidebar.number_input("Section width b (mm)", value=150.0)
        depth_d = st.sidebar.number_input("Effective depth d (mm)", value=300.0)
    else:
        st.sidebar.markdown("**I / Double‑T geometry**")
        b_top = st.sidebar.number_input("Top flange width b_top (mm)", value=1200.0)
        t_top = st.sidebar.number_input("Top flange thickness t_top (mm)", value=120.0)
        b_web = st.sidebar.number_input("Web width b_web (mm)", value=300.0)
        b_bot = st.sidebar.number_input("Bottom flange width b_bot (mm)", value=800.0)
        t_bot = st.sidebar.number_input("Bottom flange thickness t_bot (mm)", value=120.0)
        h_tot = st.sidebar.number_input("Total height h (mm)", value=700.0)
        depth_d = st.sidebar.number_input("Tension steel depth d (mm)", value=620.0)

    phi_base = st.sidebar.number_input("Bottom bar diameter (mm)", value=20.0)
    n_bars = st.sidebar.number_input("Number of bottom bars", value=2)
    fck = st.sidebar.number_input("Concrete strength fck (MPa)", value=25.0)
    fy = st.sidebar.number_input("Steel yield fy (MPa)", value=500.0)
    r2 = st.sidebar.number_input("Top cover r2 (mm)", value=20.0)

with st.sidebar.expander("🧪 Corrosion parameters", expanded=True):
    i_corr = st.sidebar.number_input("i_corr (μA/cm²)", value=1.0)


# =========================================================
# INPUT DICTIONARY
# =========================================================
inputs_calc: Dict[str, Any] = {
    "geometry_type": geometry_type,
    "t_analisis": float(t_analysis),
    "recubrimiento": float(cover_mm),
    "phi_base": float(phi_base),
    "n_barras": int(n_bars),
    "fck": float(fck),
    "fy": float(fy),
    "r2": float(r2),
    "i_corr": float(i_corr),
}

if geometry_type == "rect":
    inputs_calc.update(
        {
            "ancho_b": float(width_b),
            "canto_d": float(depth_d),
        }
    )
else:
    inputs_calc.update(
        {
            "i_b_top": float(b_top),
            "i_t_top": float(t_top),
            "i_b_web": float(b_web),
            "i_b_bot": float(b_bot),
            "i_t_bot": float(t_bot),
            "i_h": float(h_tot),
            "canto_d": float(depth_d),
        }
    )


# =========================================================
# CACHED RUNS
# =========================================================
@st.cache_data(show_spinner=False)
def run_contevect_cached(
    atype: str,
    gtype: str,
    frozen_inputs: Tuple[Tuple[str, Any], ...],
    ti: float,
):
    inputs = unfreeze_inputs(frozen_inputs)
    return ejecutar_simulacion_completa_actualizada_en_tiempo(
        atype, inputs, ti
    )


@st.cache_data(show_spinner=False)
def run_model_code_cached(
    atype: str,
    gtype: str,
    frozen_inputs: Tuple[Tuple[str, Any], ...],
    ti: float,
):
    inputs = unfreeze_inputs(frozen_inputs)
    return simulacion_total(atype, inputs, ti)


# =========================================================
# RUN
# =========================================================
try:
    ti, _, _ = calcular_iniciacion(analysis_type, inputs_calc)

    frozen = freeze_inputs(copy.deepcopy(inputs_calc))

    df_cv, t_v_cv, lim_cv, df_events = run_contevect_cached(
        analysis_type, geometry_type, frozen, ti
    )
    df_mc, t_v_mc = run_model_code_cached(
        analysis_type, geometry_type, frozen, ti
    )

    df_cv = normalize_time_column(df_cv)
    df_mc = normalize_time_column(df_mc)

    tab1, tab2, tab3 = st.tabs(["📊 CONTEVECT", "🏗️ Model Code", "🧾 Events"])

    with tab1:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(df_cv["Time (y)"], df_cv["Mu (kNm)"], label="CONTEVECT")
        ax.axvline(t_v_cv, linestyle="--", color="red", label="Vertical line")
        ax.set_xlabel("Years")
        ax.set_ylabel("Mu (kNm)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        st.dataframe(df_cv.head(20), use_container_width=True)

    with tab2:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(df_mc["Time (y)"], df_mc["Mu (kNm)"], label="Model Code")
        ax.plot(
            df_mc["Time (y)"],
            df_mc["Mu Cons (kNm)"],
            linestyle="--",
            label="Conservative",
        )
        ax.axvline(t_v_mc, linestyle="--", color="red", label="Vertical line")
        ax.set_xlabel("Years")
        ax.set_ylabel("Mu (kNm)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        st.dataframe(df_mc.head(20), use_container_width=True)

    with tab3:
        st.write("Detected events:")
        if df_events.empty:
            st.info("No events detected with current parameters.")
        else:
            st.dataframe(df_events, use_container_width=True)

except Exception as exc:
    st.error(f"Error detected: {exc}")
