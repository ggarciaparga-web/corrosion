import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as plt

# --- YOUR MODULE IMPORTS ---
from calculos.tiempo import calcular_iniciacion
from calculos.CONTEVECT import ejecutar_simulacion_completa as ejecutar_cv_base
from calculos.ModelCode import simulacion_total

# NEW: localized corrosion options (your new file)
from calculos.opciones_corrosion import ejecutar_simulacion_completa as ejecutar_cv_opciones


st.set_page_config(page_title="Structural Corrosion Viewer", layout="wide")


def draw_section_2d(inputs, df_sim, year, px0_value):
    """2D view of the degraded cross-section."""
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

    # Original outline (reference)
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

    # Current concrete (shrinks if spalling)
    off_x = (b_0 - b_current) / 2
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

    # Cracking lines if Px >= Px0 and no spalling yet
    if px_actual >= px0_value and d_current == d_0:
        for i in range(n_bars):
            x_f = (b_0 / (n_bars + 1)) * (i + 1)
            ax.plot([x_f, x_f], [0, cover], color="black", lw=1.5, alpha=0.7)
            ax.plot([x_f - 3, x_f + 3], [cover / 2, cover / 2], color="black", lw=1)

    # Bottom reinforcement
    for i in range(n_bars):
        x_pos = (b_0 / (n_bars + 1)) * (i + 1)
        steel_color = "red" if px_actual == 0 else "#8B4513"
        circ = plt.Circle(
            (x_pos, cover),
            phi_current / 2,
            facecolor=steel_color,
            edgecolor="black",
            lw=1,
            zorder=5,
        )
        ax.add_patch(circ)

    ax.set_xlim(-b_0 * 0.1, b_0 * 1.1)
    ax.set_ylim(-d_0 * 0.1, d_0 * 1.1)
    ax.set_aspect("equal")
    ax.axis("off")

    state = "SOUND"
    if px_actual >= px0_value:
        state = "CRACKED"
    if d_current < d_0:
        state = "SPALLED"

    ax.set_title(
        f"YEAR {year} - STATE: {state}\n$P_x$: {px_actual:.3f} mm",
        fontsize=12,
        fontweight="bold",
        pad=20,
    )
    return fig


# =========================================================
# SIDEBAR INPUTS
# =========================================================
st.sidebar.header("⚙️ Input Parameters")
analysis_type = st.sidebar.selectbox("Analysis Type", ["Carbonatación", "Cloruros"])

with st.sidebar.expander("📐 Geometry and Materials", expanded=True):
    cover_mm = st.sidebar.number_input("Concrete cover (mm)", value=30.0)
    t_analysis = st.sidebar.slider("Total analysis time (years)", 50, 500, 200)
    width_b = st.sidebar.number_input("Section width b (mm)", value=150)
    depth_d = st.sidebar.number_input("Effective depth d (mm)", value=300)
    phi_base = st.sidebar.number_input("Bottom bar diameter (mm)", value=20)
    n_bars = st.sidebar.number_input("Number of bottom bars", value=2)
    fck = st.sidebar.number_input("Concrete strength fck (MPa)", value=25)
    fy = st.sidebar.number_input("Steel yield fy (MPa)", value=500)
    r2 = st.sidebar.number_input("Top cover r2 (mm)", value=20)

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
}

if analysis_type == "Carbonatación":
    st.sidebar.subheader("☁️ Carbonation Parameters")
    inputs_calc["c_cemento"] = st.sidebar.number_input("Cement content (kg/m3)", value=450.0)
    inputs_calc["cs_co2"] = st.sidebar.number_input("CO2 concentration Cs (mg/m3)", value=800.0)
    inputs_calc["i_corr"] = st.sidebar.number_input("Corrosion current i_corr (μA/cm²)", value=1.0)
else:
    st.sidebar.subheader("🌊 Chlorides Parameters")
    inputs_calc["i_corr"] = st.sidebar.number_input("Corrosion current i_corr (μA/cm²)", value=2.58)

st.title("🏗️ Structural Corrosion Models")


@st.cache_data(show_spinner=False)
def run_cv_base(analysis_type_in, inputs_in, ti_in):
    return ejecutar_cv_base(analysis_type_in, inputs_in, ti_in)


@st.cache_data(show_spinner=False)
def run_cv_options(analysis_type_in, inputs_in, ti_in, zone_in):
    return ejecutar_cv_opciones(analysis_type_in, inputs_in, ti_in, corrosion_zone=zone_in)


@st.cache_data(show_spinner=False)
def run_model_code(analysis_type_in, inputs_in, ti_in):
    return simulacion_total(analysis_type_in, inputs_in, ti_in)


try:
    # 1) Initiation
    ti, times_px, px_plot = calcular_iniciacion(analysis_type, inputs_calc)

    # 2) Models
    df_cv, t_v_cv, lim_cv, pts_crit = run_cv_base(analysis_type, inputs_calc, ti)
    df_mc, t_v_mc = run_model_code(analysis_type, inputs_calc, ti)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 CONTEVECT", "🏗️ Model Code", "🔍 Section Details", "🧪 Corrosion Zone (Tab 4)"]
    )

    # =========================================================
    # TAB 1: CONTEVECT (BASE)
    # =========================================================
    with tab1:
        st.header(f"CONTEVECT Analysis ({analysis_type})")
        st.write(f"**Initiation time:** {ti:.2f} years")

        col1, col2 = st.columns(2)

        with col1:
            fig_px, ax_px = plt.subplots(figsize=(6, 4))
            ax_px.plot(times_px, px_plot, color="blue", lw=2, label="$P_x$")
            ax_px.axvline(x=ti, color="red", linestyle="--", label=f"$t_i$ = {ti:.2f} y")
            ax_px.axvline(x=t_v_cv, color="black", linestyle=":", label="Limit")
            ax_px.set_title("Corrosion penetration")
            ax_px.set_xlabel("Years")
            ax_px.legend()
            ax_px.grid(True, alpha=0.3)
            st.pyplot(fig_px)

        with col2:
            fig_cv, ax_cv = plt.subplots(figsize=(6, 4))
            ax_cv.plot(
                df_cv["Tiempo (y)"],
                df_cv["Mu (kNm)"],
                color="navy",
                lw=2,
                zorder=1,
                label="Mrd",
            )
            ax_cv.scatter(
                pts_crit["Tiempo (y)"],
                pts_crit["Mu (kNm)"],
                color="red",
                s=100,
                edgecolor="white",
                linewidth=1.5,
                zorder=3,
            )
            ax_cv.set_title("Residual resistance and critical points", fontweight="bold")
            ax_cv.set_xlabel("Years")
            ax_cv.set_ylabel("Mrd [kNm]")
            ax_cv.grid(True, alpha=0.3)
            st.pyplot(fig_cv)

    # =========================================================
    # TAB 2: MODEL CODE
    # =========================================================
    with tab2:
        st.header("Model Code 2023 Analysis")

        col_a, col_b = st.columns(2)
        with col_a:
            fig_mc_px, ax_mc_px = plt.subplots(figsize=(6, 4))
            ax_mc_px.plot(df_mc["Time"], df_mc["Px"], color="blue", lw=2)
            ax_mc_px.set_title("Penetration Px [mm]")
            ax_mc_px.grid(True, alpha=0.3)
            st.pyplot(fig_mc_px)

        with col_b:
            fig_mc_area, ax_mc_area = plt.subplots(figsize=(6, 4))
            ax_mc_area.plot(df_mc["Time"], df_mc["A_corr"], color="darkgreen", lw=2)
            ax_mc_area.set_title("Steel area [mm²]")
            ax_mc_area.grid(True, alpha=0.3)
            st.pyplot(fig_mc_area)

        fig_mc, ax_mc = plt.subplots(figsize=(10, 4))
        ax_mc.plot(df_mc["Time"], df_mc["Mu (kNm)"], label="Standard", color="navy", lw=2)
        ax_mc.plot(
            df_mc["Time"],
            df_mc["Mu Cons (kNm)"],
            label="Conservative",
            color="orange",
            linestyle="--",
            lw=2,
        )
        ax_mc.set_title("Residual resistance Mrd [kNm]")
        ax_mc.legend()
        ax_mc.grid(True, alpha=0.3)
        st.pyplot(fig_mc)

    # =========================================================
    # TAB 3: SECTION INSPECTION (BASE CONTEVECT)
    # =========================================================
    with tab3:
        st.header("🔍 Visual section inspection")

        fci = 0.333 * inputs_calc["fck"] ** (2 / 3)
        px0_val = max(
            0.0,
            (83.8 + 7.4 * (inputs_calc["recubrimiento"] / inputs_calc["phi_base"]) - 22.6 * fci)
            * 1e-3,
        )

        col_ui, col_render = st.columns([1, 2])

        with col_ui:
            st.info("Move the slider to view the physical evolution (base CONTEVECT).")
            year_sel = st.select_slider(
                "Select year", options=list(df_cv["Tiempo (y)"]), value=0
            )
            f_sel = df_cv[df_cv["Tiempo (y)"] == year_sel].iloc[0]
            st.metric("Bottom bar diameter", f"{f_sel['phi1 (mm)']:.2f} mm")
            st.metric("Penetration Px", f"{f_sel['Px (mm)']:.3f} mm")
            st.metric("Residual moment", f"{f_sel['Mu (kNm)']:.2f} kNm")

        with col_render:
            fig_insp = draw_section_2d(inputs_calc, df_cv, year_sel, px0_val)
            st.pyplot(fig_insp)

    # =========================================================
    # TAB 4: CORROSION ZONE OPTIONS (YOUR NEW MODULE)
    # =========================================================
    with tab4:
        st.header("🧪 Corrosion zone sensitivity (Tab 4)")

        left, right = st.columns([1, 2], vertical_alignment="top")

        with left:
            st.subheader("Inputs (local to this tab)")
            zone = st.radio(
                "Where does corrosion act?",
                options=["tension", "compression", "both"],
                index=2,
                help=(
                    "tension: only phi1 reduces\n"
                    "compression: only phi2/phiw reduce\n"
                    "both: phi1/phi2/phiw reduce"
                ),
            )
            compare_all = st.checkbox("Compare all zones on the same chart", value=True)
            st.caption(
                "Note: if zone = tension, the code skips cracking/spalling events (b and d remain constant)."
            )

        with right:
            if compare_all:
                zones = ["tension", "compression", "both"]
                results = {}
                for z in zones:
                    results[z] = run_cv_options(analysis_type, inputs_calc, ti, z)

                fig_mu, ax_mu = plt.subplots(figsize=(10, 4))
                colors = {"tension": "crimson", "compression": "darkgreen", "both": "navy"}
                for z in zones:
                    df_z = results[z][0]
                    ax_mu.plot(
                        df_z["Tiempo (y)"],
                        df_z["Mu (kNm)"],
                        lw=2,
                        color=colors[z],
                        label=z,
                    )
                ax_mu.set_title("Residual moment Mu by corrosion zone")
                ax_mu.set_xlabel("Years")
                ax_mu.set_ylabel("Mu [kNm]")
                ax_mu.grid(True, alpha=0.3)
                ax_mu.legend()
                st.pyplot(fig_mu)

                fig_phi, ax_phi = plt.subplots(figsize=(10, 4))
                for z in zones:
                    df_z = results[z][0]
                    ax_phi.plot(
                        df_z["Tiempo (y)"],
                        df_z["phi1 (mm)"],
                        lw=2,
                        color=colors[z],
                        label=f"phi1 ({z})",
                    )
                ax_phi.set_title("Bottom bar diameter phi1 by corrosion zone")
                ax_phi.set_xlabel("Years")
                ax_phi.set_ylabel("phi1 [mm]")
                ax_phi.grid(True, alpha=0.3)
                ax_phi.legend()
                st.pyplot(fig_phi)

                st.subheader("Preview tables")
                col_t1, col_t2, col_t3 = st.columns(3)
                with col_t1:
                    st.caption("tension (head)")
                    st.dataframe(results["tension"][0].head(10), use_container_width=True)
                with col_t2:
                    st.caption("compression (head)")
                    st.dataframe(results["compression"][0].head(10), use_container_width=True)
                with col_t3:
                    st.caption("both (head)")
                    st.dataframe(results["both"][0].head(10), use_container_width=True)

            else:
                df_opt, t_v_opt, lim_opt, pts_opt = run_cv_options(
                    analysis_type, inputs_calc, ti, zone
                )

                fig_opt, ax_opt = plt.subplots(figsize=(10, 4))
                ax_opt.plot(df_opt["Tiempo (y)"], df_opt["Mu (kNm)"], color="navy", lw=2)
                if pts_opt is not None and len(pts_opt) > 0:
                    ax_opt.scatter(
                        pts_opt["Tiempo (y)"],
                        pts_opt["Mu (kNm)"],
                        color="red",
                        s=80,
                        edgecolor="white",
                        linewidth=1.2,
                        zorder=3,
                    )
                ax_opt.set_title(f"Residual moment Mu (zone = {zone})")
                ax_opt.set_xlabel("Years")
                ax_opt.set_ylabel("Mu [kNm]")
                ax_opt.grid(True, alpha=0.3)
                st.pyplot(fig_opt)

                year_opt = st.select_slider(
                    "Select year (options simulation)",
                    options=list(df_opt["Tiempo (y)"]),
                    value=0,
                )
                row_opt = df_opt[df_opt["Tiempo (y)"] == year_opt].iloc[0]
                m1, m2, m3 = st.columns(3)
                m1.metric("Px", f"{row_opt['Px (mm)']:.3f} mm")
                m2.metric("phi1", f"{row_opt['phi1 (mm)']:.2f} mm")
                m3.metric("Mu", f"{row_opt['Mu (kNm)']:.2f} kNm")

                st.dataframe(df_opt, use_container_width=True)

except Exception as exc:
    st.error(f"Error detected: {exc}")
    st.info("Verify that the modules in /calculos return the expected values and column names.")
