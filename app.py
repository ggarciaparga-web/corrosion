import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erf
import math

# CONFIGURACIÓN DE PÁGINA
st.set_page_config(page_title="Modelos de Corrosión", layout="wide")

st.title("🏗️ Modelos de Corrosión de Estructuras")

# =========================================================
# BARRA LATERAL: INPUTS (Sincronizados)
# =========================================================
st.sidebar.header("⚙️ Parámetros de Entrada")

# Selección del modelo
opcion_web = st.sidebar.selectbox("Tipo de Análisis", ["Carbonatación", "Ataque por Cloruros"])
opcion = "1" if opcion_web == "Carbonatación" else "2"

# Inputs compartidos (Sincronizados con tus códigos)
recubrimiento = st.sidebar.number_input("Recubrimiento [mm] (r2 / cover)", value=30.0)
t_analisis = st.sidebar.slider("Tiempo total de estudio (años)", 50, 700, 250)
i_corr_manual = st.sidebar.number_input("Intensidad de corrosión (μA/cm2) [Solo si quieres sobreescribir]", value=0.0)

st.sidebar.subheader("Geometría de la Sección")
b_initial = st.sidebar.number_input("Ancho sección b (mm)", value=150)
d_initial = st.sidebar.number_input("Canto útil d (mm)", value=300)
phi1_initial = st.sidebar.number_input("Diámetro barras inferiores (mm)", value=20)
n_bottom = st.sidebar.number_input("Número de barras inferiores", value=2)

st.sidebar.subheader("Materiales")
fck = st.sidebar.number_input("fck (MPa)", value=25)
fy = st.sidebar.number_input("fy (MPa)", value=500)

# =========================================================
# BLOQUE 1: CÁLCULO DE INICIACIÓN SEGÚN EXCEL
# =========================================================

if opcion == "1":
    # Datos exactos Hoja Carbonatación
    c_cemento = 450.0
    cao_perc = 65.0
    d_co2 = 2e-08
    cs_co2 = 800.0
    i_corr = 0.1 if i_corr_manual == 0 else i_corr_manual
    alpha_val = 2.0
    
    a_param = c_cemento * (cao_perc / 100.0) * (44.0 / 56.0) * 0.6
    cs_kg_m3 = cs_co2 / 1e6
    v_co2_seg = np.sqrt((2 * d_co2 * cs_kg_m3) / a_param) * 1000
    v_co2_año = v_co2_seg * 5600
    ti = (recubrimiento / v_co2_año)**2
    metodo_text = "Carbonatación"
else:
    # Datos exactos Hoja Cloruros
    c_crit = 0.6; c_surf = 2.0; c_0 = 0.1
    d_ref = 7.12e-12; n_ageing = 0.4288; t_0_cl = 0.0767
    i_corr = 2.58 if i_corr_manual == 0 else i_corr_manual
    alpha_val = 10.0
    
    tiempos_fick = np.linspace(0.001, t_analisis, 5000)
    ti = 0; ti_encontrado = False
    for t in tiempos_fick:
        d_cl = d_ref * (t_0_cl / t)**n_ageing
        t_seg = t * 31536000
        arg = (recubrimiento / 1000.0) / (2 * np.sqrt(d_cl * t_seg))
        c_t = c_0 + (c_surf - c_0) * (1 - erf(arg))
        if not ti_encontrado and c_t >= c_crit:
            ti = t
            ti_encontrado = True; break
    metodo_text = "Ataque por Cloruros"

# Gráfica Px común
tiempos_px = np.linspace(0, t_analisis, 500)
px_plot = [0.0116 * i_corr * (t - ti) if t > ti else 0.0 for t in tiempos_px]

# =========================================================
# PESTAÑAS
# =========================================================
tab1, tab2 = st.tabs(["📊 CONTEVECT", "💻 Model Code"])

with tab1:
    st.header(f"Modelo {metodo_text} (Lógica CONTEVECT)")
    st.write(f"**Tiempo de iniciación ($t_i$):** {ti:.2f} años")

    # Gráfica 1: Px
    fig_px, ax_px = plt.subplots(figsize=(10, 4))
    ax_px.plot(tiempos_px, px_plot, color='blue', lw=2, label='$P_x$')
    ax_px.axvline(x=ti, color='red', linestyle='--', label=f'$t_i$ = {ti:.2f} años')
    ax_px.set_title("Penetración de Corrosión ($P_x$)")
    ax_px.set_xlabel("Tiempo [años]"); ax_px.set_ylabel("Px [mm]"); ax_px.legend(); ax_px.grid(True, alpha=0.3)
    st.pyplot(fig_px)

    # --- LÓGICA ESTRUCTURAL CONTEVECT (SIN SIMPLIFICAR) ---
    def calc_mu_simple_contevect(row, fyd_val, fck_val):
        a1 = row["A1 (mm2)"]; d_act = row["d"]; b_act = row["b"]; fcd = fck_val / 1.5
        if a1 <= 0: return 0.0
        x = (a1 * fyd_val) / (0.8 * b_act * fcd)
        z = d_act - 0.4 * x
        return max((a1 * fyd_val * z) / 1e6, 0.0)

    t_end_corr = t_analisis - ti
    if t_end_corr > 0:
        times_cv = np.arange(0, t_end_corr + 1, 1)
        rows_cv = []
        for t in times_cv:
            px = 0.0116 * i_corr * t
            p1 = max(0.0, phi1_initial - alpha_val * px)
            p2 = max(0.0, 20 - alpha_val * px) # phi2 fijo según tu código
            pw = max(0.0, 0.0001 - alpha_val * px) # phiw fijo
            a1 = (np.pi * p1 ** 2 / 4.0) * n_bottom
            a2 = (np.pi * p2 ** 2 / 4.0); aw = (np.pi * pw ** 2 / 4.0)
            rows_cv.append({
                "Tiempo (y)": t, "Px (mm)": px, "A1 (mm2)": a1, "Aw (mm2)": aw,
                "rho1": a1 / (b_initial * d_initial), "rho2": a2 / (b_initial * d_initial)
            })
        df_base_cv = pd.DataFrame(rows_cv)

        # Puntos Críticos (Exacto)
        fci = 0.333 * fck ** (2/3)
        px0_spall = max(0.0, (83.8 + 7.4 * (recubrimiento / phi1_initial) - 22.6 * fci) * 1e-3)
        
        points = []
        def prep(r, b, d):
            new = r.copy(); new["b"] = b; new["d"] = d; return new
        
        points.append(prep(df_base_cv.iloc[0], b_initial, d_initial))
        idx_px0 = (df_base_cv["Px (mm)"] >= px0_spall).idxmax()
        if df_base_cv["Px (mm)"].iloc[idx_px0] >= px0_spall:
            points.append(prep(df_base_cv.loc[idx_px0], b_initial, d_initial))
        
        ev3, ev4 = None, None
        for _, row in df_base_cv.iterrows():
            r1, r2, px, aw = row["rho1"]*100, row["rho2"]*100, row["Px (mm)"], row["Aw (mm2)"]
            if r1 > 1.5 and aw > (0.0036 * b_initial) and px > 0.2 and ev4 is None:
                ev4 = prep(row, b_initial - 2.0 * recubrimiento, d_initial - recubrimiento)
            if ev3 is None:
                if (r1 < 1.0 and r2 < 5.0 and px > 0.4) or (r1 < 1.0 and r2 > 5.0 and px > 0.2) or (r1 > 1.5 and r2 > 0.5 and px > 0.2):
                    ev3 = prep(row, b_initial, d_initial - recubrimiento)
        
        if ev3 is not None: points.append(ev3)
        if ev4 is not None: points.append(ev4)
        mat_crit = pd.DataFrame(points).sort_values("Px (mm)").drop_duplicates("Px (mm)")
        mat_crit["Mu (kNm)"] = mat_crit.apply(lambda r: calc_mu_simple_contevect(r, fy/1.15, fck), axis=1)

        # Continuación
        last = mat_crit.iloc[-1]
        rem_times = np.arange(last["Tiempo (y)"] + 1, t_end_corr + 1, 1)
        rem_rows = []
        for t in rem_times:
            px = 0.0116 * i_corr * t
            a1 = (np.pi * max(0.0, phi1_initial - alpha_val * px)**2 / 4.0) * n_bottom
            r_c = {"Tiempo (y)": t, "Px (mm)": px, "b": last["b"], "d": last["d"], "A1 (mm2)": a1}
            r_c["Mu (kNm)"] = calc_mu_simple_contevect(r_c, fy/1.15, fck)
            rem_rows.append(r_c)
        
        df_active_cv = pd.concat([mat_crit, pd.DataFrame(rem_rows)], ignore_index=True)
        df_active_cv["TimeReal"] = df_active_cv["Tiempo (y)"] + ti
        
        # Pasivo
        a1_ini = (np.pi * phi1_initial**2 / 4.0) * n_bottom
        mu_ini = calc_mu_simple_contevect({"A1 (mm2)": a1_ini, "b": b_initial, "d": d_initial}, fy/1.15, fck)
        df_pasivo_cv = pd.DataFrame({"TimeReal": [0, ti], "Mu (kNm)": [mu_ini, mu_ini], "A1 (mm2)": [a1_ini, a1_ini]})
        df_final_cv = pd.concat([df_pasivo_cv, df_active_cv], ignore_index=True)

        st.subheader("Capacidad y Área (CONTEVECT)")
        col1, col2 = st.columns(2)
        with col1:
            fig_cv1, ax_cv1 = plt.subplots()
            ax_cv1.plot(df_final_cv["TimeReal"], df_final_cv["Mu (kNm)"], color="navy")
            ax_cv1.set_title("Mrd vs Tiempo"); ax_cv1.grid(True, alpha=0.3)
            st.pyplot(fig_cv1)
        with col2:
            fig_cv2, ax_cv2 = plt.subplots()
            ax_cv2.plot(df_final_cv["TimeReal"], df_final_cv["A1 (mm2)"], color="darkgreen")
            ax_cv2.set_title("Área vs Tiempo"); ax_cv2.grid(True, alpha=0.3)
            st.pyplot(fig_cv2)

with tab2:
    st.header("Análisis Model Code (fib 2023)")
    
    # --- LÓGICA MODEL CODE (SIN SIMPLIFICAR) ---
    def calc_capacidad_mc(a_corr, d_act, b_act, fck_val, fy_val, r2_val):
        if a_corr <= 0: return 0.0, 0.0
        fyd_mc = fy_val / 1.15
        fcd_mc = (fck_val / 1.5) * (0.75 * min(1.0, (30.0 / fck_val) ** (1/3)))
        x_mc = (a_corr * fyd_mc) / (0.8 * b_act * fcd_mc)
        z_std = d_act - 0.4 * x_mc
        mu_res = (a_corr * fyd_mc * z_std) / 1e6
        z_cons = max(0.0, (d_act - r2_val) - 0.4 * x_mc)
        mu_cons = (a_corr * fyd_mc * z_cons) / 1e6
        return max(mu_res, 0.0), max(mu_cons, 0.0)

    tiempos_mc = np.arange(0, t_analisis + 1, 1)
    a_ini_mc = (math.pi * phi1_initial ** 2 / 4.0) * n_bottom
    m_s0, m_c0 = calc_capacidad_mc(a_ini_mc, d_initial, b_initial, fck, fy, recubrimiento)

    results_mc = []
    for t in tiempos_mc:
        if t <= ti:
            results_mc.append({"T": t, "A1": a_ini_mc, "Mu": m_s0, "MuC": m_c0})
        else:
            px_mc = 0.0116 * i_corr * (t - ti)
            p_mc = max(0.0, phi1_initial - alpha_val * px_mc)
            a_mc = (math.pi * p_mc**2 / 4.0) * n_bottom
            ms, mc = calc_capacidad_mc(a_mc, d_initial, b_initial, fck, fy, recubrimiento)
            results_mc.append({"T": t, "A1": a_mc, "Mu": ms, "MuC": mc})
    
    df_mc = pd.DataFrame(results_mc)

    st.subheader("Capacidad Mrd (Model Code)")
    fig_mc_mu, ax_mc_mu = plt.subplots(figsize=(10, 4))
    ax_mc_mu.plot(df_mc["T"], df_mc["Mu"], label="Mrd Standard", color="navy")
    ax_mc_mu.plot(df_mc["T"], df_mc["MuC"], label="Mrd Conservador", color="orange", linestyle="--")
    ax_mc_mu.axvline(x=ti, color="red", linestyle="--")
    ax_mc_mu.legend(); ax_mc_mu.grid(True, alpha=0.3)
    st.pyplot(fig_mc_mu)

    st.subheader("Área de Acero (Model Code)")
    fig_mc_a, ax_mc_a = plt.subplots(figsize=(10, 4))
    ax_mc_a.plot(df_mc["T"], df_mc["A1"], color="darkgreen", marker="o", markersize=2)
    ax_mc_a.axvline(x=ti, color="red", linestyle="--")
    ax_mc_a.grid(True, alpha=0.3)
    st.pyplot(fig_mc_a)