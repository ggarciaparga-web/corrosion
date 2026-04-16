import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

st.set_page_config(page_title="Visor Corrosión Colab", layout="wide")

st.title("🏗️ Análisis de Capacidad Flexural")
st.info("Configura los parámetros en la izquierda y pulsa 'ACTUALIZAR GRÁFICA'")

# --- FORMULARIO EN EL SIDEBAR (Para evitar errores de carga) ---
with st.sidebar.form("my_form"):
    st.header("⚙️ Parámetros")

    t_end = st.number_input("Años de estudio", value=100)
    i_corr = st.number_input("i_corr (µA/cm²)", value=2.0)
    alpha = st.number_input("Alpha", value=10.0)

    st.divider()
    b = st.number_input("Ancho b (mm)", value=150)
    d = st.number_input("Canto d (mm)", value=300)
    r2 = st.number_input("r2 (mm)", value=20)

    st.divider()
    fck = st.number_input("fck (MPa)", value=25)
    fy = st.number_input("fy (MPa)", value=500)
    phi0 = st.number_input("Ø inicial (mm)", value=20.0)
    n_bar = st.number_input("Nº barras", value=2)

    # Este botón es la clave: evita que la web parpadee y de error
    submit_button = st.form_submit_button(label='🚀 ACTUALIZAR GRÁFICA')

# --- LÓGICA DE CÁLCULO ---
times = np.arange(0, t_end + 1, 1)
results = []
for t in times:
    px = 0.0116 * i_corr * t
    phi_t = max(0.0, phi0 - alpha * px)
    a_t = (math.pi * phi_t**2 / 4.0) * n_bar

    if phi_t <= 0:
        mu_std, mu_cons = 0.0, 0.0
    else:
        fyd = fy / 1.15
        fcd = (fck / 1.5) * (0.75 * min(1.0, (30.0 / fck)**(1/3)))
        x = (a_t * fyd) / (0.8 * b * fcd)
        mu_std = a_t * fyd * (d - 0.4 * x) / 1e6
        mu_cons = a_t * fyd * max(0.0, (d - r2 - 0.4 * x)) / 1e6
    results.append([t, phi_t, mu_std, mu_cons])

df = pd.DataFrame(results, columns=["Time", "Phi", "Mu", "Mu_Cons"])

# --- GRÁFICA ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["Time"], df["Mu"], label="Mu Standard (fib)", color="#1f77b4", lw=3)
ax.plot(df["Time"], df["Mu_Cons"], label="Mu Conservador", color="#ff7f0e", ls="--", lw=2)

# Líneas Verdes (Tendencias)
def get_point(m):
    idx = np.where(m <= 0)[0]
    i = idx[0] if idx.size > 0 else len(m)-1
    return df["Time"].iloc[i], m.iloc[i]

t1, y1 = get_point(df["Mu"])
t2, y2 = get_point(df["Mu_Cons"])
ax.plot([0, t1], [df["Mu"].iloc[0], y1], color="green", ls=":", label="Tendencia Lineal")
ax.plot([0, t2], [df["Mu_Cons"].iloc[0], y2], color="green", ls=":")

ax.set_xlabel("Tiempo (años)")
ax.set_ylabel("Capacidad Mu (kNm)")
ax.legend()
ax.grid(True, alpha=0.3)

st.pyplot(fig)

# Tabla simple (más estable que st.dataframe)
st.write("### Resumen de degradación")