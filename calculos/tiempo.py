import numpy as np
from scipy.special import erf

def calcular_iniciacion(tipo_ataque, inputs):
    t_analisis = inputs['t_analisis']
    recubrimiento = inputs['recubrimiento']
    i_corr = inputs['i_corr']
    
    ti = 0.0
    
    # --- Lógica de cálculo de ti (Carbonatación o Cloruros) ---
    if tipo_ataque == "Carbonatación":
        c_cemento = inputs.get('c_cemento', 450.0)
        cs_co2 = inputs.get('cs_co2', 800.0)
        a_param = c_cemento * (65.0 / 100.0) * (44.0 / 56.0) * 0.6
        cs_kg_m3 = cs_co2 / 1e6
        v_co2_seg = np.sqrt((2 * 2e-08 * cs_kg_m3) / a_param) * 1000
        v_co2_año = v_co2_seg * np.sqrt(31536000)
        ti = (recubrimiento / v_co2_año)**2
    else:
        tiempos_fick = np.linspace(0.001, t_analisis, 5000)
        for t in tiempos_fick:
            d_cl = 7.12e-12 * (0.0767 / t)**0.4288
            arg = (recubrimiento / 1000.0) / (2 * np.sqrt(d_cl * t * 31536000))
            c_t = 0.1 + (2.0 - 0.1) * (1 - erf(arg))
            if c_t >= 0.6:
                ti = t
                break

    # --- Generación de los datos para la gráfica de Px ---
    tiempos_px = np.linspace(0, t_analisis, 1000)
    px_plot = [0.0116 * i_corr * (t - ti) if t > ti else 0.0 for t in tiempos_px]

    # Devolvemos ti (número) y los datos de la gráfica (listas)
    return float(ti), tiempos_px, px_plot
