import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

def calcular_iniciacion(tipo, inputs):
    # Inputs comunes
    t_analisis = inputs['t_analisis']
    recubrimiento = inputs['recubrimiento'] # mm
    i_corr = inputs['i_corr']
    
    ti = 0
    tiempos_px = np.linspace(0, t_analisis, 1000)
    limite_vertical = 0
    
    if tipo == "Carbonatación":
        alpha = 2
        limite_vertical = 50 / 1000 # Convertimos 50μm a mm para la gráfica si px está en mm
        
        # Variables específicas
        a_param = inputs['c_cemento'] * (65 / 100.0) * (44.0 / 56.0) * 0.6
        cs_kg_m3 = inputs['cs_co2'] / 1e6
        d_co2 = 2e-8 # m2/s
        
        # v_co2 en mm/año^0.5
        v_co2_seg = np.sqrt((2 * d_co2 * cs_kg_m3) / a_param) * 1000
        v_co2_año = v_co2_seg * np.sqrt(31536000) 
        ti = (recubrimiento / v_co2_año)**2
        
    elif tipo == "Cloruros":
        alpha = 10
        limite_vertical = 500 / 1000 # 500μm a mm
        
        # Variables específicas
        c_crit = 0.6
        c_surf = 2.0
        c_0 = 0.1
        d_ref = 7.12e-12
        n_ageing = 0.4288
        t_0_cl = 0.0767
        
        # Iteración Fick
        tiempos_fick = np.linspace(0.001, t_analisis, 5000)
        for t in tiempos_fick:
            d_cl = d_ref * (t_0_cl / t)**n_ageing
            t_seg = t * 31536000
            arg = (recubrimiento / 1000.0) / (2 * np.sqrt(d_cl * t_seg))
            c_t = c_0 + (c_surf - c_0) * (1 - erf(arg))
            if c_t >= c_crit:
                ti = t
                break

    # Cálculo de px (Penetración de corrosión) en mm
    # px = 0.0116 * i_corr * t_corrosion
    px_plot = [0.0116 * i_corr * (t - ti) if t > ti else 0.0 for t in tiempos_px]
    
    return tiempos_px, px_plot, ti, limite_vertical
