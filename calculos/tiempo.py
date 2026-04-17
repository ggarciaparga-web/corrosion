import numpy as np
from scipy.special import erf

def calcular_iniciacion(tipo_ataque, inputs):
    # 1. Recuperar inputs básicos
    t_analisis = inputs['t_analisis']
    recubrimiento = inputs['recubrimiento'] # mm
    
    ti = 0 # Valor por defecto
    
    if tipo_ataque == "Carbonatación":
        # Variables específicas de carbonatación
        c_cemento = inputs.get('c_cemento', 350)
        cs_co2 = inputs.get('cs_co2', 800)
        
        # Parámetros fijos
        cao_perc = 65.0
        d_co2 = 2e-8 # m2/s
        
        # Cálculo de la constante de carbonatación
        a_param = c_cemento * (cao_perc / 100.0) * (44.0 / 56.0) * 0.6
        cs_kg_m3 = cs_co2 / 1e6
        
        # v_co2 en mm/año^0.5
        v_co2_seg = np.sqrt((2 * d_co2 * cs_kg_m3) / a_param) * 1000
        v_co2_año = v_co2_seg * np.sqrt(31536000) 
        
        # Tiempo de iniciación
        ti = (recubrimiento / v_co2_año)**2
        
    elif tipo_ataque == "Cloruros":
        # Parámetros para modelo de Fick
        c_crit = 0.6
        c_surf = 2.0
        c_0 = 0.1
        d_ref = 7.12e-12
        n_ageing = 0.4288
        t_0_cl = 0.0767
        
        # Iteración para encontrar el año en el que C(t) >= Ccrit
        tiempos_fick = np.linspace(0.001, t_analisis, 5000)
        for t in tiempos_fick:
            d_cl = d_ref * (t_0_cl / t)**n_ageing
            t_seg = t * 31536000
            # Recubrimiento pasado a metros para coherencia con d_cl
            arg = (recubrimiento / 1000.0) / (2 * np.sqrt(d_cl * t_seg))
            c_t = c_0 + (c_surf - c_0) * (1 - erf(arg))
            
            if c_t >= c_crit:
                ti = t
                break
    
    # IMPORTANTE: Devolvemos solo ti para que app.py lo reciba correctamente
    return float(ti)
