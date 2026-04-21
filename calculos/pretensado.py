import math
import numpy as np
import pandas as pd

def ejecutar_simulacion_pretensado(inputs, ti, i_corr, alpha_val):
    """
    Calcula la evolución de tensiones en una sección pretensada.
    Extrae los datos geométricos directamente del diccionario 'inputs'.
    """
    # --- Extracción de datos del diccionario ---
    b_v = inputs["ancho_b"]
    h_v = inputs["canto_d"] # Usamos d como altura total según tu lógica previa
    t_analisis_v = inputs["t_analisis"]
    
    # Datos específicos de pretensado
    phi0_v = inputs["phi0_prestress"]
    n_pres = inputs["n_prestress"]
    fpu_v = inputs["fpu_prestress"]
    d_prima_v = inputs["d_prima_prestress"]

    # --- Propiedades geométricas ---
    a_concrete = b_v * h_v
    i_beam = b_v * h_v**3 / 12.0
    y_inf = h_v / 2.0
    y_sup = h_v / 2.0
    e = (h_v / 2.0) - d_prima_v  # Excentricidad

    # Área inicial y fuerzas
    a_pres_0 = n_pres * math.pi * phi0_v**2 / 4.0
    p0 = 0.75 * fpu_v * a_pres_0
    p_losses = 0.25 * p0  # Pérdidas estimadas iniciales

    tiempos = np.arange(0, t_analisis_v + 1, 1)
    rows = []

    for t in tiempos:
        if t <= ti:
            # Fase Pasiva: Sin corrosión
            px = 0.0
            mcorr = 0.0
        else:
            # Fase Activa: Corrosión efectiva
            t_efectivo = t - ti
            px = 0.0116 * i_corr * t_efectivo
            phi_f = max(phi0_v - alpha_val * px, 0.0)
            a_pres_f = n_pres * math.pi * phi_f**2 / 4.0
            # Relación de pérdida de área (mcorr)
            mcorr = max(0.0, min(1.0, 1.0 - (a_pres_f / a_pres_0)))

        # Tensiones base efectivas (P0 + pérdidas)
        sig_inf_eff = (p0 + p_losses) / a_concrete + (((p0 + p_losses) * e) * y_inf) / i_beam
        sig_sup_eff = (p0 - p_losses) / a_concrete - (((p0 - p_losses) * e) * y_sup) / i_beam

        # Aplicar reducción de la fuerza pretensora por corrosión
        sig_inf_final = sig_inf_eff * (1.0 - mcorr)
        sig_sup_final = sig_sup_eff * (1.0 - mcorr)

        rows.append({
            "time": t,
            "sigma_inferior": sig_inf_final,
            "sigma_superior": sig_sup_final,
            "mcorr": mcorr,
            "px": px
        })
    
    return pd.DataFrame(rows)
