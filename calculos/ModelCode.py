import numpy as np
import pandas as pd

def simulacion_total(tipo_ataque, inputs, ti):
    """
    Cálculo de resistencia residual según Model Code 2023.
    Recibe el ti ya calculado por el módulo de iniciación.
    """
    # 1. RECUPERAR INPUTS GENERALES
    t_end = inputs['t_analisis']
    recubrimiento = inputs['recubrimiento']
    i_corr = inputs['i_corr']
    fck = inputs['fck']
    fy = inputs['fy']
    b_section = inputs['ancho_b']
    d_initial = inputs['canto_d']
    phi1_initial = inputs['phi_base']
    n_bottom = inputs['n_barras']
    r2 = inputs['r2'] # Distancia armadura superior
    
    # Parámetros según tipo de ataque para la propagación
    if tipo_ataque == "Carbonatación":
        alpha = 2
        limite_px_um = 0.05  # 50 µm en mm
    else:
        alpha = 10
        limite_px_um = 0.5   # 500 µm en mm

    # 2. SIMULACIÓN DE PROPAGACIÓN (MODEL CODE LOGIC)
    times = np.arange(0, t_end + 1, 1)
    results = []
    
    # Tiempo para la recta vertical (cuando px alcanza el límite tras ti)
    t_vertical = ti + (limite_px_um / (0.0116 * i_corr))

    for t in times:
        if t <= ti:
            px = 0.0
            phi1_current = phi1_initial
        else:
            px = 0.0116 * i_corr * (t - ti)
            phi1_current = max(0.0, phi1_initial - alpha * px)

        # Áreas
        a_initial = (np.pi * phi1_initial ** 2 / 4.0) * n_bottom
        a_corr = (np.pi * phi1_current ** 2 / 4.0) * n_bottom
        m_corr = (a_initial - a_corr) / a_initial if a_initial > 0 else 1.0

        if phi1_current <= 0 or m_corr >= 1.0:
            mu_res, mu_cons = 0.0, 0.0
        else:
            # Lógica Model Code fib 2023
            fyd = fy / 1.15
            fcd_nom = fck / 1.5
            # Factor nfc basado en fck
            nfc = min(1.0, (30.0 / fck) ** (1.0 / 3.0))
            kc = 0.75 * nfc
            fcd_red = kc * fcd_nom

            # Eje neutro y momentos
            x = (a_corr * fyd) / (0.8 * b_section * fcd_red)
            
            # Brazo estándar
            z_std = d_initial - 0.4 * x
            mu_res = a_corr * fyd * z_std / 1e6
            
            # Brazo conservador (restando r2)
            z_cons = max(0.0, d_initial - r2 - 0.4 * x)
            mu_cons = a_corr * fyd * z_cons / 1e6

        results.append({
            "Time": t,
            "Px": px,
            "phi": phi1_current,
            "Mu (kNm)": max(mu_res, 0.0),
            "Mu Cons (kNm)": max(mu_cons, 0.0)
        })

    # IMPORTANTE: Retornamos exactamente lo que app.py espera recibir
    return pd.DataFrame(results), t_vertical
