import pandas as pd
import numpy as np

def ejecutar_simulacion_completa(tipo_ataque, inputs, ti):
    """
    Integra la lógica de iniciación con la matriz de degradación y eventos de CONTEVECT.
    """
    # --- 1. Parámetros desde la Interfaz ---
    t_end = inputs['t_analisis']
    i_corr = inputs['i_corr']
    recubrimiento = inputs['recubrimiento']
    fck = inputs['fck']
    
    # Asignación de Alpha según ataque
    alpha = 2 if tipo_ataque == "Carbonatación" else 10
    # Límite vertical para la gráfica (μm a mm)
    limite_px = 0.05 if tipo_ataque == "Carbonatación" else 0.5

    # Parámetros geométricos y materiales
    phi1_0 = inputs['phi_base']
    phi_w0 = 0.0001 
    phi2_0 = 20
    n_bottom = inputs['n_barras']
    b_initial = inputs['ancho_b']
    d_initial = inputs['canto_d']
    
    fyd = inputs['fy'] / 1.15
    fci = 0.333 * fck ** (2 / 3)

    # --- 2. Lógica de Simulación (Matriz Base) ---
    times = np.arange(0, t_end + 1, 1)
    rows = []

    for t in times:
        # Px empieza a contar solo después de ti
        px = 0.0116 * i_corr * (t - ti) if t > ti else 0.0

        p1 = max(0.0, phi1_0 - alpha * px)
        p2 = max(0.0, phi2_0 - alpha * px)
        pw = max(0.0, phi_w0 - alpha * px)

        a1 = (np.pi * p1 ** 2 / 4.0) * n_bottom
        a2 = (np.pi * p2 ** 2 / 4.0)
        aw = (np.pi * pw ** 2 / 4.0)

        rows.append({
            "Tiempo (y)": t,
            "Px (mm)": px,
            "phi1 (mm)": p1,
            "phi2 (mm)": p2,
            "phiw (mm)": pw,
            "A1 (mm2)": a1,
            "A2 (mm2)": a2,
            "Aw (mm2)": aw,
            "rho1": a1 / (b_initial * d_initial),
            "rho2": a2 / (b_initial * d_initial),
        })

    df_base = pd.DataFrame(rows)

    # --- 3. Cálculo de Puntos Críticos (Lógica de Eventos) ---
    px0 = max(0.0, (83.8 + 7.4 * (recubrimiento / phi1_0) - 22.6 * fci) * 1e-3)
    
    def calc_mu_local(a1, b_act, d_act):
        if a1 <= 0: return 0.0
        fcd = fck / 1.5
        x = (a1 * fyd) / (0.8 * b_act * fcd)
        z = d_act - 0.4 * x
        return max((a1 * fyd * z) / 1e6, 0.0)

    points = []
    # Estado inicial (t=0)
    row0 = df_base.iloc[0].copy()
    row0["b"], row0["d"] = b_initial, d_initial
    points.append(row0)

    # Punto de fisuración Px >= px0
    mask_px0 = df_base["Px (mm)"] >= px0
    if mask_px0.any():
        idx_px0 = mask_px0.idxmax()
        row_px0 = df_base.loc[idx_px0].copy()
        row_px0["b"], row_px0["d"] = b_initial, d_initial
        points.append(row_px0)

    ev3 = None
    ev4 = None
    for _, row in df_base.iterrows():
        r1, r2, px, aw = row["rho1"]*100, row["rho2"]*100, row["Px (mm)"], row["Aw (mm2)"]
        # Evento 4: Desprendimiento total
        if r1 > 1.5 and aw > (0.0036 * b_initial) and px > 0.2 and ev4 is None:
            ev4 = row.copy()
            ev4["b"], ev4["d"] = b_initial - 2.0 * recubrimiento, d_initial - recubrimiento
        # Evento 3: Pérdida de recubrimiento
        if ev3 is None:
            if (r1 < 1.0 and r2 < 5.0 and px > 0.4) or (r1 < 1.0 and r2 > 5.0 and px > 0.2) or (r1 > 1.5 and r2 > 0.5 and px > 0.2):
                ev3 = row.copy()
                ev3["b"], ev3["d"] = b_initial, d_initial - recubrimiento

    if ev3 is not None: points.append(ev3)
    if ev4 is not None: points.append(ev4)

    df_points = pd.DataFrame(points).sort_values("Px (mm)").drop_duplicates("Px (mm)")
    
    # --- 4. Generación de Matriz Final Combinada ---
    last_crit = df_points.iloc[-1]
    df_remaining = df_base[df_base["Tiempo (y)"] > last_crit["Tiempo (y)"]].copy()
    df_remaining["b"] = last_crit["b"]
    df_remaining["d"] = last_crit["d"]

    df_final = pd.concat([df_points, df_remaining], ignore_index=True)
    df_final["Mu (kNm)"] = df_final.apply(lambda r: calc_mu_local(r["A1 (mm2)"], r["b"], r["d"]), axis=1)

    # Tiempo exacto de la recta vertical (cuando px alcanza el límite tras ti)
    t_vertical = ti + (limite_px / (0.0116 * i_corr))

    return df_final, t_vertical, limite_px, df_points
