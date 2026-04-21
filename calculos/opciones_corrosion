import pandas as pd
import numpy as np


def ejecutar_simulacion_completa(tipo_ataque, inputs, ti, corrosion_zone=None):
    """
    Runs the full simulation with optional corrosion localization.

    corrosion_zone:
        - "tension": corrosion affects only tension reinforcement (phi1)
        - "compression": corrosion affects only compression-side reinforcement (phi2) and stirrups (phiw)
        - "both": corrosion affects phi1, phi2 and phiw
    If corrosion_zone is None, it is taken from inputs["corrosion_zone"] or asked via input().
    """
    # --- 0. Corrosion zone selection ---
    zone = corrosion_zone or inputs.get("corrosion_zone")
    if zone is None:
        zone = input(
            'Corrosion zone ("tension", "compression", "both"): '
        ).strip().lower()

    zone = zone.lower()
    valid_zones = {"tension", "compression", "both"}
    if zone not in valid_zones:
        raise ValueError(
            f'Invalid corrosion_zone="{zone}". Must be one of {sorted(valid_zones)}.'
        )

    # --- 1. Parameters from UI ---
    t_end = inputs["t_analisis"]
    i_corr = inputs["i_corr"]
    recubrimiento = inputs["recubrimiento"]
    fck = inputs["fck"]

    alpha = 2 if tipo_ataque == "Carbonatación" else 10
    limite_px = 0.05 if tipo_ataque == "Carbonatación" else 0.5

    # Geometry and materials
    phi1_0 = inputs["phi_base"]
    phi_w0 = 0.0001
    phi2_0 = 20
    n_bottom = inputs["n_barras"]
    b_initial = inputs["ancho_b"]
    d_initial = inputs["canto_d"]

    fyd = inputs["fy"] / 1.15
    fci = 0.333 * fck ** (2 / 3)

    def calc_mu_local(a1, b_act, d_act):
        if a1 <= 0:
            return 0.0
        fcd = fck / 1.5
        x = (a1 * fyd) / (0.8 * b_act * fcd)
        z = d_act - 0.4 * x
        return max((a1 * fyd * z) / 1e6, 0.0)

    # --- 2. Base simulation matrix ---
    times = np.arange(0, t_end + 1, 1)
    rows = []

    for t in times:
        px = 0.0116 * i_corr * (t - ti) if t > ti else 0.0

        # Apply corrosion loss depending on zone
        if zone == "tension":
            p1 = max(0.0, phi1_0 - alpha * px)
            p2 = phi2_0
            pw = phi_w0
        elif zone == "compression":
            p1 = phi1_0
            p2 = max(0.0, phi2_0 - alpha * px)
            pw = max(0.0, phi_w0 - alpha * px)
        else:  # zone == "both"
            p1 = max(0.0, phi1_0 - alpha * px)
            p2 = max(0.0, phi2_0 - alpha * px)
            pw = max(0.0, phi_w0 - alpha * px)

        a1 = (np.pi * p1**2 / 4.0) * n_bottom
        a2 = (np.pi * p2**2 / 4.0)
        aw = (np.pi * pw**2 / 4.0)

        rows.append(
            {
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
            }
        )

    df_base = pd.DataFrame(rows)

    # --- 3. Critical points ---
    # If corrosion only affects tension, skip cracking/event points and keep b,d constant.
    if zone == "tension":
        df_final = df_base.copy()
        df_final["b"] = b_initial
        df_final["d"] = d_initial
        df_final["Mu (kNm)"] = df_final.apply(
            lambda r: calc_mu_local(r["A1 (mm2)"], r["b"], r["d"]), axis=1
        )

        df_points = df_final.iloc[[0]].copy()

        if i_corr > 0:
            t_vertical = ti + (limite_px / (0.0116 * i_corr))
        else:
            t_vertical = np.inf

        return df_final, t_vertical, limite_px, df_points

    # Otherwise (compression or both), keep your original point logic:
    px0 = max(
        0.0, (83.8 + 7.4 * (recubrimiento / phi1_0) - 22.6 * fci) * 1e-3
    )

    points = []
    row0 = df_base.iloc[0].copy()
    row0["b"], row0["d"] = b_initial, d_initial
    points.append(row0)

    # Cracking point
    mask_px0 = df_base["Px (mm)"] >= px0
    if mask_px0.any():
        idx_px0 = mask_px0.idxmax()
        row_px0 = df_base.loc[idx_px0].copy()
        row_px0["b"], row_px0["d"] = b_initial, d_initial
        points.append(row_px0)

    # Events 3 and 4
    ev3, ev4 = None, None
    for _, row in df_base.iterrows():
        r1 = row["rho1"] * 100
        r2 = row["rho2"] * 100
        px = row["Px (mm)"]
        aw = row["Aw (mm2)"]

        if r1 > 1.5 and aw > (0.0036 * b_initial) and px > 0.2 and ev4 is None:
            ev4 = row.copy()
            ev4["b"], ev4["d"] = b_initial - 2.0 * recubrimiento, d_initial - recubrimiento

        if ev3 is None:
            cond1 = (r1 < 1.0 and r2 < 5.0 and px > 0.4)
            cond2 = (r1 < 1.0 and r2 > 5.0 and px > 0.2)
            cond3 = (r1 > 1.5 and r2 > 0.5 and px > 0.2)
            if cond1 or cond2 or cond3:
                ev3 = row.copy()
                ev3["b"], ev3["d"] = b_initial, d_initial - recubrimiento

    if ev3 is not None:
        points.append(ev3)
    if ev4 is not None:
        points.append(ev4)

    df_points = (
        pd.DataFrame(points)
        .sort_values("Px (mm)")
        .drop_duplicates("Px (mm)")
        .reset_index(drop=True)
    )
    df_points["Mu (kNm)"] = df_points.apply(
        lambda r: calc_mu_local(r["A1 (mm2)"], r["b"], r["d"]), axis=1
    )

    # --- 4. Final matrix generation ---
    last_crit = df_points.iloc[-1]
    df_remaining = df_base[df_base["Tiempo (y)"] > last_crit["Tiempo (y)"]].copy()
    df_remaining["b"] = last_crit["b"]
    df_remaining["d"] = last_crit["d"]

    df_final = pd.concat([df_points, df_remaining], ignore_index=True)
    df_final["Mu (kNm)"] = df_final.apply(
        lambda r: calc_mu_local(r["A1 (mm2)"], r["b"], r["d"]), axis=1
    )

    if i_corr > 0:
        t_vertical = ti + (limite_px / (0.0116 * i_corr))
    else:
        t_vertical = np.inf

    return df_final, t_vertical, limite_px, df_points
``
