import numpy as np
import pandas as pd


def ejecutar_simulacion_corrosion_zona(tipo_ataque, inputs, ti, corrosion_zone=None):
    """
    CONTEVECT-like simulation with corrosion acting in:
      - "tension": only phi1 reduces
      - "compression": only phi2 and phiw reduce
      - "both": phi1, phi2, phiw reduce

    Key fixes vs. your current version:
      1) Removes HTML entities (&lt; &gt;) that break Python.
      2) Corrects the return variable (df_points).
      3) Applies b/d reductions as PIECEWISE CONSTANT over ALL years after events
         (your previous approach only changed geometry at the event rows + after last event).
      4) Makes phi_w0 configurable (your phi_w0=0.0001 mm makes Aw ~ 0 always, so Event 4 never triggers).
    """
    zone = (corrosion_zone or inputs.get("corrosion_zone", "both")).strip().lower()
    if zone not in {"tension", "compression", "both"}:
        raise ValueError('corrosion_zone must be "tension", "compression", or "both".')

    # Inputs
    t_end = int(inputs["t_analisis"])
    i_corr = float(inputs["i_corr"])
    cover = float(inputs["recubrimiento"])
    fck = float(inputs["fck"])

    # Attack parameters
    alpha = 2.0 if tipo_ataque == "Carbonatación" else 10.0
    limite_px = 0.05 if tipo_ataque == "Carbonatación" else 0.5

    # Geometry/materials
    phi1_0 = float(inputs["phi_base"])
    phi2_0 = float(inputs.get("phi2_0", 20.0))

    # IMPORTANT: if you keep 0.0001 mm, Aw is ~0 and Event 4 condition will never be met.
    # Use a realistic stirrup diameter (e.g., 6–10 mm), or provide it from the UI as inputs["phi_w0"].
    phi_w0 = float(inputs.get("phi_w0", 8.0))

    n_bottom = int(inputs["n_barras"])
    b_initial = float(inputs["ancho_b"])
    d_initial = float(inputs["canto_d"])

    fyd = float(inputs["fy"]) / 1.15
    fci = 0.333 * fck ** (2.0 / 3.0)

    def calc_mu_local(a1, b_act, d_act):
        if a1 <= 0.0 or b_act <= 0.0 or d_act <= 0.0:
            return 0.0
        fcd = fck / 1.5
        x = (a1 * fyd) / (0.8 * b_act * fcd)
        z = d_act - 0.4 * x
        return max((a1 * fyd * z) / 1e6, 0.0)

    # -----------------------------
    # 1) Base time history (all years)
    # -----------------------------
    times = np.arange(0, t_end + 1, 1)
    rows = []

    for t in times:
        px = 0.0116 * i_corr * (t - ti) if t > ti else 0.0

        if zone == "tension":
            p1 = max(0.0, phi1_0 - alpha * px)
            p2 = phi2_0
            pw = phi_w0
        elif zone == "compression":
            p1 = phi1_0
            p2 = max(0.0, phi2_0 - alpha * px)
            pw = max(0.0, phi_w0 - alpha * px)
        else:  # both
            p1 = max(0.0, phi1_0 - alpha * px)
            p2 = max(0.0, phi2_0 - alpha * px)
            pw = max(0.0, phi_w0 - alpha * px)

        a1 = (np.pi * p1**2 / 4.0) * n_bottom
        a2 = (np.pi * p2**2 / 4.0)
        aw = (np.pi * pw**2 / 4.0)

        rows.append(
            {
                "Tiempo (y)": int(t),
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

    # If only tension corrodes: no events, no geometry loss (as you requested originally)
    if zone == "tension":
        df_final = df_base.copy()
        df_final["b"] = b_initial
        df_final["d"] = d_initial
        df_final["Mu (kNm)"] = df_final.apply(
            lambda r: calc_mu_local(r["A1 (mm2)"], r["b"], r["d"]), axis=1
        )

        df_points = df_final.iloc[[0]].copy()
        t_vertical = ti + (limite_px / (0.0116 * i_corr)) if i_corr > 0 else np.inf
        return df_final, t_vertical, limite_px, df_points

    # -----------------------------
    # 2) Critical thresholds / events
    # -----------------------------
    px0 = max(0.0, (83.8 + 7.4 * (cover / phi1_0) - 22.6 * fci) * 1e-3)

    # Find first time crossing px0 (cracking point)
    t_px0 = None
    mask_px0 = df_base["Px (mm)"] >= px0
    if mask_px0.any():
        t_px0 = int(df_base.loc[mask_px0.idxmax(), "Tiempo (y)"])

    # Events 3 and 4 detection
    t_ev3 = None
    t_ev4 = None

    for _, row in df_base.iterrows():
        r1 = float(row["rho1"]) * 100.0
        r2 = float(row["rho2"]) * 100.0
        px_val = float(row["Px (mm)"])
        aw_val = float(row["Aw (mm2)"])

        # Event 4 (spalling) - your original logic
        if (
            t_ev4 is None
            and r1 > 1.5
            and aw_val > (0.0036 * b_initial)
            and px_val > 0.2
        ):
            t_ev4 = int(row["Tiempo (y)"])

        # Event 3 (cover loss) - your original logic
        if t_ev3 is None:
            cond1 = (r1 < 1.0 and r2 < 5.0 and px_val > 0.4)
            cond2 = (r1 < 1.0 and r2 > 5.0 and px_val > 0.2)
            cond3 = (r1 > 1.5 and r2 > 0.5 and px_val > 0.2)
            if cond1 or cond2 or cond3:
                t_ev3 = int(row["Tiempo (y)"])

    # -----------------------------
    # 3) Apply geometry changes over ALL years (piecewise)
    # -----------------------------
    df_final = df_base.copy()
    df_final["b"] = b_initial
    df_final["d"] = d_initial

    # After Event 3: reduce d
    if t_ev3 is not None:
        df_final.loc[df_final["Tiempo (y)"] >= t_ev3, "d"] = d_initial - cover

    # After Event 4: reduce b and d (overrides Event 3 if later)
    if t_ev4 is not None:
        df_final.loc[df_final["Tiempo (y)"] >= t_ev4, "b"] = b_initial - 2.0 * cover
        df_final.loc[df_final["Tiempo (y)"] >= t_ev4, "d"] = d_initial - cover

    # Compute Mu for every year with the updated geometry
    df_final["Mu (kNm)"] = df_final.apply(
        lambda r: calc_mu_local(r["A1 (mm2)"], r["b"], r["d"]), axis=1
    )

    # -----------------------------
    # 4) Points table (for plotting markers)
    # -----------------------------
    points_times = [0]
    if t_px0 is not None:
        points_times.append(t_px0)
    if t_ev3 is not None:
        points_times.append(t_ev3)
    if t_ev4 is not None:
        points_times.append(t_ev4)

    points_times = sorted(set(points_times))
    df_points = df_final[df_final["Tiempo (y)"].isin(points_times)].copy()
    df_points = df_points.sort_values("Px (mm)").drop_duplicates("Px (mm)").reset_index(drop=True)

    # Vertical line time
    t_vertical = ti + (limite_px / (0.0116 * i_corr)) if i_corr > 0 else np.inf

    return df_final, t_vertical, limite_px, df_points



