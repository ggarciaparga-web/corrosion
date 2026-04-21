def calcular_tensiones_pretensado(ti_v, i_corr_v, alpha_v, t_analisis_v, phi0_v, n_pres, fpu_v, b_v, h_v, d_prima_v):
    # Propiedades geométricas de la sección de hormigón
    a_concrete = b_v * h_v
    i_beam = b_v * h_v**3 / 12.0
    y_inf = h_v / 2.0
    y_sup = h_v / 2.0
    e = (h_v / 2.0) - d_prima_v # Excentricidad

    # Área inicial y fuerzas
    a_pres_0 = n_pres * math.pi * phi0_v**2 / 4.0
    p0 = 0.75 * fpu_v * a_pres_0
    p_losses = 0.25 * p0 # Pérdidas estimadas

    tiempos = np.arange(0, t_analisis_v + 1, 1)
    rows = []

    for t in tiempos:
        if t <= ti_v:
            # Fase Pasiva
            px = 0.0
            mcorr = 0.0
        else:
            # Fase Activa
            t_efectivo = t - ti_v
            px = 0.0116 * i_corr_v * t_efectivo
            phi_f = max(phi0_v - alpha_v * px, 0.0)
            a_pres_f = n_pres * math.pi * phi_f**2 / 4.0
            mcorr = max(0.0, min(1.0, 1.0 - (a_pres_f / a_pres_0)))

        # Tensiones base (P0 + pérdidas)
        # Nota: Usamos la convención de signos del código original
        sig_inf_eff = (p0 + p_losses) / a_concrete + (((p0 + p_losses) * e) * y_inf) / i_beam
        sig_sup_eff = (p0 - p_losses) / a_concrete - (((p0 - p_losses) * e) * y_sup) / i_beam

        # Aplicar reducción por corrosión
        sig_inf_final = sig_inf_eff * (1.0 - mcorr)
        sig_sup_final = sig_sup_eff * (1.0 - mcorr)

        rows.append({
            "Time": t,
            "sigma_inf": sig_inf_final,
            "sigma_sup": sig_sup_final,
            "mcorr": mcorr
        })
    
    return pd.DataFrame(rows)
