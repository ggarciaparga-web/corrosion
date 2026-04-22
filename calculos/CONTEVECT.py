from __future__ import annotations

from typing import Dict, Optional, Tuple, List
import numpy as np
import pandas as pd

# Se asume que ya tienes definidas:
# - RcMaterials
# - Geometry, RectangularSection, ISection
# - build_geometry


def _flatten_geometry_to_row(geom, d_mm: float, d2_mm: float, damage_state: str) -> Dict:
    """
    Mete en el row la info necesaria para reconstruir la geometría en cada instante.
    """
    out = {
        "damage_state": damage_state,
        "d (mm)": float(d_mm),
        "d2 (mm)": float(d2_mm),
    }

    if isinstance(geom, RectangularSection):
        out.update(
            {
                "geom_type": "rect",
                "b (mm)": float(geom.b_mm),
                "h (mm)": float(geom.h_mm),
            }
        )
    elif isinstance(geom, ISection):
        out.update(
            {
                "geom_type": "i",
                "b_top (mm)": float(geom.b_top_mm),
                "t_top (mm)": float(geom.t_top_mm),
                "b_web (mm)": float(geom.b_web_mm),
                "b_bot (mm)": float(geom.b_bot_mm),
                "t_bot (mm)": float(geom.t_bot_mm),
                "h (mm)": float(geom.h_mm),
            }
        )
    else:
        out.update({"geom_type": "custom"})

    return out


def _compute_mu_from_row(r: pd.Series, materials: RcMaterials) -> float:
    """
    Calcula Mu (kNm) reconstruyendo la geometría desde la fila.
    """
    as_t = float(r["A1 (mm2)"])
    as_c = float(r.get("A2 (mm2)", 0.0))
    d_mm = float(r["d (mm)"])
    d2_mm = float(r["d2 (mm)"])

    geom_type = str(r.get("geom_type", "rect"))
    if geom_type == "rect":
        geom = RectangularSection(b_mm=float(r["b (mm)"]), h_mm=float(r["h (mm)"]))
    elif geom_type == "i":
        geom = ISection(
            b_top_mm=float(r["b_top (mm)"]),
            t_top_mm=float(r["t_top (mm)"]),
            b_web_mm=float(r["b_web (mm)"]),
            b_bot_mm=float(r["b_bot (mm)"]),
            t_bot_mm=float(r["t_bot (mm)"]),
            h_mm=float(r["h (mm)"]),
        )
    else:
        return 0.0

    return geom.moment_capacity_kNm(
        as_t_mm2=as_t,
        as_c_mm2=as_c,
        d_mm=d_mm,
        d2_mm=d2_mm,
        materials=materials,
    )


def ejecutar_simulacion_completa_actualizada_en_tiempo(
    tipo_ataque: str, inputs: Dict, ti: float, dt_years: float = 1.0
) -> Tuple[pd.DataFrame, float, float, pd.DataFrame]:
    """
    Versión "time-update": recorre el tiempo y va actualizando el estado (geometría, d, d2)
    en el mismo loop cuando saltan eventos.

    Devuelve:
      - df_time: fila por instante (año si dt=1), con estado, geometría y Mu(t)
      - t_vertical: línea vertical (como tenías)
      - limite_px
      - df_events: tabla con los eventos detectados (tiempos y estados)
    """
    # --- inputs base ---
    t_end = float(inputs["t_analisis"])
    i_corr = float(inputs["i_corr"])
    recubrimiento_mm = float(inputs["recubrimiento"])
    fck_mpa = float(inputs["fck"])
    fyk_mpa = float(inputs["fy"])

    alpha = 2.0 if tipo_ataque == "Carbonatación" else 10.0
    limite_px = 0.05 if tipo_ataque == "Carbonatación" else 0.5

    # Steel diameters (mm) and bar count
    phi1_0_mm = float(inputs["phi_base"])
    phi_w0_mm = 0.0001
    phi2_0_mm = 20.0
    n_bottom = int(inputs["n_barras"])

    # Geometría inicial
    geom0, d0_mm, d2_0_mm = build_geometry(inputs)

    materials = RcMaterials(fck_mpa=fck_mpa, fyk_mpa=fyk_mpa)

    # px0 (fisuración)
    fci = 0.333 * fck_mpa ** (2.0 / 3.0)
    px0 = max(0.0, (83.8 + 7.4 * (recubrimiento_mm / phi1_0_mm) - 22.6 * fci) * 1e-3)

    # Referencias para detección (legacy)
    b_initial = float(inputs.get("ancho_b", getattr(geom0, "b_mm", 1.0)))

    # Área de referencia para rhos (mantengo tu criterio b*d para compatibilidad)
    ref_b = float(inputs.get("ancho_b", getattr(geom0, "b_mm", 1.0)))
    ref_d = float(inputs.get("canto_d", d0_mm))
    ref_area = max(ref_b * ref_d, 1.0)

    # --- estado actual (se actualiza con el tiempo) ---
    geom = geom0
    d_mm = d0_mm
    d2_mm = d2_0_mm
    damage_state = "intact"

    cracked = False
    spall_bottom_done = False
    spall_bottom_sides_done = False

    rows: List[Dict] = []
    events: List[Dict] = []

    # --- loop temporal ---
    times = np.arange(0.0, t_end + 1e-9, dt_years, dtype=float)

    for t in times:
        # corrosión / penetración equivalente
        px = 0.0116 * i_corr * (t - ti) if t > ti else 0.0  # mm

        # diámetros degradados
        p1 = max(0.0, phi1_0_mm - alpha * px)
        p2 = max(0.0, phi2_0_mm - alpha * px)
        pw = max(0.0, phi_w0_mm - alpha * px)

        # áreas
        a1 = (np.pi * p1**2 / 4.0) * n_bottom
        a2 = (np.pi * p2**2 / 4.0)
        aw = (np.pi * pw**2 / 4.0)

        rho1 = a1 / ref_area
        rho2 = a2 / ref_area

        # -------- eventos "en tiempo" --------
        # Evento fisuración (solo marca, no cambia geom)
        if (not cracked) and (px >= px0):
            cracked = True
            events.append(
                {
                    "Time (y)": t,
                    "event": "cracking_px0",
                    "Px (mm)": px,
                    "damage_state": damage_state,
                }
            )

        # Evento 4: bottom + sides spall (según tu regla) — prioriza sobre 3 si coincide
        r1_pct = rho1 * 100.0
        r2_pct = rho2 * 100.0

        if (not spall_bottom_sides_done) and (r1_pct > 1.5) and (aw > (0.0036 * b_initial)) and (px > 0.2):
            geom, d_mm, d2_mm = geom.apply_damage(
                state="bottom_and_sides_spall",
                cover_mm=recubrimiento_mm,
                d_mm=d_mm,
                d2_mm=d2_mm,
            )
            damage_state = "bottom_and_sides_spall"
            spall_bottom_sides_done = True
            # Si ocurre este, ya tiene sentido dar por ocurrido el bottom_spall también
            spall_bottom_done = True

            events.append(
                {
                    "Time (y)": t,
                    "event": "bottom_and_sides_spall",
                    "Px (mm)": px,
                    "rho1(%)": r1_pct,
                    "rho2(%)": r2_pct,
                    "Aw (mm2)": aw,
                }
            )

        # Evento 3: bottom spall (solo si no ha ocurrido aún el 4)
        if (not spall_bottom_done) and (not spall_bottom_sides_done):
            cond = (
                (r1_pct < 1.0 and r2_pct < 5.0 and px > 0.4)
                or (r1_pct < 1.0 and r2_pct > 5.0 and px > 0.2)
                or (r1_pct > 1.5 and r2_pct > 0.5 and px > 0.2)
            )
            if cond:
                geom, d_mm, d2_mm = geom.apply_damage(
                    state="bottom_spall",
                    cover_mm=recubrimiento_mm,
                    d_mm=d_mm,
                    d2_mm=d2_mm,
                )
                damage_state = "bottom_spall"
                spall_bottom_done = True

                events.append(
                    {
                        "Time (y)": t,
                        "event": "bottom_spall",
                        "Px (mm)": px,
                        "rho1(%)": r1_pct,
                        "rho2(%)": r2_pct,
                        "Aw (mm2)": aw,
                    }
                )

        # -------- guardar fila temporal con el estado ACTUAL --------
        row = {
            "Time (y)": t,
            "Px (mm)": px,
            "phi1 (mm)": p1,
            "phi2 (mm)": p2,
            "phiw (mm)": pw,
            "A1 (mm2)": a1,
            "A2 (mm2)": a2,
            "Aw (mm2)": aw,
            "rho1": rho1,
            "rho2": rho2,
        }
        row.update(_flatten_geometry_to_row(geom, d_mm, d2_mm, damage_state))
        rows.append(row)

    df_time = pd.DataFrame(rows)
    df_time["Mu (kNm)"] = df_time.apply(lambda r: _compute_mu_from_row(r, materials), axis=1)

    df_events = pd.DataFrame(events)

    # Línea vertical (igual que antes)
    t_vertical = ti + (limite_px / (0.0116 * i_corr))

    return df_time, t_vertical, limite_px, df_events

