from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


# =========================================================
# MATERIALS
# =========================================================
class RcMaterials:
    def __init__(self, fck_mpa: float, fyk_mpa: float):
        self.fck = fck_mpa
        self.fyk = fyk_mpa
        self.fcd = fck_mpa / 1.5
        self.fyd = fyk_mpa / 1.15


# =========================================================
# GEOMETRIES
# =========================================================
class Geometry:
    def moment_capacity_kNm(
        self,
        as_t_mm2: float,
        as_c_mm2: float,
        d_mm: float,
        d2_mm: float,
        materials: RcMaterials,
    ) -> float:
        raise NotImplementedError

    def apply_damage(
        self,
        state: str,
        cover_mm: float,
        d_mm: float,
        d2_mm: float,
    ) -> Tuple["Geometry", float, float]:
        raise NotImplementedError


class RectangularSection(Geometry):
    def __init__(self, b_mm: float, h_mm: float):
        self.b_mm = b_mm
        self.h_mm = h_mm

    def moment_capacity_kNm(
        self,
        as_t_mm2: float,
        as_c_mm2: float,
        d_mm: float,
        d2_mm: float,
        materials: RcMaterials,
    ) -> float:
        if as_t_mm2 <= 0.0:
            return 0.0

        x = (as_t_mm2 * materials.fyd) / (
            0.8 * self.b_mm * materials.fcd
        )
        z = d_mm - 0.4 * x
        mu = as_t_mm2 * materials.fyd * z
        return max(mu / 1e6, 0.0)

    def apply_damage(
        self,
        state: str,
        cover_mm: float,
        d_mm: float,
        d2_mm: float,
    ) -> Tuple["RectangularSection", float, float]:
        if state == "bottom_spall":
            return (
                RectangularSection(self.b_mm, self.h_mm - cover_mm),
                d_mm - cover_mm,
                d2_mm,
            )

        if state == "bottom_and_sides_spall":
            return (
                RectangularSection(
                    self.b_mm - 2.0 * cover_mm,
                    self.h_mm - cover_mm,
                ),
                d_mm - cover_mm,
                d2_mm,
            )

        return self, d_mm, d2_mm


class ISection(Geometry):
    def __init__(
        self,
        b_top_mm: float,
        t_top_mm: float,
        b_web_mm: float,
        b_bot_mm: float,
        t_bot_mm: float,
        h_mm: float,
    ):
        self.b_top_mm = b_top_mm
        self.t_top_mm = t_top_mm
        self.b_web_mm = b_web_mm
        self.b_bot_mm = b_bot_mm
        self.t_bot_mm = t_bot_mm
        self.h_mm = h_mm

    def moment_capacity_kNm(
        self,
        as_t_mm2: float,
        as_c_mm2: float,
        d_mm: float,
        d2_mm: float,
        materials: RcMaterials,
    ) -> float:
        if as_t_mm2 <= 0.0:
            return 0.0

        # Conservative: compression only in top flange
        b_eff = self.b_top_mm
        x = (as_t_mm2 * materials.fyd) / (
            0.8 * b_eff * materials.fcd
        )
        z = d_mm - 0.4 * x
        mu = as_t_mm2 * materials.fyd * z
        return max(mu / 1e6, 0.0)

    def apply_damage(
        self,
        state: str,
        cover_mm: float,
        d_mm: float,
        d2_mm: float,
    ) -> Tuple["ISection", float, float]:
        if state == "bottom_spall":
            return (
                ISection(
                    self.b_top_mm,
                    self.t_top_mm,
                    self.b_web_mm,
                    self.b_bot_mm - 2.0 * cover_mm,
                    self.t_bot_mm - cover_mm,
                    self.h_mm - cover_mm,
                ),
                d_mm - cover_mm,
                d2_mm,
            )

        if state == "bottom_and_sides_spall":
            return (
                ISection(
                    self.b_top_mm,
                    self.t_top_mm,
                    self.b_web_mm - 2.0 * cover_mm,
                    self.b_bot_mm - 2.0 * cover_mm,
                    self.t_bot_mm - cover_mm,
                    self.h_mm - cover_mm,
                ),
                d_mm - cover_mm,
                d2_mm,
            )

        return self, d_mm, d2_mm


# =========================================================
# GEOMETRY BUILDER
# =========================================================
def build_geometry(inputs: Dict):
    if inputs["geometry_type"] == "rect":
        geom = RectangularSection(
            b_mm=inputs["ancho_b"],
            h_mm=inputs["canto_d"],
        )
        return geom, inputs["canto_d"], inputs["r2"]

    geom = ISection(
        b_top_mm=inputs["i_b_top"],
        t_top_mm=inputs["i_t_top"],
        b_web_mm=inputs["i_b_web"],
        b_bot_mm=inputs["i_b_bot"],
        t_bot_mm=inputs["i_t_bot"],
        h_mm=inputs["i_h"],
    )
    return geom, inputs["canto_d"], inputs["r2"]


# =========================================================
# CONTEVECT TIME SIMULATION
# =========================================================
def ejecutar_simulacion_completa_actualizada_en_tiempo(
    tipo_ataque: str,
    inputs: Dict,
    ti: float,
    dt_years: float = 1.0,
) -> Tuple[pd.DataFrame, float, float, pd.DataFrame]:

    t_end = float(inputs["t_analisis"])
    i_corr = float(inputs["i_corr"])
    cover_mm = float(inputs["recubrimiento"])
    fck = float(inputs["fck"])
    fy = float(inputs["fy"])

    alpha = 2.0 if tipo_ataque == "Carbonatación" else 10.0
    limite_px = 0.05 if tipo_ataque == "Carbonatación" else 0.5

    phi1_0 = float(inputs["phi_base"])
    phi2_0 = 20.0
    phi_w0 = 0.0001
    n_bars = int(inputs["n_barras"])

    geom, d_mm, d2_mm = build_geometry(inputs)
    materials = RcMaterials(fck, fy)

    fci = 0.333 * fck ** (2.0 / 3.0)
    px0 = max(
        0.0,
        (83.8 + 7.4 * (cover_mm / phi1_0) - 22.6 * fci) * 1e-3,
    )

    ref_area = max(
        inputs.get("ancho_b", 300.0) * inputs.get("canto_d", d_mm),
        1.0,
    )

    rows: List[Dict] = []
    events: List[Dict] = []

    times = np.arange(0.0, t_end + 1e-9, dt_years)

    for t in times:
        px = 0.0116 * i_corr * (t - ti) if t > ti else 0.0

        p1 = max(0.0, phi1_0 - alpha * px)
        p2 = max(0.0, phi2_0 - alpha * px)
        pw = max(0.0, phi_w0 - alpha * px)

        a1 = np.pi * p1**2 / 4.0 * n_bars
        a2 = np.pi * p2**2 / 4.0
        aw = np.pi * pw**2 / 4.0

        rho1 = a1 / ref_area
        rho2 = a2 / ref_area

        if px >= px0 and not any(e["event"] == "cracking" for e in events):
            events.append({"Time (y)": t, "event": "cracking"})

        rows.append(
            {
                "Tiempo (y)": t,
                "Px (mm)": px,
                "phi1 (mm)": p1,
                "A1 (mm2)": a1,
                "rho1": rho1,
                "d (mm)": d_mm,
                "Mu (kNm)": geom.moment_capacity_kNm(
                    a1, a2, d_mm, d2_mm, materials
                ),
            }
        )

    df_time = pd.DataFrame(rows)
    df_events = pd.DataFrame(events)

    t_vertical = ti + limite_px / (0.0116 * i_corr)

    return df_time, t_vertical, limite_px, df_events
``
