from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple


# ============================================================
# MATERIALS – Model Code 2023 style (simplified)
# ============================================================

@dataclass(frozen=True)
class ModelCodeMaterials:
    fck_mpa: float
    fy_mpa: float
    gamma_c: float = 1.50
    gamma_s: float = 1.15

    @property
    def fyd(self) -> float:
        return self.fy_mpa / self.gamma_s

    @property
    def fcd_nom(self) -> float:
        return self.fck_mpa / self.gamma_c

    @property
    def kc(self) -> float:
        nfc = min(1.0, (30.0 / self.fck_mpa) ** (1.0 / 3.0))
        return 0.75 * nfc

    @property
    def fcd_red(self) -> float:
        return self.kc * self.fcd_nom


# ============================================================
# GEOMETRY INTERFACE
# ============================================================

class Geometry:
    def apply_spalling(
        self, cover_mm: float, d_mm: float, r2_mm: float
    ) -> Tuple["Geometry", float, float]:
        raise NotImplementedError

    def lever_arm_mm(self, a_corr_mm2: float, d_mm: float, mat: ModelCodeMaterials) -> float:
        raise NotImplementedError


# ============================================================
# RECTANGULAR SECTION
# ============================================================

@dataclass(frozen=True)
class RectangularSection(Geometry):
    b_mm: float
    h_mm: float

    def apply_spalling(
        self, cover_mm: float, d_mm: float, r2_mm: float
    ) -> Tuple["Geometry", float, float]:
        new_h = max(self.h_mm - cover_mm, 1.0)
        new_d = max(d_mm - cover_mm, 1.0)
        return RectangularSection(self.b_mm, new_h), new_d, r2_mm

    def lever_arm_mm(self, a_corr_mm2: float, d_mm: float, mat: ModelCodeMaterials) -> float:
        x = (a_corr_mm2 * mat.fyd) / (0.8 * self.b_mm * mat.fcd_red)
        return max(d_mm - 0.4 * x, 0.0)


# ============================================================
# I / DOUBLE-T SECTION
# ============================================================

@dataclass(frozen=True)
class ISection(Geometry):
    b_top_mm: float
    t_top_mm: float
    b_web_mm: float
    b_bot_mm: float
    t_bot_mm: float
    h_mm: float

    def apply_spalling(
        self, cover_mm: float, d_mm: float, r2_mm: float
    ) -> Tuple["Geometry", float, float]:
        new_h = max(self.h_mm - cover_mm, 1.0)
        new_t_bot = max(self.t_bot_mm - cover_mm, 1.0)
        new_d = max(d_mm - cover_mm, 1.0)

        return (
            ISection(
                b_top_mm=self.b_top_mm,
                t_top_mm=self.t_top_mm,
                b_web_mm=self.b_web_mm,
                b_bot_mm=self.b_bot_mm,
                t_bot_mm=new_t_bot,
                h_mm=new_h,
            ),
            new_d,
            r2_mm,
        )

    def lever_arm_mm(self, a_corr_mm2: float, d_mm: float, mat: ModelCodeMaterials) -> float:
        """
        Simplified Model Code approach:
        - Equivalent rectangular stress block
        - Effective width depends on where the neutral axis falls
        """
        fcd = mat.fcd_red
        fyd = mat.fyd

        def effective_width(x_mm: float) -> float:
            if x_mm <= self.t_top_mm:
                return self.b_top_mm
            if x_mm <= (self.h_mm - self.t_bot_mm):
                return self.b_web_mm
            return self.b_bot_mm

        x = 1.0
        for _ in range(50):
            b_eff = effective_width(x)
            x = (a_corr_mm2 * fyd) / (0.8 * b_eff * fcd)

        return max(d_mm - 0.4 * x, 0.0)


# ============================================================
# GEOMETRY FACTORY
# ============================================================

def build_geometry_mc(inputs: Dict) -> Tuple[Geometry, float, float]:
    gtype = inputs.get("geometry_type", "rect").lower()

    if gtype == "rect":
        geom = RectangularSection(
            b_mm=inputs["ancho_b"],
            h_mm=inputs["canto_d"],
        )
        return geom, inputs["canto_d"], inputs["r2"]

    if gtype == "i":
        geom = ISection(
            b_top_mm=inputs["i_b_top"],
            t_top_mm=inputs["i_t_top"],
            b_web_mm=inputs["i_b_web"],
            b_bot_mm=inputs["i_b_bot"],
            t_bot_mm=inputs["i_t_bot"],
            h_mm=inputs["i_h"],
        )
        return geom, inputs["canto_d"], inputs["r2"]

    raise ValueError("Unsupported geometry_type")


# ============================================================
# MAIN MODEL CODE SIMULATION (GEOMETRY-AWARE)
# ============================================================

def simulacion_total(tipo_ataque: str, inputs: Dict, ti: float):
    """
    Residual bending resistance according to fib Model Code 2023,
    supporting rectangular and I / double-T sections.
    """

    t_end = inputs["t_analisis"]
    recubrimiento = inputs["recubrimiento"]
    i_corr = inputs["i_corr"]

    phi1_initial = inputs["phi_base"]
    n_bottom = inputs["n_barras"]

    if tipo_ataque == "Carbonatación":
        alpha = 2.0
        limite_px = 0.05
    else:
        alpha = 10.0
        limite_px = 0.5

    materials = ModelCodeMaterials(
        fck_mpa=inputs["fck"],
        fy_mpa=inputs["fy"],
    )

    geometry, d0_mm, r2_mm = build_geometry_mc(inputs)

    times = np.arange(0, t_end + 1, 1)
    results = []

    t_vertical = ti + (limite_px / (0.0116 * i_corr))

    a_initial = (np.pi * phi1_initial**2 / 4.0) * n_bottom

    for t in times:
        if t <= ti:
            px = 0.0
            phi = phi1_initial
            geom_t = geometry
            d_t = d0_mm
        else:
            px = 0.0116 * i_corr * (t - ti)
            phi = max(0.0, phi1_initial - alpha * px)
            geom_t, d_t, r2_mm = geometry.apply_spalling(recubrimiento, d0_mm, r2_mm)

        a_corr = (np.pi * phi**2 / 4.0) * n_bottom

        if phi <= 0.0 or a_corr <= 0.0:
            mu_res = 0.0
            mu_cons = 0.0
        else:
            z = geom_t.lever_arm_mm(a_corr, d_t, materials)
            mu_res = a_corr * materials.fyd * z / 1e6
            mu_cons = a_corr * materials.fyd * max(z - r2_mm, 0.0) / 1e6

        results.append(
            {
                "Time (y)": t,
                "Px (mm)": px,
                "phi (mm)": phi,
                "A_corr (mm2)": a_corr,
                "Mu (kNm)": max(mu_res, 0.0),
                "Mu Cons (kNm)": max(mu_cons, 0.0),
            }
        )

    return pd.DataFrame(results), t_vertical
