from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


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

    def lever_arm_mm(
        self, a_corr_mm2: float, d_mm: float, mat: ModelCodeMaterials
    ) -> float:
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

    def lever_arm_mm(
        self, a_corr_mm2: float, d_mm: float, mat: ModelCodeMaterials
    ) -> float:
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

    def lever_arm_mm(
        self, a_corr_mm2: float, d_mm: float, mat: ModelCodeMaterials
    ) -> float:
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
            b_eff = max(effective_width(x), 1.0)
            x = (a_corr_mm2 * fyd) / (0.8 * b_eff * fcd)

        return max(d_mm - 0.4 * x, 0.0)


# ============================================================
# GEOMETRY FACTORY
# ============================================================
def build_geometry_mc(inputs: Dict) -> Tuple[Geometry, float, float]:
    gtype = str(inputs.get("geometry_type", "rect")).lower()

    if gtype == "rect":
        geom = RectangularSection(
            b_mm=float(inputs["ancho_b"]),
            h_mm=float(inputs["canto_d"]),
        )
        return geom, float(inputs["canto_d"]), float(inputs["r2"])

    if gtype == "i":
        geom = ISection(
            b_top_mm=float(inputs["i_b_top"]),
            t_top_mm=float(inputs["i_t_top"]),
            b_web_mm=float(inputs["i_b_web"]),
            b_bot_mm=float(inputs["i_b_bot"]),
            t_bot_mm=float(inputs["i_t_bot"]),
            h_mm=float(inputs["i_h"]),
        )
        return geom, float(inputs["canto_d"]), float(inputs["r2"])

    raise ValueError("Unsupported geometry_type. Use 'rect' or 'i'.")


# ============================================================
# MAIN MODEL CODE SIMULATION (GEOMETRY-AWARE)
# ============================================================
def simulacion_total(tipo_ataque: str, inputs: Dict, ti: float):
    """
    Residual bending resistance according to a simplified fib Model Code 2023 approach,
    supporting rectangular and I / double-T sections.
    """
    t_end = int(inputs["t_analisis"])
    cover_mm = float(inputs["recubrimiento"])
    i_corr = float(inputs["i_corr"])

    phi1_initial = float(inputs["phi_base"])
    n_bottom = int(inputs["n_barras"])

    if tipo_ataque == "Carbonatación":
        alpha = 2.0
        limite_px = 0.05
    else:
        alpha = 10.0
        limite_px = 0.5

    materials = ModelCodeMaterials(
        fck_mpa=float(inputs["fck"]),
        fy_mpa=float(inputs["fy"]),
    )

    geometry_0, d0_mm, r2_mm_0 = build_geometry_mc(inputs)

    times = np.arange(0, t_end + 1, 1)
    results = []

    t_vertical = float(ti + (limite_px / (0.0116 * i_corr)))

    for t in times:
        if t <= ti:
            px = 0.0
            phi = phi1_initial
            geometry_t = geometry_0
            d_t = d0_mm
            r2_mm = r2_mm_0
        else:
            px = 0.0116 * i_corr * (t - ti)
            phi = max(0.0, phi1_initial - alpha * px)

            # Apply spalling once after initiation (kept simplified and stable)
            geometry_t, d_t, r2_mm = geometry_0.apply_spalling(
                cover_mm=cover_mm,
                d_mm=d0_mm,
                r2_mm=r2_mm_0,
            )

        a_corr = (np.pi * phi**2 / 4.0) * n_bottom

        if phi <= 0.0 or a_corr <= 0.0:
            mu_res = 0.0
            mu_cons = 0.0
        else:
            z = geometry_t.lever_arm_mm(a_corr, d_t, materials)
            mu_res = a_corr * materials.fyd * z / 1e6
            mu_cons = a_corr * materials.fyd * max(z - r2_mm, 0.0) / 1e6

        results.append(
            {
                "Time (y)": int(t),
                "Px (mm)": float(px),
                "phi (mm)": float(phi),
                "A_corr (mm2)": float(a_corr),
                "Mu (kNm)": float(max(mu_res, 0.0)),
                "Mu Cons (kNm)": float(max(mu_cons, 0.0)),
            }
        )

    return pd.DataFrame(results), t_vertical
