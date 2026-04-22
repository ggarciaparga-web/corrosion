from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Materials helpers (Eurocode-like defaults)
# Units convention in this file:
# - Geometry: mm
# - Steel areas: mm2
# - Stresses: MPa = N/mm2
# - Forces: N
# - Moments: N*mm (converted to kNm at the end)
# ----------------------------

@dataclass(frozen=True)
class RcMaterials:
    fck_mpa: float
    fyk_mpa: float
    es_mpa: float = 200_000.0
    gamma_c: float = 1.50
    gamma_s: float = 1.15
    alpha_cc: float = 0.85
    eta: float = 1.0
    lam: float = 0.8
    eps_cu: float = 3.5e-3

    @property
    def fcd_mpa(self) -> float:
        return self.alpha_cc * self.fck_mpa / self.gamma_c

    @property
    def fyd_mpa(self) -> float:
        return self.fyk_mpa / self.gamma_s


# ----------------------------
# Geometry interface + implementations
# ----------------------------

class Geometry:
    """
    Base geometry interface.

    Each geometry must implement:
    - apply_damage(state, cover_mm): returns (new_geometry, new_d_mm, new_d2_mm)
    - moment_capacity_kNm(As_t_mm2, As_c_mm2, d_mm, d2_mm, materials)
    """

    def apply_damage(
        self, state: str, cover_mm: float, d_mm: float, d2_mm: float
    ) -> Tuple["Geometry", float, float]:
        raise NotImplementedError

    def moment_capacity_kNm(
        self,
        as_t_mm2: float,
        as_c_mm2: float,
        d_mm: float,
        d2_mm: float,
        materials: RcMaterials,
    ) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class RectangularSection(Geometry):
    b_mm: float
    h_mm: float

    def apply_damage(
        self, state: str, cover_mm: float, d_mm: float, d2_mm: float
    ) -> Tuple["Geometry", float, float]:
        """
        Simple spalling model aligned with your existing logic:

        - intact: no change
        - bottom_spall: height reduces by cover, tension steel depth reduces by cover
        - bottom_and_sides_spall: width reduces by 2*cover, height reduces by cover,
          tension steel depth reduces by cover
        """
        if state == "intact":
            return self, d_mm, d2_mm

        if state == "bottom_spall":
            new_h = max(self.h_mm - cover_mm, 1.0)
            new_d = max(d_mm - cover_mm, 1.0)
            return RectangularSection(self.b_mm, new_h), new_d, d2_mm

        if state == "bottom_and_sides_spall":
            new_b = max(self.b_mm - 2.0 * cover_mm, 1.0)
            new_h = max(self.h_mm - cover_mm, 1.0)
            new_d = max(d_mm - cover_mm, 1.0)
            return RectangularSection(new_b, new_h), new_d, d2_mm

        raise ValueError(f"Unknown damage state: {state}")

    def moment_capacity_kNm(
        self,
        as_t_mm2: float,
        as_c_mm2: float,
        d_mm: float,
        d2_mm: float,
        materials: RcMaterials,
    ) -> float:
        """
        Same simplified rectangular capacity you had:
        - ignores compression steel (as_c_mm2, d2_mm) for compatibility with your code
        - EC2-like rectangular stress block
        """
        if as_t_mm2 <= 0.0 or self.b_mm <= 0.0 or d_mm <= 0.0:
            return 0.0

        fcd = materials.fcd_mpa
        fyd = materials.fyd_mpa

        x_mm = (as_t_mm2 * fyd) / (0.8 * self.b_mm * fcd)
        z_mm = d_mm - 0.4 * x_mm
        m_nmm = max(as_t_mm2 * fyd * z_mm, 0.0)
        return m_nmm / 1e6  # kNm


@dataclass(frozen=True)
class ISection(Geometry):
    """
    Doubly symmetric-ish I/T section defined by:
    - top flange: width b_top, thickness t_top
    - web: width b_web
    - bottom flange: width b_bot, thickness t_bot
    - total height: h
    """
    b_top_mm: float
    t_top_mm: float
    b_web_mm: float
    b_bot_mm: float
    t_bot_mm: float
    h_mm: float

    def apply_damage(
        self, state: str, cover_mm: float, d_mm: float, d2_mm: float
    ) -> Tuple["Geometry", float, float]:
        """
        Practical spalling model (approximation):
        - intact: no change
        - bottom_spall:
            * total height reduces by cover
            * bottom flange thickness reduces by cover (limited)
            * tension steel depth reduces by cover (since bottom surface moves upward)
        - bottom_and_sides_spall:
            * widths reduce by 2*cover (top, web, bottom) (limited)
            * total height reduces by cover
            * bottom flange thickness reduces by cover (limited)
            * tension steel depth reduces by cover
        Note: This is a geometric approximation; refine if you have a specific spalling pattern.
        """
        if state == "intact":
            return self, d_mm, d2_mm

        def clamp_positive(val: float) -> float:
            return max(val, 1.0)

        if state in {"bottom_spall", "bottom_and_sides_spall"}:
            new_h = clamp_positive(self.h_mm - cover_mm)
            new_t_bot = clamp_positive(self.t_bot_mm - cover_mm)
            new_d = clamp_positive(d_mm - cover_mm)

            if state == "bottom_spall":
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
                    d2_mm,
                )

            new_b_top = clamp_positive(self.b_top_mm - 2.0 * cover_mm)
            new_b_web = clamp_positive(self.b_web_mm - 2.0 * cover_mm)
            new_b_bot = clamp_positive(self.b_bot_mm - 2.0 * cover_mm)
            return (
                ISection(
                    b_top_mm=new_b_top,
                    t_top_mm=self.t_top_mm,
                    b_web_mm=new_b_web,
                    b_bot_mm=new_b_bot,
                    t_bot_mm=new_t_bot,
                    h_mm=new_h,
                ),
                new_d,
                d2_mm,
            )

        raise ValueError(f"Unknown damage state: {state}")

    def _compressed_block_resultant(self, a_mm: float, fcd_mpa: float, eta: float) -> Tuple[float, float]:
        """
        Returns (Cc_N, yCc_mm) for the equivalent rectangular block of depth 'a_mm'
        from top fiber (y=0 at top).
        """
        a_mm = max(a_mm, 0.0)
        if a_mm <= 0.0:
            return 0.0, 0.0

        t_top = self.t_top_mm
        h = self.h_mm
        t_bot = self.t_bot_mm

        if a_mm <= t_top:
            area = self.b_top_mm * a_mm
            cc = eta * fcd_mpa * area
            y_cc = a_mm / 2.0
            return cc, y_cc

        if a_mm <= (h - t_bot):
            a1 = self.b_top_mm * t_top
            a2 = self.b_web_mm * (a_mm - t_top)
            ac = a1 + a2
            cc = eta * fcd_mpa * ac
            y1 = t_top / 2.0
            y2 = t_top + (a_mm - t_top) / 2.0
            y_cc = (a1 * y1 + a2 * y2) / ac
            return cc, y_cc

        a3 = a_mm - (h - t_bot)
        a3 = min(a3, t_bot)
        a1 = self.b_top_mm * t_top
        a2 = self.b_web_mm * ((h - t_bot) - t_top)
        a3_area = self.b_bot_mm * a3
        ac = a1 + a2 + a3_area
        cc = eta * fcd_mpa * ac
        y1 = t_top / 2.0
        y2 = t_top + (((h - t_bot) - t_top) / 2.0)
        y3 = (h - t_bot) + a3 / 2.0
        y_cc = (a1 * y1 + a2 * y2 + a3_area * y3) / ac
        return cc, y_cc

    def moment_capacity_kNm(
        self,
        as_t_mm2: float,
        as_c_mm2: float,
        d_mm: float,
        d2_mm: float,
        materials: RcMaterials,
    ) -> float:
        """
        Full equilibrium with:
        - concrete compressed block with variable width (top flange / web / bottom flange)
        - tension steel As_t at depth d
        - compression steel As_c at depth d2
        Using bisection on neutral axis depth x.
        """
        if as_t_mm2 <= 0.0 or d_mm <= 0.0 or self.h_mm <= 0.0:
            return 0.0

        fcd = materials.fcd_mpa
        fyd = materials.fyd_mpa
        es = materials.es_mpa
        eps_cu = materials.eps_cu
        lam = materials.lam
        eta = materials.eta

        def steel_stress_mpa(eps: float) -> float:
            sig = es * eps
            return float(np.clip(sig, -fyd, fyd))

        def equilibrium(x_mm: float) -> Tuple[float, float, float, float, float]:
            a_mm = lam * x_mm
            cc_n, y_cc_mm = self._compressed_block_resultant(a_mm, fcd, eta)

            eps_t = eps_cu * (d_mm / x_mm - 1.0)
            ts_n = as_t_mm2 * steel_stress_mpa(eps_t)

            cs_n = 0.0
            if as_c_mm2 > 0.0 and d2_mm > 0.0:
                eps_c_steel = eps_cu * (1.0 - d2_mm / x_mm)
                cs_n = as_c_mm2 * steel_stress_mpa(eps_c_steel)

            # equilibrium: Cc + Cs - Ts = 0
            return (cc_n + cs_n - ts_n), cc_n, y_cc_mm, cs_n, ts_n

        x_min = 1e-6
        x_max = min(0.999 * d_mm, 0.999 * self.h_mm)

        f_min, *_ = equilibrium(x_min)
        f_max, *_ = equilibrium(x_max)

        if f_min * f_max > 0:
            # No sign change; return 0 to keep simulation running (or raise if preferred)
            return 0.0

        x_lo, x_hi = x_min, x_max
        for _ in range(250):
            x_mid = 0.5 * (x_lo + x_hi)
            f_mid, cc_n, y_cc_mm, cs_n, ts_n = equilibrium(x_mid)

            if abs(f_mid) < 1e-3:
                break

            if f_min * f_mid < 0:
                x_hi = x_mid
                f_max = f_mid
            else:
                x_lo = x_mid
                f_min = f_mid

        _, cc_n, y_cc_mm, cs_n, ts_n = equilibrium(x_mid)
        c_total = cc_n + cs_n
        if abs(c_total) < 1e-9:
            return 0.0

        y_c_mm = (cc_n * y_cc_mm + cs_n * d2_mm) / c_total if abs(cs_n) > 0.0 else y_cc_mm
        z_mm = d_mm - y_c_mm
        m_nmm = ts_n * z_mm
        return max(m_nmm / 1e6, 0.0)  # kNm


# ----------------------------
# Geometry factory
# ----------------------------

def build_geometry(inputs: Dict) -> Tuple[Geometry, float, float]:
    """
    Returns (geometry, d_mm, d2_mm).

    Expected inputs:
    - geometry_type: "rect" or "i"
    For "rect":
      - ancho_b (mm), canto_h (mm) OR canto_d used as d (mm)
      - d_tension (mm) optional; defaults to inputs["canto_d"]
      - d_compression (mm) optional; defaults to 0
    For "i":
      - i_b_top, i_t_top, i_b_web, i_b_bot, i_t_bot, i_h (all mm)
      - d_tension (mm), d_compression (mm)
    """
    gtype = str(inputs.get("geometry_type", "rect")).strip().lower()

    if gtype in {"rect", "rectangle", "rectangular"}:
        b_mm = float(inputs["ancho_b"])
        # If you only had "canto_d" before, treat it as effective depth d (not total height).
        # Provide "canto_h" to define total height; otherwise assume h ~= d.
        h_mm = float(inputs.get("canto_h", inputs.get("canto_d", 0.0)))
        d_mm = float(inputs.get("d_tension", inputs.get("canto_d", h_mm)))
        d2_mm = float(inputs.get("d_compression", 0.0))
        return RectangularSection(b_mm=b_mm, h_mm=h_mm), d_mm, d2_mm

    if gtype in {"i", "isection", "doublet", "doblet"}:
        geom = ISection(
            b_top_mm=float(inputs["i_b_top"]),
            t_top_mm=float(inputs["i_t_top"]),
            b_web_mm=float(inputs["i_b_web"]),
            b_bot_mm=float(inputs["i_b_bot"]),
            t_bot_mm=float(inputs["i_t_bot"]),
            h_mm=float(inputs["i_h"]),
        )
        d_mm = float(inputs["d_tension"])
        d2_mm = float(inputs["d_compression"])
        return geom, d_mm, d2_mm

    raise ValueError(f"Unsupported geometry_type: {gtype}")


# ----------------------------
# Main simulation (your original logic, geometry-aware)
# ----------------------------

def ejecutar_simulacion_completa(tipo_ataque: str, inputs: Dict, ti: float):
    """
    Integrates initiation logic + degradation matrix + CONTEVECT-like events,
    but now the resistant moment is computed depending on selected geometry.
    """
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

    geometry, d0_mm, d2_0_mm = build_geometry(inputs)

    materials = RcMaterials(fck_mpa=fck_mpa, fyk_mpa=fyk_mpa)

    # Base matrix
    times = np.arange(0, int(t_end) + 1, 1, dtype=float)
    rows = []

    for t in times:
        px = 0.0116 * i_corr * (t - ti) if t > ti else 0.0  # mm
        p1 = max(0.0, phi1_0_mm - alpha * px)
        p2 = max(0.0, phi2_0_mm - alpha * px)
        pw = max(0.0, phi_w0_mm - alpha * px)

        a1 = (np.pi * p1**2 / 4.0) * n_bottom
        a2 = (np.pi * p2**2 / 4.0)
        aw = (np.pi * pw**2 / 4.0)

        # rho values for event detection: need a reference area.
        # For compatibility with your prior rectangular implementation, use b*d (effective).
        # For I-sections, you can change this to gross area if you prefer.
        ref_b = float(inputs.get("ancho_b", getattr(geometry, "b_mm", 1.0)))
        ref_d = float(inputs.get("canto_d", d0_mm))
        ref_area = max(ref_b * ref_d, 1.0)

        rows.append(
            {
                "Time (y)": t,
                "Px (mm)": px,
                "phi1 (mm)": p1,
                "phi2 (mm)": p2,
                "phiw (mm)": pw,
                "A1 (mm2)": a1,
                "A2 (mm2)": a2,
                "Aw (mm2)": aw,
                "rho1": a1 / ref_area,
                "rho2": a2 / ref_area,
            }
        )

    df_base = pd.DataFrame(rows)

    # Critical points
    fci = 0.333 * fck_mpa ** (2.0 / 3.0)
    px0 = max(0.0, (83.8 + 7.4 * (recubrimiento_mm / phi1_0_mm) - 22.6 * fci) * 1e-3)

    points = []

    def build_point(
        base_row: pd.Series,
        geom: Geometry,
        d_mm: float,
        d2_mm: float,
        damage_state: str,
    ) -> pd.Series:
        r = base_row.copy()
        r["damage_state"] = damage_state
        r["d (mm)"] = d_mm
        r["d2 (mm)"] = d2_mm
        # Store some geometry metadata (useful for debugging / plots)
        if isinstance(geom, RectangularSection):
            r["geom_type"] = "rect"
            r["b (mm)"] = geom.b_mm
            r["h (mm)"] = geom.h_mm
        elif isinstance(geom, ISection):
            r["geom_type"] = "i"
            r["b_top (mm)"] = geom.b_top_mm
            r["t_top (mm)"] = geom.t_top_mm
            r["b_web (mm)"] = geom.b_web_mm
            r["b_bot (mm)"] = geom.b_bot_mm
            r["t_bot (mm)"] = geom.t_bot_mm
            r["h (mm)"] = geom.h_mm
        else:
            r["geom_type"] = "custom"
        return r

    # Milestone 1: initial state
    points.append(build_point(df_base.iloc[0], geometry, d0_mm, d2_0_mm, "intact"))

    # Milestone 2: cracking at px >= px0
    mask_px0 = df_base["Px (mm)"] >= px0
    if mask_px0.any():
        idx_px0 = int(mask_px0.idxmax())
        points.append(build_point(df_base.loc[idx_px0], geometry, d0_mm, d2_0_mm, "intact"))

    # Events 3 & 4 detection (kept as in your code)
    ev3: Optional[pd.Series] = None
    ev4: Optional[pd.Series] = None

    # For ev4 check "b_initial": use ancho_b as reference width (legacy rule).
    b_initial = float(inputs.get("ancho_b", getattr(geometry, "b_mm", 1.0)))

    for _, row in df_base.iterrows():
        r1 = float(row["rho1"]) * 100.0
        r2 = float(row["rho2"]) * 100.0
        px = float(row["Px (mm)"])
        aw = float(row["Aw (mm2)"])

        if r1 > 1.5 and aw > (0.0036 * b_initial) and px > 0.2 and ev4 is None:
            # bottom + sides spalling
            geom_dmg, d_dmg, d2_dmg = geometry.apply_damage(
                state="bottom_and_sides_spall",
                cover_mm=recubrimiento_mm,
                d_mm=d0_mm,
                d2_mm=d2_0_mm,
            )
            ev4 = build_point(row, geom_dmg, d_dmg, d2_dmg, "bottom_and_sides_spall")

        if ev3 is None:
            cond = (
                (r1 < 1.0 and r2 < 5.0 and px > 0.4)
                or (r1 < 1.0 and r2 > 5.0 and px > 0.2)
                or (r1 > 1.5 and r2 > 0.5 and px > 0.2)
            )
            if cond:
                # bottom spalling
                geom_dmg, d_dmg, d2_dmg = geometry.apply_damage(
                    state="bottom_spall",
                    cover_mm=recubrimiento_mm,
                    d_mm=d0_mm,
                    d2_mm=d2_0_mm,
                )
                ev3 = build_point(row, geom_dmg, d_dmg, d2_dmg, "bottom_spall")

    if ev3 is not None:
        points.append(ev3)
    if ev4 is not None:
        points.append(ev4)

    df_points = (
        pd.DataFrame(points)
        .sort_values("Px (mm)")
        .drop_duplicates(subset=["Px (mm)"])
        .reset_index(drop=True)
    )

    # Capacity computation for each point (geometry-aware)
    def compute_mu_kNm(r: pd.Series) -> float:
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

    df_points["Mu (kNm)"] = df_points.apply(compute_mu_kNm, axis=1)

    # Final matrix: after last critical point, keep last damaged geometry constant
    last_crit = df_points.iloc[-1]
    df_remaining = df_base[df_base["Time (y)"] > float(last_crit["Time (y)"])].copy()

    # propagate last known geometry/d/d2 columns
    for col in ["geom_type", "damage_state", "d (mm)", "d2 (mm)", "h (mm)"]:
        if col in last_crit:
            df_remaining[col] = last_crit[col]

    # propagate geometry-specific columns
    if str(last_crit.get("geom_type", "rect")) == "rect":
        df_remaining["b (mm)"] = last_crit["b (mm)"]
        df_remaining["h (mm)"] = last_crit["h (mm)"]
    else:
        for col in ["b_top (mm)", "t_top (mm)", "b_web (mm)", "b_bot (mm)", "t_bot (mm)", "h (mm)"]:
            if col in last_crit:
                df_remaining[col] = last_crit[col]

    df_final = pd.concat([df_points, df_remaining], ignore_index=True)
    df_final["Mu (kNm)"] = df_final.apply(compute_mu_kNm, axis=1)

    # Vertical line time (same as your code)
    t_vertical = ti + (limite_px / (0.0116 * i_corr))

    return df_final, t_vertical, limite_px, df_points


# ----------------------------
# Suggested GitHub folder layout (optional, just a note in code)
# ----------------------------
# You can place geometries in separate modules, e.g.:
#   geometries/
#     __init__.py
#     rectangular.py
#     isection.py
# and import them into the simulation module.
