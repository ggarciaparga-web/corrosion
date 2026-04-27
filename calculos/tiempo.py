from __future__ import annotations

from typing import Dict, Tuple
import numpy as np


def calcular_iniciacion(tipo_ataque: str, inputs: Dict) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Minimal initiation model so the app runs end-to-end.

    Replace this with your real initiation model. For now:
      - Carbonation: ti depends weakly on cover and fck
      - Chlorides: fixed ti
    """
    t_end = float(inputs.get("t_analisis", 250.0))
    cover = float(inputs.get("recubrimiento", 30.0))
    fck = float(inputs.get("fck", 25.0))

    if tipo_ataque == "Carbonatación":
        # Simple placeholder: larger cover and higher strength delay initiation
        ti = max(1.0, 8.0 + 0.08 * cover + 0.15 * (fck - 25.0))
    else:
        ti = 10.0

    times = np.arange(0.0, t_end + 1.0, 1.0)
    px_plot = np.zeros_like(times, dtype=float)
    return float(ti), times, px_plot
