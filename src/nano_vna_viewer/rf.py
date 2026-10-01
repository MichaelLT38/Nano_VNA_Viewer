"""RF quantities derived from S-parameters."""

from __future__ import annotations

import numpy as np

# Floor for |S| before taking a log, so a perfect match gives -240 dB rather than -inf.
_MIN_MAGNITUDE = 1e-12


def s_db(s: np.ndarray) -> np.ndarray:
    """Log magnitude, 20·log10|S|, in dB."""
    return 20 * np.log10(np.maximum(np.abs(s), _MIN_MAGNITUDE))


def vswr(s11: np.ndarray) -> np.ndarray:
    """Voltage standing wave ratio from the reflection coefficient.

    |S11| >= 1 (total reflection, or slightly above 1 from calibration error)
    has no finite VSWR and returns inf instead of a meaningless negative value.
    """
    mag = np.abs(s11)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (1 + mag) / (1 - mag)
    return np.where(mag >= 1, np.inf, result)


def phase_deg(s: np.ndarray) -> np.ndarray:
    """Phase in degrees, wrapped to (-180, 180]."""
    return np.angle(s, deg=True)


def impedance(s11: np.ndarray, z0: complex) -> np.ndarray:
    """Input impedance from the reflection coefficient: Z = Z0 (1 + Γ) / (1 - Γ).

    Γ = 1 exactly (an ideal open) gives an infinite impedance.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        z = z0 * (1 + s11) / (1 - s11)
    return np.where(s11 == 1, complex(np.inf, 0), z)


def min_vswr_index(s11: np.ndarray) -> int:
    """Index of the best match (lowest |S11|, hence lowest VSWR)."""
    return int(np.argmin(np.abs(s11)))
