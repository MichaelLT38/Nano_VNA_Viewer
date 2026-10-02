"""RF quantities derived from S-parameters."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

# Floor for |S| before taking a log, so a perfect match gives -240 dB rather than -inf.
_MIN_MAGNITUDE = 1e-12

# Drawn segments per measured interval when curve smoothing is on.
SMOOTHING_FACTOR = 10

# A VSWR at or below this counts as matched, for the bandwidth readout.
BAND_VSWR_LIMIT = 2.0


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


def vswr_band(
    frequency: np.ndarray, s11: np.ndarray, limit: float = BAND_VSWR_LIMIT
) -> tuple[float | None, float | None] | None:
    """The band around the best match where VSWR stays at or below ``limit``, as (low, high) in Hz.

    Each edge is where |S11| crosses the limit, placed by linear interpolation between
    the measured points either side of the crossing. An edge is None when the band runs
    past that end of the sweep. Returns None when even the best match is above the limit.
    """
    mag = np.abs(s11)
    threshold = (limit - 1) / (limit + 1)
    best = min_vswr_index(s11)
    if not mag[best] <= threshold:
        return None

    def edge(step: int) -> float | None:
        inside = best
        while 0 <= inside + step < len(mag) and mag[inside + step] <= threshold:
            inside += step
        outside = inside + step
        if not 0 <= outside < len(mag):
            return None
        if not np.isfinite(mag[outside]):
            return float(frequency[inside])
        fraction = (threshold - mag[inside]) / (mag[outside] - mag[inside])
        return float(frequency[inside] + fraction * (frequency[outside] - frequency[inside]))

    return edge(-1), edge(1)


def interpolate_s(
    frequency: np.ndarray, s: np.ndarray, factor: int = SMOOTHING_FACTOR
) -> tuple[np.ndarray, np.ndarray] | None:
    """A denser (frequency, S) pair for drawing a smooth curve through the measured points.

    The complex value is interpolated with a cubic spline, so dB, VSWR and phase should
    be derived from the result rather than interpolated themselves. Every measured point
    is kept: it is element ``i * factor`` of the result.

    Returns None when the data can't be splined (fewer than 3 points, frequency not
    strictly increasing, or non-finite values); the caller then draws the plain trace.
    """
    n = len(frequency)
    if n < 3 or not np.all(np.diff(frequency) > 0) or not np.all(np.isfinite(s)):
        return None
    # Interpolating the frequency over its own index keeps uneven spacing intact.
    dense_frequency = np.interp(
        np.linspace(0, n - 1, (n - 1) * factor + 1), np.arange(n), frequency
    )
    dense_frequency[::factor] = frequency  # exact, free of rounding
    return dense_frequency, CubicSpline(frequency, s)(dense_frequency)
