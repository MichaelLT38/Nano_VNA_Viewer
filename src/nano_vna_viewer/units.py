"""Human-readable formatting for frequencies and impedances."""

from __future__ import annotations

import numpy as np

_FREQUENCY_UNITS = ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz"))


def format_frequency(hz: float) -> str:
    """Format a frequency with an SI prefix, e.g. 9049500 -> '9.0495 MHz'."""
    for scale, unit in _FREQUENCY_UNITS:
        if abs(hz) >= scale:
            return f"{hz / scale:.6g} {unit}"
    return f"{hz:.6g} Hz"


def format_impedance(z: complex, decimals: int | None = None) -> str:
    """Format an impedance, omitting the imaginary part when it is zero.

    With ``decimals``, both parts use that many decimal places (for live readouts
    where the width should stay steady); otherwise up to 6 significant digits.
    """
    if not (np.isfinite(z.real) and np.isfinite(z.imag)):
        return "∞ Ω"
    fmt = f".{decimals}f" if decimals is not None else ".6g"
    if z.imag == 0 and decimals is None:
        return f"{z.real:{fmt}} Ω"
    sign = "+" if z.imag >= 0 else "−"
    return f"{z.real:{fmt}} {sign} j{abs(z.imag):{fmt}} Ω"
