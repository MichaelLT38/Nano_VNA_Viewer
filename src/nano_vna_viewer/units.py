"""Human-readable formatting for frequencies and impedances."""

from __future__ import annotations

import numpy as np

_FREQUENCY_UNITS = ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz"))
_COMPONENT_PREFIXES = ((1.0, ""), (1e-3, "m"), (1e-6, "µ"), (1e-9, "n"), (1e-12, "p"))


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


def format_series_component(reactance_ohm: float, hz: float) -> str:
    """The inductor or capacitor that has this reactance at this frequency, e.g. '8.75 pF'.

    Positive reactance is an inductance (L = X / 2πf), negative a capacitance
    (C = 1 / 2πf|X|). Returns '' when the reactance is zero or not finite.
    """
    if not np.isfinite(reactance_ohm) or reactance_ohm == 0 or not hz > 0:
        return ""
    omega = 2 * np.pi * hz
    if reactance_ohm > 0:
        value, unit = reactance_ohm / omega, "H"
    else:
        value, unit = 1 / (omega * -reactance_ohm), "F"
    scale, prefix = next(
        (pair for pair in _COMPONENT_PREFIXES if value >= pair[0]), _COMPONENT_PREFIXES[-1]
    )
    # Round to 3 significant figures, then print plainly (no "1e+03").
    return f"{float(f'{value / scale:.3g}'):g} {prefix}{unit}"
