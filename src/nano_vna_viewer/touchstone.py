"""Loading and saving Touchstone (.s1p / .s2p) files, as exported by a NanoVNA."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import skrf

SUPPORTED_SUFFIX = re.compile(r"^\.s[12]p$", re.IGNORECASE)


class TouchstoneError(Exception):
    """Raised when a file cannot be loaded as Touchstone data."""


@dataclass(frozen=True)
class Measurement:
    """A sweep, from a file or captured live. S11 is always available; S21 only for two-port data.

    ``path`` is None for a live capture; ``source`` then describes where it came from.
    """

    path: Path | None
    network: skrf.Network
    source: str = ""

    @property
    def name(self) -> str:
        return self.path.name if self.path is not None else self.source

    @property
    def nports(self) -> int:
        return self.network.nports

    @property
    def frequency_hz(self) -> np.ndarray:
        return self.network.f

    @property
    def points(self) -> int:
        return len(self.network.f)

    @property
    def start_hz(self) -> float:
        return float(self.network.f[0])

    @property
    def stop_hz(self) -> float:
        return float(self.network.f[-1])

    @property
    def z0(self) -> complex:
        return complex(self.network.z0[0, 0])

    @property
    def s11(self) -> np.ndarray:
        return self.network.s[:, 0, 0]

    @property
    def s21(self) -> np.ndarray | None:
        return self.network.s[:, 1, 0] if self.nports >= 2 else None


def load_touchstone(path: str | Path) -> Measurement:
    """Load a one- or two-port Touchstone file.

    Uses ``Network.read_touchstone`` rather than ``skrf.Network(path)``:
    the latter tries to unpickle the file first, which would let a
    malicious file run arbitrary code when opened.
    """
    path = Path(path).resolve()

    if not SUPPORTED_SUFFIX.match(path.suffix):
        raise TouchstoneError(
            f"Unsupported file type '{path.suffix or '(none)'}'. "
            "Expected a .s1p or .s2p Touchstone file."
        )
    if not path.is_file():
        raise TouchstoneError(f"File not found: {path}")

    network = skrf.Network()
    try:
        network.read_touchstone(str(path))
    except Exception as exc:  # skrf raises a variety of exception types
        raise TouchstoneError(f"Could not read '{path.name}': {exc}") from exc

    if len(network.f) == 0:
        raise TouchstoneError(f"'{path.name}' contains no data points.")

    return Measurement(path=path, network=network)


def measurement_from_sweep(
    frequency_hz: np.ndarray,
    s11: np.ndarray,
    s21: np.ndarray | None = None,
    z0: float = 50.0,
    source: str = "NanoVNA sweep",
) -> Measurement:
    """Wrap raw sweep arrays (e.g. from a live NanoVNA) as a Measurement.

    With S21 the result is two-port, with S12 and S22 left at zero as the NanoVNA does.
    """
    nports = 1 if s21 is None else 2
    s = np.zeros((len(frequency_hz), nports, nports), dtype=complex)
    s[:, 0, 0] = s11
    if s21 is not None:
        s[:, 1, 0] = s21
    network = skrf.Network(
        frequency=skrf.Frequency.from_f(np.asarray(frequency_hz, dtype=float), unit="hz"),
        s=s,
        z0=z0,
        name=source,
    )
    return Measurement(path=None, network=network, source=source)


def touchstone_suffix(m: Measurement) -> str:
    return f".s{m.nports}p"


def write_touchstone(m: Measurement, path: str | Path) -> None:
    """Save in the NanoVNA's own layout: Hz, real/imaginary, one line per point.

    Two-port files list S11 S21 S12 S22, the Touchstone v1 order.
    """
    s = m.network.s
    order = [(0, 0)] if m.nports == 1 else [(0, 0), (1, 0), (0, 1), (1, 1)]
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("! Saved by Nano VNA Viewer\n")
        f.write(f"# Hz S RI R {m.z0.real:g}\n")
        for i, freq in enumerate(m.frequency_hz):
            values = " ".join(f"{s[i, r, c].real:.9g} {s[i, r, c].imag:.9g}" for r, c in order)
            f.write(f"{freq:.0f} {values}\n")
