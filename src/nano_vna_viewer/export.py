"""Exporting a measurement's data as CSV."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .rf import impedance, phase_deg, s_db, vswr
from .touchstone import Measurement


def measurement_columns(m: Measurement) -> dict[str, np.ndarray]:
    """All exported columns, in order. S21 columns are included only for .s2p data."""
    z = impedance(m.s11, m.z0)
    columns = {
        "frequency_hz": m.frequency_hz,
        "s11_real": m.s11.real,
        "s11_imag": m.s11.imag,
        "s11_db": s_db(m.s11),
        "s11_phase_deg": phase_deg(m.s11),
        "vswr": vswr(m.s11),
        "z_real_ohm": z.real,
        "z_imag_ohm": z.imag,
    }
    if m.s21 is not None:
        columns.update(
            {
                "s21_real": m.s21.real,
                "s21_imag": m.s21.imag,
                "s21_db": s_db(m.s21),
                "s21_phase_deg": phase_deg(m.s21),
            }
        )
    return columns


def _cell(value: float) -> str:
    # repr-style formatting round-trips floats exactly; inf/nan become "inf"/"nan".
    if np.isnan(value):
        return "nan"
    if np.isinf(value):
        return "inf" if value > 0 else "-inf"
    return repr(float(value))


def write_csv(m: Measurement, path: str | Path) -> None:
    columns = measurement_columns(m)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in zip(*columns.values()):
            writer.writerow(_cell(v) for v in row)
