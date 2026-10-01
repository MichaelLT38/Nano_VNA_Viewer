import csv
from pathlib import Path

import numpy as np
import pytest

from nano_vna_viewer.export import measurement_columns, write_csv
from nano_vna_viewer.touchstone import load_touchstone

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
SAMPLE = SAMPLES / "data.s1p"

ONE_PORT_COLUMNS = [
    "frequency_hz",
    "s11_real",
    "s11_imag",
    "s11_db",
    "s11_phase_deg",
    "vswr",
    "z_real_ohm",
    "z_imag_ohm",
]
S21_COLUMNS = ["s21_real", "s21_imag", "s21_db", "s21_phase_deg"]


@pytest.fixture
def two_port(tmp_path):
    path = tmp_path / "thru.s2p"
    path.write_text(
        "# Hz S RI R 50\n"
        "1000000 0.1 0.0 0.9 0.1 0.9 0.1 0.2 0.0\n"
        "2000000 0.2 0.0 0.8 0.2 0.8 0.2 0.3 0.0\n"
    )
    return load_touchstone(path)


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def test_one_port_columns():
    assert list(measurement_columns(load_touchstone(SAMPLE))) == ONE_PORT_COLUMNS


def test_two_port_adds_s21_columns(two_port):
    assert list(measurement_columns(two_port)) == ONE_PORT_COLUMNS + S21_COLUMNS


def test_csv_round_trips_the_data(tmp_path):
    m = load_touchstone(SAMPLE)
    out = tmp_path / "out.csv"
    write_csv(m, out)
    header, rows = _read(out)
    assert header == ONE_PORT_COLUMNS
    assert len(rows) == 101
    data = np.array(rows, dtype=float)  # "inf" parses as float
    np.testing.assert_array_equal(data[:, 0], m.frequency_hz)
    np.testing.assert_array_equal(data[:, 1] + 1j * data[:, 2], m.s11)  # exact, not rounded


def test_csv_values(tmp_path):
    m = load_touchstone(SAMPLE)
    out = tmp_path / "out.csv"
    write_csv(m, out)
    _, rows = _read(out)
    first, at_540 = dict(zip(ONE_PORT_COLUMNS, rows[0])), dict(zip(ONE_PORT_COLUMNS, rows[60]))
    assert first["frequency_hz"] == "50000.0"
    assert first["vswr"] == "inf"  # |S11| > 1 at the first point
    assert float(at_540["vswr"]) == pytest.approx(11.797, abs=1e-3)
    assert float(at_540["z_real_ohm"]) == pytest.approx(5.649, abs=1e-3)
    assert float(at_540["z_imag_ohm"]) == pytest.approx(-28.707, abs=1e-3)


def test_csv_two_port_values(tmp_path, two_port):
    out = tmp_path / "out.csv"
    write_csv(two_port, out)
    header, rows = _read(out)
    row = dict(zip(header, rows[0]))
    assert float(row["s21_real"]) == pytest.approx(0.9)
    assert float(row["s21_imag"]) == pytest.approx(0.1)
    assert float(row["s21_db"]) == pytest.approx(20 * np.log10(abs(0.9 + 0.1j)))


@pytest.mark.parametrize("name", ["50.S1P", "MYSAVE2.S2P"])
def test_csv_for_bundled_samples(tmp_path, name):
    m = load_touchstone(SAMPLES / name)
    out = tmp_path / "out.csv"
    write_csv(m, out)
    header, rows = _read(out)
    assert len(rows) == m.points
    assert ("s21_db" in header) == (m.nports == 2)
