import pickle
import shutil
from pathlib import Path

import numpy as np
import pytest

from nano_vna_viewer.touchstone import TouchstoneError, load_touchstone

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "data.s1p"


def test_loads_sample():
    m = load_touchstone(SAMPLE)
    assert m.name == "data.s1p"
    assert m.nports == 1
    assert m.points == 101
    assert m.start_hz == 50_000
    assert m.stop_hz == 900_000_000
    assert m.z0 == 50
    assert m.s21 is None


def test_sample_values_match_file():
    m = load_touchstone(SAMPLE)
    # First and last data lines of samples/data.s1p (RI format)
    assert m.s11[0] == pytest.approx(1.000393629 + 0.000130278j)
    assert m.s11[-1] == pytest.approx(-0.829960768 - 0.303223040j)


def test_relative_path_is_made_absolute(monkeypatch):
    monkeypatch.chdir(SAMPLE.parent.parent)
    m = load_touchstone(Path("samples") / "data.s1p")
    assert m.path == SAMPLE


def test_uppercase_extension(tmp_path):
    upper = tmp_path / "DATA.S1P"
    shutil.copy(SAMPLE, upper)
    assert load_touchstone(upper).points == 101


@pytest.mark.parametrize("form", ["ma", "db"])
def test_other_data_formats_give_same_values(tmp_path, form):
    original = load_touchstone(SAMPLE)
    original.network.write_touchstone(
        filename="converted", dir=str(tmp_path), form=form
    )
    converted = load_touchstone(tmp_path / "converted.s1p")
    assert converted.frequency_hz == pytest.approx(original.frequency_hz)
    np.testing.assert_allclose(converted.s11, original.s11, rtol=1e-6, atol=1e-9)


def test_two_port_file(tmp_path):
    path = tmp_path / "thru.s2p"
    path.write_text(
        "# Hz S RI R 50\n"
        "1000000 0.1 0.0 0.9 0.1 0.9 0.1 0.2 0.0\n"
        "2000000 0.2 0.0 0.8 0.2 0.8 0.2 0.3 0.0\n"
    )
    m = load_touchstone(path)
    assert m.nports == 2
    # Touchstone v1 two-port order is S11 S21 S12 S22
    assert m.s11[0] == pytest.approx(0.1)
    assert m.s21[0] == pytest.approx(0.9 + 0.1j)


def test_missing_file(tmp_path):
    with pytest.raises(TouchstoneError, match="not found"):
        load_touchstone(tmp_path / "missing.s1p")


@pytest.mark.parametrize("name", ["data.txt", "data.csv", "data", "data.s3p"])
def test_unsupported_extension(tmp_path, name):
    path = tmp_path / name
    shutil.copy(SAMPLE, path)
    with pytest.raises(TouchstoneError, match="Unsupported file type"):
        load_touchstone(path)


@pytest.mark.parametrize(
    "content",
    ["", "# Hz S RI R 50\n", "! only a comment\n"],
    ids=["empty", "header-only", "comment-only"],
)
def test_no_data_points(tmp_path, content):
    path = tmp_path / "empty.s1p"
    path.write_text(content)
    with pytest.raises(TouchstoneError, match="no data points"):
        load_touchstone(path)


@pytest.mark.parametrize(
    "content",
    ["hello world\n", "# Hz S RI R 50\n1 abc 2\n", "# Hz S RI R 50\n1 2 3 4 5\n"],
    ids=["text", "non-numeric", "wrong-column-count"],
)
def test_malformed_file(tmp_path, content):
    path = tmp_path / "bad.s1p"
    path.write_text(content)
    with pytest.raises(TouchstoneError, match="Could not read"):
        load_touchstone(path)


class _Payload:
    ran = False

    def __reduce__(self):
        return (_mark_ran, ())


def _mark_ran():
    _Payload.ran = True


def test_pickle_disguised_as_touchstone_is_not_executed(tmp_path):
    path = tmp_path / "evil.s1p"
    path.write_bytes(pickle.dumps(_Payload()))
    with pytest.raises(TouchstoneError):
        load_touchstone(path)
    assert not _Payload.ran


# -- live sweeps and saving ------------------------------------------------------------

from nano_vna_viewer.touchstone import (  # noqa: E402
    measurement_from_sweep,
    touchstone_suffix,
    write_touchstone,
)


def test_measurement_from_sweep_one_port():
    f = np.array([1e6, 2e6, 3e6])
    m = measurement_from_sweep(f, np.array([0.1, 0.2j, -0.3]), source="NanoVNA (COM12) 13:00:00")
    assert m.path is None
    assert m.name == "NanoVNA (COM12) 13:00:00"
    assert m.nports == 1
    assert m.s21 is None
    assert m.z0 == 50
    np.testing.assert_array_equal(m.frequency_hz, f)


def test_measurement_from_sweep_two_port():
    f = np.array([1e6, 2e6])
    m = measurement_from_sweep(f, np.array([0.1, 0.2]), np.array([0.5j, 0.6j]))
    assert m.nports == 2
    np.testing.assert_array_equal(m.s21, [0.5j, 0.6j])
    np.testing.assert_array_equal(m.network.s[:, 0, 1], 0)  # S12 and S22 left at zero
    np.testing.assert_array_equal(m.network.s[:, 1, 1], 0)


@pytest.mark.parametrize("name", ["data.s1p", "MYSAVE2.S2P", "50.S1P"])
def test_write_touchstone_round_trip(tmp_path, name):
    original = load_touchstone(SAMPLE.parent / name)
    out = tmp_path / f"copy{touchstone_suffix(original)}"
    write_touchstone(original, out)
    copy = load_touchstone(out)
    np.testing.assert_array_equal(copy.frequency_hz, original.frequency_hz)
    np.testing.assert_allclose(copy.network.s, original.network.s, rtol=1e-8, atol=1e-12)


def test_written_file_matches_nanovna_layout(tmp_path):
    m = measurement_from_sweep(np.array([50_000.0]), np.array([1.000393629 + 0.000130278j]))
    out = tmp_path / "x.s1p"
    write_touchstone(m, out)
    assert out.read_text().splitlines() == [
        "! Saved by Nano VNA Viewer",
        "# Hz S RI R 50",
        "50000 1.00039363 0.000130278",
    ]


def test_touchstone_suffix():
    f = np.array([1e6, 2e6])
    assert touchstone_suffix(measurement_from_sweep(f, np.zeros(2))) == ".s1p"
    assert touchstone_suffix(measurement_from_sweep(f, np.zeros(2), np.zeros(2))) == ".s2p"
