from pathlib import Path

import numpy as np
import pytest

from nano_vna_viewer.rf import impedance, min_vswr_index, phase_deg, s_db, vswr
from nano_vna_viewer.touchstone import load_touchstone

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "data.s1p"


def test_s_db():
    np.testing.assert_allclose(s_db(np.array([1, 0.1, 0.1j, 0.5])), [0, -20, -20, -6.0206], atol=1e-4)


def test_s_db_of_zero_is_finite():
    assert s_db(np.array([0j]))[0] == pytest.approx(-240)


def test_vswr_known_values():
    gamma = np.array([0, 1 / 3, -1 / 3j, 0.5])
    np.testing.assert_allclose(vswr(gamma), [1, 2, 2, 3])


@pytest.mark.parametrize("gamma", [1, -1, 1j, 1.0004 + 0.0001j])
def test_vswr_is_infinite_at_or_above_total_reflection(gamma):
    assert np.isinf(vswr(np.array([gamma]))[0])


def test_sample_vswr_is_never_negative():
    # The sample has |S11| slightly above 1 at its first point.
    swr = vswr(load_touchstone(SAMPLE).s11)
    assert np.isinf(swr[0])
    assert np.all(swr >= 1)


def test_phase_deg():
    np.testing.assert_allclose(phase_deg(np.array([1, 1j, -1, -1j])), [0, 90, 180, -90])


def test_impedance_known_values():
    gamma = np.array([0, -1, 1 / 3, -1 / 3, 1j, -1j])
    np.testing.assert_allclose(impedance(gamma, 50), [50, 0, 100, 25, 50j, -50j], atol=1e-9)


def test_impedance_scales_with_z0():
    assert impedance(np.array([1 / 3]), 75)[0] == pytest.approx(150)


def test_impedance_of_ideal_open_is_infinite():
    z = impedance(np.array([1 + 0j]), 50)[0]
    assert np.isinf(z.real)


def test_min_vswr_index():
    assert min_vswr_index(np.array([0.9, 0.5j, -0.2, 0.7])) == 2
    assert min_vswr_index(load_touchstone(SAMPLE).s11) == 38
