from pathlib import Path

import numpy as np
import pytest

from nano_vna_viewer.rf import (
    SMOOTHING_FACTOR,
    impedance,
    interpolate_s,
    min_vswr_index,
    phase_deg,
    s_db,
    vswr,
    vswr_band,
)
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


def _dip(frequency, centre=150.4e6):
    """|S11| rising in a straight line either side of a perfect match at ``centre``."""
    return np.abs(frequency - centre) / 60e6 + 0j


def test_vswr_band_edges_are_interpolated_between_points():
    frequency = np.linspace(100e6, 200e6, 101)  # 1 MHz steps; the edges fall between points
    # VSWR 2 is |S11| = 1/3, reached 20 MHz either side of the match.
    low, high = vswr_band(frequency, _dip(frequency))
    assert low == pytest.approx(130.4e6)
    assert high == pytest.approx(170.4e6)


def test_vswr_band_with_another_limit():
    frequency = np.linspace(100e6, 200e6, 101)
    # VSWR 1.5 is |S11| = 0.2, reached 12 MHz either side.
    low, high = vswr_band(frequency, _dip(frequency), limit=1.5)
    assert low == pytest.approx(138.4e6)
    assert high == pytest.approx(162.4e6)


def test_vswr_band_is_none_when_the_best_match_is_above_the_limit():
    frequency = np.linspace(100e6, 200e6, 101)
    assert vswr_band(frequency, np.full(101, 0.5 + 0j)) is None
    assert vswr_band(load_touchstone(SAMPLE).frequency_hz, load_touchstone(SAMPLE).s11) is None


def test_vswr_band_edge_outside_the_sweep_is_none():
    frequency = np.linspace(140e6, 200e6, 61)  # the low edge (130.4 MHz) is below the sweep
    low, high = vswr_band(frequency, _dip(frequency))
    assert low is None
    assert high == pytest.approx(170.4e6)
    assert vswr_band(frequency, np.full(61, 0.1 + 0j)) == (None, None)


def test_vswr_band_stops_at_the_first_crossing():
    # A second matched region further out is not part of the band around the best match.
    frequency = np.linspace(100e6, 200e6, 11)
    s11 = np.array([0.1, 0.1, 0.9, 0.9, 0.3, 0.05, 0.3, 0.9, 0.9, 0.9, 0.9], dtype=complex)
    low, high = vswr_band(frequency, s11)
    assert 130e6 < low < 140e6
    assert 160e6 < high < 170e6


def test_interpolate_s_keeps_every_measured_point():
    m = load_touchstone(SAMPLE)
    f, s = interpolate_s(m.frequency_hz, m.s11)
    assert len(f) == len(s) == (m.points - 1) * SMOOTHING_FACTOR + 1
    np.testing.assert_array_equal(f[::SMOOTHING_FACTOR], m.frequency_hz)
    np.testing.assert_allclose(s[::SMOOTHING_FACTOR], m.s11, rtol=1e-12)
    assert np.all(np.diff(f) > 0)


def test_interpolate_s_follows_a_smooth_response():
    # A mismatched load behind a delay line: |Γ| = 0.5, phase turning 3 times over the sweep.
    def gamma(freq):
        return 0.5 * np.exp(-2j * np.pi * freq * 3e-8)

    frequency = np.linspace(1e6, 101e6, 101)
    f, s = interpolate_s(frequency, gamma(frequency))
    np.testing.assert_allclose(s, gamma(f), atol=1e-4)
    # Straight lines between the same points cut the corners far more.
    chords = np.interp(f, frequency, gamma(frequency))
    assert np.max(np.abs(s - gamma(f))) < np.max(np.abs(chords - gamma(f))) / 50


def test_interpolate_s_handles_uneven_spacing():
    frequency = np.array([1e6, 2e6, 5e6, 10e6, 20e6])
    f, s = interpolate_s(frequency, 0.1 * np.arange(5) + 0j, factor=4)
    np.testing.assert_array_equal(f[::4], frequency)
    np.testing.assert_allclose(f[:5], [1e6, 1.25e6, 1.5e6, 1.75e6, 2e6])


@pytest.mark.parametrize(
    "frequency, s",
    [
        ([1e6, 2e6], [0.1, 0.2]),  # too few points
        ([1e6, 3e6, 2e6, 4e6], [0.1, 0.2, 0.3, 0.4]),  # frequency goes backwards
        ([1e6, 2e6, 2e6, 4e6], [0.1, 0.2, 0.3, 0.4]),  # repeated frequency
        ([1e6, 2e6, 3e6, 4e6], [0.1, np.nan, 0.3, 0.4]),
    ],
)
def test_interpolate_s_declines_data_it_cannot_spline(frequency, s):
    assert interpolate_s(np.array(frequency), np.array(s, dtype=complex)) is None
