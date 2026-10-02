import pytest

from nano_vna_viewer.units import format_frequency, format_impedance, format_series_component


@pytest.mark.parametrize(
    "hz, expected",
    [
        (0, "0 Hz"),
        (999, "999 Hz"),
        (1_000, "1 kHz"),
        (50_000, "50 kHz"),
        (9_049_500, "9.0495 MHz"),
        (900_000_000, "900 MHz"),
        (1_500_000_000, "1.5 GHz"),
    ],
)
def test_format_frequency(hz, expected):
    assert format_frequency(hz) == expected


@pytest.mark.parametrize(
    "z, expected",
    [
        (50 + 0j, "50 Ω"),
        (75 + 0j, "75 Ω"),
        (25 + 10j, "25 + j10 Ω"),
        (25 - 10j, "25 − j10 Ω"),
    ],
)
def test_format_impedance(z, expected):
    assert format_impedance(z) == expected


@pytest.mark.parametrize(
    "z, expected",
    [
        (50 + 0j, "50.00 + j0.00 Ω"),
        (5.649 - 28.707j, "5.65 − j28.71 Ω"),
        (complex(float("inf"), 0), "∞ Ω"),
        (complex(float("inf"), float("nan")), "∞ Ω"),
    ],
)
def test_format_impedance_fixed_decimals(z, expected):
    assert format_impedance(z, decimals=2) == expected


@pytest.mark.parametrize(
    "reactance, hz, expected",
    [
        (-53.18, 342.031e6, "8.75 pF"),  # C = 1 / (2π · 342.031 MHz · 53.18 Ω)
        (-28.71, 540.02e6, "10.3 pF"),
        (100, 145e6, "110 nH"),  # L = 100 Ω / (2π · 145 MHz)
        (50, 1e6, "7.96 µH"),
        (-1, 100, "1.59 mF"),
        (0, 1e6, ""),
        (float("inf"), 1e6, ""),
        (float("nan"), 1e6, ""),
        (50, 0, ""),
    ],
)
def test_format_series_component(reactance, hz, expected):
    assert format_series_component(reactance, hz) == expected
