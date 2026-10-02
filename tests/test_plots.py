from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from nano_vna_viewer.plots import (
    DARK,
    LIGHT,
    SMITH_GRID_VALUES,
    PlotPanel,
    current_theme,
    smith_grid_lines,
    smith_outline,
)
from nano_vna_viewer.rf import SMOOTHING_FACTOR
from nano_vna_viewer.touchstone import load_touchstone, measurement_from_sweep

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "data.s1p"


@pytest.fixture
def sample():
    return load_touchstone(SAMPLE)


@pytest.fixture
def matched():
    # A resonance near 274 MHz with a best VSWR of 1.336, so there is a VSWR ≤ 2 band.
    return load_touchstone(SAMPLE.parent / "50.S1P")


@pytest.fixture
def two_port(tmp_path):
    path = tmp_path / "thru.s2p"
    path.write_text(
        "# Hz S RI R 50\n"
        "1000000 0.1 0.0 0.9 0.1 0.9 0.1 0.2 0.0\n"
        "2000000 0.2 0.0 0.8 0.2 0.8 0.2 0.3 0.0\n"
        "3000000 0.3 0.0 0.7 0.3 0.7 0.3 0.4 0.0\n"
    )
    return load_touchstone(path)


@pytest.fixture
def panel(qtbot):
    p = PlotPanel()
    qtbot.addWidget(p)
    p.resize(900, 600)
    p.show()
    qtbot.waitExposed(p)
    return p


TAB_NAMES = ["Return Loss", "VSWR", "Phase", "Smith Chart", "Transmission"]


def _visible_tabs(panel):
    return [panel.tabs.tabText(i) for i in range(panel.tabs.count()) if panel.tabs.isTabVisible(i)]


def test_empty_panel(panel):
    assert panel.measurement is None
    assert panel.readout.text() == "No data loaded."
    assert [panel.tabs.tabText(i) for i in range(panel.tabs.count())] == TAB_NAMES
    assert _visible_tabs(panel) == TAB_NAMES[:4]  # Transmission hidden until an .s2p loads
    panel.set_marker(5)  # ignored without data
    assert panel.marker_index is None


def test_one_port_traces(panel, sample):
    panel.set_measurement(sample)
    x, y = panel.magnitude_plot.curve.getData()
    assert len(x) == 101
    assert y[0] == pytest.approx(20 * np.log10(abs(sample.s11[0])))
    np.testing.assert_allclose(panel.phase_plot.curve.yData, np.angle(sample.s11, deg=True))
    assert _visible_tabs(panel) == TAB_NAMES[:4]


def test_infinite_vswr_is_a_gap(panel, sample):
    panel.set_measurement(sample)
    y = panel.vswr_plot.curve.yData
    assert np.isnan(y[0])  # |S11| > 1 at the first point
    assert np.all(np.isfinite(y[1:]))


def test_two_port_puts_s21_on_its_own_tab(panel, two_port):
    panel.set_measurement(two_port)
    assert _visible_tabs(panel) == TAB_NAMES
    # S11 tabs show S11 only
    np.testing.assert_allclose(panel.magnitude_plot.curve.yData, 20 * np.log10(abs(two_port.s11)))
    np.testing.assert_allclose(panel.phase_plot.curve.yData, np.angle(two_port.s11, deg=True))
    # Transmission tab shows S21
    np.testing.assert_allclose(
        panel.s21_magnitude_plot.curve.yData, 20 * np.log10(abs(two_port.s21))
    )
    np.testing.assert_allclose(panel.s21_phase_plot.curve.yData, np.angle(two_port.s21, deg=True))
    assert "S21:" in panel.readout.text()


def test_loading_one_port_after_two_port(panel, sample, two_port):
    panel.set_measurement(two_port)
    panel.tabs.setCurrentIndex(TAB_NAMES.index("Transmission"))
    panel.set_measurement(sample)
    assert _visible_tabs(panel) == TAB_NAMES[:4]
    assert panel.tabs.currentIndex() == 0  # moved off the hidden tab
    assert panel.s21_magnitude_plot.curve is None
    assert panel.s21_phase_plot.curve is None
    assert len(panel.magnitude_plot.getPlotItem().listDataItems()) == 1
    assert "S21:" not in panel.readout.text()


def test_marker_moves_transmission_plots(panel, two_port):
    panel.set_measurement(two_port)
    panel.set_marker(2)
    for plot in (panel.s21_magnitude_plot, panel.s21_phase_plot):
        assert plot.marker_line.value() == pytest.approx(two_port.frequency_hz[2])
    line = panel.s21_phase_plot.marker_line
    line.setValue(two_port.frequency_hz[1])
    line.sigDragged.emit(line)
    assert panel.marker_index == 1


def test_reset_view_resets_both_transmission_plots(qtbot, panel, two_port):
    panel.set_measurement(two_port)
    panel.tabs.setCurrentIndex(TAB_NAMES.index("Transmission"))
    qtbot.waitExposed(panel.s21_phase_plot)
    panel.reset_view()  # baseline at the tab's on-screen size
    views = [p.getPlotItem().getViewBox() for p in (panel.s21_magnitude_plot, panel.s21_phase_plot)]
    full = [v.viewRange() for v in views]
    for v in views:
        v.scaleBy((0.25, 0.25))
    panel.reset_view()
    for v, r in zip(views, full):
        np.testing.assert_allclose(v.viewRange(), r)


def test_no_legends_and_names_in_titles(panel):
    for plot in (panel.magnitude_plot, panel.vswr_plot, panel.phase_plot):
        assert plot.getPlotItem().legend is None
        assert "(S11)" in plot.getPlotItem().titleLabel.text
    for plot in (panel.s21_magnitude_plot, panel.s21_phase_plot):
        assert "(S21)" in plot.getPlotItem().titleLabel.text


def test_marker_line_is_2px(panel):
    for plot in (panel.magnitude_plot, panel.vswr_plot, panel.phase_plot, panel.s21_phase_plot):
        assert plot.marker_line.pen.widthF() == 2


def test_vswr_axis_labels_are_plain_1_2_5_numbers(panel):
    axis = panel.vswr_plot.getAxis("left")
    values = np.log10([1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 500, 1000])
    assert axis.logTickStrings(values, 1, 1) == [
        "1", "2", "", "5", "10", "20", "", "50", "100", "200", "500", "1000"
    ]


def test_marker_starts_at_minimum_vswr(panel, sample):
    panel.set_measurement(sample)
    assert panel.min_vswr_index == 38  # deepest dip, 342.031 MHz
    assert panel.marker_index == 38
    assert panel.readout.text() == (
        "Marker: 342.031 MHz    S11: -2.99 dB, -82.7°    VSWR: 5.869"
        "    Z: 18.81 − j53.18 Ω (series 8.75 pF)"
    )


def test_readout_shows_infinite_vswr_and_impedance_at_total_reflection(panel, sample):
    panel.set_measurement(sample)
    panel.set_marker(0)  # |S11| slightly above 1
    text = panel.readout.text()
    assert "Marker: 50 kHz" in text
    assert "VSWR: ∞" in text
    assert "Z: " in text


def test_set_marker_moves_every_plot(panel, sample):
    panel.set_measurement(sample)
    panel.set_marker(60)
    f = sample.frequency_hz[60]
    for plot in (panel.magnitude_plot, panel.vswr_plot, panel.phase_plot):
        assert plot.marker_line.value() == pytest.approx(f)
    x, y = panel.smith_chart.marker_dot.getData()
    assert complex(x[0], y[0]) == pytest.approx(sample.s11[60])
    # Z = 50 (1 + Γ) / (1 - Γ) with Γ = -0.419291008 - 0.73215136j, checked by hand;
    # C = 1 / (2π · 540.02 MHz · 28.71 Ω) = 10.3 pF
    assert panel.readout.text() == (
        "Marker: 540.02 MHz    S11: -1.48 dB, -119.8°    VSWR: 11.797"
        "    Z: 5.65 − j28.71 Ω (series 10.3 pF)"
    )


@pytest.mark.parametrize("index, expected", [(-5, 0), (1000, 100)])
def test_set_marker_clamps(panel, sample, index, expected):
    panel.set_measurement(sample)
    panel.set_marker(index)
    assert panel.marker_index == expected


def test_dragging_marker_snaps_to_nearest_point(panel, sample):
    panel.set_measurement(sample)
    line = panel.phase_plot.marker_line
    line.setValue(sample.frequency_hz[30] + 1_000_000)  # 1 MHz past point 30
    line.sigDragged.emit(line)
    assert panel.marker_index == 30
    assert line.value() == pytest.approx(sample.frequency_hz[30])  # snapped
    assert panel.vswr_plot.marker_line.value() == pytest.approx(sample.frequency_hz[30])


def _click_at(qtbot, plot, x, y):
    view = plot.getPlotItem().getViewBox()
    scene_pos = view.mapViewToScene(QPointF(x, y))
    qtbot.mouseClick(plot.viewport(), Qt.MouseButton.LeftButton, pos=plot.mapFromScene(scene_pos))


def test_click_on_frequency_plot_moves_marker(qtbot, panel, sample):
    panel.set_measurement(sample)
    plot = panel.magnitude_plot
    _click_at(qtbot, plot, sample.frequency_hz[42], -1.0)
    assert panel.marker_index == 42


def test_click_on_smith_chart_moves_marker(qtbot, panel, sample):
    panel.set_measurement(sample)
    panel.tabs.setCurrentIndex(3)
    qtbot.waitExposed(panel.smith_chart)
    target = sample.s11[75]
    _click_at(qtbot, panel.smith_chart, target.real, target.imag)
    assert panel.marker_index == 75


def test_reset_view_restores_auto_range(panel, sample):
    panel.set_measurement(sample)
    view = panel.magnitude_plot.getPlotItem().getViewBox()
    full = view.viewRange()
    view.scaleBy((0.25, 0.25))
    assert view.viewRange() != full
    panel.reset_view()
    np.testing.assert_allclose(view.viewRange(), full)


def test_smith_chart_shows_whole_unit_circle(qtbot, panel, sample):
    panel.set_measurement(sample)
    panel.tabs.setCurrentIndex(3)
    qtbot.waitExposed(panel.smith_chart)
    (x0, x1), (y0, y1) = panel.smith_chart.getPlotItem().getViewBox().viewRange()
    assert x0 <= -1 and x1 >= 1 and y0 <= -1 and y1 >= 1


def test_smith_outline_is_unit_circle_and_real_axis():
    circle, axis = smith_outline()
    np.testing.assert_allclose(np.abs(circle), 1)
    np.testing.assert_allclose(axis, [-1, 1])


def _unit_circle(admittance):
    """The constant-R (or constant-G) circle for normalized value 1."""
    return smith_grid_lines(admittance)[SMITH_GRID_VALUES.index(1.0)]


def test_impedance_grid_geometry():
    # r = 1 circle: centre 0.5, radius 0.5, passing through the chart centre
    np.testing.assert_allclose(np.abs(_unit_circle(False) - 0.5), 0.5)
    for line in smith_grid_lines():
        assert np.all(np.abs(line) <= 1 + 1e-9)


def test_admittance_grid_geometry():
    # g = 1 circle: centre -0.5, radius 0.5
    np.testing.assert_allclose(np.abs(_unit_circle(True) + 0.5), 0.5)
    # Every admittance curve is the matching impedance curve rotated 180 degrees.
    for z_line, y_line in zip(smith_grid_lines(), smith_grid_lines(admittance=True)):
        np.testing.assert_allclose(y_line, -z_line)


def test_admittance_grid_matches_admittance_formula():
    # A point with normalized admittance y = 1 + j1 must lie on both the g = 1
    # circle and the b = +1 arc of the admittance grid.
    y = 1 + 1j
    gamma = (1 - y) / (1 + y)
    lines = smith_grid_lines(admittance=True)
    g1 = lines[SMITH_GRID_VALUES.index(1.0)]
    b1 = lines[len(SMITH_GRID_VALUES) + 2 * SMITH_GRID_VALUES.index(1.0)]  # +b arc
    assert np.min(np.abs(g1 - gamma)) < 1e-2
    assert np.min(np.abs(b1 - gamma)) < 1e-2
    assert gamma.imag < 0  # capacitive susceptance sits in the lower half


def _admittance_shown(panel):
    shown = {item.isVisible() for item in panel.smith_chart.admittance_items}
    assert len(shown) == 1  # all admittance items share one state
    return shown.pop()


def test_admittance_overlay_is_off_by_default(panel):
    assert panel.admittance_toggle.text() == "Show admittance grid"
    assert not panel.admittance_toggle.isChecked()
    assert not panel.smith_chart.admittance_visible
    assert not _admittance_shown(panel)


def test_admittance_toggle_shows_and_hides_overlay(panel):
    panel.admittance_toggle.click()
    assert panel.smith_chart.admittance_visible
    assert _admittance_shown(panel)
    panel.admittance_toggle.click()
    assert not _admittance_shown(panel)


def test_impedance_grid_is_always_shown(panel):
    chart = panel.smith_chart
    impedance_items = [
        item
        for item in chart.getPlotItem().items
        if item not in chart.admittance_items
        and item not in (chart.marker_dot, chart.reference_dot)
    ]
    # Outline + every impedance curve + one label per grid value
    assert len(impedance_items) == 2 + len(smith_grid_lines()) + len(SMITH_GRID_VALUES)
    for checked in (True, False):
        panel.admittance_toggle.setChecked(checked)
        assert all(item.isVisible() for item in impedance_items)


def test_grid_change_keeps_trace_and_marker(panel, sample):
    panel.set_measurement(sample)
    panel.set_marker(60)
    panel.admittance_toggle.click()
    x, y = panel.smith_chart.curve.getData()
    np.testing.assert_allclose(x + 1j * y, sample.s11)
    assert panel.marker_index == 60


def test_reset_view_on_smith_tab(qtbot, panel, sample):
    panel.set_measurement(sample)
    panel.tabs.setCurrentIndex(3)
    qtbot.waitExposed(panel.smith_chart)
    view = panel.smith_chart.getPlotItem().getViewBox()
    view.scaleBy((0.25, 0.25))
    panel.reset_view()
    (x0, x1), (y0, y1) = view.viewRange()
    assert x0 <= -1 and x1 >= 1 and y0 <= -1 and y1 >= 1


def test_theme_follows_palette(qtbot):
    app = QApplication.instance()
    original = app.palette()
    try:
        dark = QPalette(original)
        dark.setColor(QPalette.ColorRole.Window, QColor("#202020"))
        app.setPalette(dark)
        assert current_theme() is DARK
        light = QPalette(original)
        light.setColor(QPalette.ColorRole.Window, QColor("#f0f0f0"))
        app.setPalette(light)
        assert current_theme() is LIGHT
    finally:
        app.setPalette(original)


def test_minimum_vswr_reference_on_every_plot(panel, sample):
    panel.set_measurement(sample)
    f_min = sample.frequency_hz[38]
    for plot in (panel.magnitude_plot, panel.vswr_plot, panel.phase_plot):
        assert plot.reference_line.isVisible()
        assert plot.reference_line.value() == pytest.approx(f_min)
    x, y = panel.smith_chart.reference_dot.getData()
    assert complex(x[0], y[0]) == pytest.approx(sample.s11[38])


def test_reference_stays_put_when_marker_moves(panel, sample):
    panel.set_measurement(sample)
    panel.set_marker(80)
    assert panel.vswr_plot.reference_line.value() == pytest.approx(sample.frequency_hz[38])


def test_go_to_min_vswr(panel, sample):
    panel.go_to_min_vswr()  # no data: ignored
    assert panel.marker_index is None
    panel.set_measurement(sample)
    panel.set_marker(80)
    panel.go_to_min_vswr()
    assert panel.marker_index == 38


def test_reference_follows_new_file(panel, sample, two_port):
    panel.set_measurement(two_port)
    assert panel.min_vswr_index == 0  # |S11| = 0.1, 0.2, 0.3
    assert panel.s21_magnitude_plot.reference_line.isVisible()
    panel.set_measurement(sample)
    assert panel.min_vswr_index == 38
    assert not panel.s21_magnitude_plot.reference_line.isVisible()


def test_export_widget(panel, two_port):
    panel.set_measurement(two_port)
    panel.tabs.setCurrentIndex(0)
    assert panel.export_widget() is panel.magnitude_plot
    panel.tabs.setCurrentIndex(3)
    assert panel.export_widget() is panel.smith_chart  # without the grid checkbox
    panel.tabs.setCurrentIndex(4)
    page = panel.export_widget()
    assert panel.s21_magnitude_plot in page.findChildren(type(panel.s21_magnitude_plot))
    assert panel.current_tab_name() == "Transmission"


SMOOTH_POINTS = (101 - 1) * SMOOTHING_FACTOR + 1


def test_smoothing_is_off_by_default(panel, sample):
    panel.set_measurement(sample)
    assert not panel.smoothing
    for plot in (panel.magnitude_plot, panel.vswr_plot, panel.phase_plot, panel.smith_chart):
        assert len(plot.curve.getData()[0]) == 101
        assert plot.points is None


def test_smoothing_draws_dense_curve_and_measured_dots(panel, sample):
    panel.set_measurement(sample)
    panel.set_smoothing(True)
    for plot in (panel.magnitude_plot, panel.phase_plot):
        assert len(plot.curve.getData()[0]) == SMOOTH_POINTS
        np.testing.assert_array_equal(plot.points.xData, sample.frequency_hz)
    np.testing.assert_allclose(panel.magnitude_plot.points.yData, 20 * np.log10(abs(sample.s11)))
    np.testing.assert_allclose(panel.phase_plot.points.yData, np.angle(sample.s11, deg=True))
    # The curve passes through the measured points.
    np.testing.assert_allclose(
        panel.magnitude_plot.curve.yData[::SMOOTHING_FACTOR], 20 * np.log10(abs(sample.s11))
    )
    # The dots have no connecting line, and the line has no dots.
    assert panel.magnitude_plot.points.opts["pen"] is None
    assert panel.magnitude_plot.curve.opts["symbol"] is None


def test_smoothed_smith_chart(panel, sample):
    panel.set_measurement(sample)
    panel.set_smoothing(True)
    chart = panel.smith_chart
    x, y = chart.curve.getData()
    assert len(x) == SMOOTH_POINTS
    np.testing.assert_allclose((x + 1j * y)[::SMOOTHING_FACTOR], sample.s11)
    x, y = chart.points.getData()
    np.testing.assert_allclose(x + 1j * y, sample.s11)
    np.testing.assert_array_equal(chart.gamma, sample.s11)  # the marker still uses measured data


def test_smoothed_vswr_leaves_out_infinite_points(panel, sample):
    panel.set_measurement(sample)
    panel.set_smoothing(True)
    # |S11| > 1 at the first point: no dot there, and the curve starts with a gap.
    np.testing.assert_array_equal(panel.vswr_plot.points.xData, sample.frequency_hz[1:])
    assert np.isnan(panel.vswr_plot.curve.yData[0])


def test_smoothing_keeps_marker_readout_and_zoom(panel, sample):
    panel.set_measurement(sample)
    panel.set_marker(60)
    readout = panel.readout.text()
    view = panel.magnitude_plot.getPlotItem().getViewBox()
    view.scaleBy((0.25, 0.25))
    zoomed = view.viewRange()
    for enabled in (True, False):
        panel.set_smoothing(enabled)
        assert panel.marker_index == 60
        assert panel.min_vswr_index == 38
        assert panel.readout.text() == readout
        np.testing.assert_allclose(view.viewRange(), zoomed)
        for plot in (panel.magnitude_plot, panel.vswr_plot, panel.phase_plot):
            assert plot.marker_line.isVisible()
            assert plot.marker_line.value() == pytest.approx(sample.frequency_hz[60])
            assert plot.reference_line.isVisible()
            assert plot.reference_line.value() == pytest.approx(sample.frequency_hz[38])
        x, y = panel.smith_chart.marker_dot.getData()
        assert complex(x[0], y[0]) == pytest.approx(sample.s11[60])
        x, y = panel.smith_chart.reference_dot.getData()
        assert complex(x[0], y[0]) == pytest.approx(sample.s11[38])


def test_marker_snaps_to_measured_points_when_smoothed(qtbot, panel, sample):
    panel.set_measurement(sample)
    panel.set_smoothing(True)
    # Between points 42 and 43, nearer to 42: there is a curve point here but no measurement.
    step = sample.frequency_hz[43] - sample.frequency_hz[42]
    _click_at(qtbot, panel.magnitude_plot, sample.frequency_hz[42] + 0.3 * step, -1.0)
    assert panel.marker_index == 42
    assert panel.magnitude_plot.marker_line.value() == pytest.approx(sample.frequency_hz[42])


def test_turning_smoothing_off_restores_plain_traces(panel, sample):
    panel.set_measurement(sample)
    panel.set_smoothing(True)
    panel.set_smoothing(False)
    for plot in (panel.magnitude_plot, panel.vswr_plot, panel.phase_plot, panel.smith_chart):
        assert len(plot.curve.getData()[0]) == 101
        assert plot.points is None
    assert len(panel.magnitude_plot.getPlotItem().listDataItems()) == 1


def test_smoothing_applies_to_later_measurements(panel, sample, two_port):
    panel.set_smoothing(True)  # nothing loaded yet: remembered for later
    assert panel.magnitude_plot.curve is None
    panel.set_measurement(two_port)
    for plot in (panel.magnitude_plot, panel.s21_magnitude_plot, panel.s21_phase_plot):
        assert len(plot.curve.getData()[0]) == (3 - 1) * SMOOTHING_FACTOR + 1
        np.testing.assert_array_equal(plot.points.xData, two_port.frequency_hz)
    panel.set_measurement(sample)
    assert len(panel.magnitude_plot.curve.getData()[0]) == SMOOTH_POINTS
    assert panel.s21_magnitude_plot.curve is None
    assert panel.s21_magnitude_plot.points is None
    assert len(panel.magnitude_plot.getPlotItem().listDataItems()) == 2  # curve + dots


def test_data_that_cannot_be_splined_is_drawn_plain(panel, tmp_path):
    path = tmp_path / "two_points.s1p"
    path.write_text("# Hz S RI R 50\n1000000 0.1 0.0\n2000000 0.2 0.0\n")
    panel.set_smoothing(True)
    panel.set_measurement(load_touchstone(path))
    assert len(panel.magnitude_plot.curve.getData()[0]) == 2
    assert panel.magnitude_plot.points is None
    assert panel.smith_chart.points is None


def test_matched_band_is_shaded_on_return_loss_and_vswr(panel, matched):
    panel.set_measurement(matched)
    low, high = panel.vswr_band
    assert low == pytest.approx(271.514e6, rel=1e-5)
    assert high == pytest.approx(276.071e6, rel=1e-5)
    for plot in (panel.magnitude_plot, panel.vswr_plot):
        assert plot.band_region.isVisible()
        assert plot.band_region.getRegion() == pytest.approx((low, high))
        assert not plot.band_region.movable
    assert not panel.phase_plot.band_region.isVisible()


def test_no_band_is_shaded_without_a_good_match(panel, sample, matched):
    panel.set_measurement(sample)  # best VSWR 5.869
    assert panel.vswr_band is None
    assert not panel.vswr_plot.band_region.isVisible()
    panel.set_measurement(matched)
    assert panel.vswr_plot.band_region.isVisible()
    panel.set_measurement(sample)
    assert not panel.vswr_plot.band_region.isVisible()


def test_band_edge_outside_the_sweep_is_drawn_at_the_sweep_end(panel):
    frequency = np.linspace(140e6, 150e6, 11)
    panel.set_measurement(measurement_from_sweep(frequency, np.full(11, 0.1 + 0j)))
    assert panel.vswr_band == (None, None)
    assert panel.vswr_plot.band_region.getRegion() == pytest.approx((140e6, 150e6))


def test_band_survives_smoothing_and_does_not_block_clicks(qtbot, panel, matched):
    panel.set_measurement(matched)
    region = panel.magnitude_plot.band_region.getRegion()
    panel.set_smoothing(True)
    assert panel.magnitude_plot.band_region.isVisible()
    assert panel.magnitude_plot.band_region.getRegion() == pytest.approx(region)
    panel.set_marker(10)
    _click_at(qtbot, panel.magnitude_plot, 274.5e6, -5.0)  # inside the shaded band
    assert panel.marker_index == 44


def _memory_data(plot):
    x, y = plot.memory.getData()
    return np.asarray(x), np.asarray(y)


def test_memory_trace_on_every_s11_plot(panel, sample):
    panel.set_memory(sample)
    assert panel.memory is sample
    x, y = _memory_data(panel.magnitude_plot)
    np.testing.assert_array_equal(x, sample.frequency_hz)
    np.testing.assert_allclose(y, 20 * np.log10(abs(sample.s11)))
    np.testing.assert_allclose(_memory_data(panel.phase_plot)[1], np.angle(sample.s11, deg=True))
    assert len(_memory_data(panel.vswr_plot)[0]) == 101
    x, y = _memory_data(panel.smith_chart)
    np.testing.assert_allclose(x + 1j * y, sample.s11)
    for plot in (panel.s21_magnitude_plot, panel.s21_phase_plot):
        assert plot.memory is None  # the held measurement has no S21


def test_memory_trace_stays_when_a_new_measurement_is_shown(panel, sample, two_port):
    panel.set_memory(sample)
    held = panel.magnitude_plot.memory
    panel.set_measurement(two_port)
    assert panel.magnitude_plot.memory is held
    assert held in panel.magnitude_plot.getPlotItem().items
    np.testing.assert_allclose(panel.magnitude_plot.curve.yData, 20 * np.log10(abs(two_port.s11)))
    # The marker and its readout belong to the live measurement.
    assert panel.measurement is two_port
    assert "Marker: 1 MHz" in panel.readout.text()


def test_memory_trace_is_left_out_of_the_auto_range(panel, sample, two_port):
    panel.set_memory(sample)  # 50 kHz to 900 MHz
    panel.set_measurement(two_port)  # 1 to 3 MHz
    (x0, x1), _ = panel.magnitude_plot.getPlotItem().getViewBox().viewRange()
    assert 0 < x0 and x1 < 4e6


def test_memory_trace_draws_behind_the_live_trace(panel, sample, two_port):
    panel.set_memory(sample)
    panel.set_measurement(two_port)
    for plot in (panel.magnitude_plot, panel.smith_chart):
        assert plot.memory.zValue() < plot.curve.zValue()
    assert panel.magnitude_plot.memory.opts["pen"].color().name() == panel.theme.memory


def test_memory_trace_with_s21(panel, sample, two_port):
    panel.set_memory(two_port)
    np.testing.assert_allclose(
        _memory_data(panel.s21_magnitude_plot)[1], 20 * np.log10(abs(two_port.s21))
    )
    np.testing.assert_allclose(
        _memory_data(panel.s21_phase_plot)[1], np.angle(two_port.s21, deg=True)
    )
    panel.set_memory(sample)
    assert panel.s21_magnitude_plot.memory is None


def test_memory_trace_follows_smoothing(panel, sample):
    panel.set_memory(sample)
    panel.set_smoothing(True)  # no live measurement: only the memory trace is redrawn
    for plot in (panel.magnitude_plot, panel.smith_chart):
        assert len(_memory_data(plot)[0]) == SMOOTH_POINTS
    panel.set_measurement(sample)
    assert len(panel.magnitude_plot.getPlotItem().listDataItems()) == 3  # curve, dots, memory
    panel.set_smoothing(False)
    assert len(_memory_data(panel.magnitude_plot)[0]) == 101


def test_clearing_the_memory_trace(panel, sample):
    panel.set_measurement(sample)
    panel.set_memory(sample)
    panel.set_memory(None)
    assert panel.memory is None
    for plot in panel.plots:
        assert plot.memory is None
    assert len(panel.magnitude_plot.getPlotItem().listDataItems()) == 1
    np.testing.assert_allclose(panel.magnitude_plot.curve.yData, 20 * np.log10(abs(sample.s11)))


def test_marker_draws_over_reference(panel):
    for plot in (panel.magnitude_plot, panel.vswr_plot, panel.phase_plot, panel.s21_magnitude_plot):
        assert plot.marker_line.zValue() > plot.reference_line.zValue()
    chart = panel.smith_chart
    assert chart.marker_dot.zValue() > chart.reference_dot.zValue()
