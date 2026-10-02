from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtGui import QImage

from fake_nanovna import FakePorts
from nano_vna_viewer.device_panel import DevicePanel
from nano_vna_viewer.touchstone import load_touchstone
from nano_vna_viewer.main_window import (
    APP_NAME,
    LAST_DIR_KEY,
    NO_VALUE,
    SMOOTH_KEY,
    MainWindow,
    band_text,
)

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "data.s1p"
# A resonance near 274 MHz with a best VSWR of 1.336, so there is a VSWR ≤ 2 band.
MATCHED = SAMPLE.parent / "50.S1P"


@pytest.fixture
def settings(tmp_path):
    # A throwaway INI file so tests never touch the real user preferences.
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


@pytest.fixture
def ports():
    return FakePorts()


def _panel(ports):
    return DevicePanel(device_factory=ports.device_factory, port_lister=ports.list_ports)


@pytest.fixture
def window(qtbot, settings, monkeypatch, ports):
    # A device panel wired to the simulated NanoVNA, so tests never touch real ports.
    w = MainWindow(settings, device_panel=_panel(ports))
    qtbot.addWidget(w)
    errors = []
    # No blocking dialogs: record messages instead.
    monkeypatch.setattr(w, "show_error", lambda message, title=None: errors.append(message))
    w.errors = errors
    return w


def test_initial_state(window):
    assert window.windowTitle() == APP_NAME
    assert window.measurement is None
    assert window.file_label.text() == NO_VALUE
    assert "Open" in window.statusBar().currentMessage()


def test_menus(window):
    menus = [a.text() for a in window.menuBar().actions()]
    assert menus == ["&File", "&View", "&Help"]
    assert window.open_action.shortcut().toString() == "Ctrl+O"


def test_open_sample_shows_details(window):
    assert window.open_file(SAMPLE)
    assert window.measurement is not None
    assert window.windowTitle() == f"data.s1p — {APP_NAME}"
    assert window.file_label.text() == str(SAMPLE)
    assert window.ports_label.text() == "1"
    assert window.range_label.text() == "50 kHz to 900 MHz"
    assert window.points_label.text() == "101"
    assert window.z0_label.text() == "50 Ω"
    assert window.statusBar().currentMessage() == "Loaded data.s1p"
    assert window.errors == []


def test_open_bad_file_reports_error_and_keeps_previous(window, tmp_path):
    window.open_file(SAMPLE)
    bad = tmp_path / "bad.s1p"
    bad.write_text("not touchstone\n")

    assert not window.open_file(bad)
    assert len(window.errors) == 1
    assert "bad.s1p" in window.errors[0]
    assert window.statusBar().currentMessage() == "Failed to open file."
    # The previously loaded file is still shown.
    assert window.measurement.name == "data.s1p"
    assert window.points_label.text() == "101"


def test_remembers_last_folder(window, settings):
    window.open_file(SAMPLE)
    assert settings.value(LAST_DIR_KEY) == str(SAMPLE.parent)
    assert window.start_dir() == str(SAMPLE.parent)


def test_last_folder_persists_to_new_window(qtbot, settings, ports):
    first = MainWindow(settings, device_panel=_panel(ports))
    qtbot.addWidget(first)
    first.open_file(SAMPLE)
    settings.sync()

    second = MainWindow(settings, device_panel=_panel(ports))
    qtbot.addWidget(second)
    assert second.start_dir() == str(SAMPLE.parent)


def test_start_dir_falls_back_to_home(window, settings, tmp_path):
    assert window.start_dir() == str(Path.home())
    settings.setValue(LAST_DIR_KEY, str(tmp_path / "deleted-folder"))
    assert window.start_dir() == str(Path.home())


def test_failed_open_does_not_change_last_folder(window, settings, tmp_path):
    window.open_file(tmp_path / "missing.s1p")
    assert settings.value(LAST_DIR_KEY) is None


def test_choose_file_uses_dialog_result(window, monkeypatch):
    calls = []

    def fake_dialog(parent, caption, directory, file_filter):
        calls.append(directory)
        return str(SAMPLE), ""

    monkeypatch.setattr(
        "nano_vna_viewer.main_window.QFileDialog.getOpenFileName", fake_dialog
    )
    window.choose_file()
    assert calls == [str(Path.home())]
    assert window.measurement.name == "data.s1p"


def test_choose_file_cancelled(window, monkeypatch):
    monkeypatch.setattr(
        "nano_vna_viewer.main_window.QFileDialog.getOpenFileName",
        lambda *args: ("", ""),
    )
    window.choose_file()
    assert window.measurement is None
    assert window.errors == []


def test_open_file_fills_plots(window):
    window.open_file(SAMPLE)
    assert window.plots.measurement is window.measurement
    assert window.plots.marker_index == 38  # starts at minimum VSWR
    assert window.reset_view_action.shortcut().toString() == "Ctrl+0"
    assert window.min_vswr_action.shortcut().toString() == "Ctrl+M"


def test_minimum_vswr_detail(window):
    assert window.min_vswr_label.text() == NO_VALUE
    window.open_file(SAMPLE)
    assert window.min_vswr_label.text() == "5.869 at 342.031 MHz  (S11 -2.99 dB)"


def test_smooth_curves_action(window, settings):
    action = window.smooth_action
    assert action.text() == "&Smooth Curves"
    assert action.isCheckable() and not action.isChecked()
    assert not window.plots.smoothing
    window.open_file(SAMPLE)
    assert len(window.plots.magnitude_plot.curve.getData()[0]) == 101

    action.trigger()
    assert window.plots.smoothing
    assert len(window.plots.magnitude_plot.curve.getData()[0]) == 1001
    assert settings.value(SMOOTH_KEY, type=bool) is True
    # Display only: the readouts still describe measured points.
    assert window.min_vswr_label.text() == "5.869 at 342.031 MHz  (S11 -2.99 dB)"
    assert window.points_label.text() == "101"

    action.trigger()
    assert not window.plots.smoothing
    assert settings.value(SMOOTH_KEY, type=bool) is False


def test_smoothing_persists_to_new_window(qtbot, settings, ports):
    first = MainWindow(settings, device_panel=_panel(ports))
    qtbot.addWidget(first)
    first.smooth_action.trigger()
    settings.sync()

    second = MainWindow(settings, device_panel=_panel(ports))
    qtbot.addWidget(second)
    assert second.smooth_action.isChecked()
    assert second.plots.smoothing
    second.open_file(SAMPLE)
    assert len(second.plots.magnitude_plot.curve.getData()[0]) == 1001


def test_exports_are_unaffected_by_smoothing(window, tmp_path):
    window.open_file(SAMPLE)
    window.smooth_action.trigger()
    assert window.export_csv(tmp_path / "data.csv")
    assert len((tmp_path / "data.csv").read_text(encoding="utf-8").splitlines()) == 102
    assert window.save_touchstone(tmp_path / "copy.s1p")
    assert load_touchstone(tmp_path / "copy.s1p").points == 101


def test_matched_band_row(window):
    assert window.band_label.text() == NO_VALUE
    window.open_file(SAMPLE)  # best VSWR 5.869
    assert window.band_label.text() == "none"
    window.open_file(MATCHED)
    assert window.band_label.text() == "271.514 MHz to 276.071 MHz  (4.55706 MHz wide)"


def test_band_text_for_edges_outside_the_sweep():
    m = load_touchstone(MATCHED)  # 135 to 450 MHz
    assert band_text(m, (None, 276e6)) == "below 135 MHz to 276 MHz"
    assert band_text(m, (271.5e6, None)) == "271.5 MHz to above 450 MHz"
    assert band_text(m, (None, None)) == "below 135 MHz to above 450 MHz"


def test_reference_actions_start_disabled(window):
    assert window.hold_reference_action.shortcut().toString() == "Ctrl+R"
    assert not window.hold_reference_action.isEnabled()  # nothing to hold yet
    assert not window.clear_reference_action.isEnabled()
    assert window.load_reference_action.isEnabled()
    assert window.reference_label.text() == NO_VALUE
    window.hold_reference()  # ignored without data
    assert window.plots.memory is None


def test_hold_and_clear_reference(window):
    window.open_file(SAMPLE)
    assert window.hold_reference_action.isEnabled()
    window.hold_reference_action.trigger()
    held = window.measurement
    assert window.plots.memory is held
    assert window.reference_label.text() == (
        "data.s1p: minimum VSWR 5.869 at 342.031 MHz  (S11 -2.99 dB)"
    )
    assert window.clear_reference_action.isEnabled()

    window.open_file(MATCHED)  # the reference stays while other data is shown
    assert window.plots.memory is held
    assert window.plots.measurement is window.measurement
    assert window.min_vswr_label.text() == "1.336 at 273.6 MHz  (S11 -16.84 dB)"
    assert window.reference_label.text().startswith("data.s1p: ")

    window.clear_reference_action.trigger()
    assert window.plots.memory is None
    assert window.reference_label.text() == NO_VALUE
    assert not window.clear_reference_action.isEnabled()


def test_load_reference_file_leaves_displayed_data_alone(window, monkeypatch):
    window.open_file(SAMPLE)
    shown = window.measurement
    monkeypatch.setattr(
        "nano_vna_viewer.main_window.QFileDialog.getOpenFileName",
        lambda *args: (str(MATCHED), ""),
    )
    window.load_reference_action.trigger()
    assert window.measurement is shown
    assert window.windowTitle() == f"data.s1p — {APP_NAME}"
    assert window.plots.memory.name == "50.S1P"
    assert window.reference_label.text() == (
        "50.S1P: minimum VSWR 1.336 at 273.6 MHz  (S11 -16.84 dB)"
    )
    assert window.statusBar().currentMessage() == "Loaded 50.S1P as the reference"


def test_reference_can_be_loaded_before_any_data(window):
    assert window.load_reference(MATCHED)
    assert window.measurement is None
    assert window.file_label.text() == NO_VALUE
    assert window.plots.memory.name == "50.S1P"
    assert window.clear_reference_action.isEnabled()


def test_bad_reference_file_keeps_the_current_reference(window, tmp_path):
    window.open_file(SAMPLE)
    window.hold_reference()
    bad = tmp_path / "bad.s1p"
    bad.write_text("not touchstone\n")
    assert not window.load_reference(bad)
    assert len(window.errors) == 1 and "bad.s1p" in window.errors[0]
    assert window.plots.memory is window.measurement
    assert window.reference_label.text().startswith("data.s1p: ")


def test_reference_is_not_exported(window, tmp_path):
    window.open_file(SAMPLE)
    window.load_reference(MATCHED)
    assert window.export_csv(tmp_path / "data.csv")
    lines = (tmp_path / "data.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 102
    assert lines[1].startswith("50000.0,")  # data.s1p's first point, not the reference's


def test_export_actions_need_a_file(window):
    assert not window.export_png_action.isEnabled()
    assert not window.export_csv_action.isEnabled()
    window.open_file(SAMPLE)
    assert window.export_png_action.isEnabled()
    assert window.export_csv_action.isEnabled()


@pytest.mark.parametrize("tab", [0, 1, 2, 3])
def test_export_png(qtbot, window, tmp_path, tab):
    window.show()
    qtbot.waitExposed(window)
    window.open_file(SAMPLE)
    window.plots.tabs.setCurrentIndex(tab)
    out = tmp_path / "plot.png"
    assert window.export_png(out)
    image = QImage(str(out))
    assert not image.isNull()
    expected = window.plots.export_widget().size()
    assert (image.width(), image.height()) == (expected.width(), expected.height())
    assert window.statusBar().currentMessage() == "Saved plot to plot.png"


def test_export_csv(window, tmp_path):
    window.open_file(SAMPLE)
    out = tmp_path / "data.csv"
    assert window.export_csv(out)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("frequency_hz,s11_real,s11_imag")
    assert len(lines) == 102  # header + 101 points
    assert window.statusBar().currentMessage() == "Saved data to data.csv"


def test_export_failure_is_reported(window, tmp_path):
    window.open_file(SAMPLE)
    missing_dir = tmp_path / "no-such-folder"
    assert not window.export_csv(missing_dir / "data.csv")
    assert not window.export_png(missing_dir / "plot.png")
    assert len(window.errors) == 2


def _fake_save_dialog(monkeypatch, returned_name, calls):
    def fake(parent, caption, start, file_filter):
        calls.append(start)
        return returned_name, ""

    monkeypatch.setattr("nano_vna_viewer.main_window.QFileDialog.getSaveFileName", fake)


def test_png_dialog_defaults_and_adds_suffix(window, monkeypatch, tmp_path):
    window.open_file(SAMPLE)
    window.plots.tabs.setCurrentIndex(3)
    calls = []
    _fake_save_dialog(monkeypatch, str(tmp_path / "chart"), calls)
    window.choose_png_export()
    assert calls == [str(SAMPLE.parent / "data_smith_chart.png")]
    assert (tmp_path / "chart.png").is_file()


def test_csv_dialog_defaults_and_keeps_suffix(window, monkeypatch, tmp_path):
    window.open_file(SAMPLE)
    calls = []
    _fake_save_dialog(monkeypatch, str(tmp_path / "out.CSV"), calls)
    window.choose_csv_export()
    assert calls == [str(SAMPLE.parent / "data.csv")]
    assert (tmp_path / "out.CSV").is_file()


def test_export_dialog_cancelled(window, monkeypatch, tmp_path):
    window.open_file(SAMPLE)
    _fake_save_dialog(monkeypatch, "", [])
    window.choose_csv_export()
    window.choose_png_export()
    assert list(tmp_path.iterdir()) == []
    assert window.errors == []


def _live_sweep(qtbot, window, continuous=False):
    panel = window.device_panel
    if panel.device is None:
        panel.connect_button.click()
        qtbot.waitUntil(lambda: panel.device is not None, timeout=5000)
    with qtbot.waitSignal(window.device_panel.measurement_ready, timeout=5000):
        if continuous:
            panel.continuous_button.setChecked(True)
        else:
            panel.sweep_button.click()


def test_live_sweep_is_displayed(qtbot, window):
    _live_sweep(qtbot, window)
    m = window.measurement
    assert m.path is None
    assert window.file_label.text() == f"{m.source} (live sweep)"
    assert window.windowTitle() == f"{m.source} — {APP_NAME}"
    assert window.ports_label.text() == "2"
    assert window.plots.measurement is m
    assert window.plots.tabs.isTabVisible(4)  # Transmission
    assert window.save_touchstone_action.isEnabled()
    qtbot.waitUntil(lambda: window.statusBar().currentMessage().startswith("Sweep complete: 101 points"))


def test_segmented_live_sweep_is_displayed_and_saved(qtbot, window, tmp_path):
    window.device_panel.segments_spin.setValue(3)
    _live_sweep(qtbot, window)
    assert window.measurement.points == 303
    assert window.points_label.text() == "303"
    assert len(window.plots.magnitude_plot.curve.getData()[0]) == 303
    assert window.save_touchstone(tmp_path / "wide.s2p")
    assert load_touchstone(tmp_path / "wide.s2p").points == 303


def test_save_live_sweep_as_touchstone(qtbot, window, tmp_path, monkeypatch):
    _live_sweep(qtbot, window)
    calls = []

    def fake_dialog(parent, caption, start, file_filter):
        calls.append((start, file_filter))
        return str(tmp_path / "capture"), ""

    monkeypatch.setattr("nano_vna_viewer.main_window.QFileDialog.getSaveFileName", fake_dialog)
    window.choose_touchstone_save()
    start, file_filter = calls[0]
    assert Path(start).name.startswith("nanovna_") and start.endswith(".s2p")
    assert "*.s2p" in file_filter
    saved = load_touchstone(tmp_path / "capture.s2p")
    np.testing.assert_allclose(saved.s11, window.measurement.s11, rtol=1e-8)
    np.testing.assert_allclose(saved.s21, window.measurement.s21, rtol=1e-8)


def test_save_loaded_file_as_touchstone(window, tmp_path):
    window.open_file(SAMPLE)
    assert window.save_touchstone(tmp_path / "copy.s1p")
    assert load_touchstone(tmp_path / "copy.s1p").points == 101
    assert window.statusBar().currentMessage() == "Saved sweep to copy.s1p"


def test_continuous_sweeps_keep_zoom_and_marker(qtbot, window):
    _live_sweep(qtbot, window)
    window.plots.set_marker(10)
    view = window.plots.magnitude_plot.getPlotItem().getViewBox()
    view.scaleBy((0.5, 0.5))
    zoomed = view.viewRange()
    first = window.measurement
    with qtbot.waitSignal(window.device_panel.measurement_ready, timeout=5000):
        window.device_panel.continuous_button.setChecked(True)
    qtbot.waitUntil(lambda: window.measurement is not first, timeout=5000)
    window.device_panel.continuous_button.setChecked(False)
    assert window.plots.marker_index == 10
    np.testing.assert_allclose(view.viewRange(), zoomed)


def test_single_sweep_resets_view(qtbot, window):
    _live_sweep(qtbot, window)
    window.plots.set_marker(10)
    qtbot.waitUntil(lambda: window.device_panel.sweep_button.isEnabled())
    _live_sweep(qtbot, window)
    assert window.plots.marker_index == window.plots.min_vswr_index


def test_device_errors_are_shown(qtbot, window):
    panel = window.device_panel
    panel.connect_button.click()
    qtbot.waitUntil(lambda: panel.device is not None, timeout=5000)
    panel.start_spin.setValue(500)
    panel.stop_spin.setValue(100)
    panel.sweep_button.click()
    qtbot.waitUntil(lambda: len(window.errors) == 1, timeout=5000)
    assert "below stop" in window.errors[0]


def test_closing_window_disconnects(qtbot, window, ports):
    panel = window.device_panel
    panel.connect_button.click()
    qtbot.waitUntil(lambda: panel.device is not None, timeout=5000)
    window.close()
    assert not ports.serials[-1].is_open


def test_stopping_continuous_keeps_view(qtbot, window):
    _live_sweep(qtbot, window)
    panel = window.device_panel
    qtbot.waitUntil(lambda: panel.sweep_button.isEnabled())
    window.plots.set_marker(10)
    panel.continuous_button.setChecked(True)
    qtbot.waitUntil(lambda: panel.busy, timeout=5000)
    panel.continuous_button.setChecked(False)  # last sweep finishes after this
    qtbot.waitUntil(lambda: not panel.busy, timeout=5000)
    assert window.plots.marker_index == 10


def test_range_change_in_continuous_starts_fresh(qtbot, window):
    _live_sweep(qtbot, window)
    panel = window.device_panel
    qtbot.waitUntil(lambda: panel.sweep_button.isEnabled())
    window.plots.set_marker(10)
    panel.start_spin.setValue(100)
    panel.stop_spin.setValue(450)
    first = window.measurement
    panel.continuous_button.setChecked(True)
    qtbot.waitUntil(lambda: window.measurement is not first, timeout=5000)
    panel.continuous_button.setChecked(False)
    qtbot.waitUntil(lambda: not panel.busy, timeout=5000)
    assert window.measurement.start_hz == 100e6
    # Marker moved to the new sweep's minimum VSWR rather than keeping point 10.
    assert window.plots.marker_index == window.plots.min_vswr_index
