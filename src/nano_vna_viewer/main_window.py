"""The application's main window."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from . import __version__
from .device_panel import DevicePanel
from .export import write_csv
from .plots import PlotPanel, format_vswr
from .rf import BAND_VSWR_LIMIT, min_vswr_index, s_db, vswr
from .touchstone import (
    Measurement,
    TouchstoneError,
    load_touchstone,
    touchstone_suffix,
    write_touchstone,
)
from .units import format_frequency, format_impedance

APP_NAME = "Nano VNA Viewer"
LAST_DIR_KEY = "last_open_dir"
SMOOTH_KEY = "smooth_curves"
# Linux file dialogs match patterns case-sensitively, so list both cases.
FILE_FILTER = "Touchstone files (*.s1p *.s2p *.S1P *.S2P);;All files (*)"
PNG_FILTER = "PNG images (*.png *.PNG)"
CSV_FILTER = "CSV files (*.csv *.CSV)"
NO_VALUE = "—"


def best_match_text(m: Measurement) -> str:
    """The minimum VSWR and where it is, e.g. '1.336 at 274.05 MHz  (S11 -16.84 dB)'."""
    i = min_vswr_index(m.s11)
    best = m.s11[i : i + 1]
    return (
        f"{format_vswr(vswr(best)[0])} at {format_frequency(m.frequency_hz[i])}"
        f"  (S11 {s_db(best)[0]:.2f} dB)"
    )


def band_text(m: Measurement, band: tuple[float | None, float | None] | None) -> str:
    """Describe the matched band from ``rf.vswr_band``; an edge outside the sweep is None."""
    if band is None:
        return "none"
    low, high = band
    low_text = f"below {format_frequency(m.start_hz)}" if low is None else format_frequency(low)
    high_text = f"above {format_frequency(m.stop_hz)}" if high is None else format_frequency(high)
    width = "" if low is None or high is None else f"  ({format_frequency(high - low)} wide)"
    return f"{low_text} to {high_text}{width}"


class MainWindow(QMainWindow):
    def __init__(
        self, settings: QSettings | None = None, device_panel: DevicePanel | None = None
    ) -> None:
        super().__init__()
        self.settings = settings if settings is not None else QSettings()
        self.measurement: Measurement | None = None
        self.device_panel = device_panel if device_panel is not None else DevicePanel()

        self.setWindowTitle(APP_NAME)
        self.resize(1000, 800)
        self._build_menus()
        self._build_central_widget()
        self.smooth_action.setChecked(self.settings.value(SMOOTH_KEY, False, type=bool))
        self.device_panel.measurement_ready.connect(self.show_live_measurement)
        self.device_panel.error.connect(lambda message: self.show_error(message, title="NanoVNA"))
        self.device_panel.status.connect(self.statusBar().showMessage)
        self.statusBar().showMessage("Open a .s1p or .s2p file (File → Open), or connect a NanoVNA.")

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        self.open_action = QAction("&Open…", self)
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_action.triggered.connect(self.choose_file)
        file_menu.addAction(self.open_action)

        self.save_touchstone_action = QAction("&Save Sweep as Touchstone…", self)
        self.save_touchstone_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_touchstone_action.triggered.connect(self.choose_touchstone_save)
        file_menu.addAction(self.save_touchstone_action)

        file_menu.addSeparator()

        self.export_png_action = QAction("Export &Plot as PNG…", self)
        self.export_png_action.setShortcut("Ctrl+E")
        self.export_png_action.triggered.connect(self.choose_png_export)
        file_menu.addAction(self.export_png_action)

        self.export_csv_action = QAction("Export &Data as CSV…", self)
        self.export_csv_action.setShortcut("Ctrl+Shift+E")
        self.export_csv_action.triggered.connect(self.choose_csv_export)
        file_menu.addAction(self.export_csv_action)

        file_menu.addSeparator()

        self.exit_action = QAction("E&xit", self)
        self.exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        self.exit_action.triggered.connect(self.close)
        file_menu.addAction(self.exit_action)

        view_menu = self.menuBar().addMenu("&View")
        self.reset_view_action = QAction("&Reset Zoom", self)
        self.reset_view_action.setShortcut("Ctrl+0")
        self.reset_view_action.triggered.connect(lambda: self.plots.reset_view())
        view_menu.addAction(self.reset_view_action)

        self.min_vswr_action = QAction("Go to &Minimum VSWR", self)
        self.min_vswr_action.setShortcut("Ctrl+M")
        self.min_vswr_action.triggered.connect(lambda: self.plots.go_to_min_vswr())
        view_menu.addAction(self.min_vswr_action)

        view_menu.addSeparator()

        # Display only: markers, readouts and exports keep using the measured points.
        self.smooth_action = QAction("&Smooth Curves", self)
        self.smooth_action.setCheckable(True)
        self.smooth_action.toggled.connect(self.set_smoothing)
        view_menu.addAction(self.smooth_action)

        view_menu.addSeparator()

        # The reference is a second, grey trace kept on screen to compare against.
        self.hold_reference_action = QAction("&Hold Trace as Reference", self)
        self.hold_reference_action.setShortcut("Ctrl+R")
        self.hold_reference_action.triggered.connect(self.hold_reference)
        view_menu.addAction(self.hold_reference_action)

        self.load_reference_action = QAction("&Load Reference File…", self)
        self.load_reference_action.triggered.connect(self.choose_reference_file)
        view_menu.addAction(self.load_reference_action)

        self.clear_reference_action = QAction("&Clear Reference", self)
        self.clear_reference_action.triggered.connect(lambda: self.set_reference(None))
        self.clear_reference_action.setEnabled(False)  # until there is one
        view_menu.addAction(self.clear_reference_action)

        help_menu = self.menuBar().addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

        for action in self._data_actions():
            action.setEnabled(False)  # until data is loaded

    def _build_central_widget(self) -> None:
        self.file_label = QLabel(NO_VALUE)
        self.ports_label = QLabel(NO_VALUE)
        self.range_label = QLabel(NO_VALUE)
        self.points_label = QLabel(NO_VALUE)
        self.z0_label = QLabel(NO_VALUE)
        self.min_vswr_label = QLabel(NO_VALUE)
        self.band_label = QLabel(NO_VALUE)
        self.reference_label = QLabel(NO_VALUE)
        for label in (
            self.file_label,
            self.ports_label,
            self.range_label,
            self.points_label,
            self.z0_label,
            self.min_vswr_label,
            self.band_label,
            self.reference_label,
        ):
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        info = QFormLayout()
        info.addRow("Source:", self.file_label)
        info.addRow("Ports:", self.ports_label)
        info.addRow("Frequency range:", self.range_label)
        info.addRow("Points:", self.points_label)
        info.addRow("Reference impedance:", self.z0_label)
        info.addRow("Minimum VSWR:", self.min_vswr_label)
        info.addRow(f"VSWR ≤ {BAND_VSWR_LIMIT:g} band:", self.band_label)
        info.addRow("Reference trace:", self.reference_label)

        self.plots = PlotPanel()

        layout = QVBoxLayout()
        layout.addWidget(self.device_panel)
        layout.addLayout(info)
        layout.addWidget(self.plots, stretch=1)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

    def start_dir(self) -> str:
        """Folder the Open dialog starts in: the last one used, if it still exists."""
        last = self.settings.value(LAST_DIR_KEY, "", type=str)
        if last and Path(last).is_dir():
            return last
        return str(Path.home())

    def choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Touchstone File", self.start_dir(), FILE_FILTER
        )
        if path:
            self.open_file(path)

    def open_file(self, path: str | Path) -> bool:
        """Load a file and display it. Returns True on success."""
        try:
            measurement = load_touchstone(path)
        except TouchstoneError as exc:
            self.statusBar().showMessage("Failed to open file.")
            self.show_error(str(exc))
            return False

        self.measurement = measurement
        self.settings.setValue(LAST_DIR_KEY, str(measurement.path.parent))
        self._show_measurement(measurement)
        self.statusBar().showMessage(f"Loaded {measurement.name}")
        return True

    def show_live_measurement(self, m: Measurement, continuous: bool = False) -> None:
        """Display a sweep captured from the NanoVNA.

        During a continuous run the zoom and marker carry over from the previous sweep,
        as long as the frequency points are the same (a new range starts fresh).
        """
        previous = self.measurement
        same_sweep = (
            previous is not None
            and previous.path is None
            and np.array_equal(previous.frequency_hz, m.frequency_hz)
        )
        self.measurement = m
        self._show_measurement(m, keep_view=continuous and same_sweep)

    def _show_measurement(self, m: Measurement, keep_view: bool = False) -> None:
        self.setWindowTitle(f"{m.name} — {APP_NAME}")
        self.file_label.setText(str(m.path) if m.path is not None else f"{m.source} (live sweep)")
        self.ports_label.setText(str(m.nports))
        self.range_label.setText(
            f"{format_frequency(m.start_hz)} to {format_frequency(m.stop_hz)}"
        )
        self.points_label.setText(str(m.points))
        self.z0_label.setText(format_impedance(m.z0))
        self.plots.set_measurement(m, keep_view=keep_view)

        self.min_vswr_label.setText(best_match_text(m))
        self.band_label.setText(band_text(m, self.plots.vswr_band))
        for action in self._data_actions():
            action.setEnabled(True)

    def hold_reference(self) -> None:
        """Keep the trace now on screen as the reference."""
        if self.measurement is not None:
            self.set_reference(self.measurement)
            self.statusBar().showMessage(f"Holding {self.measurement.name} as the reference")

    def choose_reference_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Reference File", self.start_dir(), FILE_FILTER
        )
        if path:
            self.load_reference(path)

    def load_reference(self, path: str | Path) -> bool:
        """Load a file as the reference, leaving the displayed data as it is. True on success."""
        try:
            reference = load_touchstone(path)
        except TouchstoneError as exc:
            self.show_error(str(exc), title="Could not load reference")
            return False
        self.set_reference(reference)
        self.statusBar().showMessage(f"Loaded {reference.name} as the reference")
        return True

    def set_reference(self, m: Measurement | None) -> None:
        """Show a measurement as the grey comparison trace, or remove it with None."""
        self.plots.set_memory(m)
        self.reference_label.setText(
            NO_VALUE if m is None else f"{m.name}: minimum VSWR {best_match_text(m)}"
        )
        self.clear_reference_action.setEnabled(m is not None)

    def set_smoothing(self, enabled: bool) -> None:
        """Switch curve smoothing on or off, and remember the choice."""
        self.plots.set_smoothing(enabled)
        self.settings.setValue(SMOOTH_KEY, enabled)

    def _data_actions(self) -> tuple[QAction, ...]:
        """Actions that need loaded data."""
        return (
            self.save_touchstone_action,
            self.export_png_action,
            self.export_csv_action,
            self.hold_reference_action,
        )

    def _default_stem(self) -> str:
        """Base file name for saving: the loaded file's name, or a timestamp for live data."""
        if self.measurement.path is not None:
            return self.measurement.path.stem
        return f"nanovna_{datetime.now():%Y%m%d_%H%M%S}"

    def _save_path(self, caption: str, default_name: str, file_filter: str, suffix: str) -> Path | None:
        """Ask where to save, starting next to the loaded file. Adds the suffix if missing."""
        folder = (
            self.measurement.path.parent if self.measurement.path is not None else Path(self.start_dir())
        )
        start = str(folder / default_name)
        path, _ = QFileDialog.getSaveFileName(self, caption, start, file_filter)
        if not path:
            return None
        path = Path(path)
        return path if path.suffix.lower() == suffix else path.with_name(path.name + suffix)

    def choose_png_export(self) -> None:
        if self.measurement is None:
            return
        tab = self.plots.current_tab_name().lower().replace(" ", "_")
        path = self._save_path(
            "Export Plot as PNG", f"{self._default_stem()}_{tab}.png", PNG_FILTER, ".png"
        )
        if path:
            self.export_png(path)

    def export_png(self, path: str | Path) -> bool:
        """Save the visible tab's plots as a PNG image. Returns True on success."""
        if not self.plots.export_widget().grab().save(str(path), "PNG"):
            self.show_error(f"Could not write '{path}'.", title="Export failed")
            return False
        self.statusBar().showMessage(f"Saved plot to {Path(path).name}")
        return True

    def choose_csv_export(self) -> None:
        if self.measurement is None:
            return
        path = self._save_path(
            "Export Data as CSV", f"{self._default_stem()}.csv", CSV_FILTER, ".csv"
        )
        if path:
            self.export_csv(path)

    def export_csv(self, path: str | Path) -> bool:
        """Save the loaded data as CSV. Returns True on success."""
        try:
            write_csv(self.measurement, path)
        except OSError as exc:
            self.show_error(f"Could not write '{path}': {exc.strerror or exc}", title="Export failed")
            return False
        self.statusBar().showMessage(f"Saved data to {Path(path).name}")
        return True

    def choose_touchstone_save(self) -> None:
        if self.measurement is None:
            return
        suffix = touchstone_suffix(self.measurement)
        upper = suffix.upper()
        path = self._save_path(
            "Save Sweep as Touchstone",
            f"{self._default_stem()}{suffix}",
            f"Touchstone {self.measurement.nports}-port (*{suffix} *{upper})",
            suffix,
        )
        if path:
            self.save_touchstone(path)

    def save_touchstone(self, path: str | Path) -> bool:
        """Save the current data as a Touchstone file. Returns True on success."""
        try:
            write_touchstone(self.measurement, path)
        except OSError as exc:
            self.show_error(f"Could not write '{path}': {exc.strerror or exc}", title="Save failed")
            return False
        self.statusBar().showMessage(f"Saved sweep to {Path(path).name}")
        return True

    def show_error(self, message: str, title: str = "Could not open file") -> None:
        QMessageBox.warning(self, title, message)

    def closeEvent(self, event) -> None:
        self.device_panel.shutdown()
        super().closeEvent(event)

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            f"About {APP_NAME}",
            f"<b>{APP_NAME}</b> {__version__}<br>"
            "A cross-platform viewer for NanoVNA data, from files or a live device.<br><br>"
            "MIT License. Copyright © 2026 bitswizzler.io",
        )
