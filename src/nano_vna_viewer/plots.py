"""Plot widgets: frequency-domain plots, a Smith chart, and the tabbed panel holding them.

All plots share one marker. Clicking a plot or dragging its marker line moves the
marker to the nearest measured point, and every plot plus the readout follows.

Curve smoothing is optional and display-only: the line is drawn through interpolated
values, while the marker and readout stay on the measured points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .rf import impedance, interpolate_s, min_vswr_index, phase_deg, s_db, vswr
from .touchstone import Measurement
from .units import format_frequency, format_impedance

# Normalized values drawn on the Smith chart grid: resistance and reactance for the
# impedance grid, conductance and susceptance for the admittance grid.
SMITH_GRID_VALUES = (0.2, 0.5, 1.0, 2.0, 5.0)


@dataclass(frozen=True)
class Theme:
    background: str
    foreground: str
    s11: str
    s21: str
    marker: str
    grid: tuple[int, int, int, int]
    admittance_grid: tuple[int, int, int, int]
    reference: str  # minimum-VSWR marker


LIGHT = Theme(
    "#ffffff", "#202020", "#1f6fb4", "#d9730d", "#c2185b", (0, 0, 0, 60), (0, 140, 110, 90), "#2e7d32"
)
DARK = Theme(
    "#1e1e1e", "#d0d0d0", "#4ea1ff", "#ffa64d", "#ff4fa3", (255, 255, 255, 60), (90, 210, 170, 80), "#66bb6a"
)


def current_theme() -> Theme:
    """Light or dark, following the system palette Qt picked up."""
    window = QApplication.palette().color(QPalette.ColorRole.Window)
    return DARK if window.lightness() < 128 else LIGHT


def nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


class VswrAxis(pg.AxisItem):
    """Log-scale axis labelled with plain numbers on a 1-2-5 sequence (1, 2, 5, 10, 20 …).

    pyqtgraph's default log labels ("2·10¹", "3·10¹" …) pile up on short decades.
    """

    def logTickStrings(self, values, scale, spacing):
        labels = []
        for v in values:
            value = 10.0**v
            mantissa = round(value / 10.0 ** np.floor(v + 1e-9), 6)
            labels.append(str(int(round(value))) if mantissa in (1, 2, 5) else "")
        return labels


class FrequencyPlot(pg.PlotWidget):
    """One trace against frequency, with a draggable marker line."""

    marker_requested = Signal(int)

    def __init__(
        self, title: str, y_label: str, y_units: str, theme: Theme, log_y: bool = False
    ) -> None:
        axis_items = {"left": VswrAxis(orientation="left")} if log_y else None
        super().__init__(axisItems=axis_items)
        self.theme = theme
        self.frequency = np.empty(0)
        self.curve: pg.PlotDataItem | None = None
        self.points: pg.PlotDataItem | None = None  # measured points, shown under a smoothed curve

        self.setTitle(title)
        self.setLabel("bottom", "Frequency", units="Hz")
        self.setLabel("left", y_label, units=y_units)
        self.getAxis("left").enableAutoSIPrefix(False)  # no "mdB" or "k°"
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setLogMode(y=log_y)

        self.marker_line = pg.InfiniteLine(
            angle=90, movable=True, pen=pg.mkPen(theme.marker, width=2)
        )
        self.marker_line.setVisible(False)
        self.addItem(self.marker_line, ignoreBounds=True)
        self.marker_line.sigDragged.connect(self._on_marker_dragged)
        self.scene().sigMouseClicked.connect(self._on_click)

        self.reference_line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(theme.reference, width=1.5, style=Qt.PenStyle.DashLine),
            label="Min VSWR",
            labelOpts={"position": 0.95, "color": theme.reference, "anchors": [(0, 0), (0, 0)]},
        )
        self.reference_line.setVisible(False)
        self.addItem(self.reference_line, ignoreBounds=True)
        # The marker draws over the reference line, so it stays visible when they coincide
        # (as they do when a file opens).
        self.reference_line.setZValue(10)
        self.marker_line.setZValue(11)

    def set_trace(
        self,
        frequency: np.ndarray,
        y: np.ndarray,
        color: str,
        reset_view: bool = True,
        smooth: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        """Replace the trace. NaN values leave a gap.

        With ``reset_view=False`` the current zoom is kept (for repeated live sweeps).
        ``smooth`` is an interpolated (frequency, y) pair to draw as the line instead;
        the measured points are then shown as dots. The marker follows the measured
        points either way.
        """
        self.clear_trace()
        self.frequency = frequency
        line_x, line_y = (frequency, y) if smooth is None else smooth
        self.curve = self.plot(line_x, line_y, pen=pg.mkPen(color, width=2), connect="finite")
        if smooth is not None:
            finite = np.isfinite(y)
            # A PlotDataItem rather than a ScatterPlotItem, so the log VSWR axis applies.
            self.points = self.plot(
                frequency[finite],
                y[finite],
                pen=None,
                symbol="o",
                symbolSize=5,
                symbolBrush=color,
                symbolPen=None,
            )
        self.marker_line.setVisible(True)
        if reset_view:
            self.reset_view()

    def clear_trace(self) -> None:
        for item in (self.curve, self.points):
            if item is not None:
                self.removeItem(item)
        self.curve = None
        self.points = None
        self.marker_line.setVisible(False)
        self.reference_line.setVisible(False)
        self.frequency = np.empty(0)

    def set_marker(self, index: int) -> None:
        if self.frequency.size:
            self.marker_line.setValue(self.frequency[index])

    def set_reference(self, index: int) -> None:
        """Mark the minimum-VSWR point with a fixed dashed line."""
        if self.frequency.size:
            self.reference_line.setValue(self.frequency[index])
            self.reference_line.setVisible(True)

    def reset_view(self) -> None:
        self.enableAutoRange()
        self.autoRange()

    def _on_marker_dragged(self) -> None:
        if self.frequency.size:
            self.marker_requested.emit(nearest_index(self.frequency, self.marker_line.value()))

    def _on_click(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or not self.frequency.size:
            return
        view = self.getPlotItem().getViewBox()
        if not view.sceneBoundingRect().contains(event.scenePos()):
            return
        x = view.mapSceneToView(event.scenePos()).x()
        self.marker_requested.emit(nearest_index(self.frequency, x))


# Dense near zero, reaching far enough out to close the curves at gamma = 1.
_SWEEP = np.tan(np.linspace(0, np.pi / 2, 400, endpoint=False))
_FULL_SWEEP = np.concatenate([-_SWEEP[::-1], _SWEEP[1:]])


def _to_gamma(z: np.ndarray) -> np.ndarray:
    return (z - 1) / (z + 1)


def smith_outline() -> list[np.ndarray]:
    """The unit circle and the real axis, shared by the impedance and admittance grids."""
    return [_to_gamma(1j * _FULL_SWEEP), np.array([-1, 1], dtype=complex)]


def smith_grid_lines(admittance: bool = False) -> list[np.ndarray]:
    """Grid lines as arrays of complex reflection coefficients.

    Impedance: constant-resistance circles and constant-reactance arcs, traced by
    mapping lines of the normalized impedance plane through gamma = (z - 1) / (z + 1).
    Admittance: gamma = (1 - y) / (1 + y), the same curves rotated 180°, giving
    constant-conductance circles and constant-susceptance arcs.
    """
    lines = [_to_gamma(r + 1j * _FULL_SWEEP) for r in SMITH_GRID_VALUES]
    for x in SMITH_GRID_VALUES:
        for sign in (1, -1):
            lines.append(_to_gamma(_SWEEP + 1j * sign * x))
    return [-line for line in lines] if admittance else lines


class SmithChart(pg.PlotWidget):
    """S11 on a Smith chart, with a marker dot."""

    marker_requested = Signal(int)

    def __init__(self, theme: Theme) -> None:
        super().__init__()
        self.theme = theme
        self.gamma = np.empty(0, dtype=complex)
        self.curve: pg.PlotDataItem | None = None
        self.points: pg.PlotDataItem | None = None  # measured points, shown under a smoothed curve

        self.setTitle("Smith Chart (S11)")
        self.setAspectLocked(True)
        self.hideAxis("left")
        self.hideAxis("bottom")
        self._draw_grid()
        self.set_admittance_visible(False)

        self.marker_dot = pg.ScatterPlotItem(
            size=10, brush=pg.mkBrush(theme.marker), pen=pg.mkPen(theme.background)
        )
        self.marker_dot.setZValue(10)
        self.addItem(self.marker_dot)

        self.reference_dot = pg.ScatterPlotItem(
            size=16, brush=None, pen=pg.mkPen(theme.reference, width=2)
        )
        self.reference_dot.setZValue(9)
        self.addItem(self.reference_dot)
        self.scene().sigMouseClicked.connect(self._on_click)
        self.reset_view()

    def _draw_grid(self) -> None:
        """The impedance grid is always drawn; the admittance grid is an optional overlay."""
        pen = pg.mkPen(self.theme.grid, width=1)
        for line in smith_outline() + smith_grid_lines():
            self.addItem(pg.PlotCurveItem(line.real, line.imag, pen=pen))

        y_color = self.theme.admittance_grid
        y_pen = pg.mkPen(y_color, width=1)
        self.admittance_items = [
            pg.PlotCurveItem(line.real, line.imag, pen=y_pen)
            for line in smith_grid_lines(admittance=True)
        ]

        # Impedance labels sit below the real axis and admittance labels above it,
        # so they don't collide when the overlay is on.
        for value in SMITH_GRID_VALUES:
            gamma = (value - 1) / (value + 1)
            z_label = pg.TextItem(f"{value:g}", color=self.theme.grid[:3] + (160,), anchor=(1, 0))
            z_label.setPos(gamma, 0)
            self.addItem(z_label)
            y_label = pg.TextItem(f"{value:g}", color=y_color[:3] + (200,), anchor=(0, 1))
            y_label.setPos(-gamma, 0)
            self.admittance_items.append(y_label)

        for item in self.admittance_items:
            self.addItem(item)

    def set_admittance_visible(self, visible: bool) -> None:
        """Overlay the admittance grid on the impedance grid, or hide it."""
        self.admittance_visible = visible
        for item in self.admittance_items:
            item.setVisible(visible)

    def set_trace(self, s11: np.ndarray, smooth: np.ndarray | None = None) -> None:
        """Replace the trace.

        ``smooth`` is an interpolated S11 to draw as the line instead; the measured
        points are then shown as dots. The marker follows the measured points either way.
        """
        self.clear_trace()
        self.gamma = s11
        line = s11 if smooth is None else smooth
        self.curve = self.plot(line.real, line.imag, pen=pg.mkPen(self.theme.s11, width=2))
        if smooth is not None:
            self.points = self.plot(
                s11.real,
                s11.imag,
                pen=None,
                symbol="o",
                symbolSize=5,
                symbolBrush=self.theme.s11,
                symbolPen=None,
            )

    def clear_trace(self) -> None:
        for item in (self.curve, self.points):
            if item is not None:
                self.removeItem(item)
        self.curve = None
        self.points = None
        self.marker_dot.clear()
        self.reference_dot.clear()
        self.gamma = np.empty(0, dtype=complex)

    def set_marker(self, index: int) -> None:
        point = self.gamma[index]
        self.marker_dot.setData([point.real], [point.imag])

    def set_reference(self, index: int) -> None:
        """Ring the minimum-VSWR point."""
        point = self.gamma[index]
        self.reference_dot.setData([point.real], [point.imag])

    def reset_view(self) -> None:
        # Auto-range keeps the chart fitted to the widget as it resizes, until the user zooms.
        self.enableAutoRange()
        self.autoRange()

    def _on_click(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or not self.gamma.size:
            return
        view = self.getPlotItem().getViewBox()
        if not view.sceneBoundingRect().contains(event.scenePos()):
            return
        p = view.mapSceneToView(event.scenePos())
        self.marker_requested.emit(int(np.argmin(np.abs(self.gamma - complex(p.x(), p.y())))))


SMITH_TAB = 3
TRANSMISSION_TAB = 4


def format_vswr(value: float) -> str:
    return "∞" if np.isinf(value) else f"{value:.3f}"


def finite_vswr(s11: np.ndarray) -> np.ndarray:
    """VSWR for plotting: infinite values (total reflection) become NaN, a gap in the trace."""
    swr = vswr(s11)
    return np.where(np.isfinite(swr), swr, np.nan)


class PlotPanel(QWidget):
    """Tabbed plots plus a readout line for the shared marker."""

    def __init__(self) -> None:
        super().__init__()
        self.theme = current_theme()
        pg.setConfigOptions(
            antialias=True, background=self.theme.background, foreground=self.theme.foreground
        )
        self.measurement: Measurement | None = None
        self.marker_index: int | None = None
        self.min_vswr_index: int | None = None
        self.smoothing = False

        self.magnitude_plot = FrequencyPlot("Return Loss (S11)", "Magnitude", "dB", self.theme)
        # Log scale: VSWR runs from 1 to thousands near total reflection.
        self.vswr_plot = FrequencyPlot("VSWR (S11)", "VSWR", "", self.theme, log_y=True)
        self.phase_plot = FrequencyPlot("Phase (S11)", "Phase", "°", self.theme)
        self.smith_chart = SmithChart(self.theme)
        self.s21_magnitude_plot = FrequencyPlot("Transmission (S21)", "Magnitude", "dB", self.theme)
        self.s21_phase_plot = FrequencyPlot("Transmission Phase (S21)", "Phase", "°", self.theme)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.magnitude_plot, "Return Loss")
        self.tabs.addTab(self.vswr_plot, "VSWR")
        self.tabs.addTab(self.phase_plot, "Phase")
        self.tabs.addTab(self._build_smith_page(), "Smith Chart")
        self.tabs.addTab(self._build_transmission_page(), "Transmission")
        self.tabs.setTabVisible(TRANSMISSION_TAB, False)  # only for two-port files

        # The plots on each tab, in tab order.
        self.tab_plots = (
            (self.magnitude_plot,),
            (self.vswr_plot,),
            (self.phase_plot,),
            (self.smith_chart,),
            (self.s21_magnitude_plot, self.s21_phase_plot),
        )
        self.plots = tuple(plot for plots in self.tab_plots for plot in plots)

        self.readout = QLabel("No data loaded.")
        self.readout.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs, stretch=1)
        layout.addWidget(self.readout)

        for plot in self.plots:
            plot.marker_requested.connect(self.set_marker)

    def _build_smith_page(self) -> QWidget:
        """The Smith chart with the admittance-overlay toggle above it."""
        controls = QHBoxLayout()
        self.admittance_toggle = QCheckBox("Show admittance grid")
        self.admittance_toggle.toggled.connect(self.smith_chart.set_admittance_visible)
        controls.addWidget(self.admittance_toggle)
        controls.addStretch()

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.addLayout(controls)
        layout.addWidget(self.smith_chart, stretch=1)
        return page

    def _build_transmission_page(self) -> QWidget:
        """S21 magnitude above S21 phase."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.s21_magnitude_plot)
        layout.addWidget(self.s21_phase_plot)
        return page

    def set_measurement(self, m: Measurement, keep_view: bool = False) -> None:
        """Show a measurement.

        Normally zoom is reset and the marker moves to the minimum VSWR. With
        ``keep_view`` (continuous live sweeps) the zoom is kept, and so is the marker
        when the point count hasn't changed.
        """
        previous = self.measurement
        keep_marker = (
            keep_view
            and previous is not None
            and self.marker_index is not None
            and previous.points == m.points
        )
        self.measurement = m
        self._draw_traces(reset=not keep_view)

        if m.s21 is None and self.tabs.currentIndex() == TRANSMISSION_TAB:
            self.tabs.setCurrentIndex(0)
        self.tabs.setTabVisible(TRANSMISSION_TAB, m.s21 is not None)

        self.min_vswr_index = min_vswr_index(m.s11)
        for plot in self.plots:
            plot.set_reference(self.min_vswr_index)
        self.set_marker(self.marker_index if keep_marker else self.min_vswr_index)

    def _draw_traces(self, reset: bool) -> None:
        """Draw every trace of the loaded measurement, smoothed if that is switched on."""
        m, t = self.measurement, self.theme
        f = m.frequency_hz
        dense_s11 = interpolate_s(f, m.s11) if self.smoothing else None

        def draw(plot, s, dense, quantity, color):
            # The quantity is derived from the interpolated S-parameter, not interpolated itself.
            smooth = None if dense is None else (dense[0], quantity(dense[1]))
            plot.set_trace(f, quantity(s), color, reset, smooth)

        draw(self.magnitude_plot, m.s11, dense_s11, s_db, t.s11)
        draw(self.vswr_plot, m.s11, dense_s11, finite_vswr, t.s11)
        draw(self.phase_plot, m.s11, dense_s11, phase_deg, t.s11)
        self.smith_chart.set_trace(m.s11, None if dense_s11 is None else dense_s11[1])

        if m.s21 is not None:
            dense_s21 = interpolate_s(f, m.s21) if self.smoothing else None
            draw(self.s21_magnitude_plot, m.s21, dense_s21, s_db, t.s21)
            draw(self.s21_phase_plot, m.s21, dense_s21, phase_deg, t.s21)
        else:
            self.s21_magnitude_plot.clear_trace()
            self.s21_phase_plot.clear_trace()

    def set_smoothing(self, enabled: bool) -> None:
        """Draw smooth interpolated curves, or straight lines between the measured points.

        Display only: the marker, readout and minimum VSWR always use the measured points.
        Zoom and marker are kept.
        """
        self.smoothing = enabled
        if self.measurement is None:
            return
        self._draw_traces(reset=False)
        for plot in self.plots:
            plot.set_reference(self.min_vswr_index)
        self.set_marker(self.marker_index)

    def go_to_min_vswr(self) -> None:
        if self.min_vswr_index is not None:
            self.set_marker(self.min_vswr_index)

    def set_marker(self, index: int) -> None:
        if self.measurement is None:
            return
        index = int(np.clip(index, 0, self.measurement.points - 1))
        self.marker_index = index
        for plot in self.plots:
            plot.set_marker(index)
        self.readout.setText(self.readout_text())

    def readout_text(self) -> str:
        m, i = self.measurement, self.marker_index
        if m is None or i is None:
            return "No data loaded."
        s11 = m.s11[i : i + 1]
        parts = [
            f"Marker: {format_frequency(m.frequency_hz[i])}",
            f"S11: {s_db(s11)[0]:.2f} dB, {phase_deg(s11)[0]:.1f}°",
            f"VSWR: {format_vswr(vswr(s11)[0])}",
            f"Z: {format_impedance(impedance(s11, m.z0)[0], decimals=2)}",
        ]
        if m.s21 is not None:
            s21 = m.s21[i : i + 1]
            parts.append(f"S21: {s_db(s21)[0]:.2f} dB, {phase_deg(s21)[0]:.1f}°")
        return "    ".join(parts)

    def current_tab_name(self) -> str:
        return self.tabs.tabText(self.tabs.currentIndex())

    def export_widget(self) -> QWidget:
        """What "Export Plot" captures: the visible tab's plots, without the grid toggle."""
        if self.tabs.currentIndex() == SMITH_TAB:
            return self.smith_chart
        return self.tabs.currentWidget()

    def reset_view(self) -> None:
        """Reset zoom and pan on the plots in the visible tab."""
        for plot in self.tab_plots[self.tabs.currentIndex()]:
            plot.reset_view()
