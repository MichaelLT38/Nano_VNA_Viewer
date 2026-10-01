"""The NanoVNA control panel: pick a port, connect, and run single or continuous sweeps.

Serial I/O runs on a background thread (DeviceWorker), so a slow sweep never freezes
the window. The panel talks to the worker only through queued signals.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import QMetaObject, QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
)

from .nanovna import ConnectionLost, NanoVNA, NanoVNAError, PortInfo, SweepSettings, find_ports

MHZ = 1e6


@dataclass(frozen=True)
class DeviceInfo:
    port: str
    board: str
    version: str
    max_points: int
    sweep: SweepSettings


class DeviceWorker(QObject):
    """Owns the NanoVNA connection. Lives on a background thread."""

    connected = Signal(object)  # DeviceInfo
    swept = Signal(object, float)  # Measurement, seconds taken
    failed = Signal(str)
    disconnected = Signal()

    def __init__(self, device_factory: Callable[[str], NanoVNA]) -> None:
        super().__init__()
        self._factory = device_factory
        self._device: NanoVNA | None = None

    @Slot(str)
    def open_port(self, port: str) -> None:
        self._close_device()
        try:
            device = self._factory(port)
            settings = device.get_sweep()
        except NanoVNAError as exc:
            self.failed.emit(str(exc))
            self.disconnected.emit()
            return
        self._device = device
        self.connected.emit(
            DeviceInfo(port, device.board, device.version, device.max_points, settings)
        )

    @Slot(int, int, int)
    def run_sweep(self, start_hz: int, stop_hz: int, points: int) -> None:
        if self._device is None:
            self.failed.emit("Not connected to a NanoVNA.")
            return
        started = time.monotonic()
        try:
            measurement = self._device.scan(start_hz, stop_hz, points)
        except ConnectionLost as exc:
            self._close_device()
            self.failed.emit(str(exc))
            self.disconnected.emit()
            return
        except NanoVNAError as exc:
            self.failed.emit(str(exc))
            return
        self.swept.emit(measurement, time.monotonic() - started)

    @Slot()
    def close_port(self) -> None:
        self._close_device()
        self.disconnected.emit()

    def _close_device(self) -> None:
        if self._device is not None:
            self._device.close()
            self._device = None


class DevicePanel(QGroupBox):
    """Port selection, sweep range, and sweep buttons."""

    measurement_ready = Signal(object, bool)  # Measurement, part of a continuous run
    error = Signal(str)
    status = Signal(str)

    # Requests to the worker (queued across threads).
    _open_requested = Signal(str)
    _sweep_requested = Signal(int, int, int)
    _close_requested = Signal()

    def __init__(
        self,
        device_factory: Callable[[str], NanoVNA] = NanoVNA,
        port_lister: Callable[[], list[PortInfo]] = find_ports,
    ) -> None:
        super().__init__("NanoVNA")
        self._port_lister = port_lister
        self.device: DeviceInfo | None = None
        self.busy = False  # a connect or sweep is in progress
        # Whether the sweep in flight was started by continuous mode. Recorded at the
        # start, so the last sweep after pressing Stop still counts as continuous.
        self._sweep_is_continuous = False

        self._build_widgets()

        self._thread = QThread(self)
        self.worker = DeviceWorker(device_factory)
        self.worker.moveToThread(self._thread)
        self._open_requested.connect(self.worker.open_port)
        self._sweep_requested.connect(self.worker.run_sweep)
        self._close_requested.connect(self.worker.close_port)
        self.worker.connected.connect(self._on_connected)
        self.worker.swept.connect(self._on_swept)
        self.worker.failed.connect(self._on_failed)
        self.worker.disconnected.connect(self._on_disconnected)
        self._thread.start()

        self.refresh_ports()
        self._update_controls()

    def _build_widgets(self) -> None:
        self.port_combo = QComboBox()
        self.port_combo.setMinimumContentsLength(18)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_ports)
        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self._toggle_connection)
        self.status_label = QLabel("Not connected")

        self.start_spin = self._mhz_spin()
        self.stop_spin = self._mhz_spin()
        # The NanoVNA's usual full range until a device reports its own.
        self.start_spin.setValue(0.05)
        self.stop_spin.setValue(900)
        self.points_spin = QSpinBox()
        self.points_spin.setRange(2, 101)
        self.points_spin.setValue(101)
        self.sweep_button = QPushButton("Single Sweep")
        self.sweep_button.clicked.connect(self.single_sweep)
        self.continuous_button = QPushButton("Continuous")
        self.continuous_button.setCheckable(True)
        self.continuous_button.toggled.connect(self._on_continuous_toggled)

        top = QHBoxLayout()
        top.addWidget(QLabel("Port:"))
        top.addWidget(self.port_combo)
        top.addWidget(self.refresh_button)
        top.addWidget(self.connect_button)
        top.addWidget(self.status_label, stretch=1)

        bottom = QHBoxLayout()
        for label, widget in (
            ("Start:", self.start_spin),
            ("Stop:", self.stop_spin),
            ("Points:", self.points_spin),
        ):
            bottom.addWidget(QLabel(label))
            bottom.addWidget(widget)
        bottom.addWidget(self.sweep_button)
        bottom.addWidget(self.continuous_button)
        bottom.addStretch()

        layout = QGridLayout(self)
        layout.addLayout(top, 0, 0)
        layout.addLayout(bottom, 1, 0)

    @staticmethod
    def _mhz_spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(3)  # 1 kHz steps
        spin.setRange(0.01, 6000)
        spin.setSuffix(" MHz")
        spin.setKeyboardTracking(False)
        return spin

    # -- ports and connection ---------------------------------------------------------

    def refresh_ports(self) -> None:
        current = self.port_combo.currentData()
        self.port_combo.clear()
        for port in self._port_lister():
            self.port_combo.addItem(port.label, port.device)
        index = self.port_combo.findData(current)
        if index >= 0:
            self.port_combo.setCurrentIndex(index)
        self._update_controls()

    def _toggle_connection(self) -> None:
        if self.device is not None:
            self.continuous_button.setChecked(False)
            self._close_requested.emit()
        elif self.port_combo.currentData():
            self.busy = True
            self.status_label.setText(f"Connecting to {self.port_combo.currentData()}…")
            self._update_controls()
            self._open_requested.emit(self.port_combo.currentData())

    @Slot(object)
    def _on_connected(self, info: DeviceInfo) -> None:
        self.busy = False
        self.device = info
        self.points_spin.setMaximum(info.max_points)
        self.start_spin.setValue(info.sweep.start_hz / MHZ)
        self.stop_spin.setValue(info.sweep.stop_hz / MHZ)
        self.points_spin.setValue(min(info.sweep.points, info.max_points))
        self.status_label.setText(self._connected_text(info))
        self.status.emit(
            f"Connected to {info.board} on {info.port}. Its own sweep pauses while the viewer "
            "sweeps and resumes when you disconnect."
        )
        self._update_controls()

    @staticmethod
    def _connected_text(info: DeviceInfo, paused: bool = False) -> str:
        text = f"Connected: {info.board}, firmware {info.version}"
        return text + " — NanoVNA screen paused while connected" if paused else text

    @Slot()
    def _on_disconnected(self) -> None:
        was_connected = self.device is not None
        self.busy = False
        self.device = None
        self.continuous_button.setChecked(False)
        self.status_label.setText("Not connected")
        if was_connected:
            self.status.emit("Disconnected from the NanoVNA")
        self._update_controls()

    # -- sweeps -----------------------------------------------------------------------

    def sweep_settings(self) -> tuple[int, int, int]:
        return (
            round(self.start_spin.value() * MHZ),
            round(self.stop_spin.value() * MHZ),
            self.points_spin.value(),
        )

    def single_sweep(self) -> None:
        if self.device is not None and not self.busy:
            self._start_sweep(continuous=False)

    def _start_sweep(self, continuous: bool) -> None:
        self.busy = True
        self._sweep_is_continuous = continuous
        self._update_controls()
        self._sweep_requested.emit(*self.sweep_settings())

    def _on_continuous_toggled(self, on: bool) -> None:
        if on and self.device is not None and not self.busy:
            self._start_sweep(continuous=True)
        self._update_controls()

    @Slot(object, float)
    def _on_swept(self, measurement, seconds: float) -> None:
        self.busy = False
        if self.device is not None:
            # The device's own sweep stays paused until disconnect (see nanovna.py).
            self.status_label.setText(self._connected_text(self.device, paused=True))
        self.status.emit(f"Sweep complete: {measurement.points} points in {seconds:.2f} s")
        self.measurement_ready.emit(measurement, self._sweep_is_continuous)
        if self.continuous_button.isChecked() and self.device is not None:
            self._start_sweep(continuous=True)
        else:
            self._update_controls()

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        self.busy = False
        self.continuous_button.setChecked(False)
        self.error.emit(message)
        self._update_controls()

    def _update_controls(self) -> None:
        connected = self.device is not None
        self.port_combo.setEnabled(not connected and not self.busy)
        self.refresh_button.setEnabled(not connected and not self.busy)
        self.connect_button.setText("Disconnect" if connected else "Connect")
        self.connect_button.setEnabled(
            connected or (not self.busy and self.port_combo.count() > 0)
        )
        for widget in (self.start_spin, self.stop_spin, self.points_spin):
            widget.setEnabled(connected and not self.continuous_button.isChecked())
        self.sweep_button.setEnabled(
            connected and not self.busy and not self.continuous_button.isChecked()
        )
        self.continuous_button.setEnabled(connected)

    def shutdown(self) -> None:
        """Close the connection and stop the worker thread (call when the window closes)."""
        self.continuous_button.setChecked(False)
        if self._thread.isRunning():
            QMetaObject.invokeMethod(
                self.worker, "close_port", Qt.ConnectionType.BlockingQueuedConnection
            )
            self._thread.quit()
            self._thread.wait(5000)
