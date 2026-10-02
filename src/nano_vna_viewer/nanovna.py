"""Talking to a NanoVNA over its USB serial command shell.

The shell is line based: send ``command\\r``; the device echoes the command, prints
its output, and finishes with a ``ch> `` prompt.

Only these commands are used: version, info, bandwidth, sweep, pause, scan, resume.
Nothing here writes to the device's flash (no save, saveconfig, clearconfig, cal or reset).

Firmware hang (NanoVNA-H, firmware 1.2.43): restoring the device's own sweep and
resuming it after every scan, then sending the next command ~0.2 s later, hung the
device within a few minutes when the point count changed between sweeps (it locked
up on the first command after "resume"). So the device is paused once, only "scan"
is sent per sweep, and its own sweep is restored and resumed just once, on close.
That pattern ran 500 varied sweeps without a hang. A segmented sweep follows it too:
it is just several scans in a row.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import serial
from serial.tools import list_ports

from .touchstone import Measurement, measurement_from_sweep

# USB vendor:product of the STM32 virtual COM port used by NanoVNA, -H and -H4.
NANOVNA_USB_IDS = {(0x0483, 0x5740)}

BAUD_RATE = 115200  # ignored by USB CDC devices, but required by pyserial
PROMPT = b"ch> "
COMMAND_TIMEOUT = 3.0
WRITE_TIMEOUT = 2.0  # a hung NanoVNA stops accepting data; never block forever
SCAN_TIMEOUT = 60.0  # used when the IF bandwidth is unknown
# Scan time limit = SCAN_TIMEOUT_BASE + points * SCAN_SECONDS_PER_POINT_HZ / bandwidth.
# Measured: 101 points at 1 kHz take ~0.45 s (~4.5 ms/point); this allows ~4x that.
SCAN_TIMEOUT_BASE = 5.0
SCAN_SECONDS_PER_POINT_HZ = 20.0
# After restoring the device's sweep and resuming, give the firmware time to settle
# before the port can be reused (the hang happened 0.2 s after "resume").
RESUME_SETTLE = 1.0
HANG_ADVICE = "The NanoVNA stopped responding. Switch it off and on again, then reconnect."
DEFAULT_MAX_POINTS = 101
# The "scan" output mask: 1 = frequency, 2 = S11, 4 = S21.
SCAN_MASK = 1 | 2 | 4


class NanoVNAError(Exception):
    """Raised when the NanoVNA can't be reached or rejects a command."""


class ConnectionLost(NanoVNAError):
    """The serial link failed or the device stopped answering; the connection is unusable."""


@dataclass(frozen=True)
class PortInfo:
    device: str
    description: str
    is_nanovna: bool

    @property
    def label(self) -> str:
        kind = "NanoVNA" if self.is_nanovna else self.description
        return f"{self.device} — {kind}"


def find_ports() -> list[PortInfo]:
    """All serial ports, NanoVNA-looking ones first."""
    ports = [
        PortInfo(p.device, p.description or "", (p.vid, p.pid) in NANOVNA_USB_IDS)
        for p in list_ports.comports()
    ]
    return sorted(ports, key=lambda p: (not p.is_nanovna, p.device))


@dataclass(frozen=True)
class SweepSettings:
    start_hz: int
    stop_hz: int
    points: int


class NanoVNA:
    """An open connection to a NanoVNA.

    The device's own sweep is paused at the first scan and restored on close().
    """

    def __init__(self, port: str, serial_factory=serial.Serial) -> None:
        self.port = port
        self._lost = False  # connection unusable: don't try to talk to the device again
        self._paused = False  # we paused the device; restore its sweep on close
        try:
            self._serial = serial_factory(
                port, BAUD_RATE, timeout=0.05, write_timeout=WRITE_TIMEOUT
            )
        except (serial.SerialException, OSError) as exc:
            raise NanoVNAError(f"Could not open {port}: {exc}") from exc
        try:
            self._sync()
            self.version = " ".join(self.command("version"))
            self.info = self.command("info")
            self.original_sweep = self.get_sweep()
            self.bandwidth_hz = self._read_bandwidth()
        except NanoVNAError:
            self.close()
            raise
        self.max_points = _max_points(self.info)

    @property
    def board(self) -> str:
        for line in self.info:
            if line.startswith("Board:"):
                return line.split(":", 1)[1].strip()
        return "NanoVNA"

    def close(self) -> None:
        """Give the device its own sweep back (if we paused it), then close the port."""
        if self._paused and not self._lost:
            try:
                self.set_sweep(self.original_sweep)
                self.command("resume")
                self._paused = False
                time.sleep(RESUME_SETTLE)
            except NanoVNAError:
                pass  # closing anyway
        self._serial.close()

    def __enter__(self) -> NanoVNA:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- protocol ---------------------------------------------------------------------

    def _sync(self) -> None:
        """Get a fresh prompt, discarding anything left over from earlier sessions."""
        self._serial.reset_input_buffer()
        self._write(b"\r")
        try:
            self._read_until_prompt(COMMAND_TIMEOUT)
        except ConnectionLost as exc:
            raise ConnectionLost(f"No response from {self.port}. Is it a NanoVNA?") from exc

    def _connection_lost(self, message: str) -> ConnectionLost:
        self._lost = True
        return ConnectionLost(message)

    def _write(self, data: bytes) -> None:
        try:
            self._serial.write(data)
        except serial.SerialTimeoutException as exc:
            raise self._connection_lost(HANG_ADVICE) from exc
        except (serial.SerialException, OSError) as exc:
            raise self._connection_lost(f"Lost connection to {self.port}: {exc}") from exc

    def _read_until_prompt(self, timeout: float) -> bytes:
        deadline = time.monotonic() + timeout
        buf = bytearray()
        while not buf.endswith(PROMPT):
            if time.monotonic() > deadline:
                raise self._connection_lost(HANG_ADVICE)
            try:
                buf += self._serial.read(self._serial.in_waiting or 1)
            except (serial.SerialException, OSError) as exc:
                raise self._connection_lost(f"Lost connection to {self.port}: {exc}") from exc
        return bytes(buf)

    def command(self, text: str, timeout: float = COMMAND_TIMEOUT) -> list[str]:
        """Run a shell command and return its output lines (without the echo and prompt)."""
        if self._lost:
            raise ConnectionLost(HANG_ADVICE)
        self._write(text.encode("ascii") + b"\r")
        reply = self._read_until_prompt(timeout)[: -len(PROMPT)]
        lines = [line.rstrip() for line in reply.decode("ascii", "replace").split("\n")]
        lines = [line.replace("\r", "") for line in lines]
        if lines and lines[0] == text:
            lines = lines[1:]
        lines = [line for line in lines if line]
        name = text.split()[0]
        if lines and (lines[0].startswith("usage:") or lines[0] == f"{name}?"):
            raise NanoVNAError(f"The NanoVNA rejected '{text}': {lines[0]}")
        return lines

    def _read_bandwidth(self) -> float | None:
        """IF bandwidth in Hz, from e.g. "bandwidth 3 (1000Hz)"; None if unavailable."""
        try:
            lines = self.command("bandwidth")
        except ConnectionLost:
            raise
        except NanoVNAError:
            return None  # firmware without the command
        match = re.search(r"\(([\d.]+)\s*Hz\)", " ".join(lines))
        return float(match.group(1)) if match else None

    def scan_timeout(self, points: int) -> float:
        if not self.bandwidth_hz:
            return SCAN_TIMEOUT
        return SCAN_TIMEOUT_BASE + points * SCAN_SECONDS_PER_POINT_HZ / self.bandwidth_hz

    # -- sweeps -----------------------------------------------------------------------

    def get_sweep(self) -> SweepSettings:
        lines = self.command("sweep")
        try:
            start, stop, points = (int(v) for v in lines[0].split()[:3])
        except (IndexError, ValueError) as exc:
            raise NanoVNAError(f"Unexpected reply to 'sweep': {lines}") from exc
        return SweepSettings(start, stop, points)

    def set_sweep(self, settings: SweepSettings) -> None:
        self.command(f"sweep {settings.start_hz} {settings.stop_hz} {settings.points}")

    def scan(self, start_hz: int, stop_hz: int, points: int) -> Measurement:
        """Measure S11 and S21 over the given range.

        Pauses the device's own sweep on first use; see the module notes on why it
        isn't restored after every scan.
        """
        self._check_range(start_hz, stop_hz, points)
        return self._measurement(self._scan_data(start_hz, stop_hz, points))

    def scan_segments(
        self,
        start_hz: int,
        stop_hz: int,
        points: int,
        segments: int,
        on_segment: Callable[[int, int], None] | None = None,
    ) -> Measurement:
        """Measure the range in several back-to-back scans of ``points`` each.

        This gives more measured points than one scan allows. One even grid of
        ``points * segments`` frequencies is laid over the range and each scan covers
        the next ``points`` of it, so the segments don't overlap. ``on_segment(done,
        total)`` is called after each scan.
        """
        if segments == 1:
            return self.scan(start_hz, stop_hz, points)
        self._check_range(start_hz, stop_hz, points)
        if segments < 1:
            raise NanoVNAError("Segments must be at least 1.")
        total = points * segments
        if stop_hz - start_hz < total - 1:
            raise NanoVNAError(
                f"The frequency range is too narrow for {total} points (it needs 1 Hz per point)."
            )

        step = (stop_hz - start_hz) / (total - 1)
        parts = []
        for segment in range(segments):
            first = round(start_hz + segment * points * step)
            last = round(start_hz + ((segment + 1) * points - 1) * step)
            parts.append(self._scan_data(first, last, points))
            if on_segment is not None:
                on_segment(segment + 1, segments)
        data = np.concatenate(parts)
        if not np.all(np.diff(data[:, 0]) > 0):
            raise NanoVNAError("The NanoVNA returned segments whose frequencies overlap.")
        return self._measurement(data)

    def _check_range(self, start_hz: int, stop_hz: int, points: int) -> None:
        if not 0 < start_hz < stop_hz:
            raise NanoVNAError("Start frequency must be above zero and below stop frequency.")
        if not 2 <= points <= self.max_points:
            raise NanoVNAError(f"Points must be between 2 and {self.max_points}.")

    def _scan_data(self, start_hz: int, stop_hz: int, points: int) -> np.ndarray:
        """One scan, as rows of frequency, S11 real, S11 imaginary, S21 real, S21 imaginary."""
        if not self._paused:
            self.command("pause")
            self._paused = True
        lines = self.command(
            f"scan {start_hz} {stop_hz} {points} {SCAN_MASK}", timeout=self.scan_timeout(points)
        )

        try:
            data = np.array([[float(v) for v in line.split()] for line in lines])
        except ValueError as exc:
            raise NanoVNAError(f"Unexpected scan data from the NanoVNA: {exc}") from exc
        if data.shape != (points, 5):
            raise NanoVNAError(
                f"Expected {points} points of scan data, got {len(lines)} lines."
            )
        return data

    def _measurement(self, data: np.ndarray) -> Measurement:
        source = f"NanoVNA ({self.port}) {datetime.now():%H:%M:%S}"
        return measurement_from_sweep(
            frequency_hz=data[:, 0],
            s11=data[:, 1] + 1j * data[:, 2],
            s21=data[:, 3] + 1j * data[:, 4],
            source=source,
        )


def _max_points(info: list[str]) -> int:
    """Points limit from the info line, e.g. "Version: 1.2.43 [p:101, IF:12k, ...]"."""
    for line in info:
        match = re.search(r"\bp:(\d+)", line)
        if match:
            return int(match.group(1))
    return DEFAULT_MAX_POINTS
