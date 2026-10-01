"""A stand-in for a NanoVNA's USB serial port, for tests that run without hardware.

Behaviour mirrors a real NanoVNA-H on firmware 1.2.43 (checked against one):
- every command is echoed, followed by its output and a "ch> " prompt;
- unknown commands reply "<name>?";
- "scan" pauses sweeping and sets the device's point count to the scan's.
"""

from __future__ import annotations

import time

import numpy as np
import serial

from nano_vna_viewer.nanovna import PortInfo


def fake_s11(f):
    """A load that matches best (|S11| = 0.05) at 300 MHz."""
    f = np.asarray(f, dtype=float)
    return 0.05 + 0.9 * np.abs(f - 300e6) / 900e6 * np.exp(-2j * np.pi * f / 1e9)


def fake_s21(f):
    return 0.001 * np.exp(-1j * np.asarray(f, dtype=float) / 1e8)


class FakeSerial:
    def __init__(
        self,
        port="COMFAKE",
        baudrate=115200,
        timeout=None,
        write_timeout=None,
        *,
        max_points=101,
        silent=False,
        has_bandwidth=True,
    ):
        self.port = port
        self.max_points = max_points
        self.sweep = [50_000, 900_000_000, 101]
        self.paused = False
        self.is_open = True
        self.silent = silent  # True: never answer (not a NanoVNA)
        self.hung = False  # True: firmware hang - stops reading, so writes time out
        self.has_bandwidth = has_bandwidth
        self.broken = False  # True: raise like an unplugged device
        self.scan_line_count = None  # override how many scan lines are sent
        self.commands: list[str] = []
        self._out = bytearray()
        self._in = b""

    # -- pyserial interface -----------------------------------------------------------

    @property
    def in_waiting(self) -> int:
        if self.broken:
            raise serial.SerialException("device disconnected")
        return len(self._out)

    def read(self, size=1) -> bytes:
        if self.broken:
            raise serial.SerialException("device disconnected")
        if not self._out:
            time.sleep(0.001)
            return b""
        chunk = bytes(self._out[:size])
        del self._out[:size]
        return chunk

    def write(self, data: bytes) -> int:
        if self.broken:
            raise serial.SerialException("device disconnected")
        if self.hung:
            raise serial.SerialTimeoutException("Write timeout")
        self._in += data
        while b"\r" in self._in:
            line, self._in = self._in.split(b"\r", 1)
            self._handle(line.decode("ascii"))
        return len(data)

    def reset_input_buffer(self) -> None:
        self._out.clear()

    def close(self) -> None:
        self.is_open = False

    # -- shell ------------------------------------------------------------------------

    def _handle(self, text: str) -> None:
        if self.silent:
            return
        if text:
            self.commands.append(text)
        lines = self._run(text.split()) if text else []
        reply = text + "\r\n" + "".join(line + "\r\n" for line in lines) + "ch> "
        self._out += reply.encode("ascii")

    def _run(self, args: list[str]) -> list[str]:
        name, rest = args[0], args[1:]
        if name == "version":
            return ["1.2.43"]
        if name == "info":
            return [
                "Board: NanoVNA-H",
                f"Version: 1.2.43 [p:{self.max_points}, IF:12k, ADC:192k, Lcd:320x240]",
            ]
        if name == "sweep":
            if not rest:
                return [" ".join(str(v) for v in self.sweep)]
            self.sweep = [int(v) for v in rest[:3]]
            return []
        if name == "resume":
            self.paused = False
            return []
        if name == "pause":
            self.paused = True
            return []
        if name == "bandwidth" and self.has_bandwidth:
            return ["bandwidth 3 (1000Hz)"]
        if name == "scan":
            if len(rest) < 2:
                return ["usage: scan {start(Hz)} {stop(Hz)} [points] [outmask]"]
            start, stop = int(rest[0]), int(rest[1])
            points = int(rest[2]) if len(rest) > 2 else self.sweep[2]
            if points > self.max_points:
                return ["usage: scan {start(Hz)} {stop(Hz)} [points] [outmask]"]
            self.paused = True
            self.sweep[2] = points  # what the real firmware does
            f = np.linspace(start, stop, points).round()
            s11, s21 = fake_s11(f), fake_s21(f)
            lines = [
                f"{fi:.0f} {a.real:.9f} {a.imag:.9f} {b.real:.9f} {b.imag:.9f} "
                for fi, a, b in zip(f, s11, s21)
            ]
            if self.scan_line_count is not None:
                lines = lines[: self.scan_line_count]
            return lines
        return [f"{name}?"]


class FakePorts:
    """Factory + port lister pair for DevicePanel tests."""

    def __init__(self, **serial_options):
        self.serials: list[FakeSerial] = []
        self.serial_options = serial_options
        self.ports = [
            PortInfo("COM3", "Bluetooth link", False),
            PortInfo("COMFAKE", "USB Serial Device", True),
        ]

    def list_ports(self) -> list[PortInfo]:
        return sorted(self.ports, key=lambda p: (not p.is_nanovna, p.device))

    def serial_factory(self, port, baudrate, timeout=None, write_timeout=None):
        fake = FakeSerial(port, baudrate, timeout, write_timeout, **self.serial_options)
        self.serials.append(fake)
        return fake

    def device_factory(self, port):
        from nano_vna_viewer.nanovna import NanoVNA

        return NanoVNA(port, serial_factory=self.serial_factory)
