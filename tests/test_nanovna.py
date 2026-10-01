import os
from types import SimpleNamespace

import numpy as np
import pytest
import serial

from fake_nanovna import FakeSerial, fake_s11, fake_s21
from nano_vna_viewer import nanovna
from nano_vna_viewer.nanovna import ConnectionLost, NanoVNA, NanoVNAError, SweepSettings, find_ports


@pytest.fixture
def fake():
    return FakeSerial()


@pytest.fixture
def vna(fake):
    return NanoVNA("COMFAKE", serial_factory=lambda *a, **k: fake)


def test_find_ports_lists_nanovna_first(monkeypatch):
    ports = [
        SimpleNamespace(device="COM1", description="Communications Port", vid=None, pid=None),
        SimpleNamespace(device="COM12", description="USB Serial Device", vid=0x0483, pid=0x5740),
    ]
    monkeypatch.setattr(nanovna.list_ports, "comports", lambda: ports)
    found = find_ports()
    assert [p.device for p in found] == ["COM12", "COM1"]
    assert found[0].is_nanovna and not found[1].is_nanovna
    assert found[0].label == "COM12 — NanoVNA"
    assert found[1].label == "COM1 — Communications Port"


def test_connect_reads_identity(vna):
    assert vna.version == "1.2.43"
    assert vna.board == "NanoVNA-H"
    assert vna.max_points == 101


def test_max_points_comes_from_info():
    fake = FakeSerial(max_points=401)
    assert NanoVNA("X", serial_factory=lambda *a, **k: fake).max_points == 401


def test_get_and_set_sweep(vna, fake):
    assert vna.get_sweep() == SweepSettings(50_000, 900_000_000, 101)
    vna.set_sweep(SweepSettings(1_000_000, 2_000_000, 51))
    assert fake.sweep == [1_000_000, 2_000_000, 51]


def test_scan_returns_two_port_measurement(vna):
    m = vna.scan(100_000_000, 500_000_000, 21)
    assert m.points == 21
    assert m.nports == 2
    assert m.path is None
    assert m.name.startswith("NanoVNA (COMFAKE) ")
    np.testing.assert_allclose(m.frequency_hz, np.linspace(100e6, 500e6, 21).round())
    np.testing.assert_allclose(m.s11, fake_s11(m.frequency_hz), atol=1e-9)
    np.testing.assert_allclose(m.s21, fake_s21(m.frequency_hz), atol=1e-9)


def test_scan_pauses_once_and_sends_only_scan(vna, fake):
    vna.scan(100_000_000, 500_000_000, 21)
    vna.scan(1_000_000, 30_000_000, 51)
    assert fake.paused
    assert fake.commands[-3:] == [
        "pause",
        "scan 100000000 500000000 21 7",
        "scan 1000000 30000000 51 7",
    ]  # no sweep/resume between scans (that pattern hung real hardware)


def test_close_restores_device_sweep_once(vna, fake):
    vna.scan(100_000_000, 500_000_000, 21)
    assert fake.sweep[2] == 21  # the firmware changed its point count
    vna.close()
    assert fake.sweep == [50_000, 900_000_000, 101]
    assert not fake.paused
    assert fake.commands[-2:] == ["sweep 50000 900000000 101", "resume"]
    assert not fake.is_open


def test_close_without_scanning_leaves_device_alone(vna, fake):
    sent = list(fake.commands)
    vna.close()
    assert fake.commands == sent


def test_scan_timeout_follows_bandwidth(vna):
    assert vna.bandwidth_hz == 1000
    assert vna.scan_timeout(101) == pytest.approx(5 + 101 * 20 / 1000)


def test_scan_timeout_without_bandwidth_command():
    fake = FakeSerial(has_bandwidth=False)
    vna = NanoVNA("X", serial_factory=lambda *a, **k: fake)
    assert vna.bandwidth_hz is None
    assert vna.scan_timeout(101) == nanovna.SCAN_TIMEOUT


def test_hung_device_is_reported_and_not_written_to_again(vna, fake):
    vna.scan(100_000_000, 500_000_000, 21)
    fake.hung = True
    with pytest.raises(ConnectionLost, match="Switch it off and on"):
        vna.scan(100_000_000, 500_000_000, 21)
    fake.hung = False
    sent = list(fake.commands)
    with pytest.raises(ConnectionLost):
        vna.command("version")  # connection is marked unusable
    vna.close()  # must not try to restore a hung device
    assert fake.commands == sent
    assert not fake.is_open


def test_unanswered_scan_is_reported_as_hang(vna, fake, monkeypatch):
    monkeypatch.setattr(nanovna, "SCAN_TIMEOUT_BASE", 0.1)
    monkeypatch.setattr(nanovna, "SCAN_SECONDS_PER_POINT_HZ", 0)
    vna.scan(100_000_000, 500_000_000, 21)
    fake.silent = True
    with pytest.raises(ConnectionLost, match="stopped responding"):
        vna.scan(100_000_000, 500_000_000, 21)


@pytest.mark.parametrize(
    "start, stop, points, message",
    [
        (500_000_000, 100_000_000, 21, "below stop"),
        (0, 100_000_000, 21, "above zero"),
        (100_000_000, 500_000_000, 1, "between 2 and 101"),
        (100_000_000, 500_000_000, 102, "between 2 and 101"),
    ],
)
def test_scan_validates_before_sending(vna, fake, start, stop, points, message):
    sent = list(fake.commands)
    with pytest.raises(NanoVNAError, match=message):
        vna.scan(start, stop, points)
    assert fake.commands == sent  # nothing reached the device


def test_rejected_command(vna):
    with pytest.raises(NanoVNAError, match="rejected 'bogus': bogus\\?"):
        vna.command("bogus")


def test_short_scan_reply_is_an_error(vna, fake):
    fake.scan_line_count = 5
    with pytest.raises(NanoVNAError, match="Expected 21 points"):
        vna.scan(100_000_000, 500_000_000, 21)
    vna.close()
    assert fake.sweep == [50_000, 900_000_000, 101]  # still restored on close


def test_no_response_means_not_a_nanovna(monkeypatch):
    monkeypatch.setattr(nanovna, "COMMAND_TIMEOUT", 0.1)
    fake = FakeSerial()
    fake.silent = True
    with pytest.raises(ConnectionLost, match="No response from COMFAKE"):
        NanoVNA("COMFAKE", serial_factory=lambda *a, **k: fake)
    assert not fake.is_open  # closed after failing


def test_open_failure():
    def refuse(*args, **kwargs):
        raise serial.SerialException("access denied")

    with pytest.raises(NanoVNAError, match="Could not open COM99: access denied"):
        NanoVNA("COM99", serial_factory=refuse)


def test_unplugged_device_raises_connection_lost(vna, fake):
    fake.broken = True
    with pytest.raises(ConnectionLost, match="Lost connection"):
        vna.scan(100_000_000, 500_000_000, 21)


def test_context_manager_closes(fake):
    with NanoVNA("COMFAKE", serial_factory=lambda *a, **k: fake):
        assert fake.is_open
    assert not fake.is_open


@pytest.mark.skipif("NANOVNA_PORT" not in os.environ, reason="set NANOVNA_PORT to test real hardware")
def test_real_device(monkeypatch):
    monkeypatch.setattr(nanovna, "RESUME_SETTLE", 1.0)  # real hardware needs the real delay
    port = os.environ["NANOVNA_PORT"]
    with NanoVNA(port) as vna:
        before = vna.original_sweep
        for points in (11, 51, 101):
            m = vna.scan(100_000_000, 200_000_000, points)
            assert m.points == points
            assert np.all(np.abs(m.s11) < 1.5)
    with NanoVNA(port) as vna:  # reconnect: device restored
        assert vna.get_sweep() == before
