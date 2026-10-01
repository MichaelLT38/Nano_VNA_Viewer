import pytest

from fake_nanovna import FakePorts
from nano_vna_viewer import nanovna
from nano_vna_viewer.device_panel import DevicePanel


@pytest.fixture
def ports():
    return FakePorts()


@pytest.fixture
def panel(qtbot, ports):
    p = DevicePanel(device_factory=ports.device_factory, port_lister=ports.list_ports)
    qtbot.addWidget(p)
    yield p
    p.shutdown()


@pytest.fixture
def connected(qtbot, panel):
    panel.connect_button.click()
    qtbot.waitUntil(lambda: panel.device is not None, timeout=5000)
    return panel


def test_ports_listed_with_nanovna_first(panel):
    items = [panel.port_combo.itemText(i) for i in range(panel.port_combo.count())]
    assert items == ["COMFAKE — NanoVNA", "COM3 — Bluetooth link"]
    assert panel.port_combo.currentData() == "COMFAKE"


def test_controls_before_connecting(panel):
    assert panel.connect_button.isEnabled()
    assert panel.connect_button.text() == "Connect"
    assert not panel.sweep_button.isEnabled()
    assert not panel.continuous_button.isEnabled()
    assert not panel.start_spin.isEnabled()
    assert panel.status_label.text() == "Not connected"
    assert panel.sweep_settings() == (50_000, 900_000_000, 101)  # sensible defaults


def test_no_ports_disables_connect(qtbot, ports):
    ports.ports = []
    p = DevicePanel(device_factory=ports.device_factory, port_lister=ports.list_ports)
    qtbot.addWidget(p)
    try:
        assert p.port_combo.count() == 0
        assert not p.connect_button.isEnabled()
    finally:
        p.shutdown()


def test_connect_fills_in_device_sweep(connected):
    assert connected.device.board == "NanoVNA-H"
    assert connected.status_label.text() == "Connected: NanoVNA-H, firmware 1.2.43"
    assert connected.start_spin.value() == pytest.approx(0.05)
    assert connected.stop_spin.value() == pytest.approx(900)
    assert connected.points_spin.value() == 101
    assert connected.points_spin.maximum() == 101
    assert connected.connect_button.text() == "Disconnect"
    assert not connected.port_combo.isEnabled()
    assert connected.sweep_button.isEnabled()


def test_connect_failure_is_reported(qtbot, panel, ports, monkeypatch):
    monkeypatch.setattr(nanovna, "COMMAND_TIMEOUT", 0.1)
    ports.serial_options["silent"] = True  # the port never answers
    with qtbot.waitSignal(panel.error, timeout=5000) as blocker:
        panel.connect_button.click()
    assert "No response from COMFAKE" in blocker.args[0]
    qtbot.waitUntil(lambda: panel.connect_button.isEnabled())
    assert panel.device is None
    assert panel.connect_button.text() == "Connect"
    assert not ports.serials[-1].is_open


def test_single_sweep(qtbot, connected):
    connected.start_spin.setValue(100)
    connected.stop_spin.setValue(500)
    connected.points_spin.setValue(41)
    with qtbot.waitSignal(connected.measurement_ready, timeout=5000) as blocker:
        connected.sweep_button.click()
    measurement, continuous = blocker.args
    assert not continuous
    assert measurement.points == 41
    assert measurement.start_hz == 100e6
    assert measurement.stop_hz == 500e6
    qtbot.waitUntil(lambda: connected.sweep_button.isEnabled())


def test_sweep_settings_round_to_hz(connected):
    connected.start_spin.setValue(0.05)
    connected.stop_spin.setValue(433.92)
    assert connected.sweep_settings() == (50_000, 433_920_000, 101)


def test_continuous_sweeps_until_stopped(qtbot, connected):
    results = []
    connected.measurement_ready.connect(lambda m, cont: results.append(cont))
    connected.continuous_button.click()
    assert not connected.sweep_button.isEnabled()
    assert not connected.start_spin.isEnabled()  # range locked while running
    qtbot.waitUntil(lambda: len(results) >= 3, timeout=5000)
    assert all(results[:3])
    connected.continuous_button.click()
    qtbot.waitUntil(lambda: not connected.busy, timeout=5000)
    count = len(results)
    qtbot.wait(200)
    assert len(results) == count  # stopped
    assert connected.sweep_button.isEnabled()


def test_device_error_stops_continuous(qtbot, connected):
    connected.start_spin.setValue(500)
    connected.stop_spin.setValue(100)  # invalid range
    with qtbot.waitSignal(connected.error, timeout=5000) as blocker:
        connected.continuous_button.click()
    assert "below stop" in blocker.args[0]
    assert not connected.continuous_button.isChecked()
    assert connected.device is not None  # still connected


def test_unplugging_disconnects(qtbot, connected, ports):
    ports.serials[-1].broken = True
    with qtbot.waitSignal(connected.error, timeout=5000) as blocker:
        connected.sweep_button.click()
    assert "Lost connection" in blocker.args[0]
    qtbot.waitUntil(lambda: connected.device is None, timeout=5000)
    assert connected.status_label.text() == "Not connected"
    assert connected.connect_button.text() == "Connect"


def test_disconnect_closes_port(qtbot, connected, ports):
    connected.connect_button.click()
    qtbot.waitUntil(lambda: connected.device is None, timeout=5000)
    assert not ports.serials[-1].is_open
    assert connected.connect_button.text() == "Connect"


def test_shutdown_closes_port(connected, ports):
    connected.shutdown()
    assert not ports.serials[-1].is_open


def test_last_sweep_after_stop_still_counts_as_continuous(qtbot, connected):
    results = []
    connected.measurement_ready.connect(lambda m, cont: results.append(cont))
    connected.continuous_button.click()
    qtbot.waitUntil(lambda: connected.busy and len(results) >= 1, timeout=5000)
    connected.continuous_button.click()  # stop while a sweep is in flight
    qtbot.waitUntil(lambda: not connected.busy, timeout=5000)
    assert results and all(results)


def test_single_sweep_is_not_continuous(qtbot, connected):
    with qtbot.waitSignal(connected.measurement_ready, timeout=5000) as blocker:
        connected.sweep_button.click()
    assert blocker.args[1] is False


def test_status_says_device_screen_is_paused_after_sweeping(qtbot, connected, ports):
    assert "paused" not in connected.status_label.text()
    with qtbot.waitSignal(connected.measurement_ready, timeout=5000):
        connected.sweep_button.click()
    assert connected.status_label.text() == (
        "Connected: NanoVNA-H, firmware 1.2.43 — NanoVNA screen paused while connected"
    )
    assert ports.serials[-1].paused
    connected.connect_button.click()  # disconnect
    qtbot.waitUntil(lambda: connected.device is None, timeout=5000)
    assert connected.status_label.text() == "Not connected"
    assert not ports.serials[-1].paused  # device resumed on disconnect
