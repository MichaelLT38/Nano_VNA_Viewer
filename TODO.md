# Nano VNA Viewer — TODO

A cross-platform (Windows + Linux) Python replacement for the LabVIEW-based Nano VNA LabVIEW Viewer.
No LabVIEW, NI runtime, or NI-VISA required.

**Stack:** Python 3.11+, PySide6 (GUI), pyqtgraph (plots), scikit-rf (Touchstone/RF math),
NumPy, pyserial (NanoVNA USB), PyInstaller (packaging).

---

## Phase 1 — Project setup
- [x] Create a virtual environment and `requirements.txt`
- [x] Set up the folder layout (`src/nano_vna_viewer/`, `tests/`, `samples/`)
- [x] Copy `data.S1P` from the LabVIEW project into `samples/` for testing
- [x] Add `.gitignore` and `README.md`
- [x] Add a license (MIT, bitswizzler.io)
- [x] Initialize a git repository

## Phase 2 — File loading (core)
- [x] Load `.s1p` / `.s2p` files with scikit-rf (support RI, MA, and DB formats)
  - Uses `read_touchstone`, never `skrf.Network(path)`, which unpickles files and could run malicious code
- [x] Handle bad or unsupported files with clear error messages
- [x] Write unit tests against the sample file

## Phase 3 — Basic GUI
- [x] Main window with a menu (File → Open, Exit) and a status bar
- [x] Show the loaded file name, frequency range, and point count
- [x] Remember the last-opened folder

## Phase 4 — Plots
- [x] Return loss (|S11| in dB) vs. frequency
- [x] VSWR vs. frequency (log scale, plain 1-2-5 axis labels)
  - |S11| ≥ 1 is shown as a gap and "∞" instead of a negative VSWR
- [x] Phase vs. frequency
- [x] Transmission tab (S21 magnitude and phase), shown only for .s2p files
- [x] Smith chart
  - Impedance grid always shown; optional admittance grid overlay
- [x] Tabs or a selector to switch between plots
  - Tabs (decided after review)
- [x] Cursor/marker readout (frequency, value) on hover or click
  - One marker shared by all plots: click a plot or drag the line; it snaps to the nearest point
- [x] Zoom, pan, and reset view (mouse wheel / drag; View → Reset Zoom, Ctrl+0)
- [x] Optional curve smoothing (View → Smooth Curves, off by default, remembered)
  - Display only: cubic spline of the complex S-parameter, with dB / VSWR / phase derived from it;
    measured points shown as dots; marker, readouts, and exports stay on measured points

## Phase 5 — Analysis and export
- [x] Find and mark the minimum-SWR / resonant frequency
  - Dashed "Min VSWR" line / Smith ring; marker starts there; View → Go to Minimum VSWR (Ctrl+M)
  - Minimum |S11| point; not necessarily where X = 0 (e.g. 50.S1P: 51.27 − j14.67 Ω)
- [x] Show impedance (R + jX) at the marker
- [x] Export plots as PNG (current tab; File menu, Ctrl+E)
- [x] Export data as CSV (S-params, dB, phase, VSWR, Z; File menu, Ctrl+Shift+E)

## Phase 6 — Live NanoVNA connection
- [x] Detect and list serial ports (Windows `COMx`, Linux `/dev/ttyACM*`); NanoVNA (USB 0483:5740) listed first
- [x] Connect and query device info (`info`, `version`)
- [x] Set the sweep range and points, then read frequencies and S11 data
  - Uses `scan` (S11 + S21 in one pass); pauses the device while connected and restores its own sweep on disconnect
  - Firmware hang fixed: restoring + resuming after every scan hung a NanoVNA-H (fw 1.2.43) after 162 varied
    sweeps; the scan-only pattern ran 500+ without a hang. Writes time out instead of blocking on a hung device.
- [x] Single sweep and continuous sweep modes (serial I/O on a background thread)
- [x] Save the captured sweep as `.S1P` (and `.S2P` if S21 is captured)
- [x] Document the Linux serial permission fix (add the user to the `dialout` group)
- [x] Verified against a real NanoVNA-H, firmware 1.2.43 (`NANOVNA_PORT=COM12 pytest`)

## Phase 7 — Packaging and release
- [x] Build a Windows executable with PyInstaller (`python packaging/build.py`)
  - Zipped folder, not a single .exe: single-file took 34 s to start (Defender scans the
    unpacked files), the folder build starts in 1-2 s
- [x] Build a Linux executable with PyInstaller (or an AppImage)
  - Single binary in a .tar.gz (keeps the executable bit); built on Ubuntu 22.04 for glibc compatibility
  - X11 needs `libxcb-cursor0` on the system (documented)
- [~] Test on clean Windows and Linux machines
  - Done: unzipped Windows build run from a fresh folder; Linux build tested in WSL Ubuntu 22.04
    (offscreen + Wayland); all 165 tests pass on both
  - Packaged Windows build swept the real NanoVNA; release download (built by GitHub Actions)
    verified to start. Still to do: a machine that has never had Python installed
  - Windows needs the unzip path short enough for Qt's plugins (<260 chars; documented)
- [x] Write the final README (install, usage, screenshots)
- [x] Publish a GitHub release with both builds
  - v0.1.0: https://github.com/MichaelLT38/Nano_VNA_Viewer/releases/tag/v0.1.0
  - v0.1.1 (curve smoothing): https://github.com/MichaelLT38/Nano_VNA_Viewer/releases/tag/v0.1.1
  - `.github/workflows/build.yml` tests + builds on every push and publishes a release on a `v*` tag
