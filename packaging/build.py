"""Build a standalone Nano VNA Viewer for the current platform.

usage: python packaging/build.py

The result runs without Python installed. PyInstaller can't cross-compile: build the
Windows version on Windows and the Linux version on Linux (the GitHub Actions
workflow does both).

- Windows: dist/NanoVNAViewer-<version>-windows-x64.zip, a folder with NanoVNAViewer.exe.
  A single-file .exe unpacks ~300 MB on every launch and Windows Defender scans it:
  34 s for the first launch, 5-6 s after that. The folder build doesn't unpack anything.
- Linux: dist/NanoVNAViewer-<version>-linux-x64.tar.gz, holding one executable file,
  NanoVNAViewer (a tarball keeps the executable bit, which plain downloads lose).
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAME = "NanoVNAViewer"
PLATFORMS = {"Windows": "windows", "Linux": "linux", "Darwin": "macos"}


def version() -> str:
    sys.path.insert(0, str(ROOT / "src"))
    from nano_vna_viewer import __version__

    return __version__


def main() -> int:
    system = PLATFORMS[platform.system()]
    artifact = f"{NAME}-{version()}-{system}-x64"
    one_folder = system == "windows"
    dist = ROOT / "dist"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--onedir" if one_folder else "--onefile",
            "--windowed",  # no console window on Windows
            "--name",
            NAME,
            "--paths",
            str(ROOT / "src"),
            "--distpath",
            str(dist),
            "--workpath",
            str(ROOT / "build"),
            "--specpath",
            str(ROOT / "build"),
            # Not used by the app; keeps the bundle smaller.
            "--exclude-module",
            "tkinter",
            "--exclude-module",
            "matplotlib",
            str(ROOT / "packaging" / "launcher.py"),
        ],
        check=True,
    )
    if one_folder:
        # Unzips to NanoVNAViewer/NanoVNAViewer.exe.
        output = Path(shutil.make_archive(str(dist / artifact), "zip", dist, NAME))
    else:
        output = dist / f"{artifact}.tar.gz"
        with tarfile.open(output, "w:gz") as tar:
            tar.add(dist / NAME, arcname=NAME)
    print(f"\nBuilt {output} ({output.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
