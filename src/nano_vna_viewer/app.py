"""Application entry point."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from . import __version__
from .main_window import APP_NAME, MainWindow


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv

    app = QApplication(argv)
    # Used by QSettings to decide where preferences are stored.
    app.setOrganizationName("bitswizzler.io")
    app.setOrganizationDomain("bitswizzler.io")
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)

    window = MainWindow()
    window.show()

    # Optional: open a file passed on the command line.
    if len(argv) > 1:
        window.open_file(argv[1])

    return app.exec()
