"""Entry point for the PyInstaller build (PyInstaller needs a script, not ``-m``)."""

import sys

from nano_vna_viewer.app import main

sys.exit(main())
