import os

# Run Qt without a display so tests work headless (CI, SSH sessions).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _no_resume_settle(monkeypatch):
    """Skip the post-resume settle delay; it only matters for real hardware."""
    from nano_vna_viewer import nanovna

    monkeypatch.setattr(nanovna, "RESUME_SETTLE", 0)
