from __future__ import annotations

import sys

from ai_labeller.app_qt import run_qt_mode


class GeckoAIWindowLauncher:
    startup_mode = "chooser"

    def run(self) -> None:
        run_qt_mode(self.startup_mode)


class AllModeWindowLauncher(GeckoAIWindowLauncher):
    startup_mode = "chooser"


class DetectModeWindowLauncher(GeckoAIWindowLauncher):
    startup_mode = "detect"


class LabelModeWindowLauncher(GeckoAIWindowLauncher):
    startup_mode = "label"


def build_launcher(startup_mode: str) -> GeckoAIWindowLauncher:
    normalized = (startup_mode or "chooser").strip().lower()
    mapping = {
        "chooser": AllModeWindowLauncher,
        "detect": DetectModeWindowLauncher,
        "label": LabelModeWindowLauncher,
    }
    launcher_cls = mapping.get(normalized, AllModeWindowLauncher)
    return launcher_cls()


def run_window_mode(startup_mode: str) -> None:
    try:
        build_launcher(startup_mode).run()
    except Exception as exc:
        msg = str(exc)
        if "PySide6 is required for geckoai-qt" in msg or "No module named 'PySide6'" in msg:
            sys.stderr.write(
                "PySide6 is not installed in this Python environment.\n"
                "Install with: python -m pip install -e \".[qt]\"\n"
            )
            raise SystemExit(1)
        raise
