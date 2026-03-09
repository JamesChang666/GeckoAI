from __future__ import annotations

import tkinter as tk

from ai_labeller.main import GeckoAI


class GeckoAIWindowLauncher:
    startup_mode = "chooser"

    def run(self) -> None:
        root = tk.Tk()
        GeckoAI(root, startup_mode=self.startup_mode)
        root.mainloop()


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
    build_launcher(startup_mode).run()
