import os
import sys
import tkinter as tk

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_labeller.main import GeckoAI


def main() -> None:
    root = tk.Tk()
    GeckoAI(root, startup_mode="detect")
    root.mainloop()


if __name__ == "__main__":
    main()
