import os
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_labeller.app_entry import run_window_mode


def main() -> None:
    run_window_mode("chooser")


if __name__ == "__main__":
    main()
