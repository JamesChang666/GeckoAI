"""Deprecated detect-mode compatibility shim.

Legacy Tk detect modules were removed after full PySide6 migration.
Use `ai_labeller.app_qt` / `ai_labeller.app_entry` entrypoints instead.
"""

from __future__ import annotations


def _deprecated() -> None:
    raise RuntimeError(
        "Legacy Tk detect wrappers were removed. "
        "Use PySide6 entrypoints: ai_labeller.app_qt / ai_labeller.app_entry."
    )


def show_detect_mode_page(*_args, **_kwargs):
    _deprecated()


def show_detect_source_page(*_args, **_kwargs):
    _deprecated()


def show_detect_camera_mode_page(*_args, **_kwargs):
    _deprecated()


def show_detect_file_settings_page(*_args, **_kwargs):
    _deprecated()


def open_detect_workspace(*_args, **_kwargs):
    _deprecated()


def render_current_piece_result(*_args, **_kwargs):
    _deprecated()


def detect_render_image_index(*_args, **_kwargs):
    _deprecated()


def detect_prev_image(*_args, **_kwargs):
    _deprecated()


def detect_next_image(*_args, **_kwargs):
    _deprecated()


def show_detect_plot(*_args, **_kwargs):
    _deprecated()


def refresh_detect_image(*_args, **_kwargs):
    _deprecated()


def run_detect_inference(*_args, **_kwargs):
    _deprecated()


def should_use_background_cut_detection(*_args, **_kwargs):
    _deprecated()


def get_easy_ocr_engine(*_args, **_kwargs):
    _deprecated()


def extract_ocr_id_from_result(*_args, **_kwargs):
    _deprecated()


def extract_ocr_sub_id_from_result(*_args, **_kwargs):
    _deprecated()


def evaluate_golden_match(*_args, **_kwargs):
    _deprecated()
