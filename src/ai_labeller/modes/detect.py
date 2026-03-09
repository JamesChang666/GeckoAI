"""Safe wrapper module for detect-mode functions.
This re-exports functionality from detect_controller, detect_pages and related detect runtime helpers.
Keep this module small and simple so the main app can route detect-only callers here.
"""
from typing import Any
from ai_labeller.ui import detect_pages
from ai_labeller.features import detect_controller, detect_runtime, ocr_utils, golden_controller


# UI pages
def show_detect_mode_page(app: Any) -> None:
    return detect_pages.show_detect_mode_page(app)


def show_detect_source_page(app: Any) -> None:
    return detect_pages.show_detect_source_page(app)


def show_detect_camera_mode_page(app: Any) -> None:
    return detect_pages.show_detect_camera_mode_page(app)


def show_detect_file_settings_page(app: Any) -> None:
    return detect_pages.show_detect_file_settings_page(app)


# Controller/runtime functions
def open_detect_workspace(app: Any, source_kind: str, source_value: Any, output_dir: str | None = None) -> None:
    return detect_controller.open_detect_workspace(app, source_kind, source_value, output_dir=output_dir)


def render_current_piece_result(app: Any, source_path: str) -> None:
    return detect_controller.render_current_piece_result(app, source_path)


def detect_render_image_index(app: Any) -> None:
    return detect_controller.detect_render_image_index(app)


def detect_prev_image(app: Any) -> None:
    return detect_controller.detect_prev_image(app)


def detect_next_image(app: Any) -> None:
    return detect_controller.detect_next_image(app)


def show_detect_plot(app: Any, plot_bgr: Any) -> None:
    return detect_controller.show_detect_plot(app, plot_bgr)


def refresh_detect_image(app: Any) -> None:
    return detect_controller.refresh_detect_image(app)


def run_detect_inference(app: Any, source: Any) -> Any:
    return detect_runtime.run_detect_inference(app, source)


def should_use_background_cut_detection(app: Any) -> bool:
    return detect_runtime.should_use_background_cut_detection(app)


def get_easy_ocr_engine(app: Any):
    return ocr_utils.get_easy_ocr_engine(app)


def extract_ocr_id_from_result(app: Any, result0: Any):
    return ocr_utils.extract_ocr_id_from_result(app, result0)


def extract_ocr_sub_id_from_result(app: Any, result0: Any):
    return ocr_utils.extract_ocr_sub_id_from_result(app, result0)


def evaluate_golden_match(app: Any, result0: Any):
    return golden_controller.evaluate_golden_match(app, result0)
