"""Safe wrapper module for label-mode functions.
Re-exports label-controller helpers used by the main app.
"""
from typing import Any
from ai_labeller.features import label_controller


def remove_current_from_split(app: Any) -> None:
    return label_controller.remove_current_from_split(app)


def restore_removed_file_by_name(app: Any, filename: str) -> None:
    return label_controller.restore_removed_file_by_name(app, filename)


def save_current(app: Any) -> None:
    return label_controller.save_current(app)


def reindex_dataset_labels_after_class_delete(app: Any, deleted_idx: int) -> None:
    return label_controller._reindex_dataset_labels_after_class_delete(app, deleted_idx)


def rotate_selected_boxes(app: Any, delta_deg: float) -> None:
    return label_controller.rotate_selected_boxes(app, delta_deg)
