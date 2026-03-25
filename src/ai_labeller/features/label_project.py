from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Any


def load_label_project(app: Any, path: str, kind: str, *, QMessageBox: Any) -> None:
    app._project_dir = os.path.abspath(path)
    app._label_format_mode = "detect"
    app._progress_state = app._read_progress_yaml(app._project_dir)
    images_root = os.path.join(path, "images")
    labels_root = os.path.join(path, "labels")
    has_split = any(os.path.isdir(os.path.join(images_root, s)) for s in ("train", "val", "test"))
    has_flat = os.path.isdir(images_root) and os.path.isdir(labels_root)
    if kind == "yolo_dataset":
        app._is_yolo_project = bool(has_split or has_flat)
        if not app._is_yolo_project:
            QMessageBox.warning(
                app,
                "Load Project",
                "Selected folder is not a YOLO dataset root.\nExpected images/ and labels/ (with or without train|val|test splits).",
            )
            return
    else:
        app._is_yolo_project = bool(has_split or has_flat)
    app._yolo_use_split_layout = bool(app._is_yolo_project and has_split)
    if app._is_yolo_project:
        app._project_root = path
        available = [s for s in ("train", "val", "test") if os.path.isdir(os.path.join(images_root, s))]
        app._load_class_names_from_dataset_yaml()
        progress_split = str(app._progress_state.get("split", "")).strip().lower()
        if progress_split in {"train", "val", "test"} and progress_split in available:
            app._current_split = progress_split
        progress_class_names = app._extract_class_names_from_progress(app._progress_state)
        if progress_class_names:
            app._class_names = progress_class_names
        app.combo_split.setEnabled(app._yolo_use_split_layout)
        app.combo_split.blockSignals(True)
        app.combo_split.clear()
        split_items = available if app._yolo_use_split_layout else ["all"]
        app.combo_split.addItems(split_items)
        if app._current_split not in split_items:
            app._current_split = "train" if app._yolo_use_split_layout and "train" in available else split_items[0]
        app.combo_split.setCurrentText(app._current_split)
        app.combo_split.blockSignals(False)
        app._reload_images_for_current_source(reset_classes=False)
        app._restore_progress_position()
        app._save_progress_yaml()
        return
    app._project_root = ""
    app._class_names = ["class0"]
    progress_class_names = app._extract_class_names_from_progress(app._progress_state)
    if progress_class_names:
        app._class_names = progress_class_names
    app.combo_split.setEnabled(False)
    app._reload_images_for_current_source(reset_classes=False)
    app._restore_progress_position()
    app._save_progress_yaml()


def read_progress_yaml(app: Any, project_root: str, session_progress_cache: dict[str, dict[str, str]]) -> dict[str, str]:
    key = os.path.abspath(project_root or app._project_root or app._project_dir or os.getcwd())
    data = session_progress_cache.get(key, {})
    return dict(data) if isinstance(data, dict) else {}


def extract_class_names_from_progress(progress: dict[str, str]) -> list[str]:
    try:
        class_count = int(str(progress.get("class_count", "0")).strip())
    except Exception:
        class_count = 0
    if class_count <= 0:
        return []
    out: list[str] = []
    for idx in range(class_count):
        name = str(progress.get(f"class_{idx}", "")).strip()
        if not name:
            return []
        out.append(name)
    return out


def save_progress_yaml(app: Any, session_progress_cache: dict[str, dict[str, str]]) -> None:
    if not app._project_dir and not app._project_root:
        return
    image_name = ""
    image_index = int(app._image_idx)
    if app._image_paths and 0 <= app._image_idx < len(app._image_paths):
        image_name = os.path.basename(app._image_paths[app._image_idx])
    project_root = os.path.abspath(app._project_root or app._project_dir)
    data: dict[str, str] = {
        "project_root": project_root,
        "split": str(app._current_split),
        "image_name": str(image_name),
        "image_index": str(image_index),
        "class_count": str(len(app._class_names)),
        "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    for idx, class_name in enumerate(app._class_names):
        data[f"class_{idx}"] = str(class_name)
    session_progress_cache[project_root] = data


def restore_progress_position(app: Any) -> None:
    if not app._image_paths:
        return
    progress = app._progress_state or {}
    target_name = str(progress.get("image_name", "")).strip()
    if target_name:
        for i, p in enumerate(app._image_paths):
            if os.path.basename(p) == target_name:
                app._image_idx = i
                app._show_current_image()
                return
    try:
        idx = int(str(progress.get("image_index", "0")).strip())
    except Exception:
        idx = 0
    if 0 <= idx < len(app._image_paths):
        app._image_idx = idx
        app._show_current_image()


def load_class_names_from_dataset_yaml(app: Any, *, golden_core: Any) -> None:
    if not app._project_root:
        return
    yaml_path = golden_core.find_dataset_yaml_in_folder(app._project_root)
    if not yaml_path:
        return
    mapping = golden_core.load_mapping_from_dataset_yaml(yaml_path)
    if not mapping:
        return
    max_id = max(mapping.keys())
    classes = [f"class{i}" for i in range(max_id + 1)]
    for idx, name in mapping.items():
        if 0 <= idx < len(classes):
            classes[idx] = str(name)
    if classes:
        app._class_names = classes


def scan_image_paths_for_current_source(app: Any) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    if app._is_yolo_project and app._project_root:
        if app._yolo_use_split_layout:
            split_dir = os.path.join(app._project_root, "images", app._current_split)
            os.makedirs(os.path.join(app._project_root, "labels", app._current_split), exist_ok=True)
            if not os.path.isdir(split_dir):
                return []
            return [str(p) for p in sorted(Path(split_dir).iterdir()) if p.is_file() and p.suffix.lower() in exts]
        image_dir = os.path.join(app._project_root, "images")
        os.makedirs(os.path.join(app._project_root, "labels"), exist_ok=True)
        if not os.path.isdir(image_dir):
            return []
        return [str(p) for p in sorted(Path(image_dir).iterdir()) if p.is_file() and p.suffix.lower() in exts]
    if not app._project_dir or not os.path.isdir(app._project_dir):
        return []
    return [str(p) for p in sorted(Path(app._project_dir).iterdir()) if p.is_file() and p.suffix.lower() in exts]


def refresh_combo_image_items(app: Any, *, Qt: Any) -> None:
    app.combo_image.blockSignals(True)
    app.combo_image.clear()
    for p in app._image_paths:
        base = os.path.basename(p)
        display = app._compact_name(base)
        app.combo_image.addItem(display, p)
        try:
            app.combo_image.setItemData(app.combo_image.count() - 1, base, Qt.ItemDataRole.ToolTipRole)
        except Exception:
            pass
    app.combo_image.blockSignals(False)


def reload_images_for_current_source(app: Any, reset_classes: bool = False, *, QMessageBox: Any, Qt: Any) -> None:
    app._image_paths = app._scan_image_paths_for_current_source()
    app._refresh_combo_image_items()
    if not app._image_paths:
        QMessageBox.warning(app, "Label Workspace", "No images found in selected folder.")
        app._image_idx = 0
        app._labels_by_path = {}
        app._refresh_info_labels()
        return
    if reset_classes:
        app._class_names = ["class0"]
    app._labels_by_path = {}
    app._image_idx = 0
    app._show_current_image()


def auto_refresh_tick(app: Any) -> None:
    if not app._auto_refresh_enabled:
        return
    if not app._project_dir and not app._project_root:
        return
    try:
        new_paths = app._scan_image_paths_for_current_source()
    except Exception:
        return
    if new_paths == app._image_paths:
        return
    current_path = ""
    if app._image_paths and 0 <= app._image_idx < len(app._image_paths):
        current_path = app._image_paths[app._image_idx]
    app._sync_canvas_rects_to_current_image()
    old_paths = set(app._image_paths)
    app._image_paths = new_paths
    app._refresh_combo_image_items()
    new_paths_set = set(new_paths)
    app._labels_by_path = {k: v for k, v in app._labels_by_path.items() if k in new_paths_set}
    if not app._image_paths:
        app._image_idx = 0
        app.lbl_status.setText("No images found in selected folder.")
        app._refresh_info_labels()
        return
    if current_path and current_path in app._image_paths:
        app._image_idx = app._image_paths.index(current_path)
    else:
        app._image_idx = max(0, min(app._image_idx, len(app._image_paths) - 1))
    added = len([p for p in app._image_paths if p not in old_paths])
    if added > 0:
        app.lbl_status.setText(f"Auto refresh: +{added} image(s)")
    app._show_current_image()


def on_split_changed(app: Any, split: str) -> None:
    split = str(split or "").strip().lower()
    if not app._is_yolo_project:
        return
    if not app._yolo_use_split_layout:
        return
    if split not in {"train", "val", "test"}:
        return
    app._sync_canvas_rects_to_current_image()
    app._current_split = split
    app._reload_images_for_current_source(reset_classes=False)
    app._save_progress_yaml()
