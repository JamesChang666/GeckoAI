import datetime
import json
import os
from tkinter import messagebox
from typing import Any

from ai_labeller.constants import LANG_MAP
from ai_labeller.core import SessionState, atomic_write_json, atomic_write_text


def project_progress_yaml_path(project_root: str | None = None) -> str | None:
    root = (project_root or "").strip()
    if not root:
        return None
    return os.path.join(root, ".ai_labeller_progress.yaml")


def q(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_project_progress_yaml(app: Any) -> None:
    yaml_path = project_progress_yaml_path(app.project_root)
    if not yaml_path:
        return
    image_name = ""
    image_index = app.current_idx
    if app.image_files and 0 <= app.current_idx < len(app.image_files):
        image_name = os.path.basename(app.image_files[app.current_idx])

    lines = [
        "# AI Labeller progress",
        f"project_root: {q(app.project_root)}",
        f"split: {q(app.current_split)}",
        f"image_name: {q(image_name)}",
        f"image_index: {image_index}",
        f"class_count: {len(app.class_names)}",
        f"updated_at: {q(datetime.datetime.now().isoformat(timespec='seconds'))}",
    ]
    for idx, class_name in enumerate(app.class_names):
        lines.append(f"class_{idx}: {q(class_name)}")
    try:
        atomic_write_text(yaml_path, "\n".join(lines) + "\n")
    except Exception:
        app.logger.exception("Failed to write project progress yaml: %s", yaml_path)


def read_project_progress_yaml(project_root: str) -> dict[str, str]:
    yaml_path = project_progress_yaml_path(project_root)
    if not yaml_path or not os.path.isfile(yaml_path):
        return {}
    data: dict[str, str] = {}
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if value.startswith('"') and value.endswith('"') and len(value) >= 2:
                    value = value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
                data[key] = value
    except Exception:
        # Cannot log here without an app; caller should log if needed.
        return {}
    return data


def extract_class_names_from_progress(progress: dict[str, str]) -> list[str]:
    count_raw = progress.get("class_count", "")
    try:
        class_count = int(count_raw)
    except ValueError:
        class_count = 0
    if class_count <= 0:
        return []
    names: list[str] = []
    for idx in range(class_count):
        key = f"class_{idx}"
        name = progress.get(key, "").strip()
        if not name:
            return []
        names.append(name)
    return names


def save_session_state(app: Any) -> None:
    state = SessionState(
        project_root=app.project_root,
        split=app.current_split,
        image_name="",
        detection_model_mode=app.det_model_mode.get(),
        detection_model_path=app.yolo_path.get().strip(),
    )
    if app.image_files and 0 <= app.current_idx < len(app.image_files):
        state.image_name = os.path.basename(app.image_files[app.current_idx])
    try:
        atomic_write_json(app.session_path, state.__dict__)
    except Exception:
        app.logger.exception("Failed to save session state")
    write_project_progress_yaml(app)


def load_session_state(app: Any, restore_project: bool = True) -> None:
    if not os.path.exists(app.session_path):
        return
    try:
        with open(app.session_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        app.logger.exception("Failed to load session state")
        return

    project_root = data.get("project_root", "")
    split = data.get("split", "train")
    image_name = data.get("image_name", "")
    model_mode = data.get("detection_model_mode", "Official YOLO26m.pt (Bundled)")
    model_path = data.get("detection_model_path", app.config.yolo_model_path)

    if split not in ["train", "val", "test"]:
        split = "train"
    if model_mode not in {
        "Official YOLO26m.pt (Bundled)",
        "Custom YOLO (v5/v7/v8/v9/v11/v26)",
        "Custom RF-DETR",
    }:
        model_mode = "Official YOLO26m.pt (Bundled)"
    app.det_model_mode.set(model_mode)
    if model_path:
        app.yolo_path.set(model_path)
        app._register_model_path(model_path)

    if not restore_project:
        return

    if not project_root or not os.path.exists(project_root):
        return

    app.current_split = split
    app.combo_split.set(split)
    app.load_project_from_path(project_root, preferred_image=image_name, save_session=False)
