from __future__ import annotations

import json
import os
from typing import Any


def build_removed_path(app: Any, kind: str, src_path: str) -> str:
    ext = os.path.splitext(src_path)[1]
    base = os.path.splitext(os.path.basename(src_path))[0]
    dst_dir = os.path.join(app.project_root, "removed", app.current_split, kind)
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, f"{base}{ext}")
    if not os.path.exists(dst_path):
        return dst_path

    i = 1
    while True:
        candidate = os.path.join(dst_dir, f"{base}_{i}{ext}")
        if not os.path.exists(candidate):
            return candidate
        i += 1


def unique_target_path(target_path: str) -> str:
    if not os.path.exists(target_path):
        return target_path
    folder = os.path.dirname(target_path)
    base, ext = os.path.splitext(os.path.basename(target_path))
    i = 1
    while True:
        candidate = os.path.join(folder, f"{base}_{i}{ext}")
        if not os.path.exists(candidate):
            return candidate
        i += 1


def rotation_meta_path_for_label(label_path: str) -> str:
    return f"{label_path}.rot.json"


def read_rotation_meta_angles(app: Any, rot_meta_path: str) -> list[float] | None:
    if not os.path.isfile(rot_meta_path):
        return None
    try:
        with open(rot_meta_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        raw_angles = payload.get("angles_deg", [])
        if not isinstance(raw_angles, list):
            return None
        out: list[float] = []
        for angle in raw_angles:
            out.append(app.normalize_angle_deg(float(angle)))
        return out
    except Exception:
        app.logger.exception("Failed to read rotation meta: %s", rot_meta_path)
        return None
