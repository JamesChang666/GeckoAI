from __future__ import annotations

import copy
import os
from typing import Any
from tkinter import messagebox

from ai_labeller.features import image_utils
from ai_labeller.constants import LANG_MAP, COLORS


def load_image(app: Any) -> None:
    if not app.image_files:
        return

    path = app.image_files[app.current_idx]
    prev_path = getattr(app, "_loaded_image_path", None)
    app.update_info_text()
    if getattr(app, "img_pil", None) is not None:
        try:
            app.img_pil.close()
        except Exception:
            pass
        app.img_pil = None
    app.img_pil = image_utils.open_image_as_pil(app, path, convert=None, parent=app.root)
    if app.img_pil is None:
        return

    prev_rects = copy.deepcopy(app.rects)
    if prev_path and prev_path != path:
        app._prev_image_rects = copy.deepcopy(prev_rects)
    prev_selected_indices = app._get_selected_indices()
    prev_selected_rects = [copy.deepcopy(app.rects[idx]) for idx in prev_selected_indices if 0 <= idx < len(app.rects)]
    app.rects = []
    app.history_manager.clear()
    app.selected_idx = None
    app.selected_indices = set()
    app.active_handle = None
    app.active_rotate_handle = False
    app.rotate_drag_offset_deg = 0.0
    app.is_moving_box = False
    app.is_drag_selecting = False
    app.drag_start = None
    app.temp_rect_coords = None
    app.select_rect_coords = None

    # Load labels for current image
    base = os.path.splitext(os.path.basename(path))[0]
    label_path = f"{app.project_root}/labels/{app.current_split}/{base}.txt"
    rot_meta_path = app._rotation_meta_path_for_label(label_path)

    label_exists = os.path.exists(label_path) and os.path.getsize(label_path) > 0
    propagate_mode = app.var_propagate_mode.get()
    should_propagate = False
    if app.var_propagate.get():
        if propagate_mode == "if_missing":
            should_propagate = not label_exists
        else:
            should_propagate = True

    loaded_rects: list[list[float]] = []
    if label_exists:
        W, H = app.img_pil.width, app.img_pil.height
        has_inline_angle = False
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) == 9:
                        c = int(float(parts[0]))
                        pts_norm = list(map(float, parts[1:9]))
                        loaded_rects.append(app.obb_norm_to_rect(pts_norm, W, H, c))
                    elif len(parts) >= 5:
                        c, cx, cy, w, h = map(float, parts[:5])
                        angle_deg = float(parts[5]) if len(parts) >= 6 else 0.0
                        has_inline_angle = has_inline_angle or len(parts) >= 6
                        loaded_rects.append([
                            (cx - w / 2) * W,
                            (cy - h / 2) * H,
                            (cx + w / 2) * W,
                            (cy + h / 2) * H,
                            int(c),
                            app.normalize_angle_deg(angle_deg),
                        ])
            if loaded_rects and not has_inline_angle:
                loaded_angles = app._read_rotation_meta_angles(rot_meta_path)
                if loaded_angles and len(loaded_angles) == len(loaded_rects):
                    for rect, angle in zip(loaded_rects, loaded_angles):
                        app.set_rect_angle_deg(rect, angle)
        except Exception:
            app.logger.exception("Failed to parse label file: %s", label_path)
            messagebox.showerror("Error", f"Failed to read label file: {label_path}")
            loaded_rects = []

    app.rects = loaded_rects

    if should_propagate:
        source_rects = prev_rects
        if propagate_mode == "selected":
            source_rects = prev_selected_rects
        propagated_rects = [app.clamp_box(copy.deepcopy(r)) for r in source_rects]
        if propagate_mode == "always":
            app.rects = propagated_rects
        elif propagate_mode == "selected":
            app.rects.extend(propagated_rects)
        elif not label_exists:
            app.rects = propagated_rects

    if not label_exists and not app.rects:
        if app.var_auto_yolo.get():
            app.run_yolo_detection()

    app.fit_image_to_canvas()
    app.save_session_state()
    app._loaded_image_path = path
