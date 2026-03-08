import os
import shutil
from typing import Any
import tkinter as tk
from tkinter import messagebox

from ai_labeller.core import atomic_write_text, atomic_write_json


def save_current(app: Any) -> None:
    if not app.project_root or not app.img_pil:
        return

    path = app.image_files[app.current_idx]
    base = os.path.splitext(os.path.basename(path))[0]
    W, H = app.img_pil.width, app.img_pil.height

    label_path = f"{app.project_root}/labels/{app.current_split}/{base}.txt"
    rot_meta_path = app._rotation_meta_path_for_label(label_path)

    lines = []
    angles_deg: list[float] = []
    for rect in app.rects:
        cx = (rect[0] + rect[2]) / 2 / W
        cy = (rect[1] + rect[3]) / 2 / H
        w = (rect[2] - rect[0]) / W
        h = (rect[3] - rect[1]) / H
        lines.append(f"{int(rect[4])} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        angles_deg.append(app.get_rect_angle_deg(rect))
    if not lines:
        if os.path.exists(label_path):
            try:
                os.remove(label_path)
            except OSError:
                app.logger.exception("Failed to remove empty label file: %s", label_path)
        if os.path.exists(rot_meta_path):
            try:
                os.remove(rot_meta_path)
            except OSError:
                app.logger.exception("Failed to remove empty rotation meta file: %s", rot_meta_path)
        return
    try:
        atomic_write_text(label_path, "".join(lines))
        if any(abs(a) > 1e-3 for a in angles_deg):
            atomic_write_json(rot_meta_path, {"version": 1, "angles_deg": angles_deg})
        elif os.path.exists(rot_meta_path):
            os.remove(rot_meta_path)
    except Exception:
        app.logger.exception("Failed to save label file: %s", label_path)
        messagebox.showerror("Error", f"Failed to save label file:\n{label_path}")
        return


def _reindex_dataset_labels_after_class_delete(app: Any, deleted_idx: int) -> None:
    if not app.project_root:
        return

    label_files = app._glob_label_files(app.project_root)

    for lbl_path in label_files:
        try:
            with open(lbl_path, "r", encoding="utf-8") as f:
                raw_lines = f.readlines()
        except Exception:
            app.logger.exception("Failed to read label file while deleting class: %s", lbl_path)
            continue

        updated_lines: list[str] = []
        for raw in raw_lines:
            line = raw.strip()
            if not line:
                updated_lines.append(raw if raw.endswith("\n") else raw + "\n")
                continue
            parts = line.split()
            if len(parts) < 5:
                updated_lines.append(raw if raw.endswith("\n") else raw + "\n")
                continue
            try:
                cid = int(float(parts[0]))
            except ValueError:
                updated_lines.append(raw if raw.endswith("\n") else raw + "\n")
                continue

            if cid == deleted_idx:
                continue
            if cid > deleted_idx:
                parts[0] = str(cid - 1)
                updated_lines.append(" ".join(parts) + "\n")
            else:
                updated_lines.append(raw if raw.endswith("\n") else raw + "\n")

        if updated_lines:
            try:
                atomic_write_text(lbl_path, "".join(updated_lines))
            except Exception:
                app.logger.exception("Failed to write updated label file: %s", lbl_path)
        else:
            try:
                if os.path.exists(lbl_path):
                    os.remove(lbl_path)
            except OSError:
                app.logger.exception("Failed to remove empty label file: %s", lbl_path)


def remove_current_from_split(app: Any) -> None:
    if not app.image_files:
        messagebox.showinfo(
            app.LANG_MAP[app.lang]["title"],
            app.LANG_MAP[app.lang].get("remove_none", "No image to remove.")
        )
        return
    if not app.project_root:
        return

    confirm_msg = app.LANG_MAP[app.lang].get(
        "remove_confirm",
        "Remove current image from {split}?"
    ).format(split=app.current_split)
    if not messagebox.askyesno(app.LANG_MAP[app.lang]["title"], confirm_msg):
        return

    image_path = app.image_files[app.current_idx]
    image_name = os.path.basename(image_path)
    base = os.path.splitext(image_name)[0]

    try:
        moved_image_path = app._build_removed_path("images", image_path)
        shutil.move(image_path, moved_image_path)
        label_dir = os.path.join(app.project_root, "labels", app.current_split)
        for ext in (".txt", ".json"):
            label_path = os.path.join(label_dir, f"{base}{ext}")
            if os.path.exists(label_path):
                moved_label_path = app._build_removed_path("labels", label_path)
                shutil.move(label_path, moved_label_path)
                if ext == ".txt":
                    rot_path = app._rotation_meta_path_for_label(label_path)
                    if os.path.exists(rot_path):
                        moved_rot_path = app._build_removed_path("labels", rot_path)
                        shutil.move(rot_path, moved_rot_path)
    except Exception as e:
        messagebox.showerror("Error", str(e))
        return

    del app.image_files[app.current_idx]
    if app.current_idx >= len(app.image_files):
        app.current_idx = max(0, len(app.image_files) - 1)

    if app.image_files:
        app.load_img()
    else:
        app.img_pil = None
        app.img_tk = None
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
        app.update_info_text()
        app.render()

    app.save_session_state()
    done_msg = app.LANG_MAP[app.lang].get("remove_done", "Removed: {name}").format(name=image_name)
    messagebox.showinfo(app.LANG_MAP[app.lang]["title"], done_msg)


def restore_removed_file_by_name(app: Any, filename: str) -> None:
    removed_img_path = os.path.join(app.project_root, "removed", app.current_split, "images", filename)
    if not os.path.exists(removed_img_path):
        return

    split_img_dir = os.path.join(app.project_root, "images", app.current_split)
    os.makedirs(split_img_dir, exist_ok=True)
    target_img_path = app._unique_target_path(os.path.join(split_img_dir, filename))

    base = os.path.splitext(filename)[0]
    removed_lbl_dir = os.path.join(app.project_root, "removed", app.current_split, "labels")
    split_lbl_dir = os.path.join(app.project_root, "labels", app.current_split)
    os.makedirs(split_lbl_dir, exist_ok=True)

    if app.image_files and app.img_pil:
        app.save_current()

    try:
        shutil.move(removed_img_path, target_img_path)
        for ext in (".txt", ".json"):
            removed_lbl_path = os.path.join(removed_lbl_dir, f"{base}{ext}")
            if os.path.exists(removed_lbl_path):
                target_lbl_path = app._unique_target_path(
                    os.path.join(split_lbl_dir, f"{base}{ext}")
                )
                shutil.move(removed_lbl_path, target_lbl_path)
                if ext == ".txt":
                    removed_rot_path = app._rotation_meta_path_for_label(removed_lbl_path)
                    if os.path.exists(removed_rot_path):
                        target_rot_path = app._rotation_meta_path_for_label(target_lbl_path)
                        target_rot_path = app._unique_target_path(target_rot_path)
                        shutil.move(removed_rot_path, target_rot_path)
    except Exception as e:
        messagebox.showerror("Error", str(e))
        return

    app.load_split_data()
    if target_img_path in app.image_files:
        app.current_idx = app.image_files.index(target_img_path)


def rotate_selected_boxes(app: Any, delta_deg: float) -> None:
    selected = app._get_selected_indices()
    if not selected:
        return
    app.push_history()
    for idx in selected:
        rect = app.rects[idx]
        app.set_rect_angle_deg(rect, app.get_rect_angle_deg(rect) + delta_deg)
    app.render()
