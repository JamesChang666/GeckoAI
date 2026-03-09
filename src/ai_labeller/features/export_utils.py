import os
import random
import shutil
import datetime
from typing import Any, List, Tuple

from tkinter import messagebox

from ai_labeller.core import atomic_write_text, atomic_write_json
from ai_labeller.features import image_utils


def _iter_export_images(app) -> List[Tuple[str, str, str]]:
    entries: List[Tuple[str, str, str]] = []
    if not app.project_root:
        return entries

    split_roots = [s for s in ("train", "val", "test") if os.path.isdir(f"{app.project_root}/images/{s}")]
    if split_roots:
        for split in split_roots:
            for img_path in app._list_split_images_for_root(app.project_root, split):
                base = os.path.splitext(os.path.basename(img_path))[0]
                lbl_path = f"{app.project_root}/labels/{split}/{base}.txt"
                entries.append((split, img_path, lbl_path))
        return entries

    # Flat image folder mode
    for img_path in app._glob_image_files(app.project_root):
        base = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = f"{app.project_root}/labels/train/{base}.txt"
        entries.append(("train", img_path, lbl_path))
    return entries


def export_all_by_selected_format(app) -> None:
    if not app.project_root:
        messagebox.showwarning(app.LANG_MAP[app.lang]["title"], app.LANG_MAP[app.lang].get("export_no_project"), parent=app.root)
        return
    if app.image_files and app.img_pil:
        app.save_current()

    out_dir = app.filedialog.askdirectory(
        parent=app.root,
        title=app.LANG_MAP[app.lang].get("select_export_parent_folder", "Select Export Parent Folder")
    )
    if not out_dir:
        return
    out_dir = out_dir.replace("\\", "/")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"{out_dir}/export_all_{timestamp}"
    if os.path.exists(out_dir):
        suffix = 1
        while os.path.exists(f"{out_dir}_{suffix}"):
            suffix += 1
        out_dir = f"{out_dir}_{suffix}"
    os.makedirs(out_dir, exist_ok=True)

    fmt = app.var_export_format.get()
    try:
        if fmt == "YOLO (.txt)":
            count = _export_all_yolo(app, out_dir)
            create_val = messagebox.askyesno(
                app.LANG_MAP[app.lang]["title"],
                app.LANG_MAP[app.lang].get("export_create_val_prompt", "Create validation set with aug_for_val?"),
                parent=app.root,
            )
            if create_val:
                if not app.HAS_CV2:
                    messagebox.showwarning(
                        app.LANG_MAP[app.lang]["title"],
                        app.LANG_MAP[app.lang].get("export_val_disabled_cv2", "OpenCV is not available. Validation augmentation skipped."),
                        parent=app.root,
                    )
                else:
                    val_count = _export_val_with_aug_for_val(app, out_dir)
                    if val_count > 0:
                        _write_export_yolo_dataset_yaml(app, out_dir, val_rel_path="images/val")
                        messagebox.showinfo(
                            app.LANG_MAP[app.lang]["title"],
                            app.LANG_MAP[app.lang].get("export_val_done", "Validation set created: {count} images").format(count=val_count),
                            parent=app.root,
                        )
                    else:
                        messagebox.showwarning(
                            app.LANG_MAP[app.lang]["title"],
                            app.LANG_MAP[app.lang].get("export_val_empty", "No train images found for validation augmentation."),
                            parent=app.root,
                        )
        else:
            count = _export_all_json(app, out_dir)
        messagebox.showinfo(
            app.LANG_MAP[app.lang]["title"],
            app.LANG_MAP[app.lang].get("export_done", "Export completed: {count} images\nOutput: {path}").format(
                count=count,
                path=out_dir,
            ),
            parent=app.root,
        )
    except Exception as exc:
        app.logger.exception("Export failed")
        messagebox.showerror(
            app.LANG_MAP[app.lang]["title"],
            app.LANG_MAP[app.lang].get("export_failed", "Export failed: {err}").format(err=exc),
            parent=app.root,
        )


def _export_all_yolo(app, out_dir: str) -> int:
    count = 0
    dst_img_dir = f"{out_dir}/images/train"
    dst_lbl_dir = f"{out_dir}/labels/train"
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    def build_unique_stem(split: str, stem: str, ext: str) -> str:
        candidate = stem
        target_img = f"{dst_img_dir}/{candidate}{ext}"
        if not os.path.exists(target_img):
            return candidate
        candidate = f"{split}_{stem}"
        target_img = f"{dst_img_dir}/{candidate}{ext}"
        if not os.path.exists(target_img):
            return candidate
        i = 1
        while os.path.exists(f"{dst_img_dir}/{candidate}_{i}{ext}"):
            i += 1
        return f"{candidate}_{i}"

    for split, img_path, lbl_path in _iter_export_images(app):
        img_name = os.path.basename(img_path)
        stem, ext = os.path.splitext(img_name)
        target_stem = build_unique_stem(split, stem, ext)
        shutil.copy2(img_path, f"{dst_img_dir}/{target_stem}{ext}")
        if os.path.isfile(lbl_path):
            shutil.copy2(lbl_path, f"{dst_lbl_dir}/{target_stem}.txt")
            rot_meta_path = app._rotation_meta_path_for_label(lbl_path)
            if os.path.isfile(rot_meta_path):
                shutil.copy2(rot_meta_path, app._rotation_meta_path_for_label(f"{dst_lbl_dir}/{target_stem}.txt"))
        count += 1
    _write_export_yolo_dataset_yaml(app, out_dir)
    return count


def _write_export_yolo_dataset_yaml(app, out_dir: str, val_rel_path: str = "images/train") -> None:
    yaml_path = f"{out_dir}/dataset.yaml"
    abs_out_dir = os.path.abspath(out_dir).replace("\\", "/")
    lines = [
        f"path: {abs_out_dir}",
        "train: images/train",
        f"val: {val_rel_path}",
        "nc: " + str(len(app.class_names)),
        "names:",
    ]
    for idx, cls_name in enumerate(app.class_names):
        safe_name = cls_name.replace('"', '\\"')
        lines.append(f'  {idx}: "{safe_name}"')
    atomic_write_text(yaml_path, "\n".join(lines) + "\n")


def _export_val_with_aug_for_val(app, out_dir: str) -> int:
    src_img_dir = f"{out_dir}/images/train"
    src_lbl_dir = f"{out_dir}/labels/train"
    dst_img_dir = f"{out_dir}/images/val"
    dst_lbl_dir = f"{out_dir}/labels/val"
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    if not os.path.isdir(src_img_dir):
        return 0

    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG")
    img_files = [
        f for f in os.listdir(src_img_dir)
        if os.path.isfile(os.path.join(src_img_dir, f)) and f.endswith(valid_ext)
    ]
    if not img_files:
        return 0

    def augment_brightness(image):
        hsv = app.cv2.cvtColor(image, app.cv2.COLOR_BGR2HSV)
        h, s, v = app.cv2.split(hsv)
        brightness_factor = random.uniform(0.6, 1.4)
        v = app.cv2.multiply(v, brightness_factor)
        v = app.cv2.convertScaleAbs(v)
        return app.cv2.cvtColor(app.cv2.merge((h, s, v)), app.cv2.COLOR_HSV2BGR)

    count = 0
    for img_name in img_files:
        src_img_path = os.path.join(src_img_dir, img_name)
        img = image_utils.read_cv2_image(src_img_path)
        if img is None:
            continue
        aug_img = augment_brightness(img)
        new_img_name = f"aug_{img_name}"
        image_utils.write_cv2_image(os.path.join(dst_img_dir, new_img_name), aug_img)

        label_name = os.path.splitext(img_name)[0] + ".txt"
        src_label_path = os.path.join(src_lbl_dir, label_name)
        if os.path.exists(src_label_path):
            shutil.copy2(src_label_path, os.path.join(dst_lbl_dir, f"aug_{label_name}"))
        count += 1
    return count


def _export_all_json(app, out_dir: str) -> int:
    count = 0
    for split, img_path, lbl_path in _iter_export_images(app):
        dst_img_dir = f"{out_dir}/images/{split}"
        dst_ann_dir = f"{out_dir}/annotations/{split}"
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_ann_dir, exist_ok=True)

        img_name = os.path.basename(img_path)
        base = os.path.splitext(img_name)[0]
        shutil.copy2(img_path, f"{dst_img_dir}/{img_name}")

        im = image_utils.open_image_as_pil(app, img_path, convert=None, parent=app.root)
        if im is None:
            continue
        width, height = im.width, im.height
        try:
            im.close()
        except Exception:
            pass

        anns: list[dict[str, Any]] = []
        angles_from_meta: list[float] = []
        if os.path.isfile(app._rotation_meta_path_for_label(lbl_path)):
            angles_from_meta = app._read_rotation_meta_angles(app._rotation_meta_path_for_label(lbl_path)) or []
        ann_idx = 0
        if os.path.isfile(lbl_path):
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 9:
                        cls_id = int(float(parts[0]))
                        pts_norm = list(map(float, parts[1:9]))
                        rect = app.obb_norm_to_rect(pts_norm, width, height, cls_id)
                        x1, y1, x2, y2 = rect[:4]
                        angle_deg = app.get_rect_angle_deg(rect)
                        cx = (x1 + x2) / 2 / width
                        cy = (y1 + y2) / 2 / height
                        w = (x2 - x1) / width
                        h = (y2 - y1) / height
                        cls_name = app.class_names[cls_id] if 0 <= cls_id < len(app.class_names) else str(cls_id)
                        anns.append({
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "bbox_xyxy": [x1, y1, x2, y2],
                            "bbox_yolo": [cx, cy, w, h],
                            "obb_yolo": pts_norm,
                            "angle_deg": app.normalize_angle_deg(angle_deg),
                        })
                        ann_idx += 1
                        continue
                    if len(parts) < 5:
                        continue
                    cls_id = int(float(parts[0]))
                    cx, cy, w, h = map(float, parts[1:5])
                    angle_deg = float(parts[5]) if len(parts) >= 6 else 0.0
                    if len(parts) < 6 and ann_idx < len(angles_from_meta):
                        angle_deg = float(angles_from_meta[ann_idx])
                    x1 = (cx - w / 2) * width
                    y1 = (cy - h / 2) * height
                    x2 = (cx + w / 2) * width
                    y2 = (cy + h / 2) * height
                    cls_name = app.class_names[cls_id] if 0 <= cls_id < len(app.class_names) else str(cls_id)
                    anns.append({
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "bbox_yolo": [cx, cy, w, h],
                        "angle_deg": app.normalize_angle_deg(angle_deg),
                    })
                    ann_idx += 1

        payload = {
            "image": img_name,
            "split": split,
            "width": width,
            "height": height,
            "annotations": anns,
        }
        atomic_write_json(f"{dst_ann_dir}/{base}.json", payload)
        count += 1
    return count


def export_full_coco(app) -> None:
    messagebox.showinfo("Export", "COCO export will be implemented.")
