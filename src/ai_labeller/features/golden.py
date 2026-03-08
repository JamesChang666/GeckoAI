import glob
import json
import os
import shutil
import tempfile
from typing import Any

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from PIL import Image, ImageTk

from ai_labeller.core import atomic_write_text
from ai_labeller.features import image_utils


def load_detect_background_cut_bundle(app, golden_dir: str) -> dict[str, Any] | None:
    try:
        import cv2  # noqa: F401
    except Exception:
        return None
    try:
        from ai_labeller.cut_background_detect import load_background_cut_bundle

        preferred_root = os.path.join(golden_dir, "background_cut_golden")
        search_roots: list[str] = []
        if os.path.isdir(preferred_root):
            search_roots.append(preferred_root)
        search_roots.append(golden_dir)
        for root in search_roots:
            bundle = load_background_cut_bundle(root)
            if bundle is None:
                continue
            return {
                "bundle": bundle,
                "root": bundle.root_dir,
                "rules_path": bundle.rules_path,
                "template_path": bundle.template_path,
            }
    except Exception:
        app.logger.exception("Failed to load background-cut golden bundle from: %s", golden_dir)
    return None





def parse_yolo_label_file(label_path: str) -> list[tuple[int, tuple[float, float, float, float]]]:
    items: list[tuple[int, tuple[float, float, float, float]]] = []
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return items
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
        except Exception:
            continue
        if len(parts) >= 9:
            try:
                pts = list(map(float, parts[1:9]))
            except Exception:
                continue
            xs = [pts[0], pts[2], pts[4], pts[6]]
            ys = [pts[1], pts[3], pts[5], pts[7]]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
        else:
            try:
                cx, cy, w, h = map(float, parts[1:5])
            except Exception:
                continue
            x1, y1 = cx - w / 2, cy - h / 2
            x2, y2 = cx + w / 2, cy + h / 2
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        items.append((cid, (x1, y1, x2, y2)))
    return items


def find_dataset_yaml_for_label(label_path: str) -> str | None:
    path = os.path.abspath(label_path)
    cur = os.path.dirname(path)
    for _ in range(8):
        candidate = os.path.join(cur, "dataset.yaml")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None


def find_dataset_yaml_in_folder(folder: str) -> str | None:
    candidates = [os.path.join(folder, "dataset.yaml"), os.path.join(folder, "data.yaml")]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def load_mapping_from_dataset_yaml(yaml_path: str) -> dict[int, str]:
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return {}

    mapping: dict[int, str] = {}
    lines = text.splitlines()
    in_names = False
    seq_names: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("names:"):
            in_names = True
            inline = line.split(":", 1)[1].strip()
            if inline.startswith("[") and inline.endswith("]"):
                inner = inline[1:-1]
                seq_names.extend([x.strip().strip("'\"") for x in inner.split(",") if x.strip()])
                in_names = False
            continue
        if not in_names:
            continue
        if line.startswith("-"):
            seq_names.append(line[1:].strip().strip("'\""))
            continue
        if ":" in line:
            left, right = line.split(":", 1)
            left = left.strip()
            right = right.strip().strip("'\"")
            if left.isdigit():
                mapping[int(left)] = right
                continue
        if not line.startswith(("-", "#")):
            break
    if not mapping and seq_names:
        mapping = {i: name for i, name in enumerate(seq_names)}
    return mapping


def find_golden_id_config_in_folder(folder: str) -> str | None:
    candidates = [
        os.path.join(folder, "id_config.json"),
        os.path.join(folder, "id_class.json"),
        os.path.join(folder, "golden_id.json"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def load_golden_id_config(json_path: str | None) -> dict[str, Any] | None:
    if not json_path or not os.path.isfile(json_path):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    raw_id = payload.get("id_class_id")
    cid = None
    if raw_id is not None:
        try:
            cid = int(raw_id)
        except Exception:
            cid = None
    raw_sub_id = payload.get("sub_id_class_id")
    sub_cid = None
    if raw_sub_id is not None:
        try:
            sub_cid = int(raw_sub_id)
        except Exception:
            sub_cid = None
    if cid is None and sub_cid is None:
        return None
    name = str(payload.get("id_class_name", "")).strip()
    sub_name = str(payload.get("sub_id_class_name", "")).strip()
    return {
        "id_class_id": cid,
        "id_class_name": name or None,
        "sub_id_class_id": sub_cid,
        "sub_id_class_name": sub_name or None,
        "id_config_path": os.path.abspath(json_path),
    }


def prompt_golden_id_classes(
    app,
    class_mapping: dict[int, str],
    parent: Any = None,
) -> tuple[tuple[int, str] | None, tuple[int, str] | None]:
    if not class_mapping:
        return None, None
    max_idx = max(class_mapping.keys())
    options = "\n".join(f"{idx}: {class_mapping[idx]}" for idx in sorted(class_mapping.keys()))
    id_prompt = "Select class ID for OCR image ID extraction in detect mode.\n-1: Disable OCR ID\n\n" f"{options}"
    selected_id = simpledialog.askinteger(
        "Golden ID Class",
        id_prompt,
        parent=parent or app.root,
        minvalue=-1,
        maxvalue=max_idx,
        initialvalue=-1,
    )
    id_choice = None
    if selected_id is not None and selected_id >= 0:
        id_choice = (selected_id, str(class_mapping.get(selected_id, selected_id)))

    sub_prompt = "Select class ID for OCR sub ID extraction in detect mode.\n-1: Disable OCR Sub ID\n\n" f"{options}"
    selected_sub_id = simpledialog.askinteger(
        "Golden Sub ID Class",
        sub_prompt,
        parent=parent or app.root,
        minvalue=-1,
        maxvalue=max_idx,
        initialvalue=-1,
    )
    sub_id_choice = None
    if selected_sub_id is not None and selected_sub_id >= 0:
        sub_id_choice = (selected_sub_id, str(class_mapping.get(selected_sub_id, selected_sub_id)))

    return id_choice, sub_id_choice


def write_golden_id_config(
    folder: str,
    class_id: int | None,
    class_name: str | None,
    sub_id_class_id: int | None = None,
    sub_id_class_name: str | None = None,
) -> str:
    cfg_path = os.path.join(folder, "id_config.json")
    payload = {
        "id_class_id": int(class_id) if class_id is not None else None,
        "id_class_name": str(class_name) if class_name else "",
        "sub_id_class_id": int(sub_id_class_id) if sub_id_class_id is not None else None,
        "sub_id_class_name": str(sub_id_class_name) if sub_id_class_name else "",
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    return cfg_path


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 1e-9:
        return 0.0
    return inter / union


def evaluate_golden_match(app, result0: Any) -> tuple[str | None, str]:
    if app.detect_run_mode_var.get().strip().lower() != "golden" or app._detect_golden_sample is None:
        app._detect_last_cut_piece_count = 0
        app._detect_last_ocr_id = ""
        app._detect_last_ocr_sub_id = ""
        return None, ""
    targets = app._detect_golden_sample.get("targets") or []
    if not targets:
        app._detect_last_ocr_id = ""
        app._detect_last_ocr_sub_id = ""
        return "FAIL", "golden targets missing"
    mode = app.detect_golden_mode_var.get().strip().lower()
    iou_thr = float(app.detect_golden_iou_var.get())
    h, w = getattr(result0, "orig_shape", (0, 0))
    if h <= 0 or w <= 0:
        app._detect_last_ocr_id = ""
        app._detect_last_ocr_sub_id = ""
        return "FAIL", "invalid frame shape"

    boxes = getattr(result0, "boxes", None)
    if boxes is None or getattr(boxes, "xyxy", None) is None or getattr(boxes, "cls", None) is None:
        app._detect_last_ocr_id = ""
        app._detect_last_ocr_sub_id = ""
        return "FAIL", "no detections"

    det_xyxy = boxes.xyxy.tolist()
    det_cls = boxes.cls.tolist()
    names_map = getattr(result0, "names", {}) or {}

    def normalize_name(name: str) -> str:
        return str(name).strip().lower()

    def det_name_for_cid(cid: int) -> str:
        if isinstance(names_map, dict):
            return str(names_map.get(cid, cid))
        if isinstance(names_map, (list, tuple)) and 0 <= cid < len(names_map):
            return str(names_map[cid])
        return str(cid)

    matched_targets = 0
    best_ious: list[float] = []
    for target in targets:
        rect_norm = target.get("rect_norm")
        if rect_norm is None:
            continue
        tgt_class_id = target.get("class_id")
        tgt_class_name = normalize_name(target.get("class_name")) if target.get("class_name") else ""
        target_matched = False
        target_best_iou = 0.0

        for i, box in enumerate(det_xyxy):
            det_norm = (
                float(box[0]) / float(w),
                float(box[1]) / float(h),
                float(box[2]) / float(w),
                float(box[3]) / float(h),
            )
            iou = bbox_iou(rect_norm, det_norm)
            target_best_iou = max(target_best_iou, iou)
            cid = int(det_cls[i]) if i < len(det_cls) else -1
            dname = normalize_name(det_name_for_cid(cid))

            class_match = False
            if tgt_class_name:
                class_match = dname == tgt_class_name
            elif tgt_class_id is not None:
                class_match = cid == int(tgt_class_id)

            pos_match = iou >= iou_thr
            if mode == "class" and class_match:
                target_matched = True
                break
            if mode == "position" and pos_match:
                target_matched = True
                break
            if mode == "both" and class_match and pos_match:
                target_matched = True
                break

        best_ious.append(target_best_iou)
        if target_matched:
            matched_targets += 1

    total_targets = len(targets)
    ocr_id = app._extract_ocr_id_from_result(result0)
    ocr_sub_id = app._extract_ocr_sub_id_from_result(result0)
    app._detect_last_ocr_id = ocr_id
    app._detect_last_ocr_sub_id = ocr_sub_id
    avg_iou = sum(best_ious) / max(1, len(best_ious))
    msg = f"{matched_targets}/{total_targets} matched, avg IoU={avg_iou:.3f}"
    if app._should_use_background_cut_detection():
        msg = f"{msg}, cut_pieces={int(getattr(app, '_detect_last_cut_piece_count', 0))}"
    if ocr_id:
        msg = f"{msg}, id={ocr_id}"
    if ocr_sub_id:
        msg = f"{msg}, sub_id={ocr_sub_id}"
    return ("PASS", msg) if matched_targets == total_targets else ("FAIL", msg)
