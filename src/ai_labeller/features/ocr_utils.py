from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except Exception:
    cv2 = None
    HAS_CV2 = False

# PaddleOCR support removed

try:
    import easyocr
    HAS_EASY_OCR = True
except Exception:
    easyocr = None
    HAS_EASY_OCR = False


def get_easy_ocr_engine(app: Any) -> Any:
    if not HAS_EASY_OCR:
        return None
    engine = getattr(app, "_easy_ocr_engine", None)
    if engine is not None:
        return engine
    try:
        engine = easyocr.Reader(["en"], gpu=False, verbose=False)
    except TypeError:
        engine = easyocr.Reader(["en"], gpu=False)
    except Exception:
        app.logger.exception("Failed to initialize EasyOCR engine")
        engine = None
    setattr(app, "_easy_ocr_engine", engine)
    return engine





def get_preferred_ocr_engine(app: Any) -> tuple[str | None, Any]:
    easy = get_easy_ocr_engine(app)
    if easy is not None:
        return "easyocr", easy
    return None, None


def _run_best_ocr_token(ocr_backend: str | None, ocr_engine: Any, gray_img: np.ndarray) -> tuple[str, str]:
    if ocr_engine is None or ocr_backend is None or not HAS_CV2:
        return "", "unrecognized"
    try:
        ocr_input = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        if ocr_backend == "easyocr":
            raw = ocr_engine.readtext(ocr_input, detail=1, paragraph=False)
        else:
            raw = ocr_engine.ocr(ocr_input, cls=False)
    except Exception:
        return "", "unrecognized"

    pairs: list[tuple[str, float]] = []

    def collect_pairs(node: Any) -> None:
        if isinstance(node, (list, tuple)):
            if (
                ocr_backend == "easyocr"
                and len(node) >= 3
                and isinstance(node[1], str)
            ):
                text_val = str(node[1] or "").strip()
                conf_val = float(node[2]) if node[2] is not None else 0.0
                pairs.append((text_val, conf_val))
                return
            if (
                len(node) >= 2
                and isinstance(node[1], (list, tuple))
                and len(node[1]) >= 1
                and isinstance(node[1][0], str)
            ):
                text_val = str(node[1][0] or "").strip()
                conf_val = float(node[1][1]) if len(node[1]) >= 2 else 0.0
                pairs.append((text_val, conf_val))
                return
            for item in node:
                collect_pairs(item)

    collect_pairs(raw)
    best_token = ""
    best_token_score = -1.0
    for text_raw, conf_raw in pairs:
        cleaned = re.sub(r"[^0-9A-Za-z_-]+", "", str(text_raw or "")).strip()[:128]
        if not cleaned:
            continue
        has_digit = any(ch.isdigit() for ch in cleaned)
        token_score = float(conf_raw) * 100.0 + len(cleaned) + (20.0 if has_digit else 0.0)
        if token_score > best_token_score:
            best_token_score = token_score
            best_token = cleaned
    if not best_token:
        return "", "unrecognized"
    return best_token, "ok"


def _extract_ocr_text_from_bundle_id_roi(app: Any, result0: Any) -> tuple[str, str]:
    bundle = getattr(app, "_detect_bg_cut_bundle", None)
    if bundle is None:
        return "", "disabled"
    id_mode = bool(getattr(bundle, "id_mode", False))
    roi = getattr(bundle, "id_roi_xywh", None)
    if not id_mode or not isinstance(roi, tuple) or len(roi) != 4:
        return "", "disabled"

    img = getattr(result0, "orig_img", None)
    if img is None or not HAS_CV2:
        return "", "unrecognized"
    h, w = img.shape[:2]
    try:
        rx, ry, rw, rh = [int(v) for v in roi]
    except Exception:
        return "", "missing_box"
    if rw <= 0 or rh <= 0:
        return "", "missing_box"
    x1 = max(0, min(w - 1, rx))
    y1 = max(0, min(h - 1, ry))
    x2 = max(0, min(w, rx + rw))
    y2 = max(0, min(h, ry + rh))
    if x2 <= x1 or y2 <= y1:
        return "", "missing_box"

    crop = img[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return "", "missing_box"
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if not HAS_EASY_OCR:
        if not getattr(app, "_detect_ocr_warning_shown", False):
            setattr(app, "_detect_ocr_warning_shown", True)
            app.logger.warning("OCR ID/Sub ID enabled but EasyOCR is not installed; skipping OCR.")
        return "", "unrecognized"
    ocr_backend, ocr_engine = get_preferred_ocr_engine(app)
    if ocr_engine is None:
        return "", "unrecognized"

    candidates = [
        bw,
        cv2.rotate(bw, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(bw, cv2.ROTATE_180),
        cv2.rotate(bw, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]
    for cand in candidates:
        text, state = _run_best_ocr_token(ocr_backend, ocr_engine, cand)
        if state == "ok" and text:
            return text, "ok"
    return "", "unrecognized"


def extract_ocr_text_with_state_from_result(
    app: Any,
    result0: Any,
    tgt_id: int | None,
    tgt_name: str,
) -> tuple[str, str]:
    tgt_name = str(tgt_name or "").strip().lower()
    if tgt_id is None and not tgt_name:
        return "", "disabled"

    if not HAS_EASY_OCR:
        if not getattr(app, "_detect_ocr_warning_shown", False):
            setattr(app, "_detect_ocr_warning_shown", True)
            app.logger.warning("OCR ID/Sub ID enabled but EasyOCR is not installed; skipping OCR.")
        return "", "unrecognized"
    ocr_backend, ocr_engine = get_preferred_ocr_engine(app)
    if ocr_engine is None:
        return "", "unrecognized"

    img = getattr(result0, "orig_img", None)
    if img is None:
        return "", "unrecognized"
    boxes = getattr(result0, "boxes", None)
    if boxes is None or getattr(boxes, "xyxy", None) is None or getattr(boxes, "cls", None) is None:
        return "", "missing_box"
    names_map = getattr(result0, "names", {}) or {}
    confs = boxes.conf.tolist() if getattr(boxes, "conf", None) is not None else []
    det_cls = boxes.cls.tolist()
    det_xyxy = boxes.xyxy.tolist()

    def norm_name(cid: int) -> str:
        if isinstance(names_map, dict):
            return str(names_map.get(cid, cid)).strip().lower()
        if isinstance(names_map, (list, tuple)) and 0 <= cid < len(names_map):
            return str(names_map[cid]).strip().lower()
        return str(cid).strip().lower()

    chosen_idx = None
    chosen_conf = -1.0
    for i, cid_raw in enumerate(det_cls):
        cid = int(cid_raw)
        class_match = (tgt_id is not None and cid == int(tgt_id)) or (tgt_name and norm_name(cid) == tgt_name)
        if not class_match:
            continue
        conf = float(confs[i]) if i < len(confs) else 0.0
        if conf > chosen_conf:
            chosen_conf = conf
            chosen_idx = i
    if chosen_idx is None or chosen_idx >= len(det_xyxy):
        return "", "missing_box"

    h, w = img.shape[:2]
    box = det_xyxy[chosen_idx]
    x1 = max(0, min(w - 1, int(math.floor(float(box[0])))))
    y1 = max(0, min(h - 1, int(math.floor(float(box[1])))))
    x2 = max(0, min(w, int(math.ceil(float(box[2])))))
    y2 = max(0, min(h, int(math.ceil(float(box[3])))))
    if x2 <= x1 or y2 <= y1:
        return "", "missing_box"

    crop = img[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return "", "missing_box"
    if HAS_CV2:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        return "", "unrecognized"

    candidates = [
        bw,
        cv2.rotate(bw, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(bw, cv2.ROTATE_180),
        cv2.rotate(bw, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]
    for cand in candidates:
        text, state = _run_best_ocr_token(ocr_backend, ocr_engine, cand)
        if state == "ok" and text:
            return text, "ok"
    return "", "unrecognized"


def extract_ocr_text_from_result(app: Any, result0: Any, tgt_id: int | None, tgt_name: str) -> str:
    text, _state = extract_ocr_text_with_state_from_result(app, result0, tgt_id, tgt_name)
    return text


def extract_ocr_id_from_result(app: Any, result0: Any) -> str:
    sample = getattr(app, "_detect_golden_sample", None) or {}
    return extract_ocr_text_from_result(
        app,
        result0,
        sample.get("id_class_id"),
        str(sample.get("id_class_name", "")),
    )


def extract_ocr_sub_id_from_result(app: Any, result0: Any) -> str:
    sample = getattr(app, "_detect_golden_sample", None) or {}
    return extract_ocr_text_from_result(
        app,
        result0,
        sample.get("sub_id_class_id"),
        str(sample.get("sub_id_class_name", "")),
    )


def extract_ocr_id_with_state_from_result(app: Any, result0: Any) -> tuple[str, str]:
    sample = getattr(app, "_detect_golden_sample", None) or {}
    text, state = extract_ocr_text_with_state_from_result(
        app,
        result0,
        sample.get("id_class_id"),
        str(sample.get("id_class_name", "")),
    )
    if state != "disabled":
        return text, state
    return _extract_ocr_text_from_bundle_id_roi(app, result0)


def extract_ocr_sub_id_with_state_from_result(app: Any, result0: Any) -> tuple[str, str]:
    sample = getattr(app, "_detect_golden_sample", None) or {}
    return extract_ocr_text_with_state_from_result(
        app,
        result0,
        sample.get("sub_id_class_id"),
        str(sample.get("sub_id_class_name", "")),
    )
