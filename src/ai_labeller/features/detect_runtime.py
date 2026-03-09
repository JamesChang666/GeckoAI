import hashlib
import os
import shutil
import tempfile
from typing import Any

import numpy as np

from ai_labeller.features import image_utils


def _parse_hex_color_to_bgr(hex_color: str, default_bgr: tuple[int, int, int] = (0, 255, 0)) -> tuple[int, int, int]:
    if not isinstance(hex_color, str):
        return default_bgr
    s = hex_color.strip()
    if not s:
        return default_bgr
    if s.startswith("#"):
        s = s[1:]
    if len(s) != 6:
        return default_bgr
    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
    except ValueError:
        return default_bgr
    return (b, g, r)


def _auto_class_color_bgr(class_key: str) -> tuple[int, int, int]:
    key = str(class_key or "class").encode("utf-8", errors="ignore")
    digest = hashlib.sha1(key).digest()
    # Keep colors bright enough for visibility.
    r = 64 + (digest[0] % 192)
    g = 64 + (digest[1] % 192)
    b = 64 + (digest[2] % 192)
    return (b, g, r)


def render_detect_result(app: Any, result0: Any, line_width: int = 1) -> Any:
    cv2_engine = getattr(app, "cv2", None)
    if cv2_engine is None or result0 is None:
        return result0.plot(line_width=line_width)
    class_color_map = getattr(app, "detect_class_color_map", {}) or {}

    orig_img = getattr(result0, "orig_img", None)
    if orig_img is None:
        return result0.plot(line_width=line_width)

    try:
        canvas = orig_img.copy()
    except Exception:
        return result0.plot(line_width=line_width)

    boxes = getattr(result0, "boxes", None)
    xyxy_attr = getattr(boxes, "xyxy", None) if boxes is not None else None
    if xyxy_attr is None:
        return canvas

    h, w = canvas.shape[:2]
    try:
        xyxy_list = xyxy_attr.tolist()
    except Exception:
        xyxy_list = []
    try:
        cls_list = boxes.cls.tolist() if getattr(boxes, "cls", None) is not None else []
    except Exception:
        cls_list = []
    try:
        conf_list = boxes.conf.tolist() if getattr(boxes, "conf", None) is not None else []
    except Exception:
        conf_list = []
    names = getattr(result0, "names", {}) or {}

    thickness = max(1, int(line_width))
    text_scale = 0.5
    text_thickness = max(1, thickness)
    for idx, coords in enumerate(xyxy_list):
        if not isinstance(coords, (list, tuple)) or len(coords) < 4:
            continue
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in coords[:4]]
        except Exception:
            continue
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        cls_name = ""
        if idx < len(cls_list):
            try:
                cid = int(float(cls_list[idx]))
                if isinstance(names, dict):
                    cls_name = str(names.get(cid, cid))
                elif isinstance(names, (list, tuple)) and 0 <= cid < len(names):
                    cls_name = str(names[cid])
                else:
                    cls_name = str(cid)
            except Exception:
                cls_name = ""
        class_key = str(cls_name or "").strip().lower()
        mapped_hex = class_color_map.get(class_key, "")
        if mapped_hex:
            box_color = _parse_hex_color_to_bgr(mapped_hex, default_bgr=_auto_class_color_bgr(class_key or f"idx_{idx}"))
        else:
            box_color = _auto_class_color_bgr(class_key or f"idx_{idx}")
        text_bg = box_color
        text_fg = (0, 0, 0) if (int(box_color[0]) + int(box_color[1]) + int(box_color[2])) > 382 else (255, 255, 255)
        cv2_engine.rectangle(canvas, (x1, y1), (x2, y2), box_color, thickness)

        conf_text = ""
        if idx < len(conf_list):
            try:
                conf_text = f"{float(conf_list[idx]):.2f}"
            except Exception:
                conf_text = ""
        label = cls_name if not conf_text else f"{cls_name} {conf_text}".strip()
        if not label:
            continue

        (tw, th), baseline = cv2_engine.getTextSize(
            label,
            cv2_engine.FONT_HERSHEY_SIMPLEX,
            text_scale,
            text_thickness,
        )
        by2 = max(th + baseline + 2, y1)
        by1 = max(0, by2 - th - baseline - 4)
        bx2 = min(w - 1, x1 + tw + 6)
        bx1 = max(0, x1)
        cv2_engine.rectangle(canvas, (bx1, by1), (bx2, by2), text_bg, -1)
        cv2_engine.putText(
            canvas,
            label,
            (bx1 + 3, max(th + 1, by2 - baseline - 2)),
            cv2_engine.FONT_HERSHEY_SIMPLEX,
            text_scale,
            text_fg,
            text_thickness,
            cv2_engine.LINE_AA,
        )

    return canvas


def should_use_background_cut_detection(app) -> bool:
    return (
        app.detect_run_mode_var.get().strip().lower() == "golden"
        and app._detect_bg_cut_bundle is not None
        and app.HAS_CV2
    )


def cleanup_detect_cut_piece_temp(app, remove_root: bool = False) -> None:
    last_dir = app._detect_cut_piece_last_dir
    if last_dir and os.path.isdir(last_dir):
        try:
            shutil.rmtree(last_dir, ignore_errors=True)
        except Exception:
            app.logger.exception("Failed to cleanup cut-piece temp dir: %s", last_dir)
    app._detect_cut_piece_last_dir = None
    app._detect_last_piece_paths = []
    app._detect_piece_index = 0
    if remove_root and app._detect_cut_piece_temp_root and os.path.isdir(app._detect_cut_piece_temp_root):
        try:
            shutil.rmtree(app._detect_cut_piece_temp_root, ignore_errors=True)
        except Exception:
            app.logger.exception("Failed to cleanup cut-piece temp root: %s", app._detect_cut_piece_temp_root)
        app._detect_cut_piece_temp_root = None
        app._detect_cut_piece_seq = 0
        app._detect_seen_cut_piece_hashes = set()


def ensure_detect_cut_piece_temp_root(app) -> str:
    if app._detect_cut_piece_temp_root and os.path.isdir(app._detect_cut_piece_temp_root):
        return app._detect_cut_piece_temp_root
    app._detect_cut_piece_temp_root = tempfile.mkdtemp(prefix="detect_cut_pieces_")
    app._detect_cut_piece_temp_root = app._detect_cut_piece_temp_root.replace("\\", "/")
    app._detect_cut_piece_seq = 0
    return app._detect_cut_piece_temp_root


def write_cut_pieces_to_temp_folder(app, pieces: list[np.ndarray]) -> str:
    root = ensure_detect_cut_piece_temp_root(app)
    cleanup_detect_cut_piece_temp(app, remove_root=False)
    app._detect_cut_piece_seq += 1
    run_dir = os.path.join(root, f"run_{app._detect_cut_piece_seq:06d}").replace("\\", "/")
    os.makedirs(run_dir, exist_ok=True)
    safe_pieces = [p for p in pieces if p is not None and getattr(p, "size", 0) > 0]
    if not safe_pieces:
        safe_pieces = [np.zeros((64, 64, 3), dtype=np.uint8)]
    piece_paths: list[str] = []
    for idx, piece in enumerate(safe_pieces, start=1):
        out_path = os.path.join(run_dir, f"piece_{idx:04d}.png").replace("\\", "/")
        image_utils.write_cv2_image(out_path, piece)
        piece_paths.append(out_path)
    app._detect_cut_piece_last_dir = run_dir
    app._detect_last_piece_paths = piece_paths
    app._detect_piece_index = 0
    return run_dir


def cut_piece_signature(app, piece: np.ndarray) -> str:
    ok, encoded = app.cv2.imencode(".png", piece)
    if not ok:
        return ""
    return hashlib.sha1(encoded.tobytes()).hexdigest()


def filter_unseen_cut_pieces(app, pieces: list[np.ndarray]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen = app._detect_seen_cut_piece_hashes
    for piece in pieces:
        if piece is None or getattr(piece, "size", 0) == 0:
            continue
        sig = cut_piece_signature(app, piece)
        if not sig:
            out.append(piece)
            continue
        if sig in seen:
            continue
        seen.add(sig)
        out.append(piece)
    return out


def prepare_background_cut_detect_source(app, source: Any) -> Any:
    if not should_use_background_cut_detection(app):
        app._detect_last_cut_piece_count = 0
        app._detect_last_piece_results = []
        app._detect_last_piece_paths = []
        app._detect_piece_index = 0
        cleanup_detect_cut_piece_temp(app, remove_root=False)
        return source
    try:
        from ai_labeller.cut_background_detect import extract_cut_pieces_from_bgr

        image_bgr = None
        if isinstance(source, np.ndarray):
            image_bgr = source
        elif isinstance(source, str):
            image_bgr = image_utils.read_cv2_image(source)
        if image_bgr is None or image_bgr.size == 0:
            app._detect_last_cut_piece_count = 0
            return write_cut_pieces_to_temp_folder(app, [])
        pieces = extract_cut_pieces_from_bgr(image_bgr, app._detect_bg_cut_bundle)
        new_pieces = filter_unseen_cut_pieces(app, pieces)
        app._detect_last_cut_piece_count = len(new_pieces)
        return write_cut_pieces_to_temp_folder(app, new_pieces)
    except Exception:
        app.logger.exception("Background-cut preprocessing failed; falling back to raw source.")
        app._detect_last_cut_piece_count = 0
        app._detect_last_piece_results = []
        app._detect_last_piece_paths = []
        app._detect_piece_index = 0
        cleanup_detect_cut_piece_temp(app, remove_root=False)
        return source


def select_primary_result_index(results: list[Any]) -> int:
    if not results:
        return 0
    best_idx = 0
    best_score = -1.0
    for idx, result in enumerate(results):
        boxes = getattr(result, "boxes", None)
        cls_vals = getattr(boxes, "cls", None) if boxes is not None else None
        conf_vals = getattr(boxes, "conf", None) if boxes is not None else None
        box_count = len(cls_vals.tolist()) if cls_vals is not None else 0
        conf_mean = 0.0
        if conf_vals is not None:
            conf_list = conf_vals.tolist()
            conf_mean = (sum(float(c) for c in conf_list) / len(conf_list)) if conf_list else 0.0
        score = float(box_count) * 1000.0 + conf_mean
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def run_detect_inference(app, source: Any) -> Any:
    prepared_source = prepare_background_cut_detect_source(app, source)
    try:
        results = app.yolo_model(
            prepared_source,
            verbose=False,
            device=app._detect_preferred_device,
            conf=float(app._detect_conf_threshold),
        )
        if should_use_background_cut_detection(app):
            app._detect_last_piece_results = list(results)
            best_idx = select_primary_result_index(app._detect_last_piece_results)
            if 0 <= best_idx < len(app._detect_last_piece_results):
                if best_idx != 0:
                    ordered = [app._detect_last_piece_results[best_idx]] + [
                        r for i, r in enumerate(app._detect_last_piece_results) if i != best_idx
                    ]
                    app._detect_last_piece_results = ordered
                return app._detect_last_piece_results
        else:
            app._detect_last_piece_results = []
        return results
    except RuntimeError as exc:
        if app._detect_preferred_device != "cpu" and app._is_cuda_kernel_compat_error(exc):
            app._force_cpu_detection = True
            app._detect_preferred_device = "cpu"
            results = app.yolo_model(
                prepared_source,
                verbose=False,
                device=app._detect_preferred_device,
                conf=float(app._detect_conf_threshold),
            )
            if should_use_background_cut_detection(app):
                app._detect_last_piece_results = list(results)
                best_idx = select_primary_result_index(app._detect_last_piece_results)
                if 0 <= best_idx < len(app._detect_last_piece_results):
                    if best_idx != 0:
                        ordered = [app._detect_last_piece_results[best_idx]] + [
                            r for i, r in enumerate(app._detect_last_piece_results) if i != best_idx
                        ]
                        app._detect_last_piece_results = ordered
                    return app._detect_last_piece_results
            else:
                app._detect_last_piece_results = []
            return results
        raise


def detect_class_counts(result0: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    names = getattr(result0, "names", {}) or {}
    boxes = getattr(result0, "boxes", None)
    if boxes is None or getattr(boxes, "cls", None) is None:
        return counts
    for cid in boxes.cls.tolist():
        cid_int = int(cid)
        if isinstance(names, dict):
            cls_name = names.get(cid_int, str(cid_int))
        elif isinstance(names, (list, tuple)) and 0 <= cid_int < len(names):
            cls_name = str(names[cid_int])
        else:
            cls_name = str(cid_int)
        counts[cls_name] = counts.get(cls_name, 0) + 1
    return counts
