import hashlib
import os
import shutil
import tempfile
from typing import Any

import numpy as np

from ai_labeller.features import image_utils


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
