import os
import shutil
import tempfile
from typing import Any

try:
    import cv2
except Exception:
    cv2 = None

import numpy as np
from PIL import Image, ImageTk


def scan_available_cameras(app: Any, max_probe: int = 6) -> list[int]:
    if not cv2:
        return []
    cams: list[int] = []
    for cam_idx in range(max_probe):
        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            continue
        ok, _ = cap.read()
        try:
            cap.release()
        except Exception:
            pass
        if ok:
            cams.append(cam_idx)
    return cams


def get_camera_max_fps(app: Any, camera_index: int = 0) -> float:
    if not cv2:
        return 0.0
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        try:
            cap.release()
        except Exception:
            pass
        return 0.0
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    except Exception:
        fps = 0.0
    finally:
        try:
            cap.release()
        except Exception:
            pass
    if fps > 0 and camera_index == int(app.detect_camera_index_var.get().strip() or "0"):
        app._detect_camera_max_fps = fps
    return max(0.0, fps)


def stop_detect_stream(app: Any) -> None:
    if getattr(app, "_detect_after_id", None):
        try:
            app.root.after_cancel(app._detect_after_id)
        except Exception:
            pass
        app._detect_after_id = None
    if getattr(app, "_detect_video_cap", None) is not None:
        try:
            app._detect_video_cap.release()
        except Exception:
            pass
        app._detect_video_cap = None
    app._detect_last_plot_bgr = None
    app._detect_last_piece_results = []
    app._detect_last_piece_paths = []
    app._detect_piece_index = 0
    app._detect_image_result_cache = {}
    app._detect_report_logged_keys = set()
    try:
        app._cleanup_detect_cut_piece_temp(remove_root=True)
    except Exception:
        pass
    try:
        app._close_detect_report_logger()
    except Exception:
        pass


def start_detect_video_stream(app: Any, source: Any) -> None:
    if not cv2:
        app.logger.error("OpenCV not available for video stream")
        return
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        try:
            messagebox = None
            from tkinter import messagebox as _mb
            messagebox = _mb
        except Exception:
            messagebox = None
        if messagebox:
            messagebox.showerror("Detect Mode", "Failed to open video source.")
        app._show_detect_settings_page_for_current_source()
        return
    app._detect_video_cap = cap
    src_label = f"camera {source}" if isinstance(source, int) else "video source"
    fps = 0.0
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    except Exception:
        fps = 0.0
    if fps > 0:
        try:
            app._detect_status_var.set(f"{src_label} running ({fps:.1f} FPS)")
        except Exception:
            pass
    else:
        try:
            app._detect_status_var.set(f"{src_label} running")
        except Exception:
            pass
    _detect_tick_video(app)


def _detect_tick_video(app: Any) -> None:
    if getattr(app, "_detect_video_cap", None) is None or not getattr(app, "_detect_mode_active", False):
        return
    ok, frame = app._detect_video_cap.read()
    if not ok:
        try:
            app._detect_status_var.set("Video ended")
        except Exception:
            pass
        return
    try:
        results = app._run_detect_inference(frame)
    except Exception as exc:
        app.logger.exception("Detect video frame failed")
        try:
            from tkinter import messagebox
            messagebox.showerror("Detect Mode Error", str(exc), parent=app.root)
        except Exception:
            pass
        return
    app._detect_video_frame_idx = getattr(app, "_detect_video_frame_idx", 0) + 1
    app._detect_video_frame_idx = app._detect_video_frame_idx
    verdict, detail = app._evaluate_golden_match(results[0])
    plotted = app._render_detect_result(results[0], line_width=1)
    app._set_detect_verdict(verdict, detail)
    frame_name = f"frame_{app._detect_video_frame_idx:06d}"
    app._append_detect_report_row(frame_name, results[0], verdict, detail)
    app._save_detect_result_image(frame_name, plotted)
    app._update_detect_class_panel(results[0])
    app._show_detect_plot(plotted)
    app._detect_after_id = app.root.after(max(1, int(getattr(app, "_detect_frame_interval_ms", 15))), lambda: _detect_tick_video(app))
