from __future__ import annotations

import datetime
import os
import shutil
from pathlib import Path
from typing import Any


def clear_video_label_meta(app: Any) -> None:
    app._video_label_source_name = ""
    app._video_label_total_frames = 0
    app._video_label_total_seconds = 0.0
    app._video_label_fps = 0.0
    app._video_timeline_user_dragging = False
    app._update_video_timeline_ui()


def set_video_label_meta(app: Any, source_name: str, total_frames: int, total_seconds: float, fps: float) -> None:
    app._video_label_source_name = str(source_name or "").strip()
    app._video_label_total_frames = max(0, int(total_frames or 0))
    app._video_label_total_seconds = max(0.0, float(total_seconds or 0.0))
    app._video_label_fps = max(0.0, float(fps or 0.0))
    app._video_timeline_user_dragging = False
    app._update_video_timeline_ui()


def format_video_time(seconds: float) -> str:
    total_seconds = max(0, int(round(float(seconds or 0.0))))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def update_video_timeline_ui(app: Any) -> None:
    if app._video_label_total_frames <= 0:
        app.video_timeline_slider.blockSignals(True)
        app.video_timeline_slider.setMinimum(1)
        app.video_timeline_slider.setMaximum(1)
        app.video_timeline_slider.setValue(1)
        app.video_timeline_slider.blockSignals(False)
        app.lbl_video_timeline_summary.clear()
        app.lbl_video_timeline_current.setText("00:00")
        app.lbl_video_timeline_total.setText("00:00")
        app.video_timeline_wrap.hide()
        return
    total_frames = max(1, int(app._video_label_total_frames))
    current_frame = min(max(1, app._image_idx + 1), total_frames)
    current_seconds = (float(current_frame) / float(app._video_label_fps)) if app._video_label_fps > 0 else 0.0
    total_seconds = float(app._video_label_total_seconds or 0.0)
    app.video_timeline_slider.blockSignals(True)
    app.video_timeline_slider.setMinimum(1)
    app.video_timeline_slider.setMaximum(total_frames)
    app.video_timeline_slider.setPageStep(max(1, total_frames // 20))
    app.video_timeline_slider.setValue(current_frame)
    app.video_timeline_slider.blockSignals(False)
    app.lbl_video_timeline_summary.setText(
        f"Frame {current_frame} / {total_frames} | "
        f"{format_video_time(current_seconds)} / {format_video_time(total_seconds)}"
    )
    app.lbl_video_timeline_current.setText(format_video_time(current_seconds))
    app.lbl_video_timeline_total.setText(format_video_time(total_seconds))
    app.video_timeline_wrap.show()


def on_video_timeline_changed(app: Any, value: int) -> None:
    if app._video_label_total_frames <= 0:
        return
    if getattr(app, "_video_timeline_user_dragging", False):
        current_seconds = (float(value) / float(app._video_label_fps)) if app._video_label_fps > 0 else 0.0
        app.lbl_video_timeline_summary.setText(
            f"Frame {int(value)} / {int(app._video_label_total_frames)} | "
            f"{format_video_time(current_seconds)} / {format_video_time(app._video_label_total_seconds)}"
        )
        app.lbl_video_timeline_current.setText(format_video_time(current_seconds))


def on_video_timeline_released(app: Any) -> None:
    app._video_timeline_user_dragging = False
    if app._video_label_total_frames <= 0:
        return
    target_idx = int(app.video_timeline_slider.value()) - 1
    if target_idx < 0 or target_idx >= len(app._image_paths):
        app._update_video_timeline_ui()
        return
    if target_idx == app._image_idx:
        app._update_video_timeline_ui()
        return
    app._image_idx = target_idx
    app._show_current_image()


def iter_label_images_recursive(root_dir: str) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    out: list[str] = []
    for base, _dirs, files in os.walk(root_dir):
        for name in files:
            p = os.path.join(base, name)
            if os.path.splitext(name)[1].lower() in exts and os.path.isfile(p):
                out.append(os.path.abspath(p))
    out.sort()
    return out


def prepare_cut_output_for_label(cut_output_dir: str) -> str | None:
    src_images = iter_label_images_recursive(cut_output_dir)
    if not src_images:
        return None
    ready_dir = os.path.join(cut_output_dir, "label_ready_images")
    os.makedirs(ready_dir, exist_ok=True)
    copied = 0
    for src in src_images:
        name = os.path.basename(src)
        stem, ext = os.path.splitext(name)
        rel_parent = os.path.basename(os.path.dirname(src))
        base_stem = f"{rel_parent}_{stem}".replace(" ", "_")
        dst = os.path.join(ready_dir, f"{base_stem}{ext}")
        i = 1
        while os.path.exists(dst):
            dst = os.path.join(ready_dir, f"{base_stem}_{i}{ext}")
            i += 1
        try:
            shutil.copy2(src, dst)
            copied += 1
        except Exception:
            continue
    if copied <= 0:
        return None
    return ready_dir


def maybe_run_cut_background_for_label(
    app: Any,
    path: str,
    kind: str,
    *,
    cut_background_detect: Any,
    prompt_cut_background_threshold: Any,
    QFileDialog: Any,
    QMessageBox: Any,
) -> tuple[str, str] | None:
    if kind != "image_folder":
        return os.path.abspath(path), kind
    ask_cut = QMessageBox.question(
        app,
        "Cut Background",
        "Run cut background before labeling?\n(After cut, label will open on cut results directly.)",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
    )
    if ask_cut != QMessageBox.StandardButton.Yes:
        return os.path.abspath(path), kind
    try:
        import cv2  # type: ignore
    except Exception as exc:
        QMessageBox.critical(
            app,
            "Cut Background",
            "Cut background requires full desktop OpenCV.\n"
            f"OpenCV import failed: {exc}\n\n"
            "Fix:\n"
            "pip uninstall opencv-python-headless\n"
            "pip install opencv-python",
        )
        return None
    missing = [name for name in ["imread", "namedWindow", "selectROI", "destroyWindow"] if not hasattr(cv2, name)]
    if missing:
        QMessageBox.critical(
            app,
            "Cut Background",
            "Cut background requires full desktop OpenCV.\n"
            f"OpenCV missing APIs: {', '.join(missing)}\n\n"
            "Fix:\n"
            "pip uninstall opencv-python-headless\n"
            "pip install opencv-python",
        )
        return None
    golden_image_path, _ = QFileDialog.getOpenFileName(
        app,
        "Select one golden image in this folder",
        os.path.abspath(path),
        "Image files (*.png *.jpg *.jpeg *.bmp)",
    )
    if not golden_image_path:
        return None
    try:
        if os.path.commonpath([os.path.abspath(path), os.path.abspath(golden_image_path)]) != os.path.abspath(path):
            QMessageBox.warning(app, "Cut Background", "Please select a golden image inside the selected folder.")
            return None
    except Exception:
        pass
    threshold = prompt_cut_background_threshold(app, 0.3)
    if threshold is None:
        return None
    try:
        result = cut_background_detect.run_cut_background_batch_with_golden(
            path,
            golden_image_path=golden_image_path,
            threshold=threshold,
            parent=app,
        )
    except Exception as exc:
        msg = str(exc)
        if "cvNamedWindow" in msg or "The function is not implemented" in msg:
            QMessageBox.critical(
                app,
                "Cut Background",
                "Cut background requires OpenCV GUI backend.\n"
                "Please install desktop OpenCV:\n"
                "pip uninstall opencv-python-headless\n"
                "pip install opencv-python",
            )
            return None
        QMessageBox.critical(app, "Cut Background", f"Cut background failed:\n{exc}")
        return None
    if result is None:
        return None
    ready = prepare_cut_output_for_label(result.output_dir)
    if not ready or not os.path.isdir(ready):
        QMessageBox.warning(
            app,
            "Cut Background",
            "Cut background finished but no label-ready images were produced.",
        )
        return None
    return os.path.abspath(ready), "image_folder"


def extract_video_frames_for_label(app: Any, video_path: str, output_root: str, *, QMessageBox: Any) -> tuple[str, int, float, float] | None:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        QMessageBox.critical(
            app,
            "Load from Video",
            "Video frame extraction requires OpenCV.\n"
            f"OpenCV import failed: {exc}",
        )
        return None
    video_abs = os.path.abspath(video_path)
    if not os.path.isfile(video_abs):
        QMessageBox.warning(app, "Load from Video", "Video file does not exist.")
        return None
    stem = Path(video_abs).stem or "video"
    frame_dir = os.path.join(
        os.path.abspath(output_root),
        f"{stem}_label_frames_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(frame_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_abs)
    if not cap.isOpened():
        try:
            cap.release()
        except Exception:
            pass
        QMessageBox.warning(app, "Load from Video", "Failed to open selected video.")
        return None
    fps = 0.0
    total_frame_count = 0
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    except Exception:
        fps = 0.0
    try:
        total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    except Exception:
        total_frame_count = 0
    saved_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            saved_count += 1
            out_path = os.path.join(frame_dir, f"{stem}_frame_{saved_count:06d}.jpg")
            if not cv2.imwrite(out_path, frame):
                saved_count -= 1
    finally:
        try:
            cap.release()
        except Exception:
            pass
    if saved_count <= 0:
        QMessageBox.warning(app, "Load from Video", "No frames could be extracted from selected video.")
        return None
    effective_total_frames = saved_count if saved_count > 0 else max(0, total_frame_count)
    total_seconds = (float(effective_total_frames) / float(fps)) if fps > 0 else 0.0
    return os.path.abspath(frame_dir), effective_total_frames, total_seconds, fps
