import os
import tkinter as tk
from typing import Any

from ai_labeller.features import report_utils
from ai_labeller.features import camera_utils
from ai_labeller.features import detect_runtime
from ai_labeller.features import image_utils


def open_detect_workspace(app: Any, source_kind: str, source_value: Any, output_dir: str | None = None) -> None:
    app._detect_mode_active = True
    app.hide_shortcut_tooltip()
    app._stop_detect_stream()
    app._detect_video_frame_idx = 0
    for child in app.root.winfo_children():
        child.destroy()

    frame = tk.Frame(app.root, bg=app.COLORS["bg_dark"])
    frame.pack(fill="both", expand=True)
    app._detect_workspace_frame = frame

    top = tk.Frame(frame, bg=app.COLORS["bg_white"])
    top.pack(side="top", fill="x", padx=12, pady=12)
    tk.Label(
        top,
        text="Detect Workspace",
        font=app.font_title,
        fg=app.COLORS["text_primary"],
        bg=app.COLORS["bg_white"],
    ).pack(side="left", padx=12, pady=10)
    tk.Label(
        top,
        textvariable=app._detect_status_var,
        font=app.font_primary,
        fg=app.COLORS["text_secondary"],
        bg=app.COLORS["bg_white"],
        anchor="w",
    ).pack(side="left", fill="x", expand=True, padx=8)
    app._detect_verdict_label = tk.Label(
        top,
        textvariable=app._detect_verdict_var,
        font=app.font_bold,
        fg=app.COLORS["text_secondary"],
        bg=app.COLORS["bg_white"],
        anchor="e",
    )
    app._detect_verdict_label.pack(side="right", padx=8, pady=8)
    app.create_secondary_button(
        top,
        text="Back to Source Select",
        command=app._exit_detect_workspace_to_source,
    ).pack(side="right", padx=10, pady=8)

    content = tk.Frame(frame, bg=app.COLORS["bg_dark"])
    content.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    left = tk.Frame(content, bg="#101010")
    left.pack(side="left", fill="both", expand=True)
    app._detect_image_label = tk.Label(left, bg="#101010")
    app._detect_image_label.pack(fill="both", expand=True)
    app._detect_image_label.bind("<Configure>", lambda _e: refresh_detect_image(app))

    right = tk.Frame(content, bg=app.COLORS["bg_white"], width=300)
    right.pack(side="right", fill="y")
    right.pack_propagate(False)
    tk.Label(
        right,
        text="Detected Classes",
        font=app.font_bold,
        fg=app.COLORS["text_primary"],
        bg=app.COLORS["bg_white"],
    ).pack(anchor="w", padx=12, pady=(12, 8))
    app._detect_class_listbox = tk.Listbox(
        right,
        font=app.font_primary,
        bg=app.COLORS["bg_light"],
        fg=app.COLORS["text_primary"],
        relief="flat",
        highlightthickness=0,
    )
    app._detect_class_listbox.pack(fill="both", expand=True, padx=12, pady=(0, 12))
    app._init_detect_report_logger(source_kind, source_value, output_dir=output_dir)
    app._set_detect_verdict(None, "")

    if source_kind == "camera":
        cam_source: Any = 0
        try:
            cam_source = int(str(source_value))
        except Exception:
            cam_source = 0
        app._start_detect_video_stream(cam_source)
        return

    src_path = os.path.abspath(str(source_value))
    if os.path.isdir(src_path):
        app._detect_image_paths = [p for p in app._glob_image_files(src_path, include_bmp=True)]
        app._detect_image_index = 0
        if not app._detect_image_paths:
            tk.messagebox.showwarning("Detect Mode", "No images found in selected folder.")
            app._show_detect_settings_page_for_current_source()
            return
        detect_render_image_index(app)
        return

    if src_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        app._detect_image_paths = [src_path]
        app._detect_image_index = 0
        detect_render_image_index(app)
        return

    app._start_detect_video_stream(src_path)


def render_current_piece_result(app: Any, source_path: str) -> None:
    if not app._detect_last_piece_results:
        return
    total = len(app._detect_last_piece_results)
    app._detect_piece_index = max(0, min(app._detect_piece_index, total - 1))
    idx = app._detect_piece_index
    result0 = app._detect_last_piece_results[idx]
    piece_name = (
        os.path.basename(app._detect_last_piece_paths[idx])
        if idx < len(app._detect_last_piece_paths)
        else f"piece_{idx + 1:04d}.png"
    )
    base_status = f"{os.path.basename(source_path)} ({app._detect_image_index + 1}/{len(app._detect_image_paths)})"
    app._detect_status_var.set(
        f"{base_status} | piece {idx + 1}/{total}: {piece_name} | new cut pieces: {app._detect_last_cut_piece_count}"
    )
    verdict, detail = app._evaluate_golden_match(result0)
    plotted = app._render_detect_result(result0, line_width=1)
    app._set_detect_verdict(verdict, detail)
    app._update_detect_class_panel(result0)
    app._show_detect_plot(plotted)


def detect_render_image_index(app: Any) -> None:
    if not app._detect_image_paths:
        return
    app._detect_image_index = max(0, min(app._detect_image_index, len(app._detect_image_paths) - 1))
    img_path = app._detect_image_paths[app._detect_image_index]
    cache_key = os.path.abspath(img_path)
    base_status = f"{os.path.basename(img_path)} ({app._detect_image_index + 1}/{len(app._detect_image_paths)})"
    app._detect_status_var.set(base_status)
    cached = app._detect_image_result_cache.get(cache_key)
    if cached:
        app._detect_last_cut_piece_count = int(cached.get("cut_piece_count", 0))
        app._detect_last_piece_results = list(cached.get("results") or [])
        app._detect_last_piece_paths = list(cached.get("piece_paths") or [])
        app._detect_piece_index = int(cached.get("piece_index", 0))
        if app._should_use_background_cut_detection() and app._detect_last_piece_results:
            render_current_piece_result(app, img_path)
            return
        result0 = app._detect_last_piece_results[0] if app._detect_last_piece_results else None
        if result0 is not None:
            verdict, detail = app._evaluate_golden_match(result0)
            plotted = app._render_detect_result(result0, line_width=1)
            app._set_detect_verdict(verdict, detail)
            app._update_detect_class_panel(result0)
            app._show_detect_plot(plotted)
            return
    try:
        results = app._run_detect_inference(img_path)
    except Exception as exc:
        app.logger.exception("Detect image failed")
        tk.messagebox.showerror("Detect Mode Error", str(exc), parent=app.root)
        return
    if app._should_use_background_cut_detection() and app._detect_last_piece_results:
        entries: list[dict[str, Any]] = []
        for i, piece_result in enumerate(app._detect_last_piece_results):
            piece_name = (
                os.path.basename(app._detect_last_piece_paths[i])
                if i < len(app._detect_last_piece_paths)
                else f"piece_{i + 1:04d}.png"
            )
            verdict_i, detail_i = app._evaluate_golden_match(piece_result)
            entries.append(
                {
                    "status": verdict_i,
                    "detail": detail_i,
                    "ocr_id": app._detect_last_ocr_id,
                    "ocr_sub_id": app._detect_last_ocr_sub_id,
                    "image_name": f"{os.path.basename(img_path)}::{piece_name}",
                }
            )
            app._append_detect_report_row_once(
                f"{os.path.basename(img_path)}::{piece_name}",
                piece_result,
                verdict_i,
                detail_i,
            )
            piece_plot = app._render_detect_result(piece_result, line_width=1)
            app._save_detect_result_image(f"{os.path.basename(img_path)}::{piece_name}", piece_plot)
        app._detect_image_result_cache[cache_key] = {
            "results": list(app._detect_last_piece_results),
            "piece_paths": list(app._detect_last_piece_paths),
            "entries": entries,
            "cut_piece_count": int(app._detect_last_cut_piece_count),
            "piece_index": 0,
        }
        app._detect_piece_index = 0
        render_current_piece_result(app, img_path)
        return
    verdict, detail = app._evaluate_golden_match(results[0])
    plotted = app._render_detect_result(results[0], line_width=1)
    app._set_detect_verdict(verdict, detail)
    app._append_detect_report_row_once(os.path.basename(img_path), results[0], verdict, detail)
    app._save_detect_result_image(os.path.basename(img_path), plotted)
    app._update_detect_class_panel(results[0])
    app._show_detect_plot(plotted)
    app._detect_image_result_cache[cache_key] = {
        "results": [results[0]],
        "piece_paths": [],
        "entries": [
            {
                "status": verdict,
                "detail": detail,
                "ocr_id": app._detect_last_ocr_id,
                "ocr_sub_id": app._detect_last_ocr_sub_id,
                "image_name": os.path.basename(img_path),
            }
        ],
        "cut_piece_count": 0,
        "piece_index": 0,
    }


def detect_prev_image(app: Any) -> None:
    if not app._detect_image_paths:
        return
    if app._should_use_background_cut_detection() and len(app._detect_last_piece_results) > 1 and app._detect_piece_index > 0:
        app._detect_piece_index -= 1
        cur_img = os.path.abspath(app._detect_image_paths[app._detect_image_index])
        if cur_img in app._detect_image_result_cache:
            app._detect_image_result_cache[cur_img]["piece_index"] = app._detect_piece_index
        render_current_piece_result(app, app._detect_image_paths[app._detect_image_index])
        return
    app._detect_image_index = max(0, app._detect_image_index - 1)
    detect_render_image_index(app)


def detect_next_image(app: Any) -> None:
    if not app._detect_image_paths:
        return
    if app._should_use_background_cut_detection() and len(app._detect_last_piece_results) > 1:
        if app._detect_piece_index < len(app._detect_last_piece_results) - 1:
            app._detect_piece_index += 1
            cur_img = os.path.abspath(app._detect_image_paths[app._detect_image_index])
            if cur_img in app._detect_image_result_cache:
                app._detect_image_result_cache[cur_img]["piece_index"] = app._detect_piece_index
            render_current_piece_result(app, app._detect_image_paths[app._detect_image_index])
            return
    app._detect_image_index = min(len(app._detect_image_paths) - 1, app._detect_image_index + 1)
    detect_render_image_index(app)


def show_detect_plot(app: Any, plot_bgr: Any) -> None:
    app._detect_last_plot_bgr = plot_bgr
    refresh_detect_image(app)


def refresh_detect_image(app: Any) -> None:
    if app._detect_image_label is None or app._detect_last_plot_bgr is None:
        return
    frame_rgb = app.cv2.cvtColor(app._detect_last_plot_bgr, app.cv2.COLOR_BGR2RGB)
    pil_img = image_utils.Image.fromarray(frame_rgb) if hasattr(image_utils, 'Image') else None
    if pil_img is None:
        try:
            from PIL import Image

            pil_img = Image.fromarray(frame_rgb)
        except Exception:
            app.logger.exception("Failed to create PIL image from detect plot")
            return
    lw = max(1, app._detect_image_label.winfo_width())
    lh = max(1, app._detect_image_label.winfo_height())
    scale = min(lw / pil_img.width, lh / pil_img.height)
    nw = max(1, int(pil_img.width * scale))
    nh = max(1, int(pil_img.height * scale))
    resized = pil_img.resize((nw, nh), getattr(getattr(__import__('PIL'), 'Image'), 'Resampling', getattr(__import__('PIL'), 'Image')).BILINEAR)
    from PIL import ImageTk

    app._detect_photo = ImageTk.PhotoImage(resized)
    app._detect_image_label.configure(image=app._detect_photo)
