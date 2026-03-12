import csv
import datetime
import os
import threading
from typing import Any

from ai_labeller.features import image_utils


def _resolve_golden_image_path_for_report(app: Any) -> str:
    sample = getattr(app, "_detect_golden_sample", None) or {}
    direct = str(sample.get("image_path", "")).strip()
    if direct and os.path.isfile(direct):
        return os.path.abspath(direct)

    label_path = str(sample.get("label_path", "")).strip()
    if label_path:
        base_dir = os.path.dirname(os.path.abspath(label_path))
        stem = os.path.splitext(os.path.basename(label_path))[0]
        for ext in (".png", ".jpg", ".jpeg", ".bmp"):
            candidate = os.path.join(base_dir, stem + ext)
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
    return ""


def _resolve_golden_label_path_for_report(app: Any) -> str:
    sample = getattr(app, "_detect_golden_sample", None) or {}
    label_path = str(sample.get("label_path", "")).strip()
    if label_path and os.path.isfile(label_path):
        return os.path.abspath(label_path)
    return ""


def init_detect_report_logger(app: Any, source_kind: str, source_value: Any, output_dir: str | None = None) -> None:
    app._close_detect_report_logger()
    app._detect_image_result_cache = {}
    app._detect_report_logged_keys = set()
    try:
        if output_dir:
            base_dir = os.path.abspath(output_dir)
        elif source_kind == "camera":
            base_dir = app.project_root or os.getcwd()
        else:
            src_path = os.path.abspath(str(source_value))
            base_dir = src_path if os.path.isdir(src_path) else os.path.dirname(src_path)
        os.makedirs(base_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(base_dir, f"detect_results_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        app._detect_saved_image_dir = os.path.join(run_dir, "detected_images")
        os.makedirs(app._detect_saved_image_dir, exist_ok=True)
        app._detect_saved_image_keys = set()
        csv_path = os.path.join(run_dir, f"detect_results_{timestamp}.csv")
        app._detect_report_mode = app.detect_run_mode_var.get().strip().lower()
        if app._detect_report_mode not in {"pure_detect", "golden"}:
            app._detect_report_mode = "pure_detect"
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "id",
                "sub_id",
                "image_name",
                "status",
                "detected_classes",
                "reason",
                "golden_mode",
                "iou_threshold",
                "path",
                "detect_image_path",
                "details",
                "golden_image_path",
                "golden_label_path",
            ])
        app._detect_report_csv_path = csv_path
    except Exception:
        app.logger.exception("Failed to initialize detect report logger")
        app._detect_report_csv_path = None


def _close_detect_report_logger(app: Any) -> None:
    """Finalize and clear any detect report logger state for the app.

    This will trigger report generation for the current CSV (if any) and
    clear cached report-related attributes on the `app` instance.
    """
    try:
        csv_path = getattr(app, "_detect_report_csv_path", None)
        if csv_path:
            try:
                _trigger_detect_report_generation(app, csv_path)
            except Exception:
                app.logger.exception("Failed to trigger detect report generation on close for %s", csv_path)
    finally:
        try:
            app._detect_report_csv_path = None
        except Exception:
            pass
        try:
            app._detect_report_mode = "pure_detect"
        except Exception:
            pass
        try:
            app._detect_report_logged_keys = set()
        except Exception:
            pass
        try:
            app._detect_report_generated_paths = set()
        except Exception:
            pass
        try:
            app._detect_saved_image_dir = None
        except Exception:
            pass
        try:
            app._detect_saved_image_keys = set()
        except Exception:
            pass


def save_detect_result_image(app: Any, image_name: str, plot_bgr: Any) -> None:
    out_dir = getattr(app, "_detect_saved_image_dir", None)
    if not out_dir or plot_bgr is None:
        return
    key = str(image_name or "").strip() or "detect_result"
    saved_keys = getattr(app, "_detect_saved_image_keys", set())
    if key in saved_keys:
        return
    try:
        safe_name = key.replace("\\", "_").replace("/", "_").replace(":", "_")
        safe_name = safe_name.replace("*", "_").replace("?", "_").replace('"', "_")
        safe_name = safe_name.replace("<", "_").replace(">", "_").replace("|", "_")
        root, ext = os.path.splitext(safe_name)
        if not ext:
            ext = ".jpg"
        if not root:
            root = "detect_result"
        out_path = os.path.join(out_dir, root + ext)
        suffix = 1
        while os.path.exists(out_path):
            out_path = os.path.join(out_dir, f"{root}_{suffix}{ext}")
            suffix += 1
        image_utils.write_cv2_image(out_path, plot_bgr)
        saved_keys.add(key)
        app._detect_saved_image_keys = saved_keys
    except Exception:
        app.logger.exception("Failed to save detect result image: %s", key)


def _resolve_detection_report_generator_script() -> str | None:
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "detection_report_generator.py"),
        os.path.join(os.getcwd(), "detection_report_generator.py"),
        os.path.join(os.path.expanduser("~"), "Desktop", "detection_report_generator.py"),
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return os.path.abspath(p)
    return None


def _trigger_detect_report_generation(app: Any, csv_path: str) -> None:
    csv_abs = os.path.abspath(csv_path)
    if csv_abs in app._detect_report_generated_paths:
        return
    app._detect_report_generated_paths.add(csv_abs)

    def worker() -> None:
        try:
            import inspect
            from ai_labeller import detection_report_generator as drg

            loaded = drg.load_data(csv_abs)
            if isinstance(loaded, tuple):
                records = loaded[0]
                has_golden = bool(loaded[1]) if len(loaded) > 1 else True
            else:
                records = loaded
                has_golden = any(str(r.get("status", "")).strip() for r in records) if records else False

            agg = drg.aggregate(records)
            if isinstance(agg, tuple):
                sorted_classes = agg[0]
                class_img_count = agg[1] if len(agg) > 1 else {}
                prefix_stats = agg[2] if len(agg) > 2 else {}
                status_counts = agg[3] if len(agg) > 3 else {}
                iou_values = agg[4] if len(agg) > 4 else []
            else:
                sorted_classes = []
                class_img_count = {}
                prefix_stats = {}
                status_counts = {}
                iou_values = []

            base = os.path.splitext(csv_abs)[0]
            excel_out = base + "_report.xlsx"
            html_out = base + "_dashboard.html"
            pdf_out = base + "_dashboard.pdf"

            def call_builder(fn_name: str, out_path: str) -> None:
                fn = getattr(drg, fn_name, None)
                if fn is None:
                    raise AttributeError(f"{fn_name} not found in detection_report_generator")
                sig = inspect.signature(fn)
                kwargs: dict[str, Any] = {"out_path": out_path}
                for p in sig.parameters.keys():
                    if p == "records":
                        kwargs[p] = records
                    elif p == "sorted_classes":
                        kwargs[p] = sorted_classes
                    elif p == "class_img_count":
                        kwargs[p] = class_img_count
                    elif p == "prefix_stats":
                        kwargs[p] = prefix_stats
                    elif p == "status_counts":
                        kwargs[p] = status_counts
                    elif p == "iou_values":
                        kwargs[p] = iou_values
                    elif p == "has_golden":
                        kwargs[p] = has_golden
                fn(**kwargs)

            call_builder("build_excel", excel_out)
            call_builder("build_html", html_out)
            if hasattr(drg, "build_pdf"):
                try:
                    call_builder("build_pdf", pdf_out)
                except Exception:
                    app.logger.exception("Detection PDF report generation failed for %s", csv_abs)
            app.logger.info("Detection report generated for %s", csv_abs)
        except Exception:
            app.logger.exception("Failed to generate detection report for %s", csv_abs)

    threading.Thread(target=worker, daemon=True).start()


def append_detect_report_row(app: Any, image_name: str, result0: Any, status: str | None, details: str) -> None:
    try:
        counts = app._detect_class_counts(result0)
        class_text = "; ".join(f"{k} x{v}" for k, v in sorted(counts.items())) if counts else "No detections"
        iou_text = f"{float(app.detect_golden_iou_var.get()):.2f}" if app.detect_run_mode_var.get().strip().lower() == "golden" else ""
        details_text = str(details or "")
        if getattr(app, "_detect_last_ocr_id", None):
            if details_text:
                details_text = f"{details_text}; id={app._detect_last_ocr_id}"
            else:
                details_text = f"id={app._detect_last_ocr_id}"
        if getattr(app, "_detect_last_ocr_sub_id", None):
            if details_text:
                details_text = f"{details_text}; sub_id={app._detect_last_ocr_sub_id}"
            else:
                details_text = f"sub_id={app._detect_last_ocr_sub_id}"
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if getattr(app, "_detect_report_csv_path", None):
            with open(app._detect_report_csv_path, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                if app._detect_report_mode == "golden":
                    mode_raw = app.detect_golden_mode_var.get()
                    if hasattr(app, "_normalize_golden_mode"):
                        mode = app._normalize_golden_mode(mode_raw)
                    else:
                        mode = str(mode_raw or "").strip().lower()
                    golden_image_path = _resolve_golden_image_path_for_report(app)
                    golden_label_path = _resolve_golden_label_path_for_report(app)
                    reason_text = str(getattr(app, "_detect_last_fail_reason", "") or "").strip()
                    id_text = str(getattr(app, "_detect_last_ocr_id", "") or "").strip()
                    sub_id_text = str(getattr(app, "_detect_last_ocr_sub_id", "") or "").strip()
                    writer.writerow([
                        ts,
                        id_text,
                        sub_id_text,
                        image_name,
                        status or "",
                        class_text,
                        reason_text,
                        mode,
                        iou_text,
                        "",
                        "",
                        details_text,
                        golden_image_path,
                        golden_label_path,
                    ])
                else:
                    writer.writerow([
                        ts,
                        "",
                        "",
                        image_name,
                        "",
                        class_text,
                        "",
                        "",
                        "",
                        "",
                        "",
                        details_text,
                        "",
                        "",
                    ])
    except Exception:
        app.logger.exception("Failed to append detect report row")


def append_detect_report_row_once(app: Any, image_name: str, result0: Any, status: str | None, details: str) -> None:
    csv_path = getattr(app, "_detect_report_csv_path", "") or ""
    key = f"{csv_path}|{image_name}"
    if key in getattr(app, "_detect_report_logged_keys", set()):
        return
    append_detect_report_row(app, image_name, result0, status, details)
    try:
        app._detect_report_logged_keys.add(key)
    except Exception:
        pass
