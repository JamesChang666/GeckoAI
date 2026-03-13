from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from ai_labeller.features import golden as golden_core
from ai_labeller.features import ocr_utils


class _SimpleVar:
    def __init__(self, value: Any = None) -> None:
        self._value = value

    def get(self) -> Any:
        return self._value

    def set(self, value: Any) -> None:
        self._value = value


class DetectCliRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.logger = logging.getLogger("geckoai.cli.detect")
        self._model = None
        self._csv_path = ""
        self._saved_image_dir = ""
        self._easy_ocr_engine = None
        self._detect_ocr_warning_shown = False
        self._detect_last_ocr_id = ""
        self._detect_last_ocr_sub_id = ""
        self._detect_last_cut_piece_count = 0
        self._detect_cut_piece_count_by_path: dict[str, int] = {}
        self._detect_spatial_mismatch_rects_norm: list[tuple[float, float, float, float]] = []
        self._detect_last_fail_reason = ""
        self._detect_class_name_map: dict[int, str] = {}
        self._detect_class_color_overrides: dict[int, tuple[int, int, int]] = {}
        self._detect_golden_sample: dict[str, Any] | None = None
        self._detect_bg_cut_bundle = None
        self._logged: set[str] = set()
        self._processed_paths: set[str] = set()
        self._known_source_images: set[str] = set()
        self._report_paths: dict[str, str] = {}
        self._summary: dict[str, Any] = {}
        self._runtime_device = "cpu" if str(args.device).strip().lower() == "cpu" else "gpu"
        self._gpu_fallback_reported = False
        self._watch_initial_scan_done = False
        self.detect_run_mode_var = _SimpleVar("golden" if args.golden_dir else "pure_detect")
        self.detect_golden_mode_var = _SimpleVar(args.golden_mode)
        self.detect_golden_iou_var = _SimpleVar(float(args.golden_iou))
        self._load_class_color_map()

    def run(self) -> int:
        import cv2

        self._init_report()
        self._load_golden_sample_if_needed()
        self._load_model()
        image_paths = self._resolve_pending_image_paths(cv2)
        if not image_paths and not self.args.watch:
            self.logger.error("No images found in source folder: %s", self.args.source)
            return 4
        if image_paths:
            self._run_batch(image_paths, cv2)
        if self.args.watch:
            self.logger.info("Watching folder: %s", os.path.abspath(self.args.source))
            while True:
                time.sleep(float(self.args.watch_interval))
                pending = self._resolve_pending_image_paths(cv2)
                if not pending:
                    continue
                self._run_batch(pending, cv2)
        return self._finalize_outputs()

    def _init_report(self) -> None:
        base = os.path.abspath(str(self.args.output).strip() or os.getcwd())
        os.makedirs(base, exist_ok=True)
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(base, f"detect_results_{ts}")
        os.makedirs(run_dir, exist_ok=True)
        self._saved_image_dir = os.path.join(run_dir, "detected_images")
        os.makedirs(self._saved_image_dir, exist_ok=True)
        self._csv_path = os.path.join(run_dir, f"detect_results_{ts}.csv")
        with open(self._csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
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
                ]
            )

    def _load_model(self) -> None:
        from ultralytics import YOLO

        self._model = YOLO(os.path.abspath(self.args.model))

    def _load_class_color_map(self) -> None:
        raw = str(getattr(self.args, "class_color_map", "") or "").strip()
        if not raw:
            return
        payload: dict[str, Any]
        if os.path.isfile(raw):
            with open(raw, "r", encoding="utf-8") as f:
                payload = json.load(f)
        else:
            payload = {}
            for chunk in raw.split(","):
                item = str(chunk or "").strip()
                if not item or "=" not in item:
                    continue
                left, right = item.split("=", 1)
                payload[left.strip()] = right.strip()
        for key, value in payload.items():
            try:
                cid = int(str(key).strip())
            except Exception:
                continue
            rgb = self._parse_color_value(value)
            if rgb is None:
                continue
            r, g, b = rgb
            self._detect_class_color_overrides[cid] = (b, g, r)

    def _parse_color_value(self, value: Any) -> tuple[int, int, int] | None:
        text = str(value or "").strip()
        if not text:
            return None
        if text.startswith("#") and len(text) == 7:
            try:
                return (int(text[1:3], 16), int(text[3:5], 16), int(text[5:7], 16))
            except Exception:
                return None
        parts = [p.strip() for p in text.split(",")]
        if len(parts) == 3:
            try:
                return tuple(max(0, min(255, int(float(p)))) for p in parts)  # type: ignore[return-value]
            except Exception:
                return None
        return None

    def _load_golden_sample_if_needed(self) -> None:
        if self.detect_run_mode_var.get() != "golden":
            self._detect_golden_sample = None
            self._detect_bg_cut_bundle = None
            return
        golden_dir = os.path.abspath(str(self.args.golden_dir or "").strip())
        resolved = golden_core.resolve_golden_project_folder(golden_dir)
        if resolved and resolved.get("project_root"):
            golden_dir = os.path.abspath(str(resolved.get("project_root")))
        mapping_path = str((resolved or {}).get("mapping_path", "")).strip()
        if not mapping_path:
            mapping_path = golden_core.find_dataset_yaml_in_folder(golden_dir) or ""
        label_path = str((resolved or {}).get("label_path", "")).strip()
        if not label_path:
            txt_files = sorted(str(p) for p in Path(golden_dir).glob("*.txt") if p.is_file())
            label_path = txt_files[0] if txt_files else ""
        if not mapping_path or not label_path:
            raise RuntimeError("Golden folder missing mapping or label file.")
        candidates = golden_core.parse_yolo_label_file(label_path)
        class_mapping = golden_core.load_mapping_from_dataset_yaml(mapping_path)
        targets: list[dict[str, Any]] = []
        for class_id, rect_norm in candidates:
            targets.append(
                {
                    "class_id": int(class_id),
                    "class_name": class_mapping.get(int(class_id)) if class_mapping else None,
                    "rect_norm": rect_norm,
                }
            )
        bg_cut_bundle_meta = golden_core.load_detect_background_cut_bundle(self, golden_dir)
        self._detect_bg_cut_bundle = bg_cut_bundle_meta.get("bundle") if bg_cut_bundle_meta else None
        cfg = golden_core.load_golden_id_config(golden_core.find_golden_id_config_in_folder(golden_dir)) or {}
        self._detect_golden_sample = {
            "label_path": os.path.abspath(label_path),
            "targets": targets,
            "mapping_path": os.path.abspath(mapping_path),
            "id_class_id": cfg.get("id_class_id"),
            "id_class_name": cfg.get("id_class_name"),
            "sub_id_class_id": cfg.get("sub_id_class_id"),
            "sub_id_class_name": cfg.get("sub_id_class_name"),
            "include_id_in_match": bool(self.args.include_id_in_match),
            "id_config_path": cfg.get("id_config_path", ""),
            "background_cut_root": bg_cut_bundle_meta.get("root") if bg_cut_bundle_meta else None,
            "background_cut_rules": bg_cut_bundle_meta.get("rules_path") if bg_cut_bundle_meta else None,
            "background_cut_template": bg_cut_bundle_meta.get("template_path") if bg_cut_bundle_meta else None,
        }

    def _scan_source_images(self) -> list[str]:
        src = os.path.abspath(str(self.args.source).strip())
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        return [str(p) for p in sorted(Path(src).iterdir()) if p.is_file() and p.suffix.lower() in exts]

    def _resolve_pending_image_paths(self, cv2_module) -> list[str]:
        source_images = [os.path.abspath(p) for p in self._scan_source_images()]
        if not source_images:
            return []
        if self.args.watch:
            if not self._watch_initial_scan_done:
                self._watch_initial_scan_done = True
                initial_mode = str(self.args.watch_once_initial).strip().lower()
                if initial_mode in {"false", "0", "no", "skip"}:
                    self._known_source_images = set(source_images)
                    return []
            added_sources = [p for p in source_images if p not in self._known_source_images]
            self._known_source_images = set(source_images)
        else:
            added_sources = source_images
            self._known_source_images = set(source_images)
        if not added_sources:
            return []
        if self._should_use_background_cut_detection():
            detect_paths = self._build_cut_background_source_images(added_sources, cv2_module)
        else:
            detect_paths = added_sources
        pending = [os.path.abspath(p) for p in detect_paths if os.path.abspath(p) not in self._processed_paths]
        return pending

    def _should_use_background_cut_detection(self) -> bool:
        return self.detect_run_mode_var.get() == "golden" and self._detect_bg_cut_bundle is not None

    def _build_cut_background_source_images(self, image_paths: list[str], cv2_module) -> list[str]:
        from ai_labeller import cut_background_detect

        bundle = self._detect_bg_cut_bundle
        if bundle is None or not image_paths:
            return []
        run_root = os.path.dirname(self._saved_image_dir) if self._saved_image_dir else os.getcwd()
        cut_root = os.path.join(run_root, "cut_input_images")
        os.makedirs(cut_root, exist_ok=True)
        out_paths: list[str] = []
        total_pieces = 0
        for src in image_paths:
            bgr = cv2_module.imread(src)
            if bgr is None or getattr(bgr, "size", 0) == 0:
                continue
            pieces = cut_background_detect.extract_cut_pieces_from_bgr(bgr, bundle)
            if not pieces:
                continue
            stem = os.path.splitext(os.path.basename(src))[0]
            for i, piece in enumerate(pieces, 1):
                out_path = os.path.join(cut_root, f"{stem}_cut_{i:03d}.png")
                if not cv2_module.imwrite(out_path, piece):
                    continue
                abs_out = os.path.abspath(out_path)
                out_paths.append(abs_out)
                self._detect_cut_piece_count_by_path[abs_out] = 1
                total_pieces += 1
        self._detect_last_cut_piece_count = total_pieces
        return out_paths

    def _class_color_bgr(self, class_id: int) -> tuple[int, int, int]:
        cid = int(class_id)
        if cid in self._detect_class_color_overrides:
            return self._detect_class_color_overrides[cid]
        palette = [
            (255, 85, 85),
            (85, 255, 85),
            (85, 170, 255),
            (255, 170, 85),
            (170, 85, 255),
            (255, 255, 85),
            (85, 255, 255),
            (255, 85, 255),
            (255, 255, 255),
            (0, 165, 255),
            (255, 0, 128),
            (128, 255, 0),
        ]
        return palette[cid % len(palette)]

    def _draw_detection_plot(self, result0: Any, cv2_module) -> Any:
        img = getattr(result0, "orig_img", None)
        if img is None:
            return result0.plot(line_width=1)
        out = img.copy()
        boxes = getattr(result0, "boxes", None)
        if boxes is None or getattr(boxes, "xyxy", None) is None:
            return out
        det_xyxy = boxes.xyxy.tolist()
        det_cls = boxes.cls.tolist() if getattr(boxes, "cls", None) is not None else []
        det_conf = boxes.conf.tolist() if getattr(boxes, "conf", None) is not None else []
        names_obj = getattr(result0, "names", None)
        idx_to_name: dict[int, str] = {}
        if isinstance(names_obj, dict):
            for key, value in names_obj.items():
                try:
                    idx_to_name[int(key)] = str(value)
                except Exception:
                    continue
        elif isinstance(names_obj, (list, tuple)):
            for i, value in enumerate(names_obj):
                idx_to_name[i] = str(value)
        self._detect_class_name_map = dict(idx_to_name)
        for i, box in enumerate(det_xyxy):
            try:
                cid = int(det_cls[i]) if i < len(det_cls) else -1
            except Exception:
                cid = -1
            color = self._class_color_bgr(cid)
            x1, y1, x2, y2 = map(int, [box[0], box[1], box[2], box[3]])
            cv2_module.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cname = idx_to_name.get(cid, str(cid))
            conf = float(det_conf[i]) if i < len(det_conf) else 0.0
            text = f"{cname} {conf:.2f}"
            (tw, th), _ = cv2_module.getTextSize(text, cv2_module.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty1 = max(0, y1 - th - 6)
            tx2 = x1 + tw + 6
            cv2_module.rectangle(out, (x1, ty1), (tx2, y1), color, -1)
            cv2_module.putText(out, text, (x1 + 3, max(12, y1 - 4)), cv2_module.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2_module.LINE_AA)
        return out

    def _draw_mismatch_overlay(self, plot_bgr: Any, cv2_module, result0: Any, status: str | None) -> Any:
        if str(status or "").strip().upper() != "FAIL":
            return plot_bgr
        rects = list(getattr(self, "_detect_spatial_mismatch_rects_norm", []) or [])
        if not rects:
            return plot_bgr
        out = plot_bgr.copy()
        h, w = getattr(result0, "orig_shape", (0, 0))
        if h <= 0 or w <= 0:
            return out
        for r in rects:
            try:
                x1 = int(float(r[0]) * float(w))
                y1 = int(float(r[1]) * float(h))
                x2 = int(float(r[2]) * float(w))
                y2 = int(float(r[3]) * float(h))
            except Exception:
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            cv2_module.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2_module.line(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2_module.line(out, (x2, y1), (x1, y2), (0, 0, 255), 2)
        return out

    def _infer(self, source: Any, cv2_module) -> tuple[Any, Any, dict[str, int]]:
        conf = float(self.args.conf)
        requested = str(self.args.device).strip().lower()
        first_device: Any = "cpu" if self._runtime_device == "cpu" else 0
        try:
            results = self._model.predict(source=source, conf=conf, verbose=False, device=first_device)
        except Exception as gpu_exc:
            msg = str(gpu_exc).lower()
            if requested == "cpu":
                raise
            if "cuda" in msg or "cudart" in msg or "no kernel image" in msg or "acceleratorerror" in msg:
                if not self._gpu_fallback_reported:
                    self.logger.warning("GPU inference failed, fallback to CPU for the remaining run: %s", gpu_exc)
                    self._gpu_fallback_reported = True
                self._runtime_device = "cpu"
                results = self._model.predict(source=source, conf=conf, verbose=False, device="cpu")
            else:
                raise
        result0 = results[0]
        plot_bgr = self._draw_detection_plot(result0, cv2_module)
        names_obj = getattr(result0, "names", None)
        idx_to_name: dict[int, str] = {}
        if isinstance(names_obj, dict):
            for key, value in names_obj.items():
                try:
                    idx_to_name[int(key)] = str(value)
                except Exception:
                    continue
        elif isinstance(names_obj, (list, tuple)):
            for i, value in enumerate(names_obj):
                idx_to_name[i] = str(value)
        counts: dict[str, int] = {}
        boxes = getattr(result0, "boxes", None)
        if boxes is not None and getattr(boxes, "cls", None) is not None:
            for cls_val in boxes.cls.tolist():
                try:
                    cls_idx = int(cls_val)
                except Exception:
                    continue
                cls_name = idx_to_name.get(cls_idx, str(cls_idx))
                counts[cls_name] = counts.get(cls_name, 0) + 1
        return result0, plot_bgr, counts

    def _evaluate_golden_match(self, result0: Any) -> tuple[str | None, str]:
        return golden_core.evaluate_golden_match(self, result0)

    def _resolve_golden_image_path_for_report(self) -> str:
        sample = self._detect_golden_sample or {}
        label_path = str(sample.get("label_path", "")).strip()
        if not label_path:
            return ""
        base_dir = os.path.dirname(os.path.abspath(label_path))
        stem = os.path.splitext(os.path.basename(label_path))[0]
        for ext in (".png", ".jpg", ".jpeg", ".bmp"):
            p = os.path.join(base_dir, stem + ext)
            if os.path.isfile(p):
                return os.path.abspath(p)
        return ""

    def _append_report_once(
        self,
        image_name: str,
        image_path: str,
        counts: dict[str, int],
        plot_bgr: Any,
        cv2_module,
        status: str | None,
        details: str,
    ) -> None:
        key = image_name.strip()
        if key in self._logged:
            return
        class_text = "; ".join(f"{k} x{v}" for k, v in sorted(counts.items())) if counts else "No detections"
        id_text = str(self._detect_last_ocr_id or "").strip()
        sub_id_text = str(self._detect_last_ocr_sub_id or "").strip() or "none"
        mode_text = self.detect_golden_mode_var.get() if self.detect_run_mode_var.get() == "golden" else ""
        iou_text = f"{float(self.detect_golden_iou_var.get()):.2f}" if self.detect_run_mode_var.get() == "golden" else ""
        reason_text = str(getattr(self, "_detect_last_fail_reason", "") or "").strip()
        status_text = str(status or "")
        path_text = os.path.abspath(str(image_path or "").strip()) if str(image_path or "").strip() else ""
        details_text = str(details or "")
        golden_image_path = self._resolve_golden_image_path_for_report()
        golden_label_path = str((self._detect_golden_sample or {}).get("label_path", "")).strip()
        ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        root = os.path.splitext(os.path.basename(key))[0]
        ext = os.path.splitext(os.path.basename(key))[1] or ".jpg"
        out_path = os.path.join(self._saved_image_dir, f"{root}{ext}")
        suffix = 1
        while os.path.exists(out_path):
            out_path = os.path.join(self._saved_image_dir, f"{root}_{suffix}{ext}")
            suffix += 1
        cv2_module.imwrite(out_path, plot_bgr)
        with open(self._csv_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    ts,
                    id_text,
                    sub_id_text,
                    image_name,
                    status_text,
                    class_text,
                    reason_text,
                    mode_text,
                    iou_text,
                    path_text,
                    os.path.abspath(out_path),
                    details_text,
                    golden_image_path,
                    golden_label_path,
                ]
            )
        self._logged.add(key)
        self._processed_paths.add(os.path.abspath(image_path))

    def _generate_detect_reports(self) -> None:
        from ai_labeller import detection_report_generator as drg

        loaded = drg.load_data(self._csv_path)
        if isinstance(loaded, tuple):
            records, has_golden = loaded
        else:
            records, has_golden = loaded, True
        sorted_classes, class_img_count, prefix_stats, status_counts, iou_values = drg.aggregate(records)
        base = os.path.splitext(self._csv_path)[0]
        xlsx_path = base + "_report.xlsx"
        html_path = base + "_dashboard.html"
        pdf_path = base + "_dashboard.pdf"
        drg.build_excel(records, sorted_classes, class_img_count, prefix_stats, status_counts, iou_values, has_golden, xlsx_path)
        drg.build_html(records, sorted_classes, class_img_count, prefix_stats, status_counts, iou_values, has_golden, html_path)
        try:
            drg.build_pdf(records, sorted_classes, class_img_count, prefix_stats, status_counts, iou_values, has_golden, pdf_path)
        except Exception as exc:
            self.logger.warning("PDF report generation failed: %s", exc)
            pdf_path = ""
        self._report_paths = {
            "csv": os.path.abspath(self._csv_path),
            "xlsx": os.path.abspath(xlsx_path),
            "html": os.path.abspath(html_path),
            "pdf": os.path.abspath(pdf_path) if pdf_path else "",
        }
        self._summary = self._build_summary(records, status_counts, has_golden)

    def _build_summary(self, records: list[dict[str, Any]], status_counts: dict[str, int], has_golden: bool) -> dict[str, Any]:
        status_map = {str(k): int(v) for k, v in status_counts.items()}
        total = len(records)
        sample_ids: list[str] = []
        for row in records[:10]:
            token = str(row.get("id") or row.get("image_name") or "").strip()
            if token:
                sample_ids.append(token)
        return {
            "mode": "golden" if has_golden else "pure_detect",
            "device_requested": str(self.args.device).strip().lower(),
            "device_used": self._runtime_device,
            "source": os.path.abspath(self.args.source),
            "output_dir": os.path.dirname(os.path.abspath(self._csv_path)),
            "total_records": total,
            "status_counts": status_map,
            "report_paths": self._report_paths,
            "sample_ids": sample_ids,
            "watch": bool(self.args.watch),
        }

    def _write_summary_json(self) -> None:
        if not self._summary:
            return
        target = str(getattr(self.args, "save_json", "") or "").strip()
        if not target:
            target = os.path.join(os.path.dirname(os.path.abspath(self._csv_path)), "summary.json")
        elif os.path.isdir(target):
            target = os.path.join(os.path.abspath(target), "summary.json")
        else:
            target = os.path.abspath(target)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self._summary, f, ensure_ascii=True, indent=2)
        self._report_paths["summary_json"] = target

    def _print_summary_stdout(self) -> None:
        if not self._summary:
            return
        print(json.dumps(self._summary, ensure_ascii=True, indent=2))

    def _run_batch(self, image_paths: list[str], cv2_module) -> None:
        total = len(image_paths)
        for index, image_path in enumerate(image_paths, 1):
            self.logger.info("[%s/%s] %s", index, total, os.path.basename(image_path))
            result0, plot_bgr, counts = self._infer(image_path, cv2_module)
            status, detail = self._evaluate_golden_match(result0)
            plot_bgr = self._draw_mismatch_overlay(plot_bgr, cv2_module, result0, status)
            self._append_report_once(os.path.basename(image_path), image_path, counts, plot_bgr, cv2_module, status, detail)
        if not self.args.no_report:
            self._generate_detect_reports()
            if self.args.save_json or self.args.summary_stdout:
                self._write_summary_json()
                self._summary["report_paths"] = dict(self._report_paths)
                if self.args.summary_stdout:
                    self._print_summary_stdout()

    def _finalize_outputs(self) -> int:
        if not self.args.no_report and not self._summary:
            self._generate_detect_reports()
        if (self.args.save_json or self.args.summary_stdout) and self._summary:
            self._write_summary_json()
            self._summary["report_paths"] = dict(self._report_paths)
            if self.args.summary_stdout:
                self._print_summary_stdout()
        self.logger.info("Done: %s", os.path.dirname(self._csv_path))
        if self.args.fail_exit_code and self._summary:
            fail_count = int((self._summary.get("status_counts") or {}).get("FAIL", 0))
            if fail_count > 0:
                return int(self.args.fail_exit_code)
        return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="geckoai-cli", description="GeckoAI command line tools")
    sub = parser.add_subparsers(dest="command", required=True)

    detect = sub.add_parser("detect", help="Run folder-based detection and generate reports")
    detect.add_argument("--model", required=True, help="YOLO model path (.pt/.onnx)")
    detect.add_argument("--source", required=True, help="Image folder")
    detect.add_argument("--output", required=True, help="Output folder for detect results")
    detect.add_argument("--golden-dir", default="", help="Golden project folder")
    detect.add_argument("--golden-mode", default="both", choices=["class", "position", "both"])
    detect.add_argument("--golden-iou", type=float, default=0.5)
    detect.add_argument("--include-id-in-match", action="store_true")
    detect.add_argument("--conf", type=float, default=0.5, help="Detection confidence")
    detect.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"])
    detect.add_argument("--no-report", action="store_true", help="Skip xlsx/html/pdf generation")
    detect.add_argument("--watch", action="store_true", help="Watch the source folder and detect new images continuously")
    detect.add_argument("--watch-interval", type=float, default=2.0, help="Polling interval in seconds for --watch")
    detect.add_argument("--watch-once-initial", default="true", help="For --watch: true=process existing images first, false=watch only new images")
    detect.add_argument("--class-color-map", default="", help="Inline map like 0=#FF0000,1=0,255,0 or a JSON file path")
    detect.add_argument("--save-json", default="", help="Write run summary JSON to this file or folder")
    detect.add_argument("--summary-stdout", action="store_true", help="Print JSON summary to stdout")
    detect.add_argument("--fail-exit-code", type=int, default=0, help="Return this non-zero code when any FAIL record exists")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "detect":
        runner = DetectCliRunner(args)
        return runner.run()
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
