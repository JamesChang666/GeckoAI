import gc
import os
import shutil
import warnings
from importlib import resources
from typing import Any

import numpy as np
from ai_labeller.dialogs import filedialog, messagebox

try:
    import cv2
except Exception:
    cv2 = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import torch
except Exception:
    torch = None

from ai_labeller.constants import COLORS, LANG_MAP


def use_official_yolo26n(app: Any) -> None:
    app.det_model_mode.set("Official YOLO26m.pt (Bundled)")
    app.yolo_path.set(app.config.yolo_model_path)
    app._register_model_path(app.config.yolo_model_path)
    app.yolo_model = None
    app._loaded_model_key = None


def _resolve_official_model_path(app: Any) -> str:
    candidates: list[str] = []
    try:
        packaged = resources.files("ai_labeller").joinpath("models", app.config.yolo_model_path)
        if packaged.is_file():
            return str(packaged)
    except Exception:
        app.logger.exception("Failed to resolve packaged official model path")

    candidates.append(app.config.yolo_model_path)
    candidates.append(os.path.join(os.path.dirname(__file__), "..", "models", app.config.yolo_model_path))
    candidates.append(os.path.join(os.getcwd(), app.config.yolo_model_path))

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f"Official model not found: {app.config.yolo_model_path}. Please reinstall package or choose a custom model."
    )


def _resolve_custom_model_path(app: Any, raw_path: str) -> str:
    path = (raw_path or "").strip().strip('"').strip("'")
    if not path:
        raise FileNotFoundError("Model file not found: empty path")

    normalized = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(normalized):
        return normalized

    candidates: list[str] = []
    lower_norm = normalized.lower()

    if os.path.isdir(normalized):
        candidates.extend(
            [
                os.path.join(normalized, "weights", "best.pt"),
                os.path.join(normalized, "weights", "last.pt"),
                os.path.join(normalized, "best.pt"),
                os.path.join(normalized, "last.pt"),
            ]
        )
    else:
        parent = os.path.dirname(normalized)
        name = os.path.basename(normalized).lower()
        if name == "best.pt":
            candidates.append(os.path.join(parent, "last.pt"))
        elif name == "last.pt":
            candidates.append(os.path.join(parent, "best.pt"))

        root, ext = os.path.splitext(normalized)
        if not ext:
            candidates.extend(
                [
                    os.path.join(normalized, "weights", "best.pt"),
                    os.path.join(normalized, "weights", "last.pt"),
                    os.path.join(normalized, "best.pt"),
                    os.path.join(normalized, "last.pt"),
                ]
            )
        if lower_norm.endswith(os.path.join("weights", "best.pt").lower()):
            run_dir = os.path.dirname(os.path.dirname(normalized))
            candidates.append(os.path.join(run_dir, "weights", "last.pt"))
        if lower_norm.endswith(os.path.join("weights", "last.pt").lower()):
            run_dir = os.path.dirname(os.path.dirname(normalized))
            candidates.append(os.path.join(run_dir, "weights", "best.pt"))

    for candidate in candidates:
        if os.path.isfile(candidate):
            app.logger.warning("Model path repaired: %s -> %s", normalized, candidate)
            return os.path.abspath(candidate)

    raise FileNotFoundError(f"Model file not found:\n{normalized}")


def ensure_yolo_model(app: Any, loaded_key: tuple[str, str], model_path: str) -> None:
    """Ensure `app.yolo_model` is loaded for the given `loaded_key`.

    If a different model is already loaded, it will be released and the new
    model will be loaded. Raises a RuntimeError on failure.
    """
    if YOLO is None:
        raise RuntimeError("YOLO is not available in this Python environment")
    model_path = os.path.abspath(model_path)
    if app.yolo_model is None or app._loaded_model_key != loaded_key:
        # release previous model if any
        if app.yolo_model is not None:
            try:
                del app.yolo_model
            except Exception:
                pass
            app.yolo_model = None
            gc.collect()
        try:
            app.logger.info("Loading YOLO model: %s", model_path)
            app.yolo_model = YOLO(model_path)
            app._loaded_model_key = loaded_key
        except Exception as exc:
            app.yolo_model = None
            app._loaded_model_key = None
            raise RuntimeError(f"Failed to load model: {exc}") from exc


def pick_model_file(app: Any, forced_mode: str | None = None) -> bool:
    app.logger.info("Opening model file dialog (mode=%s)", forced_mode or "auto")
    model_path = filedialog.askopenfilename(
        parent=app.root,
        title="Select model",
        filetypes=[
            ("Model files", "*.pt *.onnx"),
            ("PyTorch", "*.pt"),
            ("ONNX", "*.onnx"),
            ("All files", "*.*"),
        ],
    )
    if not model_path:
        app.logger.info("Model selection cancelled")
        return False
    model_path = os.path.abspath(model_path)
    if not os.path.isfile(model_path):
        app.logger.error("Selected model file not found: %s", model_path)
        messagebox.showerror("Model Error", f"Model file not found:\n{model_path}")
        return False
    if not model_path.lower().endswith((".pt", ".onnx")):
        proceed = messagebox.askyesno(
            "Model Warning",
            f"Selected file may not be a YOLO model:\n{os.path.basename(model_path)}\n\nContinue?",
        )
        if not proceed:
            app.logger.info("Model selection rejected by user due to extension warning")
            return False
    app.yolo_path.set(model_path)
    app._register_model_path(model_path)
    if forced_mode:
        app.det_model_mode.set(forced_mode)
    elif "rfdetr" in os.path.basename(model_path).lower():
        app.det_model_mode.set("Custom RF-DETR")
    else:
        app.det_model_mode.set("Custom YOLO (v5/v7/v8/v9/v11/v26)")
    app.yolo_model = None
    app._loaded_model_key = None
    app.save_session_state()
    app.logger.info("Model selected: %s (%s)", model_path, app.det_model_mode.get())
    return True


def browse_detection_model(app: Any) -> None:
    pick_model_file(app)


def autolabel_red(app: Any) -> None:
    if cv2 is None or not app.img_pil:
        return
    try:
        app.push_history()

        img = cv2.cvtColor(np.array(app.img_pil), cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        _, a, _ = cv2.split(lab)
        _, red = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        k = max(1, int(app.config.red_detection_kernel_size))
        kernel = np.ones((k, k), np.uint8)
        red = cv2.dilate(
            red,
            kernel,
            iterations=max(1, int(app.config.red_detection_dilate_iterations)),
        )

        contours, _ = cv2.findContours(
            red,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > app.config.auto_label_min_area:
                app.rects.append(app.clamp_box([
                    x, y, x + w, y + h,
                    app.combo_cls.current()
                ]))

        app.render()
    except Exception:
        app.logger.exception("Red auto-label failed")
        messagebox.showerror("Error", "Red auto-label failed. See logs for details.")


def _is_cuda_kernel_compat_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "no kernel image is available for execution on the device" in msg
        or "cudaerrornokernelimagefordevice" in msg
    )


def _can_use_cuda_runtime() -> bool:
    if torch is None:
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            if not torch.cuda.is_available():
                return False
            cap = torch.cuda.get_device_capability(0)
            sm = f"sm_{cap[0]}{cap[1]}"
            arch_list = []
            if hasattr(torch.cuda, "get_arch_list"):
                arch_list = list(torch.cuda.get_arch_list() or [])
            if arch_list and sm not in arch_list:
                return False
            _ = torch.zeros((1,), device="cuda")
            return True
    except Exception:
        return False


def _auto_runtime_device(app: Any, allow_forced_cpu: bool = True) -> str:
    if allow_forced_cpu and getattr(app, "_force_cpu_detection", False):
        return "cpu"
    return "0" if _can_use_cuda_runtime() else "cpu"


def run_yolo_detection(app: Any):
    if YOLO is None:
        messagebox.showwarning("YOLO Not Available", "Please install ultralytics first.")
        return
    if not app.img_pil:
        messagebox.showwarning("No Image", "Please load an image first.")
        return

    try:
        mode = app.det_model_mode.get()
        if mode == "Official YOLO26m.pt (Bundled)":
            model_path = _resolve_official_model_path(app)
        else:
            model_path = _resolve_custom_model_path(app, app.yolo_path.get().strip())

        if not model_path:
            messagebox.showwarning("Model", "Please choose a model file first.")
            return
        model_path = os.path.abspath(model_path)
        app.yolo_path.set(model_path)
        app._register_model_path(model_path)

        loaded_key = (mode, model_path)
        if app.yolo_model is None or app._loaded_model_key != loaded_key:
            app.root.config(cursor="watch")
            app.root.update_idletasks()
            if app.yolo_model is not None:
                del app.yolo_model
                app.yolo_model = None
                gc.collect()
            try:
                app.logger.info("Loading YOLO model: %s", model_path)
                app.yolo_model = YOLO(model_path)
                app._loaded_model_key = loaded_key
            except Exception as exc:
                app.yolo_model = None
                app._loaded_model_key = None
                raise RuntimeError(f"Failed to load model: {exc}") from exc
            finally:
                app.root.config(cursor="")
                app.root.update_idletasks()

        preferred_device = 0 if _auto_runtime_device(app) == "0" else "cpu"

        try:
            results = app.yolo_model(
                app.img_pil,
                conf=app.var_yolo_conf.get(),
                verbose=False,
                device=preferred_device,
            )
        except RuntimeError as exc:
            if preferred_device != "cpu" and _is_cuda_kernel_compat_error(exc):
                app.logger.warning(
                    "CUDA kernel compatibility error detected; retrying YOLO detection on CPU. error=%s",
                    exc,
                )
                app._force_cpu_detection = True
                results = app.yolo_model(
                    app.img_pil,
                    conf=app.var_yolo_conf.get(),
                    verbose=False,
                    device="cpu",
                )
            else:
                raise

        app.push_history()
        detection_count = 0
        fallback_class_idx = app.combo_cls.current()
        if fallback_class_idx < 0:
            fallback_class_idx = 0
        for result in results:
            if result.boxes is None:
                continue
            for det_idx, box in enumerate(result.boxes.xyxy):
                class_idx = app._resolve_detected_class_index(result, det_idx, fallback_class_idx)
                app.rects.append(app.clamp_box([
                    box[0].item(),
                    box[1].item(),
                    box[2].item(),
                    box[3].item(),
                    class_idx
                ]))
                detection_count += 1

        app.render()
        app.logger.info("YOLO detection complete: %s boxes", detection_count)
    except FileNotFoundError as exc:
        app.logger.error("Model path error: %s", exc)
        messagebox.showerror("Model Error", str(exc))
    except MemoryError:
        app.logger.exception("Out of memory during YOLO detection")
        messagebox.showerror("Memory Error", "Not enough memory for detection.")
    except RuntimeError as exc:
        app.logger.exception("YOLO runtime error")
        messagebox.showerror("Detection Error", str(exc))
    except Exception as exc:
        app.logger.exception("YOLO detection failed")
        messagebox.showerror("Detection Error", f"YOLO detection failed:\n{exc}")
