from __future__ import annotations

from typing import Any

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None

try:
    import cv2
    HAS_CV2 = True
except Exception:
    cv2 = None
    HAS_CV2 = False

from tkinter import messagebox


def open_image_as_pil(app: Any, path: str, convert: str | None = "RGB", parent: Any = None):
    if Image is None:
        app.logger.error("Pillow is not available to open images")
        return None
    try:
        img = Image.open(path)
        if convert:
            img = img.convert(convert)
        return img
    except Exception as exc:
        try:
            messagebox.showerror("Error", f"Failed to open image:\n{exc}", parent=parent or getattr(app, 'root', None))
        except Exception:
            pass
        app.logger.exception("Failed to open image: %s", path)
        return None


def pil_resize_to_fit(pil_img, max_w: int, max_h: int, resample=None):
    if pil_img is None:
        return None, 1.0
    w, h = pil_img.width, pil_img.height
    scale = min(max_w / w, max_h / h) if w and h else 1.0
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    resample = resample if resample is not None else getattr(Image, 'Resampling', Image).BILINEAR
    resized = pil_img.resize((nw, nh), resample)
    return resized, scale


def pil_to_tk(pil_img):
    if ImageTk is None or pil_img is None:
        return None
    return ImageTk.PhotoImage(pil_img)


def read_cv2_image(path: str):
    if not HAS_CV2:
        return None
    try:
        return cv2.imread(path)
    except Exception:
        return None


def write_cv2_image(path: str, img) -> bool:
    if not HAS_CV2:
        return False
    try:
        return cv2.imwrite(path, img)
    except Exception:
        return False
