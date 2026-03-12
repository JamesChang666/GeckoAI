import datetime
import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageEnhance

from ai_labeller.core.geometry import calculate_iou
from ai_labeller.dialogs import filedialog, messagebox


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def _select_adjustable_roi(
    image: np.ndarray,
    window_title: str,
    initial_window_size: tuple[int, int] = (1100, 760),
) -> tuple[int, int, int, int] | None:
    h, w = image.shape[:2]
    if h <= 1 or w <= 1:
        return None
    handle_radius = 8
    min_size = 4
    state: dict[str, Any] = {
        "rect": None,  # [x1, y1, x2, y2]
        "mode": "idle",  # idle | draw | move | resize
        "handle": -1,
        "start": (0, 0),
        "base": None,
    }

    def _clamp(v: int, lo: int, hi: int) -> int:
        return int(max(lo, min(hi, v)))

    def _norm(rect: list[int] | tuple[int, int, int, int] | None) -> list[int] | None:
        if rect is None:
            return None
        x1, y1, x2, y2 = map(int, rect)
        x1, x2 = sorted([_clamp(x1, 0, w - 1), _clamp(x2, 0, w - 1)])
        y1, y2 = sorted([_clamp(y1, 0, h - 1), _clamp(y2, 0, h - 1)])
        return [x1, y1, x2, y2]

    def _handles(rect: list[int]) -> list[tuple[int, int]]:
        x1, y1, x2, y2 = rect
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        return [
            (x1, y1),  # TL
            (cx, y1),  # TM
            (x2, y1),  # TR
            (x2, cy),  # RM
            (x2, y2),  # BR
            (cx, y2),  # BM
            (x1, y2),  # BL
            (x1, cy),  # LM
        ]

    def _hit_handle(x: int, y: int, rect: list[int]) -> int:
        for i, (hx, hy) in enumerate(_handles(rect)):
            if (x - hx) * (x - hx) + (y - hy) * (y - hy) <= (handle_radius + 3) * (handle_radius + 3):
                return i
        return -1

    def _inside(x: int, y: int, rect: list[int]) -> bool:
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def _resize_from_handle(base: list[int], handle_idx: int, x: int, y: int) -> list[int]:
        x1, y1, x2, y2 = base
        x = _clamp(x, 0, w - 1)
        y = _clamp(y, 0, h - 1)
        if handle_idx == 0:  # TL
            x1 = min(x, x2 - min_size)
            y1 = min(y, y2 - min_size)
        elif handle_idx == 1:  # TM
            y1 = min(y, y2 - min_size)
        elif handle_idx == 2:  # TR
            x2 = max(x, x1 + min_size)
            y1 = min(y, y2 - min_size)
        elif handle_idx == 3:  # RM
            x2 = max(x, x1 + min_size)
        elif handle_idx == 4:  # BR
            x2 = max(x, x1 + min_size)
            y2 = max(y, y1 + min_size)
        elif handle_idx == 5:  # BM
            y2 = max(y, y1 + min_size)
        elif handle_idx == 6:  # BL
            x1 = min(x, x2 - min_size)
            y2 = max(y, y1 + min_size)
        elif handle_idx == 7:  # LM
            x1 = min(x, x2 - min_size)
        return _norm([x1, y1, x2, y2]) or [x1, y1, x2, y2]

    def _mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        del flags, param
        x = _clamp(int(x), 0, w - 1)
        y = _clamp(int(y), 0, h - 1)
        rect = state.get("rect")
        if event == cv2.EVENT_LBUTTONDOWN:
            if rect is not None:
                hit = _hit_handle(x, y, rect)
                if hit >= 0:
                    state["mode"] = "resize"
                    state["handle"] = hit
                    state["base"] = list(rect)
                    return
                if _inside(x, y, rect):
                    state["mode"] = "move"
                    state["start"] = (x, y)
                    state["base"] = list(rect)
                    return
            state["mode"] = "draw"
            state["rect"] = [x, y, x, y]
            state["base"] = None
            state["handle"] = -1
            return
        if event == cv2.EVENT_MOUSEMOVE:
            mode = str(state.get("mode", "idle"))
            if mode == "draw":
                cur = state.get("rect")
                if cur is not None:
                    cur[2] = x
                    cur[3] = y
                    state["rect"] = _norm(cur)
            elif mode == "resize":
                base = state.get("base")
                if isinstance(base, list):
                    state["rect"] = _resize_from_handle(base, int(state.get("handle", -1)), x, y)
            elif mode == "move":
                base = state.get("base")
                sx, sy = state.get("start", (x, y))
                if isinstance(base, list):
                    dx = int(x - sx)
                    dy = int(y - sy)
                    bw = int(base[2] - base[0])
                    bh = int(base[3] - base[1])
                    nx1 = _clamp(int(base[0] + dx), 0, max(0, w - 1 - bw))
                    ny1 = _clamp(int(base[1] + dy), 0, max(0, h - 1 - bh))
                    state["rect"] = [nx1, ny1, nx1 + bw, ny1 + bh]
            return
        if event == cv2.EVENT_LBUTTONUP:
            state["mode"] = "idle"
            state["handle"] = -1
            state["base"] = None
            state["rect"] = _norm(state.get("rect"))

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    try:
        win_w, win_h = initial_window_size
        cv2.resizeWindow(window_title, int(win_w), int(win_h))
    except Exception:
        pass
    cv2.setMouseCallback(window_title, _mouse)
    try:
        while True:
            canvas = image.copy()
            rect = _norm(state.get("rect"))
            if rect is not None:
                x1, y1, x2, y2 = rect
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 230, 40), 2, cv2.LINE_AA)
                for hx, hy in _handles(rect):
                    cv2.circle(canvas, (int(hx), int(hy)), handle_radius, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas, (int(hx), int(hy)), handle_radius - 2, (40, 230, 40), -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                "Drag to draw | Drag 8 handles to adjust | R: redraw | Enter: confirm | Esc: cancel",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "Drag to draw | Drag 8 handles to adjust | R: redraw | Enter: confirm | Esc: cancel",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (30, 30, 30),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(window_title, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10):
                rect = _norm(state.get("rect"))
                if rect is None:
                    continue
                x1, y1, x2, y2 = rect
                if (x2 - x1) <= min_size or (y2 - y1) <= min_size:
                    continue
                return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            if key == 27:
                return None
            if key in (ord("r"), ord("R")):
                state["rect"] = None
                state["mode"] = "idle"
                state["handle"] = -1
                state["base"] = None
    finally:
        try:
            cv2.destroyWindow(window_title)
            cv2.waitKey(1)
        except Exception:
            pass


def select_adjustable_template_roi(
    image: np.ndarray,
    window_title: str = "Select Golden Template",
    initial_window_size: tuple[int, int] = (1100, 760),
) -> tuple[int, int, int, int] | None:
    return _select_adjustable_roi(image, window_title, initial_window_size=initial_window_size)


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _apply_crop_logic(
    img: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray | None:
    hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = int(params["board_hsv"][0]), int(params["board_hsv"][1]), int(params["board_hsv"][2])
    h_tol, s_tol, v_tol = int(params["h_tol"]), int(params["s_tol"]), int(params["v_tol"])

    lower_board = np.array([max(0, h - h_tol), max(0, s - s_tol), max(0, v - v_tol)])
    upper_board = np.array([min(179, h + h_tol), min(255, s + s_tol), min(255, v + v_tol)])
    board_mask_full = cv2.inRange(hsv_full, lower_board, upper_board)
    final_mask_full = board_mask_full

    if params["bg_hsv"] is not None:
        bh, bs, bv = int(params["bg_hsv"][0]), int(params["bg_hsv"][1]), int(params["bg_hsv"][2])
        lower_bg = np.array([max(0, bh - h_tol), max(0, bs - s_tol), max(0, bv - v_tol)])
        upper_bg = np.array([min(179, bh + h_tol), min(255, bs + s_tol), min(255, bv + v_tol)])
        bg_mask_full = cv2.inRange(hsv_full, lower_bg, upper_bg)
        final_mask_full = cv2.bitwise_and(board_mask_full, cv2.bitwise_not(bg_mask_full))

    kernel_full = np.ones((11, 11), np.uint8)
    final_mask_full = cv2.morphologyEx(final_mask_full, cv2.MORPH_CLOSE, kernel_full, iterations=3)

    contours, _ = cv2.findContours(final_mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    main_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(main_contour)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype="float32")
    ordered_pts = _order_points(box)
    (tl, tr, br, bl) = ordered_pts

    width_a = float(np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2)))
    width_b = float(np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2)))
    max_width = max(int(width_a), int(width_b))
    height_a = float(np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2)))
    height_b = float(np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2)))
    max_height = max(int(height_a), int(height_b))
    if max_width <= 1 or max_height <= 1:
        return None

    dst_pts = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
    warped = cv2.warpPerspective(img, matrix, (max_width, max_height))
    if warped.shape[1] > warped.shape[0]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped


def _interactive_crop(image_path: str) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    state: dict[str, Any] = {
        "board_hsv": None,
        "bg_hsv": None,
        "display_img": None,
        "hsv_preview": None,
        "h_tol": 20,
        "s_tol": 60,
        "v_tol": 60,
    }

    def mouse_callback(event: int, x: int, y: int, flags: int, param: Any) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            state["board_hsv"] = state["hsv_preview"][y, x]
            cv2.circle(state["display_img"], (x, y), 8, (255, 255, 255), -1)
            cv2.circle(state["display_img"], (x, y), 6, (0, 255, 0), -1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["bg_hsv"] = state["hsv_preview"][y, x]
            cv2.circle(state["display_img"], (x, y), 8, (255, 255, 255), -1)
            cv2.circle(state["display_img"], (x, y), 6, (0, 0, 255), -1)

    img = cv2.imread(image_path)
    if img is None:
        return None, None

    h_orig, w_orig = img.shape[:2]
    if h_orig > 1000 or w_orig > 1000:
        scale = 800.0 / max(h_orig, w_orig)
        img_preview = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    else:
        img_preview = img.copy()

    state["hsv_preview"] = cv2.cvtColor(img_preview, cv2.COLOR_BGR2HSV)
    state["display_img"] = img_preview.copy()

    cv2.namedWindow("Control Panel", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Control Panel", 600, 700)
    cv2.setMouseCallback("Control Panel", mouse_callback)

    def nothing(_: int) -> None:
        return

    cv2.createTrackbar("H Tolerance", "Control Panel", 20, 90, nothing)
    cv2.createTrackbar("S Tolerance", "Control Panel", 60, 255, nothing)
    cv2.createTrackbar("V Tolerance", "Control Panel", 60, 255, nothing)

    while True:
        cv2.imshow("Control Panel", state["display_img"])
        if state["board_hsv"] is not None:
            state["h_tol"] = cv2.getTrackbarPos("H Tolerance", "Control Panel")
            state["s_tol"] = cv2.getTrackbarPos("S Tolerance", "Control Panel")
            state["v_tol"] = cv2.getTrackbarPos("V Tolerance", "Control Panel")

            h = int(state["board_hsv"][0])
            s = int(state["board_hsv"][1])
            v = int(state["board_hsv"][2])
            h_tol = int(state["h_tol"])
            s_tol = int(state["s_tol"])
            v_tol = int(state["v_tol"])

            lower_board = np.array([max(0, h - h_tol), max(0, s - s_tol), max(0, v - v_tol)])
            upper_board = np.array([min(179, h + h_tol), min(255, s + s_tol), min(255, v + v_tol)])
            board_mask = cv2.inRange(state["hsv_preview"], lower_board, upper_board)
            final_mask = board_mask

            if state["bg_hsv"] is not None:
                bh = int(state["bg_hsv"][0])
                bs = int(state["bg_hsv"][1])
                bv = int(state["bg_hsv"][2])
                lower_bg = np.array([max(0, bh - h_tol), max(0, bs - s_tol), max(0, bv - v_tol)])
                upper_bg = np.array([min(179, bh + h_tol), min(255, bs + s_tol), min(255, bv + v_tol)])
                bg_mask = cv2.inRange(state["hsv_preview"], lower_bg, upper_bg)
                final_mask = cv2.bitwise_and(board_mask, cv2.bitwise_not(bg_mask))

            kernel = np.ones((5, 5), np.uint8)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            cv2.imshow("Mask Preview (Adjust until board is white)", final_mask)

        key = cv2.waitKey(30) & 0xFF
        if key == 13:
            break
        if key == 27:
            cv2.destroyAllWindows()
            return None, None

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    if state["board_hsv"] is None:
        return None, None

    golden_params = {
        "board_hsv": state["board_hsv"],
        "bg_hsv": state["bg_hsv"],
        "h_tol": state["h_tol"],
        "s_tol": state["s_tol"],
        "v_tol": state["v_tol"],
    }
    warped = _apply_crop_logic(img, golden_params)
    return warped, golden_params


def _find_rois_by_minmax(
    result: np.ndarray,
    templ_w: int,
    templ_h: int,
    threshold: float = 0.3,
    max_matches: int = 300,
    overlap_thresh: float = 0.9,
) -> list[tuple[tuple[int, int], float]]:
    ys, xs = np.where(result >= threshold)
    if len(xs) == 0:
        return []
    scores = result[ys, xs]
    candidates = list(zip(xs.tolist(), ys.tolist(), scores.tolist()))
    candidates.sort(key=lambda item: item[2], reverse=True)
    selected: list[tuple[int, int, float]] = []
    boxes: list[tuple[int, int, int, int]] = []

    for x, y, score in candidates:
        if len(selected) >= max_matches:
            break
        box = (x, y, x + templ_w, y + templ_h)
        if any(calculate_iou(list(box), list(existing)) > overlap_thresh for existing in boxes):
            continue
        selected.append((x, y, score))
        boxes.append(box)
    return [((int(x), int(y)), float(score)) for x, y, score in selected]


def _enhance_gray(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return np.array(ImageEnhance.Contrast(Image.fromarray(gray)).enhance(100.0))


def _iter_images_recursive(root_dir: str) -> list[str]:
    out: list[str] = []
    for base, _dirs, files in os.walk(root_dir):
        for name in files:
            if name.lower().endswith(IMAGE_EXTENSIONS):
                out.append(os.path.join(base, name))
    return sorted(out)


@dataclass
class BatchResult:
    golden_dir: str
    output_dir: str
    total_images: int
    processed_images: int
    total_crops: int


@dataclass
class BackgroundCutBundle:
    root_dir: str
    rules_path: str
    template_path: str
    board_hsv: np.ndarray
    bg_hsv: np.ndarray | None
    h_tol: int
    s_tol: int
    v_tol: int
    match_threshold: float
    template_bgr: np.ndarray
    id_mode: bool
    id_roi_xywh: tuple[int, int, int, int] | None


def run_cut_background_batch(
    root_dir: str,
    threshold: float = 0.3,
    parent: Any = None,
) -> BatchResult | None:
    root_dir = os.path.abspath(root_dir)
    if not os.path.isdir(root_dir):
        messagebox.showerror("Cut Background", f"Folder not found:\n{root_dir}", parent=parent)
        return None

    golden_image_path = filedialog.askopenfilename(
        title="Select Golden Reference Photo",
        initialdir=root_dir,
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")],
        parent=parent,
    )
    if not golden_image_path:
        return None
    return run_cut_background_batch_with_golden(
        root_dir=root_dir,
        golden_image_path=golden_image_path,
        threshold=threshold,
        parent=parent,
    )


def run_cut_background_batch_with_golden(
    root_dir: str,
    golden_image_path: str,
    threshold: float = 0.3,
    parent: Any = None,
) -> BatchResult | None:
    root_dir = os.path.abspath(root_dir)
    golden_image_path = os.path.abspath(golden_image_path)
    if not os.path.isdir(root_dir):
        messagebox.showerror("Cut Background", f"Folder not found:\n{root_dir}", parent=parent)
        return None
    if not os.path.isfile(golden_image_path):
        messagebox.showerror("Cut Background", f"Golden image not found:\n{golden_image_path}", parent=parent)
        return None
    try:
        if os.path.commonpath([root_dir, golden_image_path]) != root_dir:
            messagebox.showwarning("Cut Background", "Golden image must be inside selected folder.", parent=parent)
            return None
    except Exception:
        pass

    messagebox.showinfo(
        "Golden Setup - Step 1",
        "Pick colors for the Golden Board.\n"
        "1) Left-Click Board\n"
        "2) Right-Click Background\n"
        "3) Adjust Trackbars\n"
        "4) Press ENTER",
        parent=parent,
    )

    golden_board, golden_params = _interactive_crop(golden_image_path)
    if golden_board is None or golden_params is None:
        return None

    messagebox.showinfo(
        "Golden Setup - Step 2",
        "Draw a box around the ID region.\n"
        "After drawing, drag 8 handles to adjust.\n"
        "R: redraw | ENTER: confirm | ESC: cancel.",
        parent=parent,
    )
    id_mode_enabled = True
    id_roi_xywh: list[int] | None = None
    id_roi = _select_adjustable_roi(golden_board, "Select Golden ID Region")
    if id_roi is None:
        return None
    idx, idy, idw, idh = map(int, id_roi)
    if idw <= 0 or idh <= 0:
        return None
    id_roi_xywh = [idx, idy, idw, idh]

    messagebox.showinfo(
        "Golden Setup - Step 3",
        "Draw a box around the Golden Template.\n"
        "After drawing, drag 8 handles to adjust.\n"
        "R: redraw | ENTER: confirm | ESC: cancel.",
        parent=parent,
    )
    roi = _select_adjustable_roi(golden_board, "Select Golden Template")

    if roi is None:
        return None
    x, y, w, h = map(int, roi)
    if w <= 0 or h <= 0:
        return None
    golden_template = golden_board[y : y + h, x : x + w].copy()
    if golden_template.size == 0:
        return None

    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    golden_dir = os.path.join(root_dir, f"golden_for_cut_background_{ts}")
    output_dir = os.path.join(root_dir, f"cut_background_detect_{ts}")
    os.makedirs(golden_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    rules_json = {
        "board_hsv": [int(v) for v in golden_params["board_hsv"]],
        "bg_hsv": [int(v) for v in golden_params["bg_hsv"]] if golden_params["bg_hsv"] is not None else None,
        "h_tol": int(golden_params["h_tol"]),
        "s_tol": int(golden_params["s_tol"]),
        "v_tol": int(golden_params["v_tol"]),
        "match_threshold": float(threshold),
        "id_mode": bool(id_mode_enabled),
        "id_roi_xywh": id_roi_xywh,
    }
    with open(os.path.join(golden_dir, "golden_rules.json"), "w", encoding="utf-8") as f:
        json.dump(rules_json, f, ensure_ascii=False, indent=2)
    cv2.imwrite(os.path.join(golden_dir, "golden_template.png"), golden_template)
    cv2.imwrite(os.path.join(golden_dir, "golden_board.png"), golden_board)
    with open(os.path.join(golden_dir, "golden_source_path.txt"), "w", encoding="utf-8") as f:
        f.write(os.path.abspath(golden_image_path) + "\n")

    excluded_dirs = {os.path.abspath(golden_dir), os.path.abspath(output_dir)}
    image_paths = [
        p
        for p in _iter_images_recursive(root_dir)
        if all(not os.path.abspath(p).startswith(excluded + os.sep) for excluded in excluded_dirs)
    ]
    if not image_paths:
        messagebox.showwarning("Cut Background", "No images found to process.", parent=parent)
        return None

    templ_gray = _enhance_gray(golden_template)
    templ_h, templ_w = templ_gray.shape[:2]
    total_crops = 0
    processed = 0
    tmp_dir = tempfile.mkdtemp(prefix="cutbg_")
    try:
        for src_path in image_paths:
            img = cv2.imread(src_path)
            if img is None:
                continue
            warped = _apply_crop_logic(img, golden_params)
            if warped is None:
                continue
            processed += 1
            straight_name = f"straight_{os.path.basename(src_path)}"
            straight_path = os.path.join(tmp_dir, straight_name)
            cv2.imwrite(straight_path, warped)

            board = cv2.imread(straight_path)
            if board is None:
                continue
            board_gray = _enhance_gray(board)
            bh, bw = board_gray.shape[:2]
            if bh < templ_h or bw < templ_w:
                # Template larger than board; skip this image safely.
                meta = {
                    "source_image": os.path.abspath(src_path),
                    "straight_board_path": os.path.abspath(straight_path),
                    "matches": 0,
                    "threshold": float(threshold),
                    "skipped_reason": "board_smaller_than_template",
                    "board_shape": [int(bh), int(bw)],
                    "template_shape": [int(templ_h), int(templ_w)],
                }
                rel_path = os.path.relpath(src_path, root_dir)
                rel_stem = os.path.splitext(rel_path)[0].replace("\\", "_").replace("/", "_")
                image_dir = os.path.join(output_dir, f"{rel_stem}_detect_cut_board_golden")
                os.makedirs(image_dir, exist_ok=True)
                with open(os.path.join(image_dir, "meta.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                continue
            result = cv2.matchTemplate(board_gray, templ_gray, cv2.TM_CCOEFF_NORMED)
            matches = _find_rois_by_minmax(
                result,
                templ_w,
                templ_h,
                threshold=float(threshold),
                max_matches=500,
                overlap_thresh=0.3,
            )

            rel_path = os.path.relpath(src_path, root_dir)
            rel_stem = os.path.splitext(rel_path)[0].replace("\\", "_").replace("/", "_")
            image_dir = os.path.join(output_dir, f"{rel_stem}_detect_cut_board_golden")
            os.makedirs(image_dir, exist_ok=True)

            meta = {
                "source_image": os.path.abspath(src_path),
                "straight_board_path": os.path.abspath(straight_path),
                "matches": len(matches),
                "threshold": float(threshold),
            }
            with open(os.path.join(image_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            for i, ((mx, my), score) in enumerate(matches, start=1):
                crop = board[my : my + templ_h, mx : mx + templ_w]
                if crop.size == 0:
                    continue
                total_crops += 1
                crop_name = f"golden_cut_{i:03d}_{score:.3f}.png"
                cv2.imwrite(os.path.join(image_dir, crop_name), crop)
    finally:
        try:
            for name in os.listdir(tmp_dir):
                path = os.path.join(tmp_dir, name)
                if os.path.isfile(path):
                    os.remove(path)
            os.rmdir(tmp_dir)
        except Exception:
            pass

    return BatchResult(
        golden_dir=os.path.abspath(golden_dir),
        output_dir=os.path.abspath(output_dir),
        total_images=len(image_paths),
        processed_images=processed,
        total_crops=total_crops,
    )


def _find_background_cut_bundle_files(root_dir: str) -> tuple[str, str] | None:
    root_dir = os.path.abspath(root_dir)
    candidates: list[tuple[str, str]] = []
    for base, _dirs, files in os.walk(root_dir):
        lowered = {f.lower(): f for f in files}
        if "golden_rules.json" in lowered and "golden_template.png" in lowered:
            rules_path = os.path.join(base, lowered["golden_rules.json"])
            template_path = os.path.join(base, lowered["golden_template.png"])
            candidates.append((rules_path, template_path))
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0])
    return candidates[0]


def load_background_cut_bundle(root_dir: str) -> BackgroundCutBundle | None:
    pair = _find_background_cut_bundle_files(root_dir)
    if pair is None:
        return None
    rules_path, template_path = pair
    with open(rules_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    board_hsv = np.array(data["board_hsv"], dtype=np.uint8)
    bg_raw = data.get("bg_hsv")
    bg_hsv = np.array(bg_raw, dtype=np.uint8) if bg_raw is not None else None
    h_tol = int(data["h_tol"])
    s_tol = int(data["s_tol"])
    v_tol = int(data["v_tol"])
    thr = float(data.get("match_threshold", 0.3))
    id_mode = bool(data.get("id_mode", False))
    id_roi_raw = data.get("id_roi_xywh")
    id_roi_xywh: tuple[int, int, int, int] | None = None
    if isinstance(id_roi_raw, (list, tuple)) and len(id_roi_raw) == 4:
        try:
            x, y, w, h = [int(v) for v in id_roi_raw]
            if w > 0 and h > 0:
                id_roi_xywh = (x, y, w, h)
        except Exception:
            id_roi_xywh = None
    template_bgr = cv2.imread(template_path)
    if template_bgr is None or template_bgr.size == 0:
        raise ValueError(f"Failed to load template image: {template_path}")
    return BackgroundCutBundle(
        root_dir=os.path.abspath(root_dir),
        rules_path=os.path.abspath(rules_path),
        template_path=os.path.abspath(template_path),
        board_hsv=board_hsv,
        bg_hsv=bg_hsv,
        h_tol=h_tol,
        s_tol=s_tol,
        v_tol=v_tol,
        match_threshold=max(0.01, min(1.0, thr)),
        template_bgr=template_bgr,
        id_mode=id_mode,
        id_roi_xywh=id_roi_xywh,
    )


def extract_cut_pieces_from_bgr(image_bgr: np.ndarray, bundle: BackgroundCutBundle) -> list[np.ndarray]:
    params = {
        "board_hsv": bundle.board_hsv,
        "bg_hsv": bundle.bg_hsv,
        "h_tol": bundle.h_tol,
        "s_tol": bundle.s_tol,
        "v_tol": bundle.v_tol,
    }
    warped = _apply_crop_logic(image_bgr, params)
    if warped is None:
        return []
    templ_gray = _enhance_gray(bundle.template_bgr)
    board_gray = _enhance_gray(warped)
    templ_h, templ_w = templ_gray.shape[:2]
    bh, bw = board_gray.shape[:2]
    if bh < templ_h or bw < templ_w:
        return []
    result = cv2.matchTemplate(board_gray, templ_gray, cv2.TM_CCOEFF_NORMED)
    matches = _find_rois_by_minmax(
        result,
        templ_w,
        templ_h,
        threshold=float(bundle.match_threshold),
        max_matches=2000,
        overlap_thresh=0.9,
    )
    pieces: list[np.ndarray] = []
    for (mx, my), _score in matches:
        crop = warped[my : my + templ_h, mx : mx + templ_w]
        if crop is None or crop.size == 0:
            continue
        pieces.append(crop.copy())
    return pieces
