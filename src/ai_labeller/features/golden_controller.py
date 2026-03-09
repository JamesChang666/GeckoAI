import tkinter as tk
from typing import Any
from PIL import Image, ImageTk
import os
import glob
import shutil
import tempfile

from ai_labeller.features import image_utils
from ai_labeller.constants import COLORS
from ai_labeller.features import golden as golden_core
from ai_labeller.core import atomic_write_text


def parse_yolo_label_file(label_path: str):
    return golden_core.parse_yolo_label_file(label_path)


def find_dataset_yaml_for_label(label_path: str):
    return golden_core.find_dataset_yaml_for_label(label_path)


def find_dataset_yaml_in_folder(folder: str):
    return golden_core.find_dataset_yaml_in_folder(folder)


def load_mapping_from_dataset_yaml(yaml_path: str):
    return golden_core.load_mapping_from_dataset_yaml(yaml_path)


def find_golden_id_config_in_folder(folder: str):
    return golden_core.find_golden_id_config_in_folder(folder)


def load_golden_id_config(json_path: str | None):
    return golden_core.load_golden_id_config(json_path)


def write_golden_id_config(folder: str, class_id: int | None, class_name: str | None, sub_id_class_id: int | None = None, sub_id_class_name: str | None = None):
    return golden_core.write_golden_id_config(folder, class_id, class_name, sub_id_class_id=sub_id_class_id, sub_id_class_name=sub_id_class_name)


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    return golden_core.bbox_iou(a, b)


def evaluate_golden_match(app: Any, result0: Any) -> tuple[str | None, str]:
    return golden_core.evaluate_golden_match(app, result0)


def normalize_golden_mode(mode_raw: str) -> str:
    return golden_core.normalize_golden_mode(mode_raw)


def pick_golden_rect_on_image(app: Any, image_path: str) -> tuple[float, float, float, float] | None:
    pil_img = image_utils.open_image_as_pil(app, image_path, convert="RGB", parent=app.root)
    if pil_img is None:
        return None

    win = tk.Toplevel(app.root)
    win.title("Golden Sample - Draw Position Box")
    win.geometry("1000x740")
    win.transient(app.root)
    win.grab_set()

    canvas = tk.Canvas(win, bg="#202020", highlightthickness=0)
    canvas.pack(fill="both", expand=True, padx=8, pady=8)
    ctrl = tk.Frame(win, bg=COLORS["bg_white"])
    ctrl.pack(fill="x", padx=8, pady=(0, 8))

    state: dict[str, Any] = {
        "scale": 1.0,
        "offset_x": 0.0,
        "offset_y": 0.0,
        "start": None,
        "rect_id": None,
        "rect_canvas": None,
        "result": None,
        "img_tk": None,
    }

    def redraw() -> None:
        cw = max(1, canvas.winfo_width())
        ch = max(1, canvas.winfo_height())
        scale = min(cw / pil_img.width, ch / pil_img.height)
        nw = max(1, int(pil_img.width * scale))
        nh = max(1, int(pil_img.height * scale))
        ox = (cw - nw) / 2
        oy = (ch - nh) / 2
        state["scale"] = scale
        state["offset_x"] = ox
        state["offset_y"] = oy
        resized = pil_img.resize((nw, nh), Image.Resampling.BILINEAR)
        state["img_tk"] = ImageTk.PhotoImage(resized)
        canvas.delete("all")
        canvas.create_image(ox, oy, image=state["img_tk"], anchor="nw")
        if state["rect_canvas"] is not None:
            x1, y1, x2, y2 = state["rect_canvas"]
            state["rect_id"] = canvas.create_rectangle(x1, y1, x2, y2, outline="#00E676", width=2)

    def to_image_coords(cx: float, cy: float) -> tuple[float, float]:
        ix = (cx - float(state["offset_x"])) / max(float(state["scale"]), 1e-6)
        iy = (cy - float(state["offset_y"])) / max(float(state["scale"]), 1e-6)
        ix = max(0.0, min(float(pil_img.width), ix))
        iy = max(0.0, min(float(pil_img.height), iy))
        return ix, iy

    def on_down(e: Any) -> None:
        state["start"] = (e.x, e.y)
        state["rect_canvas"] = (e.x, e.y, e.x, e.y)
        redraw()

    def on_drag(e: Any) -> None:
        if state["start"] is None:
            return
        sx, sy = state["start"]
        state["rect_canvas"] = (sx, sy, e.x, e.y)
        redraw()

    def on_up(e: Any) -> None:
        if state["start"] is None:
            return
        sx, sy = state["start"]
        state["rect_canvas"] = (sx, sy, e.x, e.y)
        redraw()

    def confirm() -> None:
        if state["rect_canvas"] is None:
            tk.messagebox.showwarning("Golden Sample", "Please draw a box first.", parent=win)
            return
        x1, y1, x2, y2 = state["rect_canvas"]
        ix1, iy1 = to_image_coords(min(x1, x2), min(y1, y2))
        ix2, iy2 = to_image_coords(max(x1, x2), max(y1, y2))
        if abs(ix2 - ix1) < 2 or abs(iy2 - iy1) < 2:
            tk.messagebox.showwarning("Golden Sample", "Box is too small.", parent=win)
            return
        state["result"] = (
            ix1 / max(1.0, float(pil_img.width)),
            iy1 / max(1.0, float(pil_img.height)),
            ix2 / max(1.0, float(pil_img.width)),
            iy2 / max(1.0, float(pil_img.height)),
        )
        win.destroy()

    def cancel() -> None:
        state["result"] = None
        win.destroy()

    canvas.bind("<ButtonPress-1>", on_down)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_up)
    canvas.bind("<Configure>", lambda _e: redraw())

    app.create_primary_button(ctrl, text="Confirm", command=confirm, bg=COLORS["success"]).pack(side="right", padx=8, pady=8)
    app.create_secondary_button(ctrl, text="Cancel", command=cancel).pack(side="right", padx=8, pady=8)
    tk.Label(ctrl, text="Drag to draw golden position box", bg=COLORS["bg_white"], fg=COLORS["text_secondary"], font=app.font_primary).pack(side="left", padx=8)

    win.update_idletasks()
    redraw()
    app.root.wait_window(win)
    return state["result"]


def configure_detect_golden_sample(app: Any) -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    import os

    golden_dir = filedialog.askdirectory(
        parent=app.root,
        title="Select golden folder (yaml + label txt)",
    )
    if not golden_dir:
        return
    golden_dir = os.path.abspath(golden_dir)

    mapping_path = golden_core.find_dataset_yaml_in_folder(golden_dir)
    if not mapping_path:
        messagebox.showwarning(
            "Golden Sample",
            "No dataset.yaml/data.yaml found in selected folder.",
            parent=app.root,
        )
        return

    txt_files = sorted(
        p for p in glob.glob(os.path.join(golden_dir, "*.txt"))
        if os.path.isfile(p)
    )
    if not txt_files:
        messagebox.showwarning(
            "Golden Sample",
            "No label txt found in selected folder.",
            parent=app.root,
        )
        return
    if len(txt_files) == 1:
        label_path = txt_files[0]
    else:
        label_path = filedialog.askopenfilename(
            parent=app.root,
            title="Select golden label txt in folder",
            initialdir=golden_dir,
            filetypes=[("YOLO label", "*.txt"), ("All files", "*.*")],
        )
        if not label_path:
            return

    candidates = golden_core.parse_yolo_label_file(label_path)
    if not candidates:
        messagebox.showwarning("Golden Sample", "No valid YOLO labels found in selected file.", parent=app.root)
        return
    class_mapping = golden_core.load_mapping_from_dataset_yaml(mapping_path)
    id_cfg_path = golden_core.find_golden_id_config_in_folder(golden_dir)
    id_cfg = golden_core.load_golden_id_config(id_cfg_path)
    targets: list[dict[str, Any]] = []
    for class_id, rect_norm in candidates:
        class_name = class_mapping.get(int(class_id)) if class_mapping else None
        targets.append(
            {
                "class_id": int(class_id),
                "class_name": class_name,
                "rect_norm": rect_norm,
            }
        )
    if not targets:
        messagebox.showwarning("Golden Sample", "No valid target in selected label.", parent=app.root)
        return
    first_name = targets[0].get("class_name")
    app.detect_golden_class_var.set(str(first_name or targets[0].get("class_id")))
    bg_cut_bundle_meta = golden_core.load_detect_background_cut_bundle(app, golden_dir)
    app._detect_bg_cut_bundle = bg_cut_bundle_meta.get("bundle") if bg_cut_bundle_meta else None

    app._detect_golden_sample = {
        "label_path": os.path.abspath(label_path),
        "targets": targets,
        "mapping_path": os.path.abspath(mapping_path),
        "id_class_id": id_cfg.get("id_class_id") if id_cfg else None,
        "id_class_name": id_cfg.get("id_class_name") if id_cfg else None,
        "sub_id_class_id": id_cfg.get("sub_id_class_id") if id_cfg else None,
        "sub_id_class_name": id_cfg.get("sub_id_class_name") if id_cfg else None,
        "id_config_path": id_cfg.get("id_config_path") if id_cfg else None,
        "background_cut_root": bg_cut_bundle_meta.get("root") if bg_cut_bundle_meta else None,
        "background_cut_rules": bg_cut_bundle_meta.get("rules_path") if bg_cut_bundle_meta else None,
        "background_cut_template": bg_cut_bundle_meta.get("template_path") if bg_cut_bundle_meta else None,
    }
    app.detect_run_mode_var.set("golden")
    app._show_detect_settings_page_for_current_source()


def create_detect_golden_from_label_mode(app: Any) -> None:
    from tkinter import filedialog, messagebox
    import tempfile
    import shutil
    import os

    img_path = filedialog.askopenfilename(
        parent=app.root,
        title="Select one golden image to annotate in Label Mode",
        filetypes=[
            ("Images", "*.jpg *.jpeg *.png *.bmp"),
            ("All files", "*.*"),
        ],
    )
    if not img_path:
        return
    out_dir = filedialog.askdirectory(
        parent=app.root,
        title="Select output folder for golden txt/yaml",
        initialdir=app.detect_output_dir_var.get().strip() or os.path.dirname(os.path.abspath(img_path)),
    )
    if not out_dir:
        return
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    img_path = os.path.abspath(img_path)
    tmp_root = tempfile.mkdtemp(prefix="golden_label_")
    tmp_root = tmp_root.replace("\\", "/")
    os.makedirs(f"{tmp_root}/images/train", exist_ok=True)
    os.makedirs(f"{tmp_root}/labels/train", exist_ok=True)
    img_name = os.path.basename(img_path)
    tmp_img = f"{tmp_root}/images/train/{img_name}"
    shutil.copy2(img_path, tmp_img)

    class_options = [str(c).strip() for c in (app.class_names or []) if str(c).strip()]
    if not class_options:
        class_options = ["class0"]
    yaml_lines = [f"path: {tmp_root}", "train: images/train", "val: images/train", f"nc: {len(class_options)}", "names:"]
    for i, name in enumerate(class_options):
        safe = name.replace('"', '\\"')
        yaml_lines.append(f'  {i}: "{safe}"')
    golden_core.atomic_write_text(f"{tmp_root}/dataset.yaml", "\n".join(yaml_lines) + "\n")

    app._golden_capture_active = True
    app._golden_capture_temp_root = tmp_root
    app._golden_capture_output_dir = out_dir
    app._golden_capture_image_name = img_name

    app._stop_detect_stream()
    app._detect_mode_active = False
    app.rebuild_ui()
    app.load_project_from_path(tmp_root, preferred_image=img_name)
    messagebox.showinfo(
        "Golden Sample",
        "Now in full Label Mode for golden image.\nAnnotate boxes/classes, then click toolbar 'Save Golden'.",
        parent=app.root,
    )


def finalize_golden_from_label_mode(app: Any) -> None:
    from tkinter import messagebox
    import os
    import shutil

    if not app._golden_capture_active or not app._golden_capture_temp_root or not app._golden_capture_image_name:
        messagebox.showwarning("Golden Sample", "Golden capture is not active.", parent=app.root)
        return
    try:
        app.save_current()
    except Exception:
        app.logger.exception("Failed to save current annotations before golden finalize")

    tmp_root = app._golden_capture_temp_root
    img_name = app._golden_capture_image_name
    out_dir = app._golden_capture_output_dir or app.detect_output_dir_var.get().strip()
    if not out_dir:
        messagebox.showwarning("Golden Sample", "Output folder is missing.", parent=app.root)
        return
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    stem = os.path.splitext(img_name)[0]
    lbl_src = f"{tmp_root}/labels/train/{stem}.txt"
    if not os.path.isfile(lbl_src):
        messagebox.showwarning("Golden Sample", "No labels found. Please annotate at least one box.", parent=app.root)
        return
    yaml_src = f"{tmp_root}/dataset.yaml"
    lbl_dst = os.path.join(out_dir, f"{stem}.txt")
    yaml_dst = os.path.join(out_dir, "dataset.yaml")
    try:
        shutil.copy2(lbl_src, lbl_dst)
        shutil.copy2(yaml_src, yaml_dst)
    except Exception as exc:
        messagebox.showerror("Golden Sample", f"Failed to export golden files:\n{exc}", parent=app.root)
        return

    candidates = golden_core.parse_yolo_label_file(lbl_dst)
    class_mapping = golden_core.load_mapping_from_dataset_yaml(yaml_dst)
    targets: list[dict[str, Any]] = []
    for class_id, rect_norm in candidates:
        targets.append(
            {
                "class_id": int(class_id),
                "class_name": class_mapping.get(int(class_id)),
                "rect_norm": rect_norm,
            }
        )
    if not targets:
        messagebox.showwarning("Golden Sample", "Exported label has no valid targets.", parent=app.root)
        return

    app.detect_output_dir_var.set(out_dir)
    app.detect_golden_class_var.set(str(targets[0].get("class_name") or targets[0].get("class_id")))
    id_choice, sub_id_choice = pick_golden_id_classes(app, class_mapping, parent=app.root)
    id_cfg_path = None
    id_class_id = None
    id_class_name = None
    sub_id_class_id = None
    sub_id_class_name = None
    if id_choice is not None:
        id_class_id, id_class_name = id_choice
    if sub_id_choice is not None:
        sub_id_class_id, sub_id_class_name = sub_id_choice
    if id_choice is not None or sub_id_choice is not None:
        id_cfg_path = golden_core.write_golden_id_config(
            out_dir,
            id_class_id,
            id_class_name,
            sub_id_class_id=sub_id_class_id,
            sub_id_class_name=sub_id_class_name,
        )
    app._detect_bg_cut_bundle = None
    app._detect_golden_sample = {
        "label_path": lbl_dst,
        "targets": targets,
        "mapping_path": yaml_dst,
        "image_path": os.path.abspath(f"{tmp_root}/images/train/{img_name}"),
        "id_class_id": id_class_id,
        "id_class_name": id_class_name,
        "sub_id_class_id": sub_id_class_id,
        "sub_id_class_name": sub_id_class_name,
        "id_config_path": id_cfg_path,
        "background_cut_root": None,
        "background_cut_rules": None,
        "background_cut_template": None,
    }
    app.detect_run_mode_var.set("golden")
    _cleanup_golden_capture_temp(app)
    app._show_detect_settings_page_for_current_source()
    messagebox.showinfo(
        "Golden Sample",
        f"Golden exported:\nLabel: {lbl_dst}\nMapping YAML: {yaml_dst}",
        parent=app.root,
    )


def cancel_golden_capture_and_back_to_detect(app: Any) -> None:
    if not getattr(app, "_golden_capture_active", False):
        app._show_detect_settings_page_for_current_source()
        return
    _cleanup_golden_capture_temp(app)
    app._show_detect_settings_page_for_current_source()


def _cleanup_golden_capture_temp(app: Any) -> None:
    import shutil
    tmp_root = getattr(app, "_golden_capture_temp_root", None)
    app._golden_capture_active = False
    app._golden_capture_temp_root = None
    app._golden_capture_output_dir = None
    app._golden_capture_image_name = None
    if tmp_root and os.path.isdir(tmp_root):
        try:
            shutil.rmtree(tmp_root, ignore_errors=True)
        except Exception:
            app.logger.exception("Failed to cleanup golden temp root: %s", tmp_root)


def annotate_golden_image_label_style(app: Any, image_path: str, class_options: list[str]) -> list[dict[str, Any]] | None:
    from tkinter import messagebox, ttk

    pil_img = image_utils.open_image_as_pil(app, image_path, convert="RGB", parent=app.root)
    if pil_img is None:
        return None

    win = tk.Toplevel(app.root)
    win.title("Golden Annotation (Label-style)")
    win.geometry("1100x800")
    win.transient(app.root)
    win.grab_set()

    main = tk.Frame(win, bg=app.COLORS["bg_dark"])
    main.pack(fill="both", expand=True)
    left = tk.Frame(main, bg="#202020")
    left.pack(side="left", fill="both", expand=True, padx=(8, 4), pady=8)
    right = tk.Frame(main, bg=app.COLORS["bg_white"], width=280)
    right.pack(side="right", fill="y", padx=(4, 8), pady=8)
    right.pack_propagate(False)

    canvas = tk.Canvas(left, bg="#202020", highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    tk.Label(right, text="Class", bg=app.COLORS["bg_white"], fg=app.COLORS["text_primary"], font=app.font_bold).pack(
        anchor="w", padx=12, pady=(12, 4)
    )
    class_var = tk.StringVar(value=class_options[0])
    class_combo = ttk.Combobox(right, values=class_options, textvariable=class_var, state="readonly", font=app.font_primary)
    class_combo.pack(fill="x", padx=12, pady=(0, 10))

    tk.Label(right, text="Boxes", bg=app.COLORS["bg_white"], fg=app.COLORS["text_primary"], font=app.font_bold).pack(
        anchor="w", padx=12, pady=(0, 4)
    )
    box_list = tk.Listbox(right, font=app.font_primary, bg=app.COLORS["bg_light"], fg=app.COLORS["text_primary"], relief="flat")
    box_list.pack(fill="both", expand=True, padx=12, pady=(0, 10))

    state: dict[str, Any] = {
        "scale": 1.0,
        "offset_x": 0.0,
        "offset_y": 0.0,
        "start": None,
        "temp_rect": None,
        "anns": [],
        "img_tk": None,
        "result": None,
    }

    def img_to_canvas(ix: float, iy: float) -> tuple[float, float]:
        return ix * state["scale"] + state["offset_x"], iy * state["scale"] + state["offset_y"]

    def canvas_to_img(cx: float, cy: float) -> tuple[float, float]:
        ix = (cx - state["offset_x"]) / max(state["scale"], 1e-6)
        iy = (cy - state["offset_y"]) / max(state["scale"], 1e-6)
        ix = max(0.0, min(float(pil_img.width), ix))
        iy = max(0.0, min(float(pil_img.height), iy))
        return ix, iy

    def redraw() -> None:
        cw = max(1, canvas.winfo_width())
        ch = max(1, canvas.winfo_height())
        scale = min(cw / pil_img.width, ch / pil_img.height)
        nw = max(1, int(pil_img.width * scale))
        nh = max(1, int(pil_img.height * scale))
        ox = (cw - nw) / 2
        oy = (ch - nh) / 2
        state["scale"] = scale
        state["offset_x"] = ox
        state["offset_y"] = oy
        resized = pil_img.resize((nw, nh), Image.Resampling.BILINEAR)
        state["img_tk"] = ImageTk.PhotoImage(resized)
        canvas.delete("all")
        canvas.create_image(ox, oy, image=state["img_tk"], anchor="nw")

        for idx, ann in enumerate(state["anns"]):
            x1, y1, x2, y2 = ann["rect_norm"]
            c1 = img_to_canvas(x1 * pil_img.width, y1 * pil_img.height)
            c2 = img_to_canvas(x2 * pil_img.width, y2 * pil_img.height)
            canvas.create_rectangle(c1[0], c1[1], c2[0], c2[1], outline="#00E676", width=2)
            canvas.create_text(c1[0] + 4, c1[1] + 12, anchor="w", fill="#00E676", text=f"{idx+1}:{ann['class_name']}")

        if state["temp_rect"] is not None:
            x1, y1, x2, y2 = state["temp_rect"]
            canvas.create_rectangle(x1, y1, x2, y2, outline="#18A0FB", width=2, dash=(4, 2))

    def refresh_list() -> None:
        box_list.delete(0, tk.END)
        for idx, ann in enumerate(state["anns"]):
            box_list.insert(tk.END, f"{idx+1}. {ann['class_name']}")

    def on_down(e: Any) -> None:  # type: ignore[name-defined]
        state["start"] = (e.x, e.y)
        state["temp_rect"] = (e.x, e.y, e.x, e.y)
        redraw()

    def on_drag(e: Any) -> None:  # type: ignore[name-defined]
        if state["start"] is None:
            return
        sx, sy = state["start"]
        state["temp_rect"] = (sx, sy, e.x, e.y)
        redraw()

    def on_up(e: Any) -> None:  # type: ignore[name-defined]
        if state["start"] is None:
            return
        sx, sy = state["start"]
        state["temp_rect"] = (sx, sy, e.x, e.y)
        x1, y1, x2, y2 = state["temp_rect"]
        ix1, iy1 = canvas_to_img(min(x1, x2), min(y1, y2))
        ix2, iy2 = canvas_to_img(max(x1, x2), max(y1, y2))
        if abs(ix2 - ix1) >= 2 and abs(iy2 - iy1) >= 2:
            ann = {
                "class_name": class_var.get(),
                "rect_norm": (
                    ix1 / max(1.0, float(pil_img.width)),
                    iy1 / max(1.0, float(pil_img.height)),
                    ix2 / max(1.0, float(pil_img.width)),
                    iy2 / max(1.0, float(pil_img.height)),
                ),
            }
            state["anns"].append(ann)
            refresh_list()
        state["start"] = None
        state["temp_rect"] = None
        redraw()

    def undo_last() -> None:
        if state["anns"]:
            state["anns"].pop()
            refresh_list()
            redraw()

    def remove_selected() -> None:
        sel = box_list.curselection()
        if not sel:
            return
        idx = int(sel[0])
        if 0 <= idx < len(state["anns"]):
            state["anns"].pop(idx)
            refresh_list()
            redraw()

    def done() -> None:
        if not state["anns"]:
            messagebox.showwarning("Golden Sample", "Please draw at least one box.", parent=win)
            return
        state["result"] = list(state["anns"])
        win.destroy()

    def cancel() -> None:
        state["result"] = None
        win.destroy()

    app.create_secondary_button(right, text="Undo Last", command=undo_last).pack(fill="x", padx=12, pady=(0, 6))
    app.create_secondary_button(right, text="Remove Selected", command=remove_selected).pack(fill="x", padx=12, pady=(0, 10))
    app.create_primary_button(right, text="Done", command=done, bg=app.COLORS["success"]).pack(fill="x", padx=12, pady=(0, 6))
    app.create_secondary_button(right, text="Cancel", command=cancel).pack(fill="x", padx=12, pady=(0, 12))
    tk.Label(
        right,
        text="Draw box: drag left mouse\nSelect class before drawing",
        bg=app.COLORS["bg_white"],
        fg=app.COLORS["text_secondary"],
        font=app.font_primary,
        justify="left",
        anchor="w",
    ).pack(fill="x", padx=12, pady=(0, 10))

    canvas.bind("<ButtonPress-1>", on_down)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_up)
    canvas.bind("<Configure>", lambda _e: redraw())

    win.update_idletasks()
    redraw()
    app.root.wait_window(win)
    return state["result"]


def pick_golden_id_classes(app: Any, class_mapping: dict[int, str], parent: Any = None):
    return golden_core.prompt_golden_id_classes(app, class_mapping, parent=parent)
