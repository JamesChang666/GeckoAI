from typing import Any
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, colorchooser
import datetime
import glob
import tempfile
import shutil
import cv2
import numpy as np


def _reset_detect_setup_page(app) -> tk.Frame:
    app._detect_mode_active = True
    app._stop_detect_stream()
    app._detect_workspace_frame = None
    app.hide_shortcut_tooltip()
    for child in app.root.winfo_children():
        child.destroy()
    wrap = tk.Frame(app.root, bg=app.COLORS["bg_dark"]) if hasattr(app, "COLORS") else tk.Frame(app.root)
    wrap.pack(fill="both", expand=True)
    return wrap


def _create_detect_setup_card(app, wrap: tk.Frame, *, width: int, height: int, title: str, subtitle: str) -> tk.Frame:
    COLORS = getattr(app, "COLORS", {})
    card = tk.Frame(wrap, bg=COLORS.get("bg_white", "#fff"), bd=0, highlightthickness=0)
    card.place(relx=0.5, rely=0.5, anchor="center", width=width, height=height)
    tk.Label(
        card,
        text=title,
        font=app.font_title,
        fg=COLORS.get("text_primary", "#000"),
        bg=COLORS.get("bg_white", "#fff"),
        anchor="center",
    ).pack(fill="x", padx=24, pady=(28, 8))
    tk.Label(
        card,
        text=subtitle,
        font=app.font_primary,
        fg=COLORS.get("text_secondary", "#666"),
        bg=COLORS.get("bg_white", "#fff"),
        anchor="center",
    ).pack(fill="x", padx=24, pady=(0, 14))
    return card


def _add_detect_class_color_editor(app: Any, card: tk.Frame, refresh_callback) -> None:
    classes = app._get_detect_model_class_names() if hasattr(app, "_get_detect_model_class_names") else []
    section = tk.Frame(card, bg=app.COLORS.get("bg_white"))
    section.pack(fill="x", padx=28, pady=(0, 12))
    tk.Label(
        section,
        text="Class Color Mapping",
        font=app.font_primary,
        fg=app.COLORS.get("text_secondary"),
        bg=app.COLORS.get("bg_white"),
        anchor="w",
    ).pack(fill="x", pady=(0, 4))

    if not classes:
        tk.Label(
            section,
            text="No model classes loaded yet. Pick model in Step 1 first.",
            font=app.font_mono,
            fg=app.COLORS.get("text_secondary"),
            bg=app.COLORS.get("bg_white"),
            anchor="w",
            justify="left",
            wraplength=680,
        ).pack(fill="x")
        return

    if not hasattr(app, "detect_class_color_map") or not isinstance(app.detect_class_color_map, dict):
        app.detect_class_color_map = {}
    color_map = app.detect_class_color_map

    listbox = tk.Listbox(
        section,
        height=min(8, max(4, len(classes))),
        font=app.font_primary,
        bg=app.COLORS.get("bg_light"),
        fg=app.COLORS.get("text_primary"),
        relief="flat",
        highlightthickness=0,
        exportselection=False,
    )
    for cls_name in classes:
        key = cls_name.strip().lower()
        color_hex = color_map.get(key, "")
        suffix = f" -> {color_hex}" if color_hex else ""
        listbox.insert(tk.END, f"{cls_name}{suffix}")
    listbox.pack(fill="x", pady=(0, 8))

    btn_row = tk.Frame(section, bg=app.COLORS.get("bg_white"))
    btn_row.pack(fill="x")

    def _set_selected_class_color() -> None:
        sel = listbox.curselection()
        if not sel:
            messagebox.showinfo("Detect Mode", "Please select a class first.", parent=app.root)
            return
        cls_name = classes[int(sel[0])]
        key = cls_name.strip().lower()
        initial_color = color_map.get(key, "#00FF00")
        _rgb, hex_code = colorchooser.askcolor(color=initial_color, parent=app.root, title=f"Choose color for {cls_name}")
        if not hex_code:
            return
        color_map[key] = hex_code.upper()
        app.detect_class_color_map = color_map
        refresh_callback()

    def _on_listbox_double_click(_e: Any = None) -> None:
        _set_selected_class_color()

    def _clear_selected_class_color() -> None:
        sel = listbox.curselection()
        if not sel:
            messagebox.showinfo("Detect Mode", "Please select a class first.", parent=app.root)
            return
        cls_name = classes[int(sel[0])]
        key = cls_name.strip().lower()
        if key in color_map:
            color_map.pop(key, None)
            app.detect_class_color_map = color_map
            refresh_callback()

    def _clear_all_class_colors() -> None:
        app.detect_class_color_map = {}
        refresh_callback()

    app.create_secondary_button(
        btn_row,
        text="Set Selected Color",
        command=_set_selected_class_color,
    ).pack(side="left", padx=(0, 8))
    app.create_secondary_button(
        btn_row,
        text="Clear Selected",
        command=_clear_selected_class_color,
    ).pack(side="left", padx=(0, 8))
    app.create_secondary_button(
        btn_row,
        text="Clear All",
        command=_clear_all_class_colors,
    ).pack(side="left")
    listbox.bind("<Double-Button-1>", _on_listbox_double_click)


def show_detect_mode_page(app) -> None:
    if getattr(app, "_startup_mode", "chooser") == "label":
        app.show_startup_source_dialog(force=True)
        return
    wrap = _reset_detect_setup_page(app)

    if not hasattr(app, "detect_model_path_var"):
        app.detect_model_path_var = tk.StringVar(value="")
    if not hasattr(app, "detect_source_mode_var"):
        app.detect_source_mode_var = tk.StringVar(value="")
    if not hasattr(app, "detect_media_path_var"):
        app.detect_media_path_var = tk.StringVar(value="")
    if not hasattr(app, "detect_output_dir_var"):
        app.detect_output_dir_var = tk.StringVar(value="")
    if not hasattr(app, "detect_conf_var"):
        app.detect_conf_var = tk.DoubleVar(value=max(0.01, min(1.0, float(app.var_yolo_conf.get()))))
    if not hasattr(app, "_detect_source_selected"):
        app._detect_source_selected = False

    card = _create_detect_setup_card(
        app,
        wrap,
        width=700,
        height=520,
        title="Detect Mode - Step 1",
        subtitle="Choose detection model",
    )

    selected_model = app.detect_model_path_var.get().strip()
    model_hint = f"Selected: {selected_model}" if selected_model else "Selected: None"
    tk.Label(
        card,
        text=model_hint,
        font=app.font_mono,
        fg=app.COLORS.get("text_secondary", "#666"),
        bg=app.COLORS.get("bg_white", "#fff"),
        anchor="w",
        justify="left",
        wraplength=620,
    ).pack(fill="x", padx=28, pady=(0, 10))
    app.create_primary_button(
        card,
        text="Choose Model File (.pt/.onnx)",
        command=app._on_detect_pick_model,
        bg=app.COLORS.get("primary"),
    ).pack(fill="x", padx=28, pady=(0, 12))

    app.create_primary_button(
        card,
        text="Next: Choose Source",
        command=app._go_detect_source_page,
        bg=app.COLORS.get("success"),
    ).pack(fill="x", padx=28, pady=(0, 10))

    def switch_to_label_mode() -> None:
        app._stop_detect_stream()
        app._detect_mode_active = False
        app._detect_workspace_frame = None
        app.rebuild_ui()
        app.show_startup_source_dialog(force=True, bypass_detect_lock=True)

    if getattr(app, "_startup_mode", "chooser") != "detect":
        app.create_secondary_button(
            card,
            text="Switch to Label/Training Mode",
            command=switch_to_label_mode,
        ).pack(fill="x", padx=28, pady=(0, 10))


def show_detect_source_page(app) -> None:
    wrap = _reset_detect_setup_page(app)
    card = _create_detect_setup_card(
        app,
        wrap,
        width=700,
        height=520,
        title="Detect Mode - Step 2",
        subtitle="Choose source type",
    )

    current_source = app.detect_source_mode_var.get().strip().lower()
    if not app._detect_source_selected:
        source_hint = "Selected source: None"
    elif current_source == "file":
        source_text = app.detect_media_path_var.get().strip() or "None"
        source_hint = f"Selected source: Image Folder - {source_text}"
    else:
        source_hint = "Selected source: Camera (Realtime)"
    tk.Label(
        card,
        text=source_hint,
        font=app.font_mono,
        fg=app.COLORS.get("text_secondary", "#666"),
        bg=app.COLORS.get("bg_white", "#fff"),
        anchor="w",
        justify="left",
        wraplength=620,
    ).pack(fill="x", padx=28, pady=(0, 12))

    app.create_primary_button(
        card,
        text="Use Camera",
        command=lambda: app._on_detect_choose_camera(),
        bg=app.COLORS.get("primary"),
    ).pack(fill="x", padx=28, pady=(0, 10))

    app.create_primary_button(
        card,
        text="Choose Image Folder",
        command=app._on_detect_browse_media_file,
        bg=app.COLORS.get("success"),
    ).pack(fill="x", padx=28, pady=(0, 12))

    app.create_secondary_button(
        card,
        text="Back: Choose Model",
        command=app.show_detect_mode_page,
    ).pack(fill="x", padx=28, pady=(0, 10))


def show_detect_camera_mode_page(app) -> None:
    wrap = _reset_detect_setup_page(app)
    card = _create_detect_setup_card(
        app,
        wrap,
        width=760,
        height=760,
        title="Detect Mode - Step 3 (Camera)",
        subtitle="Choose camera speed mode",
    )

    cams = app._detect_available_cameras[:] or app._scan_available_cameras()
    app._detect_available_cameras = cams[:]
    if not cams:
        messagebox.showwarning("Detect Mode", "No camera found.", parent=app.root)
        app.show_detect_source_page()
        return
    cam_values = [str(c) for c in cams]
    selected_cam = app.detect_camera_index_var.get().strip()
    if selected_cam not in cam_values:
        selected_cam = cam_values[0]
        app.detect_camera_index_var.set(selected_cam)
    app.detect_media_path_var.set(selected_cam)

    if len(cam_values) > 1:
        tk.Label(
            card,
            text="Camera",
            font=app.font_primary,
            fg=app.COLORS.get("text_secondary"),
            bg=app.COLORS.get("bg_white"),
            anchor="w",
        ).pack(fill="x", padx=28, pady=(0, 4))
        camera_combo = ttk.Combobox(
            card,
            textvariable=app.detect_camera_index_var,
            values=cam_values,
            state="readonly",
            font=app.font_primary,
        )
        camera_combo.pack(fill="x", padx=28, pady=(0, 10))

        def _on_camera_changed(_e: Any = None) -> None:
            app.detect_media_path_var.set(app.detect_camera_index_var.get().strip())
            app.show_detect_camera_mode_page()

        camera_combo.bind("<<ComboboxSelected>>", _on_camera_changed)

    max_fps = app._get_camera_max_fps(int(selected_cam))
    max_hint = f"Max FPS: {max_fps:.1f}" if max_fps > 0 else "Max FPS: Unknown (camera did not report)"
    tk.Label(
        card,
        text=max_hint,
        font=app.font_mono,
        fg=app.COLORS.get("text_secondary"),
        bg=app.COLORS.get("bg_white"),
        anchor="w",
    ).pack(fill="x", padx=28, pady=(0, 10))

    mode_row = tk.Frame(card, bg=app.COLORS.get("bg_white"))
    mode_row.pack(fill="x", padx=28, pady=(0, 10))
    tk.Radiobutton(
        mode_row,
        text="Auto Mode (use camera max FPS)",
        variable=app.detect_camera_mode_var,
        value="auto",
        bg=app.COLORS.get("bg_white"),
        fg=app.COLORS.get("text_primary"),
        selectcolor=app.COLORS.get("bg_white"),
        font=app.font_primary,
        anchor="w",
        command=app.show_detect_camera_mode_page,
    ).pack(anchor="w")
    tk.Radiobutton(
        mode_row,
        text="Manual Mode (preferred FPS)",
        variable=app.detect_camera_mode_var,
        value="manual",
        bg=app.COLORS.get("bg_white"),
        fg=app.COLORS.get("text_primary"),
        selectcolor=app.COLORS.get("bg_white"),
        font=app.font_primary,
        anchor="w",
        command=app.show_detect_camera_mode_page,
    ).pack(anchor="w", pady=(4, 0))

    if app.detect_camera_mode_var.get().strip().lower() == "manual":
        manual_row = tk.Frame(card, bg=app.COLORS.get("bg_white"))
        manual_row.pack(fill="x", padx=28, pady=(0, 12))
        tk.Label(
            manual_row,
            text="Preferred FPS",
            font=app.font_primary,
            fg=app.COLORS.get("text_secondary"),
            bg=app.COLORS.get("bg_white"),
        ).pack(side="left")
        tk.Entry(
            manual_row,
            textvariable=app.detect_manual_fps_var,
            width=10,
            font=app.font_primary,
        ).pack(side="left", padx=(8, 12))
        if max_fps > 0:
            max_text = f"max {max_fps:.1f}"
        else:
            max_text = "max depends on camera"
        tk.Label(
            manual_row,
            text=max_text,
            font=app.font_mono,
            fg=app.COLORS.get("text_secondary"),
            bg=app.COLORS.get("bg_white"),
        ).pack(side="left")

    conf_value = max(0.01, min(1.0, float(app.detect_conf_var.get())))
    app.detect_conf_var.set(conf_value)
    conf_text = tk.StringVar(value=f"Conf Threshold: {conf_value:.2f}")
    tk.Label(
        card,
        textvariable=conf_text,
        font=app.font_primary,
        fg=app.COLORS.get("text_secondary"),
        bg=app.COLORS.get("bg_white"),
        anchor="w",
    ).pack(fill="x", padx=28, pady=(0, 4))
    conf_scale = ttk.Scale(
        card,
        from_=0.01,
        to=1.0,
        variable=app.detect_conf_var,
        orient="horizontal",
    )
    conf_scale.pack(fill="x", padx=28, pady=(0, 12))
    conf_scale.configure(command=lambda _v: conf_text.set(f"Conf Threshold: {float(app.detect_conf_var.get()):.2f}"))
    _add_detect_class_color_editor(app, card, app.show_detect_camera_mode_page)

    out_dir = app.detect_output_dir_var.get().strip()
    out_hint = out_dir if out_dir else "(auto) current project root"
    tk.Label(
        card,
        text=f"Output CSV folder: {out_hint}",
        font=app.font_mono,
        fg=app.COLORS.get("text_secondary"),
        bg=app.COLORS.get("bg_white"),
        anchor="w",
        justify="left",
        wraplength=680,
    ).pack(fill="x", padx=28, pady=(0, 8))
    app.create_secondary_button(
        card,
        text="Optional: Choose Output Folder",
        command=app._on_detect_choose_output_dir,
    ).pack(fill="x", padx=28, pady=(0, 14))

    app.create_primary_button(
        card,
        text="Start Detect",
        command=app._start_detect_from_setup,
        bg=app.COLORS.get("success"),
    ).pack(fill="x", padx=28, pady=(0, 10))
    app.create_secondary_button(
        card,
        text="Back: Choose Source",
        command=app.show_detect_source_page,
    ).pack(fill="x", padx=28, pady=(0, 10))


def show_detect_file_settings_page(app) -> None:
    wrap = _reset_detect_setup_page(app)
    card = _create_detect_setup_card(
        app,
        wrap,
        width=760,
        height=900,
        title="Detect Mode - Step 3 (Image Folder)",
        subtitle="Set confidence threshold and run options",
    )

    src_text = app.detect_media_path_var.get().strip() or "None"
    tk.Label(
        card,
        text=f"Source Folder: {src_text}",
        font=app.font_mono,
        fg=app.COLORS.get("text_secondary"),
        bg=app.COLORS.get("bg_white"),
        anchor="w",
        justify="left",
        wraplength=680,
    ).pack(fill="x", padx=28, pady=(0, 8))
    app.create_secondary_button(
        card,
        text="Choose Source Folder",
        command=app._on_detect_browse_media_file,
    ).pack(fill="x", padx=28, pady=(0, 12))

    conf_value = max(0.01, min(1.0, float(app.detect_conf_var.get())))
    app.detect_conf_var.set(conf_value)
    conf_text = tk.StringVar(value=f"Conf Threshold: {conf_value:.2f}")
    tk.Label(
        card,
        textvariable=conf_text,
        font=app.font_primary,
        fg=app.COLORS.get("text_secondary"),
        bg=app.COLORS.get("bg_white"),
        anchor="w",
    ).pack(fill="x", padx=28, pady=(0, 4))
    conf_scale = ttk.Scale(
        card,
        from_=0.01,
        to=1.0,
        variable=app.detect_conf_var,
        orient="horizontal",
    )
    conf_scale.pack(fill="x", padx=28, pady=(0, 12))
    conf_scale.configure(command=lambda _v: conf_text.set(f"Conf Threshold: {float(app.detect_conf_var.get()):.2f}"))
    _add_detect_class_color_editor(app, card, app.show_detect_file_settings_page)

    tk.Label(
        card,
        text="Run Type",
        font=app.font_primary,
        fg=app.COLORS.get("text_secondary"),
        bg=app.COLORS.get("bg_white"),
        anchor="w",
    ).pack(fill="x", padx=28, pady=(0, 4))
    mode_combo = ttk.Combobox(
        card,
        textvariable=app.detect_run_mode_var,
        values=["pure_detect", "golden"],
        state="readonly",
        font=app.font_primary,
    )
    mode_combo.pack(fill="x", padx=28, pady=(0, 12))
    mode_combo.bind("<<ComboboxSelected>>", lambda _e: app.show_detect_file_settings_page())

    if app.detect_run_mode_var.get().strip().lower() == "golden":
        mode_label_to_value = {
            "Label Count Match": "class",
            "Spatial Match": "position",
            "Strict Match": "both",
        }
        mode_value_to_label = {v: k for k, v in mode_label_to_value.items()}
        current_mode = app._normalize_golden_mode(app.detect_golden_mode_var.get()) if hasattr(app, "_normalize_golden_mode") else str(app.detect_golden_mode_var.get()).strip().lower()
        app.detect_golden_mode_var.set(current_mode)
        mode_display_var = tk.StringVar(value=mode_value_to_label.get(current_mode, "Strict Match"))

        golden_summary = "None"
        if app._detect_golden_sample is not None:
            targets = app._detect_golden_sample.get("targets") or []
            cls_names = sorted({str(t.get("class_name") or f"id:{t.get('class_id')}") for t in targets}) if targets else []
            cls_text = ", ".join(cls_names[:3]) + (" ..." if len(cls_names) > 3 else "") if cls_names else "None"
            lbl_name = os.path.basename(str(app._detect_golden_sample.get("label_path", "")))
            id_cfg_path = app._detect_golden_sample.get("id_config_path")
            bg_cut_root = str(app._detect_golden_sample.get("background_cut_root") or "").strip()
            bg_cut_text = f", background_cut=ON ({os.path.basename(bg_cut_root)})" if bg_cut_root else ", background_cut=OFF"
            golden_summary = (
                f"label={lbl_name}, targets={len(targets)}, classes={cls_text}{bg_cut_text}"
            )
        tk.Label(
            card,
            text=f"Golden Sample: {golden_summary}",
            font=app.font_mono,
            fg=app.COLORS.get("text_secondary"),
            bg=app.COLORS.get("bg_white"),
            anchor="w",
            justify="left",
            wraplength=680,
        ).pack(fill="x", padx=28, pady=(0, 6))

        tk.Label(
            card,
            text="Golden Match Strategy",
            font=app.font_primary,
            fg=app.COLORS.get("text_secondary"),
            bg=app.COLORS.get("bg_white"),
            anchor="w",
        ).pack(fill="x", padx=28, pady=(0, 4))
        golden_mode_combo = ttk.Combobox(
            card,
            textvariable=mode_display_var,
            values=list(mode_label_to_value.keys()),
            state="readonly",
            font=app.font_primary,
        )
        golden_mode_combo.pack(fill="x", padx=28, pady=(0, 10))

        def _on_golden_mode_changed(_e: Any = None) -> None:
            chosen_label = mode_display_var.get().strip()
            app.detect_golden_mode_var.set(mode_label_to_value.get(chosen_label, "both"))

        golden_mode_combo.bind("<<ComboboxSelected>>", _on_golden_mode_changed)
        _on_golden_mode_changed()

        app.create_secondary_button(
            card,
            text="Import Golden from Label Mode (YOLO txt + dataset.yaml mapping)",
            command=app._configure_detect_golden_sample,
        ).pack(fill="x", padx=28, pady=(0, 10))

    out_dir = app.detect_output_dir_var.get().strip() or "None"
    tk.Label(
        card,
        text=f"Output CSV folder: {out_dir}",
        font=app.font_mono,
        fg=app.COLORS.get("text_secondary"),
        bg=app.COLORS.get("bg_white"),
        anchor="w",
        justify="left",
        wraplength=680,
    ).pack(fill="x", padx=28, pady=(0, 8))
    app.create_secondary_button(
        card,
        text="Choose Output Folder",
        command=app._on_detect_choose_output_dir,
    ).pack(fill="x", padx=28, pady=(0, 12))

    app.create_primary_button(
        card,
        text="Start Detect",
        command=app._start_detect_from_setup,
        bg=app.COLORS.get("success"),
    ).pack(fill="x", padx=28, pady=(0, 10))
    app.create_secondary_button(
        card,
        text="Back: Choose Source",
        command=app.show_detect_source_page,
    ).pack(fill="x", padx=28, pady=(0, 10))
