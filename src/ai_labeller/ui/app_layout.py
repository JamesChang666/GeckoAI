import tkinter as tk
from tkinter import ttk


def setup_ui(app):
    app.setup_toolbar()
    app.setup_sidebar()
    canvas_wrap = tk.Frame(app.root, bg=app.COLORS["bg_canvas"])
    canvas_wrap.pack(side="left", fill="both", expand=True)

    app.canvas = tk.Canvas(
        canvas_wrap,
        bg=app.COLORS["bg_canvas"],
        cursor="none",
        highlightthickness=0,
        relief="flat",
    )
    app.canvas.pack(side="left", fill="both", expand=True)


def setup_toolbar(app):
    toolbar = tk.Frame(app.root, bg=app.COLORS["bg_dark"], height=56)
    toolbar.pack(side="top", fill="x")
    toolbar.pack_propagate(False)

    left_frame = tk.Frame(toolbar, bg=app.COLORS["bg_dark"])
    left_frame.pack(side="left", fill="y", padx=16)

    title_frame = tk.Frame(left_frame, bg=app.COLORS["bg_dark"])
    title_frame.pack(side="left", pady=12)

    if app.toolbar_logo_tk is not None:
        logo = tk.Label(title_frame, image=app.toolbar_logo_tk, bg=app.COLORS["bg_dark"])
    else:
        logo = tk.Label(
            title_frame,
            text="AI",
            font=("Arial", 20),
            fg=app.COLORS["primary"],
            bg=app.COLORS["bg_dark"],
        )
    logo.pack(side="left", padx=(0, 8))
    logo.bind("<Button-1>", app.return_to_source_select)
    logo.bind("<Enter>", lambda _e: logo.config(cursor="hand2"))

    title_label = tk.Label(
        title_frame,
        text=app.LANG_MAP[app.lang]["title"],
        font=app.font_title,
        fg=app.toolbar_text_color(app.COLORS["bg_dark"]),
        bg=app.COLORS["bg_dark"],
    )
    title_label.pack(side="left")
    title_label.bind("<Button-1>", app.return_to_source_select)
    title_label.bind("<Enter>", lambda _e: title_label.config(cursor="hand2"))

    tk.Frame(left_frame, width=1, bg=app.COLORS["divider"]).pack(side="left", fill="y", padx=16)

    app.create_toolbar_button(
        left_frame,
        text=app.LANG_MAP[app.lang]["load_proj"],
        command=lambda: app.show_startup_source_dialog(force=True, bypass_detect_lock=True),
        bg=app.COLORS["primary"],
    ).pack(side="left", padx=4)

    if app._golden_capture_active:
        app.create_toolbar_button(left_frame, text="Save Golden", command=app._finalize_golden_from_label_mode, bg=app.COLORS["success"]).pack(side="left", padx=4)
        app.create_toolbar_button(left_frame, text="Cancel Golden", command=app._cancel_golden_capture_and_back_to_detect, bg=app.COLORS["danger"]).pack(side="left", padx=4)

    dataset_frame = tk.Frame(left_frame, bg=app.COLORS["bg_dark"])
    dataset_frame.pack(side="left", padx=12)

    tk.Label(dataset_frame, text=app.LANG_MAP[app.lang]["dataset"], font=app.font_primary, fg=app.COLORS["text_secondary"], bg=app.COLORS["bg_dark"]).pack(side="left", padx=(0, 8))

    app.combo_split = ttk.Combobox(dataset_frame, values=["train", "val", "test"], width=10, state="readonly", font=app.font_primary)
    app.combo_split.current(0)
    app.combo_split.pack(side="left")
    app.combo_split.bind("<<ComboboxSelected>>", app.on_split_change)

    center_frame = tk.Frame(toolbar, bg=app.COLORS["bg_dark"])
    center_frame.pack(side="left", fill="y", padx=16)

    app.create_toolbar_icon_button(center_frame, text="\u21B6", command=app.undo, tooltip=app.LANG_MAP[app.lang]["undo"], bg="#000000", fg="#FFFFFF", circular=True).pack(side="left", padx=2)
    app.create_toolbar_icon_button(center_frame, text="\u21B7", command=app.redo, tooltip=app.LANG_MAP[app.lang]["redo"], bg="#000000", fg="#FFFFFF", circular=True).pack(side="left", padx=2)

    tk.Frame(center_frame, width=1, bg=app.COLORS["divider"]).pack(side="left", fill="y", padx=8, pady=10)

    ttk.Combobox(center_frame, textvariable=app.var_export_format, values=["YOLO (.txt)", "JSON", "COCO"], state="readonly", width=12, font=app.font_primary).pack(side="left", padx=(0, 6), pady=12)

    app.create_toolbar_button(center_frame, text=app.LANG_MAP[app.lang]["export"], command=app.export_all_by_selected_format, bg=app.COLORS["info"]).pack(side="left", padx=2, pady=8)
    app.create_toolbar_button(center_frame, text="Export Golden", command=app.export_golden_folder, bg=app.COLORS["warning"]).pack(side="left", padx=2, pady=8)
    app.create_toolbar_button(center_frame, text=app.LANG_MAP[app.lang].get("train_from_labels", "Train From Labels"), command=app.start_training_from_labels, bg=app.COLORS["danger"]).pack(side="left", padx=2, pady=8)

    right_frame = tk.Frame(toolbar, bg=app.COLORS["bg_dark"])
    right_frame.pack(side="right", fill="y", padx=16)

    app.create_help_icon(right_frame).pack(side="right", padx=4, pady=12)
    app.create_toolbar_button(right_frame, text=app.get_theme_switch_label(), command=app.toggle_theme, bg=app.COLORS["bg_medium"]).pack(side="right", padx=4, pady=12)


def setup_sidebar(app):
    sidebar = tk.Frame(app.root, width=320, bg=app.COLORS["bg_light"])
    sidebar.pack(side="right", fill="y")
    sidebar.pack_propagate(False)

    scroll_wrap = tk.Frame(sidebar, bg=app.COLORS["bg_light"])
    scroll_wrap.pack(side="top", fill="both", expand=True)

    app.sidebar_canvas = tk.Canvas(scroll_wrap, bg=app.COLORS["bg_light"], highlightthickness=0, relief="flat")
    app.sidebar_scrollbar = tk.Scrollbar(
        scroll_wrap,
        orient="vertical",
        command=app.sidebar_canvas.yview,
        width=24,
        bg=app.COLORS["bg_medium"],
        troughcolor=app.COLORS["bg_dark"],
        activebackground=app.COLORS["primary"],
        highlightthickness=0,
        relief="flat",
        borderwidth=0,
    )
    app.sidebar_scroll_frame = tk.Frame(app.sidebar_canvas, bg=app.COLORS["bg_light"])

    app.sidebar_window = app.sidebar_canvas.create_window((0, 0), window=app.sidebar_scroll_frame, anchor="nw")
    app.sidebar_canvas.configure(yscrollcommand=app.sidebar_scrollbar.set)
    app.sidebar_scroll_frame.bind("<Configure>", app._on_sidebar_frame_configure)
    app.sidebar_canvas.bind("<Configure>", app._on_sidebar_canvas_configure)
    app.sidebar_canvas.bind("<MouseWheel>", app._on_sidebar_mousewheel)
    app.sidebar_canvas.bind("<Button-4>", lambda e: app.sidebar_canvas.yview_scroll(-1, "units"))
    app.sidebar_canvas.bind("<Button-5>", lambda e: app.sidebar_canvas.yview_scroll(1, "units"))

    app.create_info_card(app.sidebar_scroll_frame)
    app.create_class_card(app.sidebar_scroll_frame)
    app.create_ai_card(app.sidebar_scroll_frame)
    app.create_navigation(sidebar)
    app._bind_sidebar_mousewheel(app.sidebar_scroll_frame)

    app.sidebar_scrollbar.pack(side="right", fill="y")
    app.sidebar_canvas.pack(side="left", fill="both", expand=True)
    app.root.after_idle(app._refresh_sidebar_scrollregion)


def on_sidebar_frame_configure(app, e=None):
    if hasattr(app, "sidebar_canvas"):
        app.sidebar_canvas.configure(scrollregion=app.sidebar_canvas.bbox("all"))


def on_sidebar_canvas_configure(app, e):
    if hasattr(app, "sidebar_canvas") and hasattr(app, "sidebar_window"):
        app.sidebar_canvas.itemconfigure(app.sidebar_window, width=e.width)


def on_sidebar_mousewheel(app, e):
    if e.delta:
        app.sidebar_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
    return "break"


def bind_sidebar_mousewheel(app, widget):
    widget.bind("<MouseWheel>", app._on_sidebar_mousewheel, add="+")
    widget.bind("<Button-4>", lambda e: app.sidebar_canvas.yview_scroll(-1, "units"), add="+")
    widget.bind("<Button-5>", lambda e: app.sidebar_canvas.yview_scroll(1, "units"), add="+")
    for child in widget.winfo_children():
        app._bind_sidebar_mousewheel(child)


def refresh_sidebar_scrollregion(app) -> None:
    if not hasattr(app, "sidebar_canvas") or not hasattr(app, "sidebar_scroll_frame"):
        return
    try:
        app.sidebar_scroll_frame.update_idletasks()
        app.sidebar_canvas.configure(scrollregion=app.sidebar_canvas.bbox("all"))
    except Exception:
        pass


def create_card(app, parent, title=None):
    card = tk.Frame(parent, bg=app.COLORS["bg_white"], relief="flat", borderwidth=0)
    card.pack(fill="x", padx=16, pady=8)

    card_border = tk.Frame(card, bg=app.COLORS["border"], height=1)
    card_border.pack(fill="x", side="bottom")

    content = tk.Frame(card, bg=app.COLORS["bg_white"])
    content.pack(fill="both", expand=True, padx=16, pady=16)

    if title:
        title_label = tk.Label(content, text=title, font=app.font_bold, fg=app.COLORS["text_primary"], bg=app.COLORS["bg_white"], anchor="w")
        title_label.pack(fill="x", pady=(0, 12))
    return content


def create_info_card(app, parent):
    content = app.create_card(parent, app.LANG_MAP[app.lang]["file_info"])

    app.lbl_filename = tk.Label(content, text=app.LANG_MAP[app.lang]["no_img"], font=app.font_mono, fg=app.COLORS["text_primary"], bg=app.COLORS["bg_white"], anchor="w", wraplength=260)
    app.lbl_filename.pack(fill="x")

    tk.Label(content, text=app.LANG_MAP[app.lang].get("select_image", "Select Image"), font=app.font_primary, fg=app.COLORS["text_secondary"], bg=app.COLORS["bg_white"], anchor="w").pack(fill="x", pady=(10, 6))

    image_select_row = tk.Frame(content, bg=app.COLORS["bg_white"])
    image_select_row.pack(fill="x")

    app.combo_image = ttk.Combobox(image_select_row, values=[], state="readonly", font=app.font_primary)
    app.combo_image.pack(side="left", fill="x", expand=True)
    app.combo_image.bind("<<ComboboxSelected>>", app.on_image_selected)

    app.create_toolbar_icon_button(image_select_row, text="\u2716", command=app.remove_current_from_split, tooltip=app.LANG_MAP[app.lang].get("remove_from_split", "Remove From Split"), bg=app.COLORS["danger"]).pack(side="left", padx=(6, 0))
    app.create_toolbar_icon_button(image_select_row, text="\u21BA", command=app.open_restore_removed_dialog, tooltip=app.LANG_MAP[app.lang].get("restore_from_split", "Restore Deleted Frame"), bg=app.COLORS["success"]).pack(side="left", padx=(6, 0))

    progress_frame = tk.Frame(content, bg=app.COLORS["bg_white"])
    progress_frame.pack(fill="x", pady=(8, 0))

    app.lbl_progress = tk.Label(progress_frame, text="0 / 0", font=app.font_primary, fg=app.COLORS["text_secondary"], bg=app.COLORS["bg_white"])
    app.lbl_progress.pack(side="left")

    app.lbl_box_count = tk.Label(progress_frame, text=f"{app.LANG_MAP[app.lang]['boxes']}: 0", font=app.font_primary, fg=app.COLORS["primary"], bg=app.COLORS["bg_white"])
    app.lbl_box_count.pack(side="right")

    counts_detail_frame = tk.Frame(content, bg=app.COLORS["bg_white"])
    counts_detail_frame.pack(fill="x", pady=(4, 0))

    app.lbl_class_count = tk.Label(counts_detail_frame, text=f"{app.LANG_MAP[app.lang]['class_mgmt']}: 0 / 0", font=app.font_primary, fg=app.COLORS["primary"], bg=app.COLORS["bg_white"], anchor="e")
    app.lbl_class_count.pack(side="right")


def create_class_card(app, parent):
    content = app.create_card(parent, app.LANG_MAP[app.lang]["class_mgmt"])

    tk.Label(content, text=app.LANG_MAP[app.lang]["current_class"], font=app.font_primary, fg=app.COLORS["text_secondary"], bg=app.COLORS["bg_white"], anchor="w").pack(fill="x", pady=(0, 8))

    app.combo_cls = ttk.Combobox(content, values=app.class_names, state="readonly", font=app.font_primary)
    app.combo_cls.current(0)
    app.combo_cls.pack(fill="x", pady=(0, 12))
    app.combo_cls.bind("<<ComboboxSelected>>", app.on_class_change_request)

    class_action_row = tk.Frame(content, bg=app.COLORS["bg_white"])
    class_action_row.pack(fill="x", pady=(0, 12))

    app.create_primary_button(
        class_action_row,
        text=app.LANG_MAP[app.lang]["edit_classes"],
        command=app.edit_classes_table,
    ).pack(side="left", fill="x", expand=True, padx=(0, 4))
    app.create_primary_button(
        class_action_row,
        text=app.LANG_MAP[app.lang]["clear_labels"],
        command=app.clear_current_labels,
        bg=app.COLORS["danger"],
    ).pack(side="left", fill="x", expand=True, padx=(4, 0))

    tk.Checkbutton(content, text="Show Last Photo Labels (ghost)", variable=app.var_show_prev_labels, command=app.render, bg=app.COLORS["bg_white"], fg=app.COLORS["text_primary"], font=app.font_primary, activebackground=app.COLORS["bg_white"], selectcolor=app.COLORS["bg_white"], anchor="w").pack(fill="x", pady=(0, 12))


def create_ai_card(app, parent):
    content = app.create_card(parent, app.LANG_MAP[app.lang]["ai_tools"])

    checkbox_style = {
        "bg": app.COLORS["bg_white"],
        "fg": app.COLORS["text_primary"],
        "font": app.font_primary,
        "activebackground": app.COLORS["bg_white"],
        "selectcolor": app.COLORS["bg_white"],
        "anchor": "w",
    }

    tk.Checkbutton(content, text=app.LANG_MAP[app.lang]["auto_detect"], variable=app.var_auto_yolo, **checkbox_style).pack(fill="x", pady=4)

    propagate_row = tk.Frame(content, bg=app.COLORS["bg_white"])
    propagate_row.pack(fill="x", pady=4)
    tk.Checkbutton(propagate_row, text=app.LANG_MAP[app.lang]["propagate"], variable=app.var_propagate, **checkbox_style).pack(side="left")
    app.combo_propagate_mode = ttk.Combobox(propagate_row, state="readonly", width=18, font=app.font_primary)
    app.combo_propagate_mode.pack(side="right")
    app.combo_propagate_mode.bind("<<ComboboxSelected>>", app.on_propagate_mode_changed)
    app._refresh_propagate_mode_combo()

    tk.Label(content, text=app.LANG_MAP[app.lang].get("detection_model", "Detection Model"), font=app.font_primary, fg=app.COLORS["text_secondary"], bg=app.COLORS["bg_white"], anchor="w").pack(fill="x", pady=(10, 4))

    app.combo_det_model = ttk.Combobox(content, textvariable=app.det_model_mode, values=["Official YOLO26m.pt (Bundled)", "Custom YOLO (v5/v7/v8/v9/v11/v26)", "Custom RF-DETR"], state="readonly", font=app.font_primary)
    app.combo_det_model.pack(fill="x", pady=(0, 6))
    app.combo_det_model.bind("<<ComboboxSelected>>", app.on_detection_model_mode_changed)

    app.combo_model_path = ttk.Combobox(content, textvariable=app.yolo_path, values=app.model_library, state="readonly", font=app.font_primary)
    app.combo_model_path.pack(fill="x", pady=(0, 6))
    app._refresh_model_dropdown()

    picker_row = tk.Frame(content, bg=app.COLORS["bg_white"])
    picker_row.pack(fill="x", pady=(0, 6))

    app.create_secondary_button(picker_row, text=app.LANG_MAP[app.lang].get("browse_model", "Browse Model"), command=app.browse_detection_model).pack(side="left", fill="x", expand=True, padx=(0, 4))

    app.create_primary_button(content, text=app.LANG_MAP[app.lang]["run_detection"], command=app.run_yolo_detection, bg=app.COLORS["success"]).pack(fill="x", pady=(12, 0))


def create_shortcut_card(app, parent):
    content = app.create_card(parent, app.LANG_MAP[app.lang]["shortcuts"])
    for key, desc in app._shortcut_items():
        row = tk.Frame(content, bg=app.COLORS["bg_white"])
        row.pack(fill="x", pady=2)
        tk.Label(row, text=key, font=app.font_mono, fg=app.COLORS["primary"], bg=app.COLORS["bg_white"], width=12, anchor="w").pack(side="left")
        tk.Label(row, text=desc, font=app.font_primary, fg=app.COLORS["text_secondary"], bg=app.COLORS["bg_white"], anchor="w").pack(side="left")


def create_navigation(app, parent):
    nav_frame = tk.Frame(parent, bg=app.COLORS["bg_light"], height=80)
    nav_frame.pack(side="bottom", fill="x")
    nav_frame.pack_propagate(False)

    btn_container = tk.Frame(nav_frame, bg=app.COLORS["bg_light"])
    btn_container.pack(fill="both", expand=True, padx=16, pady=16)

    app.create_nav_button(btn_container, text=app.LANG_MAP[app.lang]["prev"], command=app.prev_img, side="left")
    app.create_nav_button(btn_container, text=f"{app.LANG_MAP[app.lang]['next']} >", command=app.save_and_next, side="right", primary=True)


def get_theme_switch_label(app):
    key = "theme_light" if app.theme == "dark" else "theme_dark"
    return app.LANG_MAP[app.lang][key]
