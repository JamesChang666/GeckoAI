from __future__ import annotations

from typing import Any

import tkinter as tk
from tkinter import ttk


def show_app_mode_dialog(app: Any, colors: dict[str, str], lang_map: dict[str, dict[str, str]], force: bool = False) -> None:
    mode = getattr(app, "_startup_mode", "chooser")
    if mode == "detect":
        app.show_detect_mode_page()
        return
    if mode == "label":
        app.show_startup_source_dialog(force=True)
        return
    if app._app_mode_dialog_open:
        return
    if not force and app._app_mode_dialog_shown:
        return
    app._app_mode_dialog_shown = True
    app._app_mode_dialog_open = True
    if hasattr(app, "_app_mode_page") and app._app_mode_page is not None:
        try:
            app._app_mode_page.destroy()
        except Exception:
            pass

    page = tk.Frame(app.root, bg=colors["bg_dark"])
    page.place(relx=0, rely=0, relwidth=1, relheight=1)
    app._app_mode_page = page

    card = tk.Frame(page, bg=colors["bg_white"], bd=0, highlightthickness=0)
    card.place(relx=0.5, rely=0.5, anchor="center", width=520, height=290)

    tk.Label(
        card,
        text="Choose startup mode",
        bg=colors["bg_white"],
        fg=colors["text_primary"],
        font=app.font_title,
        anchor="center",
    ).pack(fill="x", padx=24, pady=(26, 20))

    def choose_label_mode() -> None:
        close_app_mode_dialog(app)
        app.root.after(1, lambda: app.show_startup_source_dialog(force=True, bypass_detect_lock=True))

    def choose_detect_mode() -> None:
        close_app_mode_dialog(app)
        app.root.after(1, app.show_detect_mode_page)

    app.create_primary_button(
        card,
        text="Label / Training Mode",
        command=choose_label_mode,
        bg=colors["primary"],
    ).pack(fill="x", padx=28, pady=(0, 12))

    app.create_primary_button(
        card,
        text="Detect Mode (Realtime Video / Image)",
        command=choose_detect_mode,
        bg=colors["success"],
    ).pack(fill="x", padx=28, pady=(0, 12))


def close_app_mode_dialog(app: Any) -> None:
    if hasattr(app, "_app_mode_page") and app._app_mode_page is not None:
        try:
            app._app_mode_page.destroy()
        except Exception:
            pass
        app._app_mode_page = None
    app._app_mode_dialog_open = False


def show_startup_source_dialog(
    app: Any,
    colors: dict[str, str],
    lang_map: dict[str, dict[str, str]],
    force: bool = False,
    reason: str | None = None,
    bypass_detect_lock: bool = False,
) -> None:
    mode = getattr(app, "_startup_mode", "chooser")
    if mode == "detect" and not bypass_detect_lock:
        app.show_detect_mode_page()
        return
    if app._startup_dialog_open:
        return
    if not force and getattr(app, "_startup_dialog_shown", False):
        return
    app._startup_dialog_shown = True
    app._startup_dialog_open = True
    if reason:
        app.logger.info("Showing startup source dialog: %s", reason)

    overlay = app._open_fullpage_overlay()
    card = tk.Frame(overlay, bg=colors["bg_white"], bd=0, highlightthickness=0)
    card.place(relx=0.5, rely=0.5, anchor="center", width=540, height=320)

    tk.Label(
        card,
        text=lang_map[app.lang]["startup_prompt"],
        bg=colors["bg_white"],
        fg=colors["text_primary"],
        font=app.font_title,
        anchor="center",
    ).pack(fill="x", padx=20, pady=(24, 18))

    source_choices: list[tuple[str, str]] = [
        (lang_map[app.lang]["startup_images"], "images"),
        (lang_map[app.lang]["startup_yolo"], "yolo"),
        (lang_map[app.lang]["startup_rfdetr"], "rfdetr"),
    ]
    source_label_to_mode = {label: mode_name for label, mode_name in source_choices}
    startup_source_var = tk.StringVar(value=lang_map[app.lang]["startup_images"])

    tk.Label(
        card,
        text=lang_map[app.lang]["startup_choose_source"],
        bg=colors["bg_white"],
        fg=colors["text_secondary"],
        font=app.font_primary,
        anchor="w",
    ).pack(fill="x", padx=28, pady=(0, 6))

    ttk.Combobox(
        card,
        textvariable=startup_source_var,
        values=[label for label, _mode in source_choices],
        state="readonly",
        font=app.font_primary,
    ).pack(fill="x", padx=28, pady=(0, 14))

    def choose_startup_source() -> None:
        mode_value = source_label_to_mode.get(startup_source_var.get(), "images")
        close_startup_dialog(app)
        if mode_value == "yolo":
            app.det_model_mode.set("Custom YOLO (v5/v7/v8/v9/v11/v26)")
        elif mode_value == "rfdetr":
            app.det_model_mode.set("Custom RF-DETR")
        app.root.after(1, lambda: app.startup_choose_images_folder(mode_value))

    app.create_primary_button(
        card,
        text="Start",
        command=choose_startup_source,
        bg=colors["primary"],
    ).pack(fill="x", padx=28, pady=(0, 16))

    app.create_secondary_button(
        card,
        text="Back",
        command=lambda: (close_startup_dialog(app), app.show_startup_source_dialog(force=True))
        if getattr(app, "_startup_mode", "chooser") == "label"
        else (close_startup_dialog(app), app.show_app_mode_dialog(force=True)),
    ).pack(fill="x", padx=28, pady=(0, 10))


def close_startup_dialog(app: Any) -> None:
    app._close_fullpage_overlay()
    app._startup_dialog_open = False
    app.logger.info("Startup source dialog closed")
