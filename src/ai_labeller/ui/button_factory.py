from __future__ import annotations

import tkinter as tk
from typing import Any


def is_accent_bg(bg: str, colors: dict[str, str]) -> bool:
    return bg in {
        colors["primary"],
        colors["warning"],
        colors["danger"],
        colors["success"],
        colors["info"],
        colors["primary_light"],
    }


def toolbar_text_color(bg: str, theme: str, colors: dict[str, str]) -> str:
    if is_accent_bg(bg, colors):
        return colors["text_white"]
    return colors["text_white"] if theme == "dark" else colors["text_primary"]


def lighten_color(color: str, colors: dict[str, str]) -> str:
    if color == "#000000":
        return "#1A1A1A"
    if color == colors["danger"]:
        return "#FF6B54"
    if color == colors["warning"]:
        return "#FFBB33"
    return color


def create_toolbar_button(
    parent: Any,
    text: str,
    command: Any,
    bg: str | None,
    font_primary: tuple[str, int] | tuple[str, int, str],
    theme: str,
    colors: dict[str, str],
) -> tk.Button:
    bg_val = bg or colors["bg_medium"]
    fg_val = toolbar_text_color(bg_val, theme, colors)
    btn = tk.Button(
        parent,
        text=text,
        command=command,
        bg=bg_val,
        fg=fg_val,
        font=font_primary,
        relief="flat",
        padx=16,
        pady=8,
        cursor="hand2",
        borderwidth=0,
        highlightthickness=0,
    )

    def on_enter(_e: Any) -> None:
        btn.config(bg=colors["primary_hover"] if bg == colors["primary"] else colors["bg_medium"])

    def on_leave(_e: Any) -> None:
        btn.config(bg=bg_val)

    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)
    return btn


def create_toolbar_icon_button(
    parent: Any,
    text: str,
    command: Any,
    bg: str | None,
    fg: str | None,
    circular: bool,
    theme: str,
    colors: dict[str, str],
) -> tk.Button:
    bg_val = bg or colors["bg_medium"]
    fg_val = fg or toolbar_text_color(bg_val, theme, colors)
    icon_font = ("Segoe UI Symbol", 14, "bold") if circular else ("Arial", 14)
    btn = tk.Button(
        parent,
        text=text,
        command=command,
        bg=bg_val,
        fg=fg_val,
        font=icon_font,
        relief="flat",
        width=2 if circular else 3,
        height=1,
        cursor="hand2",
        borderwidth=0,
        highlightthickness=0,
        padx=6 if circular else 0,
        pady=2 if circular else 0,
    )

    def on_enter(_e: Any) -> None:
        if bg:
            btn.config(bg=lighten_color(bg, colors))
        else:
            btn.config(bg=colors["divider"])

    def on_leave(_e: Any) -> None:
        btn.config(bg=bg_val)

    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)
    return btn


def create_primary_button(
    parent: Any,
    text: str,
    command: Any,
    bg: str | None,
    font_primary: tuple[str, int] | tuple[str, int, str],
    colors: dict[str, str],
) -> tk.Button:
    original_bg = bg or colors["primary"]
    hover_bg = colors["primary_hover"] if not bg else lighten_color(bg, colors)
    btn = tk.Button(
        parent,
        text=text,
        command=command,
        bg=original_bg,
        fg=colors["text_white"],
        font=font_primary,
        relief="flat",
        pady=10,
        cursor="hand2",
        borderwidth=0,
        highlightthickness=0,
    )
    btn.bind("<Enter>", lambda _e: btn.config(bg=hover_bg))
    btn.bind("<Leave>", lambda _e: btn.config(bg=original_bg))
    return btn


def create_secondary_button(
    parent: Any,
    text: str,
    command: Any,
    font_primary: tuple[str, int] | tuple[str, int, str],
    colors: dict[str, str],
) -> tk.Button:
    label = str(text).strip().lower()
    is_back_button = label.startswith("back")
    base_bg = colors["danger"] if is_back_button else colors["bg_white"]
    base_fg = colors["text_white"] if is_back_button else colors["text_primary"]
    hover_bg = lighten_color(base_bg, colors) if is_back_button else colors["bg_light"]
    btn = tk.Button(
        parent,
        text=text,
        command=command,
        bg=base_bg,
        fg=base_fg,
        font=font_primary,
        relief="flat",
        pady=10,
        cursor="hand2",
        borderwidth=0 if is_back_button else 1,
        highlightthickness=0,
    )
    if is_back_button:
        btn.config(highlightbackground=base_bg, highlightcolor=base_bg)
    else:
        btn.config(highlightbackground=colors["border"], highlightcolor=colors["border"])
    btn.bind("<Enter>", lambda _e: btn.config(bg=hover_bg))
    btn.bind("<Leave>", lambda _e: btn.config(bg=base_bg))
    return btn


def create_nav_button(
    parent: Any,
    text: str,
    command: Any,
    side: str,
    primary: bool,
    font_bold: tuple[str, int] | tuple[str, int, str],
    colors: dict[str, str],
) -> tk.Button:
    bg = colors["primary"] if primary else colors["bg_white"]
    fg = colors["text_white"] if primary else colors["text_primary"]
    btn = tk.Button(
        parent,
        text=text,
        command=command,
        bg=bg,
        fg=fg,
        font=font_bold,
        relief="flat",
        pady=12,
        cursor="hand2",
        borderwidth=0 if primary else 1,
        highlightthickness=0,
    )
    if not primary:
        btn.config(highlightbackground=colors["border"], highlightcolor=colors["border"])
    if primary:
        btn.bind("<Enter>", lambda _e: btn.config(bg=colors["primary_hover"]))
        btn.bind("<Leave>", lambda _e: btn.config(bg=colors["primary"]))
    else:
        btn.bind("<Enter>", lambda _e: btn.config(bg=colors["bg_light"]))
        btn.bind("<Leave>", lambda _e: btn.config(bg=colors["bg_white"]))

    if side == "left":
        btn.pack(side="left", fill="both", expand=True, padx=(0, 4))
    else:
        btn.pack(side="right", fill="both", expand=True, padx=(4, 0))
    return btn
