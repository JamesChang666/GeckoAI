from __future__ import annotations

from typing import Any
import tkinter as tk
from tkinter import ttk


def create_label(app: Any, parent: tk.Widget, text: str = "", font=None, fg=None, bg=None, anchor: str | None = "w", **kwargs) -> tk.Label:
    font = font or getattr(app, "font_primary", None)
    fg = fg or app.COLORS.get("text_secondary")
    bg = bg or app.COLORS.get("bg_light")
    lbl = tk.Label(parent, text=text, font=font, fg=fg, bg=bg, anchor=anchor, **kwargs)
    return lbl


def create_bold_label(app: Any, parent: tk.Widget, text: str = "", font=None, fg=None, bg=None, anchor: str | None = "w", **kwargs) -> tk.Label:
    font = font or getattr(app, "font_bold", None)
    fg = fg or app.COLORS.get("text_primary")
    bg = bg or app.COLORS.get("bg_light")
    lbl = tk.Label(parent, text=text, font=font, fg=fg, bg=bg, anchor=anchor, **kwargs)
    return lbl


def create_mono_label(app: Any, parent: tk.Widget, text: str = "", font=None, fg=None, bg=None, **kwargs) -> tk.Label:
    font = font or getattr(app, "font_mono", None)
    fg = fg or app.COLORS.get("text_primary")
    bg = bg or app.COLORS.get("bg_light")
    return tk.Label(parent, text=text, font=font, fg=fg, bg=bg, **kwargs)


def create_textbox(app: Any, parent: tk.Widget, **kwargs) -> tk.Text:
    font = kwargs.pop("font", getattr(app, "font_mono", None))
    fg = kwargs.pop("fg", app.COLORS.get("text_primary"))
    bg = kwargs.pop("bg", app.COLORS.get("bg_white"))
    txt = tk.Text(parent, font=font, fg=fg, bg=bg, **kwargs)
    return txt


def create_combobox(app: Any, parent: tk.Widget, values=None, textvariable=None, state="readonly", font=None, **kwargs) -> ttk.Combobox:
    font = font or getattr(app, "font_primary", None)
    cb = ttk.Combobox(parent, values=values or [], textvariable=textvariable, state=state, font=font, **kwargs)
    return cb


def create_entry(app: Any, parent: tk.Widget, textvariable=None, font=None, state=None, readonlybackground=None, fg=None, bg=None, **kwargs) -> tk.Entry:
    font = font or getattr(app, "font_mono", None)
    fg = fg or app.COLORS.get("text_primary")
    bg = bg or app.COLORS.get("bg_light")
    entry_kwargs = dict(font=font, fg=fg, bg=bg)
    if textvariable is not None:
        entry_kwargs["textvariable"] = textvariable
    if state is not None:
        entry_kwargs["state"] = state
    if readonlybackground is not None:
        entry_kwargs["readonlybackground"] = readonlybackground
    entry_kwargs.update(kwargs)
    ent = tk.Entry(parent, **entry_kwargs)
    return ent
