import tkinter as tk
from typing import List, Tuple, Any

from ai_labeller.constants import COLORS, LANG_MAP


def shortcut_items(app: Any) -> List[Tuple[str, str]]:
    return [
        ("F", LANG_MAP[app.lang]["next"]),
        ("D", LANG_MAP[app.lang]["prev"]),
        ("Q / E", "Rotate selected box"),
        ("Ctrl+Z", LANG_MAP[app.lang]["undo"]),
        ("Ctrl+Y", LANG_MAP[app.lang]["redo"]),
        ("Del", LANG_MAP[app.lang]["delete"]),
    ]


def build_shortcut_text(app: Any) -> str:
    lines = [LANG_MAP[app.lang]["shortcut_help"]]
    for key, desc in shortcut_items(app):
        lines.append(f"{key} - {desc}")
    return "\n".join(lines)


def show_shortcut_tooltip(app: Any, widget: tk.Widget) -> None:
    hide_shortcut_tooltip(app)
    _show_tooltip_now(app, widget)


def _show_tooltip_now(app: Any, widget: tk.Widget) -> None:
    if getattr(app, "_tooltip_win", None):
        return
    x = widget.winfo_rootx() + 10
    y = widget.winfo_rooty() + widget.winfo_height() + 6
    win = tk.Toplevel(app.root)
    win.wm_overrideredirect(True)
    win.configure(bg=COLORS["bg_white"])
    label = tk.Label(
        win,
        text=build_shortcut_text(app),
        font=app.font_primary,
        fg=COLORS["text_primary"],
        bg=COLORS["bg_white"],
        justify="left",
        anchor="w",
        padx=10,
        pady=8,
    )
    label.pack()
    win.update_idletasks()

    tooltip_w = win.winfo_reqwidth()
    tooltip_h = win.winfo_reqheight()
    left, top, right, bottom = app._get_widget_monitor_bounds(widget)
    margin = 8

    if x + tooltip_w > right - margin:
        x = right - tooltip_w - margin
    if x < left + margin:
        x = left + margin

    if y + tooltip_h > bottom - margin:
        y = widget.winfo_rooty() - tooltip_h - 6
    if y < top + margin:
        y = top + margin

    win.wm_geometry(f"+{int(x)}+{int(y)}")
    app._tooltip_win = win


def hide_shortcut_tooltip(app: Any) -> None:
    if getattr(app, "_tooltip_after_id", None):
        try:
            app.root.after_cancel(app._tooltip_after_id)
        except Exception:
            pass
        app._tooltip_after_id = None
    if getattr(app, "_tooltip_win", None):
        try:
            app._tooltip_win.destroy()
        except Exception:
            pass
        app._tooltip_win = None


def create_help_icon(app: Any, parent: tk.Widget) -> tk.Label:
    btn = tk.Label(
        parent,
        text="?",
        font=app.font_bold,
        fg=app.toolbar_text_color(COLORS["bg_medium"]),
        bg=COLORS["bg_medium"],
        width=3,
        height=1,
        relief="flat",
    )
    btn.bind("<Enter>", lambda e: show_shortcut_tooltip(app, btn))
    btn.bind("<Leave>", lambda e: hide_shortcut_tooltip(app))
    return btn
