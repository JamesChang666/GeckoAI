from __future__ import annotations

from typing import Any


def _qt_modules():
    try:
        from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox

        return QApplication, QFileDialog, QInputDialog, QMessageBox
    except Exception:
        return None, None, None, None


def _resolve_parent(parent: Any):
    app_cls, _, _, _ = _qt_modules()
    if app_cls is None:
        return None
    try:
        app = app_cls.instance()
    except Exception:
        app = None
    if parent is not None:
        return parent
    if app is None:
        return None
    try:
        return app.activeWindow()
    except Exception:
        return None


def _filetypes_to_filter(filetypes: Any) -> str:
    if not filetypes:
        return "All files (*.*)"
    parts: list[str] = []
    try:
        for item in filetypes:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            label = str(item[0]).strip() or "Files"
            pattern = str(item[1]).strip() or "*.*"
            parts.append(f"{label} ({pattern})")
    except Exception:
        return "All files (*.*)"
    return ";;".join(parts) if parts else "All files (*.*)"


class filedialog:
    @staticmethod
    def askopenfilename(**kwargs) -> str:
        _, file_dialog, _, _ = _qt_modules()
        if file_dialog is None:
            return ""
        parent = _resolve_parent(kwargs.get("parent"))
        title = str(kwargs.get("title", "Select File"))
        initialdir = str(kwargs.get("initialdir", ""))
        filt = _filetypes_to_filter(kwargs.get("filetypes"))
        try:
            path, _selected = file_dialog.getOpenFileName(parent, title, initialdir, filt)
            return path or ""
        except Exception:
            return ""

    @staticmethod
    def askdirectory(**kwargs) -> str:
        _, file_dialog, _, _ = _qt_modules()
        if file_dialog is None:
            return ""
        parent = _resolve_parent(kwargs.get("parent"))
        title = str(kwargs.get("title", "Select Folder"))
        initialdir = str(kwargs.get("initialdir", ""))
        try:
            path = file_dialog.getExistingDirectory(parent, title, initialdir)
            return path or ""
        except Exception:
            return ""


class messagebox:
    @staticmethod
    def showinfo(title: str, message: str, parent: Any = None) -> None:
        _, _, _, msg_box = _qt_modules()
        if msg_box is None:
            return
        msg_box.information(_resolve_parent(parent), str(title), str(message))

    @staticmethod
    def showwarning(title: str, message: str, parent: Any = None) -> None:
        _, _, _, msg_box = _qt_modules()
        if msg_box is None:
            return
        msg_box.warning(_resolve_parent(parent), str(title), str(message))

    @staticmethod
    def showerror(title: str, message: str, parent: Any = None) -> None:
        _, _, _, msg_box = _qt_modules()
        if msg_box is None:
            return
        msg_box.critical(_resolve_parent(parent), str(title), str(message))

    @staticmethod
    def askyesno(title: str, message: str, parent: Any = None) -> bool:
        _, _, _, msg_box = _qt_modules()
        if msg_box is None:
            return False
        buttons = msg_box.StandardButton.Yes | msg_box.StandardButton.No
        result = msg_box.question(_resolve_parent(parent), str(title), str(message), buttons)
        return result == msg_box.StandardButton.Yes


class simpledialog:
    @staticmethod
    def askinteger(
        title: str,
        prompt: str,
        parent: Any = None,
        minvalue: int | None = None,
        maxvalue: int | None = None,
        initialvalue: int | None = None,
    ) -> int | None:
        _, _, input_dialog, _ = _qt_modules()
        if input_dialog is None:
            return None
        low = -2147483648 if minvalue is None else int(minvalue)
        high = 2147483647 if maxvalue is None else int(maxvalue)
        value = low if initialvalue is None else int(initialvalue)
        value = max(low, min(high, value))
        try:
            result, ok = input_dialog.getInt(
                _resolve_parent(parent),
                str(title),
                str(prompt),
                value,
                low,
                high,
                1,
            )
            if not ok:
                return None
            return int(result)
        except Exception:
            return None
