from __future__ import annotations

from typing import Any


def get_widget_monitor_bounds(widget: Any) -> tuple[int, int, int, int]:
    left = widget.winfo_vrootx()
    top = widget.winfo_vrooty()
    right = left + widget.winfo_vrootwidth()
    bottom = top + widget.winfo_vrootheight()

    try:
        import ctypes
        import ctypes.wintypes

        user32 = ctypes.windll.user32
        monitor = user32.MonitorFromPoint(
            ctypes.wintypes.POINT(widget.winfo_rootx(), widget.winfo_rooty()),
            2,
        )
        if not monitor:
            return left, top, right, bottom

        class RECT(ctypes.Structure):
            _fields_ = [
                ("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long),
            ]

        class MONITORINFO(ctypes.Structure):
            _fields_ = [
                ("cbSize", ctypes.c_ulong),
                ("rcMonitor", RECT),
                ("rcWork", RECT),
                ("dwFlags", ctypes.c_ulong),
            ]

        mi = MONITORINFO()
        mi.cbSize = ctypes.sizeof(MONITORINFO)
        ok = user32.GetMonitorInfoW(monitor, ctypes.byref(mi))
        if ok:
            return (
                mi.rcWork.left,
                mi.rcWork.top,
                mi.rcWork.right,
                mi.rcWork.bottom,
            )
    except Exception:
        pass

    return left, top, right, bottom
