from __future__ import annotations

from typing import Any, Callable, Dict

from ai_labeller.ui import button_factory as ui_buttons
from ai_labeller.constants import COLORS


def bind_button_creators(app: Any) -> Dict[str, Callable[..., Any]]:
    """Return bound button creator callables using the app's fonts/theme.

    This centralizes how buttons are created with the app's defaults so
    callers don't need to repeatedly pass `font_primary`, `theme`, `COLORS`, etc.
    """

    def create_toolbar_button(parent, text, command, bg=None):
        return ui_buttons.create_toolbar_button(
            parent, text, command, bg, app.font_primary, app.theme, COLORS
        )

    def create_toolbar_icon_button(parent, text, command, tooltip="", bg=None, fg=None, circular=False):
        return ui_buttons.create_toolbar_icon_button(
            parent, text, command, bg, fg, circular, app.theme, COLORS
        )

    def create_primary_button(parent, text, command, bg=None):
        return ui_buttons.create_primary_button(parent, text, command, bg, app.font_primary, COLORS)

    def create_secondary_button(parent, text, command):
        return ui_buttons.create_secondary_button(parent, text, command, app.font_primary, COLORS)

    def create_nav_button(parent, text, command, side, primary=False):
        return ui_buttons.create_nav_button(parent, text, command, side, primary, app.font_bold, COLORS)

    def lighten_color(color: str) -> str:
        return ui_buttons.lighten_color(color, COLORS)

    def is_accent_bg(bg: str) -> bool:
        return ui_buttons.is_accent_bg(bg, COLORS)

    def toolbar_text_color(bg: str) -> str:
        return ui_buttons.toolbar_text_color(bg, app.theme, COLORS)

    return {
        "create_toolbar_button": create_toolbar_button,
        "create_toolbar_icon_button": create_toolbar_icon_button,
        "create_primary_button": create_primary_button,
        "create_secondary_button": create_secondary_button,
        "create_nav_button": create_nav_button,
        "lighten_color": lighten_color,
        "is_accent_bg": is_accent_bg,
        "toolbar_text_color": toolbar_text_color,
    }
