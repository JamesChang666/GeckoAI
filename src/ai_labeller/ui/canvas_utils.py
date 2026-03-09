from __future__ import annotations

from typing import Any, List, Tuple

from ai_labeller.core import geometry as geom


def clamp_box(app: Any, box: list[float]) -> list[float]:
    if not getattr(app, "img_pil", None):
        return box
    return geom.clamp_box(box, app.img_pil.width, app.img_pil.height)


def normalize_angle_deg(angle_deg: float) -> float:
    return geom.normalize_angle_deg(angle_deg)


def get_rect_angle_deg(rect: list[float]) -> float:
    return geom.get_rect_angle_deg(rect)


def set_rect_angle_deg(rect: list[float], angle_deg: float) -> list[float]:
    return geom.set_rect_angle_deg(rect, angle_deg)


def rotate_point_around_center(x: float, y: float, cx: float, cy: float, angle_deg: float) -> Tuple[float, float]:
    return geom.rotate_point_around_center(x, y, cx, cy, angle_deg)


def get_rotated_corners(rect: list[float]) -> List[Tuple[float, float]]:
    return geom.get_rotated_corners(rect)


def rect_to_obb_norm(rect: list[float], width: float, height: float) -> list[float]:
    return geom.rect_to_obb_norm(rect, width, height)


def obb_norm_to_rect(app: Any, pts_norm: list[float], width: float, height: float, class_id: int) -> list[float]:
    rect = geom.obb_norm_to_rect(pts_norm, width, height, class_id)
    return clamp_box(app, rect)


def point_in_rotated_box(x: float, y: float, rect: list[float]) -> bool:
    return geom.point_in_rotated_box(x, y, rect)


def get_handles(app: Any, rect: list[float]):
    x1 = min(rect[0], rect[2])
    y1 = min(rect[1], rect[3])
    x2 = max(rect[0], rect[2])
    y2 = max(rect[1], rect[3])
    xm = (x1 + x2) / 2
    ym = (y1 + y2) / 2

    local_handles = [
        (x1, y1), (xm, y1), (x2, y1), (x2, ym),
        (x2, y2), (xm, y2), (x1, y2), (x1, ym)
    ]
    angle_deg = get_rect_angle_deg(rect)
    if abs(angle_deg) <= 1e-6:
        return local_handles
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    out = []
    for xh, yh in local_handles:
        rx, ry = rotate_point_around_center(xh, yh, cx, cy, angle_deg)
        out.append((rx, ry))
    return out
