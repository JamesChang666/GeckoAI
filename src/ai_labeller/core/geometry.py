from __future__ import annotations

from .types import Rect
import math
from typing import List, Tuple


def normalize_angle_deg(angle_deg: float) -> float:
    return ((angle_deg + 180.0) % 360.0) - 180.0


def get_rect_angle_deg(rect: Rect) -> float:
    if len(rect) >= 6:
        return normalize_angle_deg(float(rect[5]))
    return 0.0


def set_rect_angle_deg(rect: Rect, angle_deg: float) -> Rect:
    normalized = normalize_angle_deg(float(angle_deg))
    out = rect[:]
    if len(out) >= 6:
        out[5] = normalized
    else:
        out.append(normalized)
    return out


def rotate_point_around_center(
    x: float,
    y: float,
    cx: float,
    cy: float,
    angle_deg: float,
) -> Tuple[float, float]:
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    dx = x - cx
    dy = y - cy
    return cx + dx * cos_t - dy * sin_t, cy + dx * sin_t + dy * cos_t


def get_rotated_corners(rect: Rect) -> List[Tuple[float, float]]:
    x1 = min(rect[0], rect[2])
    y1 = min(rect[1], rect[3])
    x2 = max(rect[0], rect[2])
    y2 = max(rect[1], rect[3])
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    angle_deg = get_rect_angle_deg(rect)

    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    if abs(angle_deg) <= 1e-6:
        return corners
    return [rotate_point_around_center(px, py, cx, cy, angle_deg) for px, py in corners]


def rect_to_obb_norm(rect: Rect, width: float, height: float) -> List[float]:
    points: List[float] = []
    for px, py in get_rotated_corners(rect):
        nx = max(0.0, min(1.0, px / max(width, 1e-6)))
        ny = max(0.0, min(1.0, py / max(height, 1e-6)))
        points.extend([nx, ny])
    return points


def obb_norm_to_rect(pts_norm: List[float], width: float, height: float, class_id: int) -> Rect:
    if len(pts_norm) != 8:
        return [0.0, 0.0, 0.0, 0.0, float(class_id), 0.0]
    pts = [
        (pts_norm[0] * width, pts_norm[1] * height),
        (pts_norm[2] * width, pts_norm[3] * height),
        (pts_norm[4] * width, pts_norm[5] * height),
        (pts_norm[6] * width, pts_norm[7] * height),
    ]
    cx = sum(p[0] for p in pts) / 4.0
    cy = sum(p[1] for p in pts) / 4.0
    dx = pts[1][0] - pts[0][0]
    dy = pts[1][1] - pts[0][1]
    angle_deg = math.degrees(math.atan2(dy, dx))

    local_pts = [rotate_point_around_center(px, py, cx, cy, -angle_deg) for px, py in pts]
    x1 = min(p[0] for p in local_pts)
    y1 = min(p[1] for p in local_pts)
    x2 = max(p[0] for p in local_pts)
    y2 = max(p[1] for p in local_pts)
    # clamp box will be handled by caller if needed
    return [x1, y1, x2, y2, int(class_id), angle_deg]


def point_in_rotated_box(x: float, y: float, rect: Rect) -> bool:
    x1 = min(rect[0], rect[2])
    y1 = min(rect[1], rect[3])
    x2 = max(rect[0], rect[2])
    y2 = max(rect[1], rect[3])
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    angle_deg = get_rect_angle_deg(rect)

    px, py = rotate_point_around_center(x, y, cx, cy, -angle_deg)
    return x1 < px < x2 and y1 < py < y2


def clamp_box(box: Rect, img_width: int, img_height: int) -> Rect:
    if img_width is None or img_height is None:
        return box
    W, H = img_width, img_height
    x1, x2 = sorted([box[0], box[2]])
    y1, y2 = sorted([box[1], box[3]])

    out = [
        max(0, min(W, x1)),
        max(0, min(H, y1)),
        max(0, min(W, x2)),
        max(0, min(H, y2)),
        int(box[4]) if len(box) > 4 else 0,
    ]
    if len(box) >= 6:
        out.append(normalize_angle_deg(float(box[5])))
    else:
        out.append(0.0)
    return out


def calculate_iou(box1: Rect, box2: Rect) -> float:
    """Compute IoU (intersection over union) for two axis-aligned boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    inter = (x2_i - x1_i) * (y2_i - y1_i)
    union = (x2_1 - x1_1) * (y2_1 - y1_1) + (x2_2 - x1_2) * (y2_2 - y1_2) - inter
    return inter / union if union > 0 else 0.0


def fuse_boxes(boxes: list[Rect], iou_thresh: float, dist_thresh: int) -> list[Rect]:
    """Iteratively merge boxes by IoU or short horizontal gap with vertical overlap."""
    if len(boxes) <= 1:
        return boxes
    keep_fusing = True
    current = [box[:] for box in boxes]
    while keep_fusing:
        keep_fusing = False
        merged: list[Rect] = []
        used = [False] * len(current)
        for i, box in enumerate(current):
            if used[i]:
                continue
            curr = box[:]
            used[i] = True
            for j in range(i + 1, len(current)):
                if used[j]:
                    continue
                other = current[j]
                should_merge = calculate_iou(curr, other) > iou_thresh
                if not should_merge:
                    h_dist = max(0, max(curr[0], other[0]) - min(curr[2], other[2]))
                    v_overlap = min(curr[3], other[3]) - max(curr[1], other[1])
                    if v_overlap > 0 and h_dist <= dist_thresh:
                        should_merge = True
                if should_merge:
                    curr = [
                        min(curr[0], other[0]),
                        min(curr[1], other[1]),
                        max(curr[2], other[2]),
                        max(curr[3], other[3]),
                        curr[4],
                    ]
                    used[j] = True
                    keep_fusing = True
            merged.append(curr)
        current = merged
    return current
