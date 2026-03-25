from __future__ import annotations

import json
import os
from typing import Any


STATE_VISIBLE = "visible"
STATE_OCCLUDED = "occluded"
STATE_OUTSIDE = "outside"
STATE_ENDED = STATE_OUTSIDE
VALID_STATES = {STATE_VISIBLE, STATE_OCCLUDED, STATE_OUTSIDE}

IDX_STATE = 6
IDX_TRACK_ID = 7
IDX_KEYFRAME = 8
IDX_GENERATED = 9


def is_video_label_mode(app: Any) -> bool:
    return int(getattr(app, "_video_label_total_frames", 0) or 0) > 0


def normalize_state(value: Any) -> str:
    state = str(value or "").strip().lower()
    if state == "ended":
        state = STATE_OUTSIDE
    return state if state in VALID_STATES else STATE_VISIBLE


def normalize_rect(rect: list[float]) -> list[float]:
    out = list(rect[:10])
    while len(out) < 6:
        out.append(0.0)
    if len(out) < 7:
        out.append(STATE_VISIBLE)
    else:
        out[IDX_STATE] = normalize_state(out[IDX_STATE])
    if len(out) < 8:
        out.append(None)
    else:
        out[IDX_TRACK_ID] = _normalize_track_id(out[IDX_TRACK_ID])
    if len(out) < 9:
        out.append(False)
    else:
        out[IDX_KEYFRAME] = bool(out[IDX_KEYFRAME])
    if len(out) < 10:
        out.append(False)
    else:
        out[IDX_GENERATED] = bool(out[IDX_GENERATED])
    return out


def _normalize_track_id(value: Any) -> int | None:
    try:
        track_id = int(value)
    except Exception:
        return None
    return track_id if track_id > 0 else None


def rect_state(rect: list[float]) -> str:
    return normalize_rect(rect)[IDX_STATE]


def rect_track_id(rect: list[float]) -> int | None:
    return normalize_rect(rect)[IDX_TRACK_ID]


def is_keyframe(rect: list[float]) -> bool:
    return bool(normalize_rect(rect)[IDX_KEYFRAME])


def is_generated(rect: list[float]) -> bool:
    return bool(normalize_rect(rect)[IDX_GENERATED])


def is_tracked_rect(rect: list[float]) -> bool:
    return rect_track_id(rect) is not None


def set_rect_state(rect: list[float], state: str) -> list[float]:
    out = normalize_rect(rect)
    out[IDX_STATE] = normalize_state(state)
    return out


def set_rect_track_id(rect: list[float], track_id: int | None) -> list[float]:
    out = normalize_rect(rect)
    out[IDX_TRACK_ID] = _normalize_track_id(track_id)
    return out


def set_rect_keyframe(rect: list[float], is_manual_keyframe: bool = True) -> list[float]:
    out = normalize_rect(rect)
    out[IDX_KEYFRAME] = bool(is_manual_keyframe)
    out[IDX_GENERATED] = False
    return out


def set_rect_generated(rect: list[float], generated: bool = True) -> list[float]:
    out = normalize_rect(rect)
    out[IDX_GENERATED] = bool(generated)
    if generated:
        out[IDX_KEYFRAME] = False
    return out


def make_keyframe_rect(rect: list[float], *, state: str | None = None, track_id: int | None = None) -> list[float]:
    out = normalize_rect(rect)
    if state is not None:
        out[IDX_STATE] = normalize_state(state)
    if track_id is not None or rect_track_id(out) is not None:
        out[IDX_TRACK_ID] = _normalize_track_id(track_id if track_id is not None else rect_track_id(out))
    out[IDX_KEYFRAME] = True
    out[IDX_GENERATED] = False
    return out


def make_generated_rect(rect: list[float], *, state: str | None = None, track_id: int | None = None) -> list[float]:
    out = normalize_rect(rect)
    if state is not None:
        out[IDX_STATE] = normalize_state(state)
    if track_id is not None or rect_track_id(out) is not None:
        out[IDX_TRACK_ID] = _normalize_track_id(track_id if track_id is not None else rect_track_id(out))
    out[IDX_KEYFRAME] = False
    out[IDX_GENERATED] = True
    return out


def is_exportable_rect(rect: list[float]) -> bool:
    return rect_state(rect) == STATE_VISIBLE


def state_display_name(state: str) -> str:
    state = normalize_state(state)
    if state == STATE_OCCLUDED:
        return "Occluded"
    if state == STATE_OUTSIDE:
        return "Outside"
    return "Visible"


def sidecar_path_for_image(app: Any, image_path: str) -> str:
    return app._label_path_for_image(image_path) + ".video.json"


def load_rects_from_sidecar(app: Any, image_path: str) -> list[list[float]] | None:
    sidecar_path = sidecar_path_for_image(app, image_path)
    if not os.path.isfile(sidecar_path):
        return None
    try:
        with open(sidecar_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    rect_items = payload.get("rects", [])
    if not isinstance(rect_items, list):
        return None
    rects: list[list[float]] = []
    for item in rect_items:
        if not isinstance(item, dict):
            continue
        try:
            rect = [
                float(item.get("x1", 0.0)),
                float(item.get("y1", 0.0)),
                float(item.get("x2", 0.0)),
                float(item.get("y2", 0.0)),
                int(item.get("class_id", 0)),
                float(item.get("angle", 0.0)),
                normalize_state(item.get("state", STATE_VISIBLE)),
                _normalize_track_id(item.get("track_id")),
                bool(item.get("keyframe", False)),
                bool(item.get("generated", False)),
            ]
        except Exception:
            continue
        rects.append(normalize_rect(rect))
    return rects


def save_rects_sidecar(app: Any, image_path: str, rects: list[list[float]]) -> None:
    sidecar_path = sidecar_path_for_image(app, image_path)
    normalized = [normalize_rect(rect) for rect in rects]
    needs_sidecar = any(
        rect_state(rect) != STATE_VISIBLE
        or rect_track_id(rect) is not None
        or is_keyframe(rect)
        or is_generated(rect)
        for rect in normalized
    )
    if not needs_sidecar:
        try:
            if os.path.isfile(sidecar_path):
                os.remove(sidecar_path)
        except Exception:
            pass
        return
    payload = {
        "version": 2,
        "rects": [
            {
                "x1": float(rect[0]),
                "y1": float(rect[1]),
                "x2": float(rect[2]),
                "y2": float(rect[3]),
                "class_id": int(rect[4]) if len(rect) >= 5 else 0,
                "angle": float(rect[5]) if len(rect) >= 6 else 0.0,
                "state": rect_state(rect),
                "track_id": rect_track_id(rect),
                "keyframe": is_keyframe(rect),
                "generated": is_generated(rect),
            }
            for rect in normalized
            if len(rect) >= 4
        ],
    }
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def find_next_track_id(app: Any) -> int:
    max_track_id = 0
    for rects in getattr(app, "_labels_by_path", {}).values():
        for rect in rects or []:
            track_id = rect_track_id(rect)
            if track_id is not None:
                max_track_id = max(max_track_id, int(track_id))
    return max_track_id + 1


def collect_tracks(app: Any) -> list[dict[str, Any]]:
    summary: dict[int, dict[str, Any]] = {}
    for image_path, rects in getattr(app, "_labels_by_path", {}).items():
        for rect in rects or []:
            track_id = rect_track_id(rect)
            if track_id is None:
                continue
            item = summary.setdefault(
                int(track_id),
                {
                    "track_id": int(track_id),
                    "class_id": int(rect[4]) if len(rect) >= 5 else 0,
                    "frames": 0,
                    "last_path": image_path,
                },
            )
            item["frames"] += 1
            item["last_path"] = image_path
    return [summary[k] for k in sorted(summary.keys())]
