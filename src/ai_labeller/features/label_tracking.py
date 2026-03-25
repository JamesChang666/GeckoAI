from __future__ import annotations

import math
from typing import Any

from ai_labeller.features import label_video_state


def is_video_tracking_active(app: Any) -> bool:
    return (
        int(getattr(app, "_video_label_total_frames", 0) or 0) > 0
        and bool(getattr(app, "_track_video_enabled", False))
    )


def get_active_track_id(app: Any) -> int | None:
    try:
        track_id = int(getattr(app, "_active_track_id", 0) or 0)
    except Exception:
        return None
    return track_id if track_id > 0 else None


def set_active_track_id(app: Any, track_id: int | None) -> int | None:
    normalized = int(track_id) if track_id else 0
    app._active_track_id = normalized if normalized > 0 else None
    return get_active_track_id(app)


def ensure_track_overlay_rects(app: Any, image_path: str, img_w: int, img_h: int) -> list[list[float]]:
    base_rects = _load_rects_for_image(app, image_path, img_w, img_h)
    if not is_video_tracking_active(app):
        return [list(r) for r in base_rects]
    generated = build_generated_rects(app, image_path, img_w, img_h, existing_rects=base_rects)
    if not generated:
        return [list(r) for r in base_rects]
    return merge_generated_rects(base_rects, generated)


def build_generated_rects(
    app: Any,
    image_path: str,
    img_w: int,
    img_h: int,
    *,
    existing_rects: list[list[float]] | None = None,
) -> list[list[float]]:
    if not is_video_tracking_active(app) or not image_path:
        return []
    try:
        target_idx = app._image_paths.index(image_path)
    except Exception:
        return []
    existing = [list(r) for r in (existing_rects if existing_rects is not None else _load_rects_for_image(app, image_path, img_w, img_h))]
    existing_track_ids = {
        int(track_id)
        for rect in existing
        for track_id in [label_video_state.rect_track_id(rect)]
        if track_id is not None
    }
    out: list[list[float]] = []
    for track_id in _collect_track_ids(app):
        if int(track_id) in existing_track_ids:
            continue
        track_rect = _build_rect_for_track(app, int(track_id), target_idx, img_w, img_h)
        if track_rect is None:
            continue
        out.append(track_rect)
    return out


def merge_generated_rects(existing_rects: list[list[float]], generated_rects: list[list[float]]) -> list[list[float]]:
    merged = [list(r) for r in existing_rects]
    existing_track_ids = {
        int(track_id)
        for rect in existing_rects
        for track_id in [label_video_state.rect_track_id(rect)]
        if track_id is not None
    }
    for rect in generated_rects:
        track_id = label_video_state.rect_track_id(rect)
        if track_id is not None and int(track_id) in existing_track_ids:
            continue
        merged.append(list(rect))
    return merged


def mark_selected_as_new_track(app: Any) -> int:
    targets = _selected_indices(app)
    if not targets:
        return 0
    changed = 0
    for idx in targets:
        track_id = label_video_state.find_next_track_id(app)
        rect = app.canvas.rects[idx]
        rect = label_video_state.set_rect_track_id(rect, track_id)
        rect = label_video_state.make_keyframe_rect(rect, state=label_video_state.STATE_VISIBLE, track_id=track_id)
        app.canvas.rects[idx] = rect
        set_active_track_id(app, track_id)
        changed += 1
    return changed


def mark_selected_as_keyframe(app) -> int:
    targets = _selected_indices(app)
    if not targets:
        return 0
    changed = 0
    for idx in targets:
        rect = app.canvas.rects[idx]
        track_id = _resolve_track_id_for_rect(app, idx)
        if track_id is None and label_video_state.rect_track_id(rect) is None:
            track_id = infer_track_id_for_selected_rect(app, idx)
            if track_id is None:
                continue
            rect = label_video_state.set_rect_track_id(rect, track_id)
        rect = label_video_state.make_keyframe_rect(rect, state=label_video_state.STATE_VISIBLE, track_id=track_id)
        app.canvas.rects[idx] = rect
        set_active_track_id(app, track_id)
        changed += 1
        _remove_generated_track_duplicates(app, keep_idx=idx, track_id=int(track_id))
    return changed


def mark_selected_with_state(app: Any, state: str) -> int:
    targets = _selected_indices(app)
    if not targets:
        return 0
    changed = 0
    current_index = int(getattr(app, "_image_idx", -1))
    for idx in targets:
        rect = app.canvas.rects[idx]
        track_id = _resolve_track_id_for_rect(app, idx)
        if track_id is None:
            continue
        rect = label_video_state.make_keyframe_rect(rect, state=state, track_id=track_id)
        app.canvas.rects[idx] = rect
        set_active_track_id(app, track_id)
        changed += 1
        _remove_generated_track_duplicates(app, keep_idx=idx, track_id=int(track_id))
        if state == label_video_state.STATE_OUTSIDE and current_index >= 0:
            changed += _remove_track_rects_after_index(app, int(track_id), current_index)
    return changed


def attach_selected_to_track(app: Any, track_id: int) -> int:
    targets = _selected_indices(app)
    if not targets or int(track_id or 0) <= 0:
        return 0
    changed = 0
    for idx in targets:
        rect = app.canvas.rects[idx]
        updated_rect = label_video_state.set_rect_track_id(rect, int(track_id))
        updated_rect = label_video_state.set_rect_state(updated_rect, label_video_state.STATE_VISIBLE)
        updated_rect = label_video_state.set_rect_generated(updated_rect, False)
        app.canvas.rects[idx] = updated_rect
        set_active_track_id(app, int(track_id))
        changed += 1
    return changed


def attach_new_box_to_generated_track(app: Any, rect_idx: int) -> int:
    if not is_video_tracking_active(app):
        return 0
    if not hasattr(app, "canvas") or app.canvas is None:
        return 0
    if not (0 <= int(rect_idx) < len(app.canvas.rects)):
        return 0
    rect = app.canvas.rects[int(rect_idx)]
    if label_video_state.rect_track_id(rect) is not None:
        return 0
    active_track_id = get_active_track_id(app)
    if active_track_id is not None:
        matched_rect = _find_generated_rect_for_track_on_canvas(app, active_track_id)
        if matched_rect is not None:
            generated_idx, generated_rect = matched_rect
            if _same_class(rect, generated_rect):
                attached_rect = label_video_state.set_rect_track_id(rect, int(active_track_id))
                attached_rect = label_video_state.set_rect_state(attached_rect, label_video_state.STATE_VISIBLE)
                attached_rect = label_video_state.set_rect_generated(attached_rect, False)
                app.canvas.rects[int(rect_idx)] = attached_rect
                _remove_generated_track_duplicates(app, keep_idx=int(rect_idx), track_id=int(active_track_id))
                set_active_track_id(app, int(active_track_id))
                return 1
    class_id = int(rect[4]) if len(rect) >= 5 else 0
    cx = (float(rect[0]) + float(rect[2])) / 2.0
    cy = (float(rect[1]) + float(rect[3])) / 2.0
    best_idx: int | None = None
    best_track_id: int | None = None
    best_distance: float | None = None
    for idx, candidate in enumerate(app.canvas.rects):
        if idx == int(rect_idx):
            continue
        candidate_track_id = label_video_state.rect_track_id(candidate)
        if candidate_track_id is None:
            continue
        if not label_video_state.is_generated(candidate):
            continue
        candidate_class_id = int(candidate[4]) if len(candidate) >= 5 else 0
        if candidate_class_id != class_id:
            continue
        candidate_cx = (float(candidate[0]) + float(candidate[2])) / 2.0
        candidate_cy = (float(candidate[1]) + float(candidate[3])) / 2.0
        distance = ((candidate_cx - cx) ** 2 + (candidate_cy - cy) ** 2) ** 0.5
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_idx = idx
            best_track_id = int(candidate_track_id)
    if best_track_id is None:
        return 0
    attached_rect = label_video_state.set_rect_track_id(rect, int(best_track_id))
    attached_rect = label_video_state.set_rect_state(attached_rect, label_video_state.STATE_VISIBLE)
    attached_rect = label_video_state.set_rect_generated(attached_rect, False)
    app.canvas.rects[int(rect_idx)] = attached_rect
    _remove_generated_track_duplicates(app, keep_idx=int(rect_idx), track_id=int(best_track_id))
    set_active_track_id(app, int(best_track_id))
    return 1


def collect_track_items(app: Any) -> list[tuple[int, str]]:
    items: list[tuple[int, str]] = []
    active_track_id = get_active_track_id(app)
    for track in label_video_state.collect_tracks(app):
        track_id = int(track["track_id"])
        class_id = int(track.get("class_id", 0))
        class_name = app._class_name_by_id(class_id) if hasattr(app, "_class_name_by_id") else f"class{class_id}"
        active_suffix = " | active" if active_track_id == track_id else ""
        items.append((track_id, f"Track {track_id} | {class_name}{active_suffix}"))
    return items


def materialize_track(app: Any, track_id: int | None = None) -> int:
    target_track_id = int(track_id or 0) if track_id is not None else int(get_active_track_id(app) or 0)
    if target_track_id <= 0:
        return 0
    changed = 0
    for target_idx, image_path in enumerate(getattr(app, "_image_paths", []) or []):
        img_w, img_h = app._image_size_for_label_path(image_path)
        if img_w <= 0 or img_h <= 0:
            continue
        rects = _load_rects_for_image(app, image_path, img_w, img_h)
        if any(label_video_state.rect_track_id(rect) == target_track_id for rect in rects):
            continue
        rect = _build_materialized_rect_for_track(app, target_track_id, target_idx, img_w, img_h)
        if rect is None:
            continue
        updated = [list(r) for r in rects]
        updated.append(rect)
        app._labels_by_path[image_path] = updated
        changed += 1
    return changed


def _collect_track_ids(app: Any) -> list[int]:
    return [int(item["track_id"]) for item in label_video_state.collect_tracks(app)]


def _build_rect_for_track(app: Any, track_id: int, target_idx: int, img_w: int, img_h: int) -> list[float] | None:
    prev_item = _find_track_keyframe(app, track_id, target_idx, -1, img_w, img_h)
    next_item = _find_track_keyframe(app, track_id, target_idx, 1, img_w, img_h)
    if prev_item is None and next_item is None:
        return None
    if prev_item is not None and prev_item["index"] == target_idx:
        return None
    if next_item is not None and next_item["index"] == target_idx:
        return None
    if prev_item is not None and label_video_state.rect_state(prev_item["rect"]) == label_video_state.STATE_OUTSIDE:
        return None
    if prev_item is not None and next_item is not None:
        left_rect = prev_item["rect"]
        right_rect = next_item["rect"]
        if label_video_state.rect_state(right_rect) == label_video_state.STATE_OUTSIDE:
            return _interpolate_rect(left_rect, right_rect, ratio=float(target_idx - prev_item["index"]) / float(next_item["index"] - prev_item["index"]), track_id=track_id)
        ratio = float(target_idx - prev_item["index"]) / float(next_item["index"] - prev_item["index"])
        return _interpolate_rect(left_rect, right_rect, ratio, track_id)
    if prev_item is not None:
        return _carry_forward_rect(prev_item["rect"], track_id)
    return None


def _build_materialized_rect_for_track(app: Any, track_id: int, target_idx: int, img_w: int, img_h: int) -> list[float] | None:
    prev_item = _find_track_keyframe(app, track_id, target_idx, -1, img_w, img_h)
    next_item = _find_track_keyframe(app, track_id, target_idx, 1, img_w, img_h)
    if prev_item is None or next_item is None:
        return None
    if prev_item["index"] == target_idx or next_item["index"] == target_idx:
        return None
    left_rect = prev_item["rect"]
    right_rect = next_item["rect"]
    right_state = label_video_state.rect_state(right_rect)
    if right_state == label_video_state.STATE_OUTSIDE:
        ratio = float(target_idx - prev_item["index"]) / float(next_item["index"] - prev_item["index"])
        generated = _interpolate_rect(left_rect, right_rect, ratio, track_id)
        return _materialize_rect(generated, track_id)
    ratio = float(target_idx - prev_item["index"]) / float(next_item["index"] - prev_item["index"])
    generated = _interpolate_rect(left_rect, right_rect, ratio, track_id)
    return _materialize_rect(generated, track_id)


def _carry_forward_rect(rect: list[float], track_id: int) -> list[float] | None:
    state = label_video_state.rect_state(rect)
    if state == label_video_state.STATE_OUTSIDE:
        return None
    out = label_video_state.make_generated_rect(rect, state=state, track_id=track_id)
    return out


def _materialize_rect(rect: list[float], track_id: int) -> list[float]:
    out = label_video_state.set_rect_track_id(rect, int(track_id))
    out = label_video_state.set_rect_generated(out, False)
    out = label_video_state.set_rect_keyframe(out, False)
    return out


def _remove_track_rects_after_index(app: Any, track_id: int, after_index: int) -> int:
    removed = 0
    image_paths = getattr(app, "_image_paths", []) or []
    for idx in range(int(after_index) + 1, len(image_paths)):
        image_path = image_paths[idx]
        img_w, img_h = app._image_size_for_label_path(image_path)
        if img_w <= 0 or img_h <= 0:
            continue
        rects = _load_rects_for_image(app, image_path, img_w, img_h)
        kept_rects = [list(rect) for rect in rects if label_video_state.rect_track_id(rect) != int(track_id)]
        removed += max(0, len(rects) - len(kept_rects))
        app._labels_by_path[image_path] = kept_rects
    return removed


def _interpolate_rect(left_rect: list[float], right_rect: list[float], ratio: float, track_id: int) -> list[float]:
    x1 = _lerp(float(left_rect[0]), float(right_rect[0]), ratio)
    y1 = _lerp(float(left_rect[1]), float(right_rect[1]), ratio)
    x2 = _lerp(float(left_rect[2]), float(right_rect[2]), ratio)
    y2 = _lerp(float(left_rect[3]), float(right_rect[3]), ratio)
    cid = int(left_rect[4]) if len(left_rect) >= 5 else 0
    left_angle = float(left_rect[5]) if len(left_rect) >= 6 else 0.0
    right_angle = float(right_rect[5]) if len(right_rect) >= 6 else 0.0
    angle = _lerp_angle(left_angle, right_angle, ratio)
    state = label_video_state.STATE_VISIBLE
    if (
        label_video_state.rect_state(left_rect) == label_video_state.STATE_OCCLUDED
        or label_video_state.rect_state(right_rect) == label_video_state.STATE_OCCLUDED
    ):
        state = label_video_state.STATE_OCCLUDED
    rect = [x1, y1, x2, y2, cid, angle]
    return label_video_state.make_generated_rect(rect, state=state, track_id=track_id)


def _find_track_keyframe(app: Any, track_id: int, start_idx: int, step: int, img_w: int, img_h: int) -> dict[str, Any] | None:
    idx = int(start_idx)
    while 0 <= idx < len(app._image_paths):
        image_path = app._image_paths[idx]
        rects = _load_rects_for_image(app, image_path, img_w, img_h)
        for rect in rects:
            if label_video_state.rect_track_id(rect) != int(track_id):
                continue
            if not label_video_state.is_keyframe(rect):
                continue
            return {"index": idx, "rect": list(rect), "path": image_path}
        idx += int(step)
    return None


def _load_rects_for_image(app: Any, image_path: str, img_w: int, img_h: int) -> list[list[float]]:
    rects = app._labels_by_path.get(image_path)
    if rects is None:
        rects = app._load_label_file_for_image(image_path, img_w, img_h) if img_w > 0 and img_h > 0 else []
        app._labels_by_path[image_path] = [list(r) for r in rects]
    return [label_video_state.normalize_rect(r) for r in rects]


def _selected_indices(app: Any) -> list[int]:
    targets = sorted(i for i in getattr(app.canvas, "selected_indices", set()) if 0 <= int(i) < len(app.canvas.rects))
    if not targets and app.canvas.selected_idx is not None and 0 <= int(app.canvas.selected_idx) < len(app.canvas.rects):
        targets = [int(app.canvas.selected_idx)]
    return targets


def infer_track_id_for_selected_rect(app: Any, selected_idx: int) -> int | None:
    if not (0 <= int(selected_idx) < len(app.canvas.rects)):
        return None
    rect = app.canvas.rects[int(selected_idx)]
    existing_track_id = label_video_state.rect_track_id(rect)
    if existing_track_id is not None:
        return int(existing_track_id)
    active_track_id = get_active_track_id(app)
    if active_track_id is not None:
        active_track = _find_track_definition(app, active_track_id)
        if active_track is not None:
            candidate_class_id = int(active_track[4]) if len(active_track) >= 5 else 0
            if candidate_class_id == (int(rect[4]) if len(rect) >= 5 else 0):
                return int(active_track_id)
    class_id = int(rect[4]) if len(rect) >= 5 else 0
    cx = (float(rect[0]) + float(rect[2])) / 2.0
    cy = (float(rect[1]) + float(rect[3])) / 2.0
    best_track_id: int | None = None
    best_score: tuple[int, float] | None = None
    for idx, candidate in enumerate(app.canvas.rects):
        if idx == int(selected_idx):
            continue
        candidate_track_id = label_video_state.rect_track_id(candidate)
        if candidate_track_id is None:
            continue
        candidate_class_id = int(candidate[4]) if len(candidate) >= 5 else 0
        if candidate_class_id != class_id:
            continue
        candidate_cx = (float(candidate[0]) + float(candidate[2])) / 2.0
        candidate_cy = (float(candidate[1]) + float(candidate[3])) / 2.0
        distance = ((candidate_cx - cx) ** 2 + (candidate_cy - cy) ** 2) ** 0.5
        priority = 0 if label_video_state.is_generated(candidate) else 1
        score = (priority, distance)
        if best_score is None or score < best_score:
            best_score = score
            best_track_id = int(candidate_track_id)
    if best_track_id is not None:
        return best_track_id
    try:
        image_idx = int(app._image_idx)
    except Exception:
        image_idx = -1
    if image_idx < 0:
        return None
    for step in (-1, 1):
        idx = image_idx + step
        while 0 <= idx < len(app._image_paths):
            image_path = app._image_paths[idx]
            img_w, img_h = app._image_size_for_label_path(image_path)
            if img_w <= 0 or img_h <= 0:
                idx += step
                continue
            for candidate in _load_rects_for_image(app, image_path, img_w, img_h):
                candidate_track_id = label_video_state.rect_track_id(candidate)
                if candidate_track_id is None:
                    continue
                candidate_class_id = int(candidate[4]) if len(candidate) >= 5 else 0
                if candidate_class_id != class_id:
                    continue
                return int(candidate_track_id)
            idx += step
    return None


def _remove_generated_track_duplicates(app: Any, *, keep_idx: int, track_id: int) -> None:
    if int(track_id or 0) <= 0:
        return
    new_rects: list[list[float]] = []
    new_keep_idx = keep_idx
    for idx, rect in enumerate(app.canvas.rects):
        same_track = label_video_state.rect_track_id(rect) == int(track_id)
        if idx != keep_idx and same_track and label_video_state.is_generated(rect):
            if idx < keep_idx:
                new_keep_idx -= 1
            continue
        new_rects.append(rect)
    app.canvas.rects = [list(r) for r in new_rects]
    app.canvas.selected_idx = new_keep_idx if 0 <= new_keep_idx < len(app.canvas.rects) else None
    if app.canvas.selected_idx is not None:
        app.canvas.selected_indices = {app.canvas.selected_idx}


def auto_select_active_track_rect(app: Any) -> bool:
    active_track_id = get_active_track_id(app)
    if active_track_id is None or not hasattr(app, "canvas") or app.canvas is None:
        return False
    match = _find_rect_index_for_track(app, active_track_id)
    if match is None:
        return False
    app.canvas.selected_idx = int(match)
    app.canvas.selected_indices = {int(match)}
    return True


def _resolve_track_id_for_rect(app: Any, selected_idx: int) -> int | None:
    if not (0 <= int(selected_idx) < len(app.canvas.rects)):
        return None
    rect = app.canvas.rects[int(selected_idx)]
    track_id = label_video_state.rect_track_id(rect)
    if track_id is not None:
        return int(track_id)
    active_track_id = get_active_track_id(app)
    if active_track_id is None:
        return None
    track_rect = _find_track_definition(app, active_track_id)
    if track_rect is None or not _same_class(rect, track_rect):
        return None
    return int(active_track_id)


def _find_generated_rect_for_track_on_canvas(app: Any, track_id: int) -> tuple[int, list[float]] | None:
    for idx, rect in enumerate(getattr(app.canvas, "rects", []) or []):
        if label_video_state.rect_track_id(rect) == int(track_id) and label_video_state.is_generated(rect):
            return idx, rect
    return None


def _find_rect_index_for_track(app: Any, track_id: int) -> int | None:
    generated_idx: int | None = None
    keyframe_idx: int | None = None
    for idx, rect in enumerate(getattr(app.canvas, "rects", []) or []):
        if label_video_state.rect_track_id(rect) != int(track_id):
            continue
        if label_video_state.is_generated(rect) and generated_idx is None:
            generated_idx = idx
        elif label_video_state.is_keyframe(rect) and keyframe_idx is None:
            keyframe_idx = idx
    return generated_idx if generated_idx is not None else keyframe_idx


def _find_track_definition(app: Any, track_id: int) -> list[float] | None:
    for image_path in getattr(app, "_image_paths", []) or []:
        img_w, img_h = app._image_size_for_label_path(image_path)
        if img_w <= 0 or img_h <= 0:
            continue
        for rect in _load_rects_for_image(app, image_path, img_w, img_h):
            if label_video_state.rect_track_id(rect) == int(track_id):
                return rect
    return None


def _same_class(left_rect: list[float], right_rect: list[float]) -> bool:
    left_class = int(left_rect[4]) if len(left_rect) >= 5 else 0
    right_class = int(right_rect[4]) if len(right_rect) >= 5 else 0
    return left_class == right_class


def _lerp(a: float, b: float, ratio: float) -> float:
    return float(a) + (float(b) - float(a)) * float(ratio)


def _lerp_angle(a: float, b: float, ratio: float) -> float:
    delta = ((float(b) - float(a) + 180.0) % 360.0) - 180.0
    out = float(a) + delta * float(ratio)
    while out <= -180.0:
        out += 360.0
    while out > 180.0:
        out -= 360.0
    return out if math.isfinite(out) else 0.0
