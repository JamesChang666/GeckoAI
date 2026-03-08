import copy
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
from typing import Any


def render(app) -> None:
    app.canvas.delete("all")
    app._cursor_line_x = None
    app._cursor_line_y = None
    app._cursor_text_id = None
    app._cursor_bg_id = None

    if not app.img_pil:
        return

    w = int(app.img_pil.width * app.scale)
    h = int(app.img_pil.height * app.scale)
    app.img_tk = ImageTk.PhotoImage(app.img_pil.resize((w, h), Image.Resampling.NEAREST))
    app.canvas.create_image(app.offset_x, app.offset_y, image=app.img_tk, anchor="nw")

    if app.var_show_prev_labels.get() and app._prev_image_rects:
        ghost_color = "#A8B0BA"
        for rect in app._prev_image_rects:
            corners = app.get_rotated_corners(rect)
            canvas_points: list[float] = []
            for px, py in corners:
                cxp, cyp = app.img_to_canvas(px, py)
                canvas_points.extend([cxp, cyp])
            app.canvas.create_polygon(canvas_points, outline=ghost_color, width=1, fill="", dash=(4, 6))

    box_colors = [app.COLORS["box_1"], app.COLORS["box_2"], app.COLORS["box_3"], app.COLORS["box_4"], app.COLORS["box_5"], app.COLORS["box_6"]]

    selected_set = set(app._get_selected_indices())
    for i, rect in enumerate(app.rects):
        x1, y1 = app.img_to_canvas(rect[0], rect[1])
        corners = app.get_rotated_corners(rect)
        canvas_points: list[float] = []
        for px, py in corners:
            cxp, cyp = app.img_to_canvas(px, py)
            canvas_points.extend([cxp, cyp])

        is_selected = i in selected_set
        class_id = int(rect[4])
        color = app.COLORS["box_selected"] if is_selected else box_colors[class_id % len(box_colors)]
        width = 3 if is_selected else 2

        app.canvas.create_polygon(canvas_points, outline=color, width=width, fill="")

        angle_deg = app.get_rect_angle_deg(rect)
        if abs(angle_deg) > 1e-3:
            cx_mid = (rect[0] + rect[2]) / 2
            cy_mid = (rect[1] + rect[3]) / 2
            xh = (rect[0] + rect[2]) / 2
            yh = min(rect[1], rect[3])
            pxh, pyh = app.rotate_point_around_center(xh, yh, cx_mid, cy_mid, angle_deg)
            cxc, cyc = app.img_to_canvas(cx_mid, cy_mid)
            cxx, cyy = app.img_to_canvas(pxh, pyh)
            app.canvas.create_line(cxc, cyc, cxx, cyy, fill=color, width=2)

        if is_selected and app.selected_idx == i and len(selected_set) == 1:
            for hx, hy in app.get_handles(rect):
                cx, cy = app.img_to_canvas(hx, hy)
                app.canvas.create_oval(cx - app.HANDLE_SIZE, cy - app.HANDLE_SIZE, cx + app.HANDLE_SIZE, cy + app.HANDLE_SIZE, fill=app.COLORS["bg_white"], outline=color, width=2)
            top_x, top_y, rot_x, rot_y = app.get_rotation_handle_points(rect)
            ctx, cty = app.img_to_canvas(top_x, top_y)
            crx, cry = app.img_to_canvas(rot_x, rot_y)
            app.canvas.create_line(ctx, cty, crx, cry, fill=color, width=2)
            knob_r = app.HANDLE_SIZE + 1
            app.canvas.create_oval(crx - knob_r, cry - knob_r, crx + knob_r, cry + knob_r, fill=color, outline=app.COLORS["bg_white"], width=2)

        if app.show_all_labels:
            class_name = app.class_names[class_id] if class_id < len(app.class_names) else f"ID:{class_id}"
            if abs(angle_deg) > 1e-3:
                class_name = f"{class_name} ({angle_deg:.1f}簞)"

            min_canvas_y = min(canvas_points[1::2]) if canvas_points else y1
            min_canvas_x = min(canvas_points[0::2]) if canvas_points else x1
            label_y = max(min_canvas_y - 24, 8)
            text_id = app.canvas.create_text(min_canvas_x + 8, label_y + 4, text=class_name, fill=app.COLORS["text_white"], font=app.font_primary, anchor="nw")
            bbox = app.canvas.bbox(text_id)
            if bbox:
                padding = 4
                bg_id = app.canvas.create_rectangle(bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding, fill=color, outline="")
                app.canvas.tag_lower(bg_id, text_id)

    if app.temp_rect_coords:
        cx, cy, ex, ey = app.temp_rect_coords
        app.canvas.create_rectangle(cx, cy, ex, ey, outline=app.COLORS["primary_light"], width=2, dash=(6, 4))
    if app.select_rect_coords:
        cx, cy, ex, ey = app.select_rect_coords
        app.canvas.create_rectangle(cx, cy, ex, ey, outline=app.COLORS["box_selected"], width=2, dash=(4, 4))

    app.update_cursor_overlay()
    app.update_info_text()


def on_mouse_move(app, e: Any) -> None:
    app.mouse_pos = (e.x, e.y)
    app.update_cursor_overlay()


def update_cursor_overlay(app) -> None:
    if not app.canvas:
        return

    mx, my = app.mouse_pos
    canvas_h = app.canvas.winfo_height()
    canvas_w = app.canvas.winfo_width()

    if app._cursor_line_x is None:
        app._cursor_line_x = app.canvas.create_line(mx, 0, mx, canvas_h, fill=app.COLORS["primary"], width=1, dash=(2, 4), tags="cursor_overlay")
    else:
        app.canvas.coords(app._cursor_line_x, mx, 0, mx, canvas_h)

    if app._cursor_line_y is None:
        app._cursor_line_y = app.canvas.create_line(0, my, canvas_w, my, fill=app.COLORS["primary"], width=1, dash=(2, 4), tags="cursor_overlay")
    else:
        app.canvas.coords(app._cursor_line_y, 0, my, canvas_w, my)

    coord_text = f"{mx}, {my}"
    if app._cursor_text_id is None:
        app._cursor_text_id = app.canvas.create_text(mx + 12, my - 12, text=coord_text, fill=app.COLORS["text_primary"] if app.theme == "light" else app.COLORS["text_white"], font=app.font_mono, anchor="nw", tags="cursor_overlay")
    else:
        app.canvas.coords(app._cursor_text_id, mx + 12, my - 12)
        app.canvas.itemconfig(app._cursor_text_id, text=coord_text)

    coord_bbox = app.canvas.bbox(app._cursor_text_id)
    if not coord_bbox:
        return

    padding = 4
    bx1 = coord_bbox[0] - padding
    by1 = coord_bbox[1] - padding
    bx2 = coord_bbox[2] + padding
    by2 = coord_bbox[3] + padding
    if app._cursor_bg_id is None:
        app._cursor_bg_id = app.canvas.create_rectangle(bx1, by1, bx2, by2, fill=app.COLORS["bg_dark"], outline=app.COLORS["primary"], width=1, tags="cursor_overlay")
    else:
        app.canvas.coords(app._cursor_bg_id, bx1, by1, bx2, by2)
    app.canvas.tag_lower(app._cursor_bg_id, app._cursor_text_id)


def on_mouse_down(app, e):
    if not app.img_pil:
        return

    ix, iy = app.canvas_to_img(e.x, e.y)
    is_additive_select = bool(e.state & 0x0001) or bool(e.state & 0x0004)
    is_ctrl_select = bool(e.state & 0x0004)
    app.active_rotate_handle = False
    app.active_handle = None
    app.is_drag_selecting = False
    app.select_rect_coords = None

    if app.selected_idx is not None and len(app._get_selected_indices()) == 1 and not is_additive_select:
        active_rect = app.rects[app.selected_idx]
        _, _, rhx, rhy = app.get_rotation_handle_points(active_rect)
        rotate_dist = np.sqrt((ix - rhx) ** 2 + (iy - rhy) ** 2) * app.scale
        if rotate_dist < (app.config.mouse_handle_hit_radius_px + 3):
            cx = (active_rect[0] + active_rect[2]) / 2
            cy = (active_rect[1] + active_rect[3]) / 2
            pointer_deg = np.degrees(np.arctan2(iy - cy, ix - cx))
            app.rotate_drag_offset_deg = pointer_deg - app.get_rect_angle_deg(active_rect)
            app.active_rotate_handle = True
            app.drag_start = (ix, iy)
            app.push_history()
            return
        for i, (hx, hy) in enumerate(app.get_handles(active_rect)):
            dist = np.sqrt((ix - hx) ** 2 + (iy - hy) ** 2) * app.scale
            if dist < app.config.mouse_handle_hit_radius_px:
                app.active_handle = i
                app.drag_start = (ix, iy)
                app.push_history()
                return

    clicked_idx = app._pick_box_at_point(ix, iy)

    if clicked_idx is not None:
        if is_additive_select:
            selected = app._get_selected_indices()
            if clicked_idx in selected:
                selected = [idx for idx in selected if idx != clicked_idx]
            else:
                selected.append(clicked_idx)
            app._set_selected_indices(selected, primary_idx=clicked_idx if clicked_idx in selected else None)
            app._sync_class_combo_with_selection()
            app.is_moving_box = False
            app.drag_start = None
        else:
            selected = app._get_selected_indices()
            if clicked_idx in selected and len(selected) > 1:
                app._set_selected_indices(selected, primary_idx=clicked_idx)
            else:
                app._set_selected_indices([clicked_idx], primary_idx=clicked_idx)
            app.is_moving_box = True
            app.drag_start = (ix, iy)
            app._sync_class_combo_with_selection()
            app.push_history()
    else:
        if is_ctrl_select:
            app.drag_start = (ix, iy)
            app.is_drag_selecting = True
            app.select_rect_coords = (e.x, e.y, e.x, e.y)
        else:
            if not is_additive_select:
                app._set_selected_indices([])
            app.drag_start = (ix, iy)
            app.temp_rect_coords = (e.x, e.y, e.x, e.y)

    app.render()


def on_mouse_down_right(app, e):
    if not app.img_pil:
        return
    if app.var_show_prev_labels.get() and app._prev_image_rects:
        ix, iy = app.canvas_to_img(e.x, e.y)
        app.active_rotate_handle = False
        app.active_handle = None
        app.is_drag_selecting = False
        app.select_rect_coords = None
        app.is_moving_box = False
        app.drag_start = None
        app.temp_rect_coords = None
        app.paste_previous_labels(ix, iy)
        return
    app.active_rotate_handle = False
    app.active_handle = None
    app.is_drag_selecting = False
    app.select_rect_coords = None
    app.is_moving_box = False
    app.drag_start = app.canvas_to_img(e.x, e.y)
    app.temp_rect_coords = (e.x, e.y, e.x, e.y)
    app.render()


def on_mouse_drag(app, e):
    app.mouse_pos = (e.x, e.y)
    if not app.img_pil or not app.drag_start:
        app.render()
        return

    ix, iy = app.canvas_to_img(e.x, e.y)
    W, H = app.img_pil.width, app.img_pil.height
    ix = max(0, min(W, ix))
    iy = max(0, min(H, iy))

    if app.selected_idx is not None and app.active_rotate_handle:
        rect = app.rects[app.selected_idx]
        cx = (rect[0] + rect[2]) / 2
        cy = (rect[1] + rect[3]) / 2
        pointer_deg = np.degrees(np.arctan2(iy - cy, ix - cx))
        app.set_rect_angle_deg(rect, pointer_deg - app.rotate_drag_offset_deg)
    elif app.selected_idx is not None and app.active_handle is not None:
        rect = app.rects[app.selected_idx]
        cx = (rect[0] + rect[2]) / 2
        cy = (rect[1] + rect[3]) / 2
        angle_deg = app.get_rect_angle_deg(rect)
        lx, ly = app.rotate_point_around_center(ix, iy, cx, cy, -angle_deg)
        if app.active_handle in [0, 6, 7]:
            rect[0] = lx
        if app.active_handle in [0, 1, 2]:
            rect[1] = ly
        if app.active_handle in [2, 3, 4]:
            rect[2] = lx
        if app.active_handle in [4, 5, 6]:
            rect[3] = ly
    elif app.is_moving_box:
        dx = ix - app.drag_start[0]
        dy = iy - app.drag_start[1]
        selected = app._get_selected_indices()
        if not selected and app.selected_idx is not None:
            selected = [app.selected_idx]
        if selected:
            min_dx = max(-app.rects[idx][0] for idx in selected)
            max_dx = min(W - app.rects[idx][2] for idx in selected)
            min_dy = max(-app.rects[idx][1] for idx in selected)
            max_dy = min(H - app.rects[idx][3] for idx in selected)
            clamped_dx = min(max(dx, min_dx), max_dx)
            clamped_dy = min(max(dy, min_dy), max_dy)
            for idx in selected:
                rect = app.rects[idx]
                rect[0] += clamped_dx
                rect[1] += clamped_dy
                rect[2] += clamped_dx
                rect[3] += clamped_dy
        app.drag_start = (ix, iy)
    elif app.is_drag_selecting:
        if app.select_rect_coords:
            app.select_rect_coords = (app.select_rect_coords[0], app.select_rect_coords[1], e.x, e.y)
    else:
        if app.temp_rect_coords:
            app.temp_rect_coords = (app.temp_rect_coords[0], app.temp_rect_coords[1], e.x, e.y)

    app.render()


def on_mouse_up(app, e):
    if app.is_drag_selecting and app.select_rect_coords:
        sx, sy, ex, ey = app.select_rect_coords
        ix1, iy1 = app.canvas_to_img(sx, sy)
        ix2, iy2 = app.canvas_to_img(ex, ey)
        hits = app._pick_boxes_in_img_rect(ix1, iy1, ix2, iy2)
        merged = sorted(set(app._get_selected_indices() + hits))
        app._set_selected_indices(merged, primary_idx=hits[-1] if hits else app.selected_idx)
        app._sync_class_combo_with_selection()
        app.is_drag_selecting = False
        app.select_rect_coords = None
        app.drag_start = None
        app.render()
        return

    if app.temp_rect_coords:
        ix, iy = app.canvas_to_img(e.x, e.y)
        new_box = app.clamp_box([app.drag_start[0], app.drag_start[1], ix, iy, app.combo_cls.current()])
        if (new_box[2] - new_box[0]) > 2 and (new_box[3] - new_box[1]) > 2:
            app.push_history()
            app.rects.append(new_box)
        app.temp_rect_coords = None

    for idx in app._get_selected_indices():
        app.rects[idx] = app.clamp_box(app.rects[idx])

    app.is_moving_box = False
    app.active_handle = None
    app.active_rotate_handle = False
    app.rotate_drag_offset_deg = 0.0
    app.is_drag_selecting = False
    app.select_rect_coords = None
    app.render()


def on_mouse_up_right(app, e):
    on_mouse_up(app, e)


def paste_previous_labels(app, ix: float, iy: float) -> None:
    if not app.img_pil or not app._prev_image_rects:
        return
    prev_idx = app._pick_prev_box_at_point(ix, iy)
    if prev_idx is None:
        return
    copied = app.clamp_box(copy.deepcopy(app._prev_image_rects[prev_idx]))
    app.push_history()
    app.rects.append(copied)
    pasted_idx = len(app.rects) - 1
    app._set_selected_indices([pasted_idx], primary_idx=pasted_idx)
    app._sync_class_combo_with_selection()
    app.render()


def on_zoom(app, e):
    factor = app.config.zoom_in_factor if e.delta > 0 else app.config.zoom_out_factor
    app.offset_x = e.x - (e.x - app.offset_x) * factor
    app.offset_y = e.y - (e.y - app.offset_y) * factor
    app.scale *= factor
    app.render()


def on_canvas_resize(app, e):
    if not app.img_pil:
        return
    app.fit_image_to_canvas()


def fit_image_to_canvas(app):
    if not app.img_pil:
        return
    cw = app.canvas.winfo_width()
    ch = app.canvas.winfo_height()
    if cw <= 1 or ch <= 1:
        return
    iw, ih = app.img_pil.width, app.img_pil.height
    if iw <= 0 or ih <= 0:
        app.logger.warning("Invalid image size: %sx%s", iw, ih)
        return
    scale = min(cw / iw, ch / ih)
    app.scale = scale
    app.offset_x = (cw - iw * scale) / 2
    app.offset_y = (ch - ih * scale) / 2
    app.render()
