import ctypes
import ctypes.wintypes
import copy
import datetime
import gc
import glob
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from collections import deque
from importlib import resources
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from ai_labeller.core import (
    AppConfig,
    AppState,
    SessionState,
    HistoryManager,
    atomic_write_json,
    atomic_write_text,
    setup_logging,
)

# Optional dependencies
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

try:
    import torch
    from groundingdino.util.inference import Model as GroundingDINOModel
    from segment_anything import sam_model_registry, SamPredictor
    HAS_FOUNDATION_STACK = True
except ImportError:
    HAS_FOUNDATION_STACK = False

LOGGER = setup_logging()

# ==================== Theme Colors ====================
COLORS = {
    # Primary
    "primary": "#5551FF",           # Figma ????????
    "primary_hover": "#4845E4",
    "primary_light": "#7B79FF",
    "primary_bg": "#F0F0FF",
    
    # Status colors
    "success": "#0FA958",           # Draw a new box?
    "danger": "#F24822",            # ???????
    "warning": "#FFAA00",           # ??????
    "info": "#18A0FB",              # Draw a new box?
    
    # Neutral colors
    "bg_dark": "#1E1E1E",           # ???????
    "bg_medium": "#2C2C2C",         # ?????
    "bg_light": "#F5F5F5",          # ????
    "bg_white": "#FFFFFF",          # ???
    "bg_canvas": "#18191B",         # Canvas ?
    
    # Text
    "text_primary": "#000000",
    "text_secondary": "#8E8E93",
    "text_tertiary": "#C7C7CC",
    "text_white": "#FFFFFF",
    
    # Borders
    "border": "#E5E5EA",
    "divider": "#38383A",
    
    # Bounding boxes
    "box_1": "#FF3B30",  # ??
    "box_2": "#FF9500",  # ??
    "box_3": "#FFCC00",  # ??
    "box_4": "#34C759",  # ??
    "box_5": "#5AC8FA",  # ??
    "box_6": "#5856D6",  # ??
    "box_selected": "#00D4FF",  # ???????
    
    # Effects
    "shadow": "rgba(0, 0, 0, 0.1)",
}

THEMES = {
    "dark": {
        "bg_dark": "#1E1E1E",
        "bg_medium": "#2C2C2C",
        "bg_light": "#F5F5F5",
        "bg_white": "#FFFFFF",
        "bg_canvas": "#18191B",
        "text_primary": "#000000",
        "text_secondary": "#8E8E93",
        "text_tertiary": "#C7C7CC",
        "border": "#E5E5EA",
        "divider": "#38383A",
    },
    "light": {
        "bg_dark": "#F5F5F7",
        "bg_medium": "#E5E5EA",
        "bg_light": "#FFFFFF",
        "bg_white": "#FFFFFF",
        "bg_canvas": "#EFEFF4",
        "text_primary": "#000000",
        "text_secondary": "#5C5C5C",
        "text_tertiary": "#8E8E93",
        "border": "#D1D1D6",
        "divider": "#D1D1D6",
    },
}

# ==================== Language ====================
LANG_MAP = {
    "zh": {
    "title": "AI 標註專業版",
    "load_proj": "載入專案",
    "undo": "復原",
    "redo": "重做",
    "autolabel": "紅色檢測",
    "fuse": "合併框",
    "file_info": "檔案資訊",
    "no_img": "沒有影像",
    "filename": "檔案",
    "progress": "進度",
    "boxes": "框",
    "class_mgmt": "類別管理",
    "current_class": "目前類別",
    "edit_classes": "編輯類別",
    "reassign_class": "重新指定所選類別",
    "clear_labels": "刪除所有標註",
    "add": "新增",
    "rename": "重新命名",
    "apply": "套用",
    "delete_class": "刪除類別",
    "delete_class_confirm": "刪除類別 '{name}' (ID {idx})？\n目前影像中此類別的標註將被重新指定。",
    "delete_class_last": "無法刪除最後一個類別。",
    "class_name": "類別名稱",
    "rename_prompt": "修改 '{name}':",
    "add_prompt": "類別名稱：",
    "current": "目前",
    "to": "至",
    "no_label_selected": "未選取任何標註。",
    "no_classes_available": "沒有可用的類別。",
    "theme_light": "淺色模式",
    "theme_dark": "深色模式",
    "export_format": "全部匯出為",
    "ai_tools": "AI 工具",
    "auto_detect": "自動檢測",
    "learning": "學習",
    "foundation_mode": "基礎輔助",
    "propagate": "傳播",
    "run_detection": "執行檢測",
    "train_from_labels": "以現有標註開始訓練",
    "detection_model": "檢測模型",
    "browse_model": "瀏覽模型",
    "train_range_start": "起始索引（從 1 開始）",
    "train_range_end": "結束索引（從 1 開始）",
    "train_epochs": "訓練回合數（Epochs）",
    "train_imgsz": "訓練影像尺寸",
    "select_train_output": "選擇訓練輸出資料夾",
    "train_no_project": "尚未載入資料集。",
    "train_no_labels": "找不到可用於訓練的已標註影像。",
    "train_bad_range": "區間無效，請輸入正確的起始與結束索引。",
    "train_done": "訓練完成。\n輸出位置：{path}",
    "train_failed": "訓練失敗：{err}",
    "train_monitor": "訓練監控",
    "train_status": "狀態",
    "train_progress": "進度",
    "train_eta": "預估剩餘時間",
    "train_idle": "閒置",
    "train_running": "訓練中",
    "train_command": "命令",
    "train_already_running": "目前已有訓練在執行中。",
    "use_official_yolo26n": "使用官方 yolo26m.pt",
    "export": "全部匯出",
    "prev": "上一個",
    "next": "下一個",
    "shortcuts": "快捷鍵",
    "shortcut_help": "快捷鍵說明",
    "dataset": "資料集",
    "lang_switch": "EN",
    "delete": "刪除所選",
    "remove_from_split": "從分割中移除",
    "remove_confirm": "要將目前影像從 {split} 移除嗎？",
    "remove_done": "已移除：{name}",
    "remove_none": "沒有可移除的影像。",
    "restore_from_split": "恢復已刪除影格",
    "restore_none": "此分割中沒有找到已刪除的影格。",
    "restore_title": "恢復已刪除影格",
    "restore_select": "選擇要恢復的影格：",
    "restore_done": "已恢復：{name}",
    "select_image": "選擇影像",
    "startup_choose_source": "選擇啟動來源",
    "startup_prompt": "您要如何開始？",
    "startup_images": "開啟影像資料夾",
    "startup_yolo": "開啟 YOLO 資料集",
    "startup_rfdetr": "開啟 RF-DETR 資料集",
    "startup_skip": "稍後",
    "back_to_source": "返回來源選擇",
    "startup_model_cancel_title": "模型選擇已取消",
    "startup_model_cancel_msg": "未選擇模型。僅繼續使用影像資料夾？",
    "pick_folder_title": "選擇資料夾",
    "loaded_from": "已載入 {count} 張影像\n來源：{path}\n分割：{split}",
    "no_supported_images": "未找到支援的影像 (png/jpg/jpeg)\n資料夾：{path}",
    "select_export_folder": "選擇匯出資料夾",
    "export_no_project": "尚未載入資料集。",
    "export_done": "匯出完成：{count} 張影像\n輸出：{path}",
    "export_failed": "匯出失敗：{err}"
},
    "en": {
        "title": "AI Labeller Pro",
        "load_proj": "Load Project",
        "undo": "Undo",
        "redo": "Redo",
        "autolabel": "Red Detection",
        "fuse": "Fuse Boxes",
        "file_info": "File Info",
        "no_img": "No Image",
        "filename": "File",
        "progress": "Progress",
        "boxes": "Boxes",
        "class_mgmt": "Classes",
        "current_class": "Current Class",
        "edit_classes": "Edit Classes",
        "reassign_class": "Reassign Selected Class",
        "clear_labels": "Delete All Labels",
        "add": "Add",
        "rename": "Rename",
        "apply": "Apply",
        "delete_class": "Delete Class",
        "delete_class_confirm": "Delete class '{name}' (ID {idx})?\nLabels with this class in current image will be reassigned.",
        "delete_class_last": "Cannot delete the last class.",
        "class_name": "Class Name",
        "rename_prompt": "Modify '{name}':",
        "add_prompt": "Class name:",
        "current": "Current",
        "to": "To",
        "no_label_selected": "No label selected.",
        "no_classes_available": "No classes available.",
        "theme_light": "Light Mode",
        "theme_dark": "Dark Mode",
        "export_format": "Export All As",
        "ai_tools": "AI Tools",
        "auto_detect": "Auto Detect",
        "learning": "Learning",
        "foundation_mode": "Foundation Assist",
        "propagate": "Propagate",
        "run_detection": "Run Detection",
        "train_from_labels": "Train From Labels",
        "detection_model": "Detection Model",
        "browse_model": "Browse Model",
        "train_range_start": "Start Index (1-based)",
        "train_range_end": "End Index (1-based)",
        "train_epochs": "Epochs",
        "train_imgsz": "Image Size",
        "select_train_output": "Select Training Output Folder",
        "train_no_project": "No dataset loaded.",
        "train_no_labels": "No labeled images found for training.",
        "train_bad_range": "Invalid range. Please input valid start/end index.",
        "train_done": "Training finished.\nOutput: {path}",
        "train_failed": "Training failed: {err}",
        "train_monitor": "Training Monitor",
        "train_status": "Status",
        "train_progress": "Progress",
        "train_eta": "ETA",
        "train_idle": "Idle",
        "train_running": "Running",
        "train_command": "Command",
        "train_already_running": "Training is already running.",
        "use_official_yolo26n": "Use Official yolo26m.pt",
        "export": "Export All",
        "prev": "Previous",
        "next": "Next",
        "shortcuts": "Shortcuts",
        "shortcut_help": "Shortcut Help",
        "dataset": "Dataset",
        "lang_switch": "ZH",
        "delete": "Delete Selected",
        "remove_from_split": "Remove From Split",
        "remove_confirm": "Remove current image from {split}?",
        "remove_done": "Removed: {name}",
        "remove_none": "No image to remove.",
        "restore_from_split": "Restore Deleted Frame",
        "restore_none": "No removed frame found in this split.",
        "restore_title": "Restore Deleted Frame",
        "restore_select": "Select a frame to restore:",
        "restore_done": "Restored: {name}",
        "select_image": "Select Image",
        "startup_choose_source": "Choose Startup Source",
        "startup_prompt": "How do you want to start?",
        "startup_images": "Open Images Folder",
        "startup_yolo": "Open YOLO Dataset",
        "startup_rfdetr": "Open RF-DETR Dataset",
        "startup_skip": "Later",
        "back_to_source": "Back to Source Select",
        "startup_model_cancel_title": "Model Selection Cancelled",
        "startup_model_cancel_msg": "No model selected. Continue with images folder only?",
        "pick_folder_title": "Select Folder",
        "loaded_from": "Loaded {count} images\nFrom: {path}\nSplit: {split}",
        "no_supported_images": "No supported images found (png/jpg/jpeg)\nFolder: {path}",
        "select_export_folder": "Select Export Folder",
        "export_no_project": "No dataset loaded.",
        "export_done": "Export completed: {count} images\nOutput: {path}",
        "export_failed": "Export failed: {err}",
    },
}
# ==================== Main App ====================
class UltimateLabeller:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.lang = "en"
        self.theme = "dark"
        self.config = AppConfig()
        self.state = AppState()
        self.history_manager = HistoryManager()
        self.logger = setup_logging(os.path.join(os.path.expanduser("~"), ".ai_labeller.log"))

        self.root.title(LANG_MAP[self.lang]["title"])
        self.root.geometry(self.config.default_window_size)
        self.root.minsize(self.config.min_window_width, self.config.min_window_height)
        self.window_icon_tk = None
        self.toolbar_logo_tk = None
        
        # ???????????????
        self.setup_fonts()
        self.apply_theme(self.theme, rebuild=False)
        self.setup_app_icon()
        self._tooltip_after_id = None
        self._tooltip_win = None
        
        # --- ????? ---
        self.project_root = self.state.project_root
        self.current_split = self.state.current_split
        self.image_files = self.state.image_files
        self.current_idx = self.state.current_idx
        self.rects = self.state.rects  # [x1, y1, x2, y2, cid, angle_deg?]
        self.class_names = self.state.class_names
        self.learning_mem = deque(maxlen=self.config.max_learning_memory)
        
        # --- ??????---
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.img_pil = None
        self.img_tk = None
        self.selected_idx = None
        self.selected_indices: set[int] = set()
        self.active_handle = None
        self.active_rotate_handle = False
        self.rotate_drag_offset_deg = 0.0
        self.is_moving_box = False
        self.is_drag_selecting = False
        self.drag_start = None
        self.temp_rect_coords = None
        self.select_rect_coords = None
        self.mouse_pos = (0, 0)
        self.HANDLE_SIZE = self.config.handle_size
        self.show_all_labels = True
        self._cursor_line_x: int | None = None
        self._cursor_line_y: int | None = None
        self._cursor_text_id: int | None = None
        self._cursor_bg_id: int | None = None
        
        # --- AI ? ---
        self.yolo_model = None
        self.yolo_path = tk.StringVar(value=self.config.yolo_model_path)
        self.det_model_mode = tk.StringVar(value="Official YOLO26m.pt (Bundled)")
        self._loaded_model_key: tuple[str, str] | None = None
        self._force_cpu_detection = False
        self.model_library: list[str] = [self.config.yolo_model_path]
        self.var_export_format = tk.StringVar(value="YOLO (.txt)")
        self.var_auto_yolo = tk.BooleanVar(value=False)
        self.var_propagate = tk.BooleanVar(value=False)
        self.var_propagate_mode = tk.StringVar(value="if_missing")
        self.var_yolo_conf = tk.DoubleVar(value=self.config.default_yolo_conf)
        self.session_path = os.path.join(os.path.expanduser("~"), self.config.session_file_name)
        self.foundation_dino = None
        self.foundation_sam_predictor = None
        self._uncertainty_cache: dict[str, float] = {}
        self._active_scan_offset = 0
        self._folder_dialog_open = False
        self._startup_dialog_shown = False
        self._startup_dialog_open = False
        self.training_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.training_thread: threading.Thread | None = None
        self.training_process: subprocess.Popen[str] | None = None
        self.training_running = False
        self.training_start_time: float | None = None
        self.training_total_epochs = 0
        self.training_current_epoch = 0
        self.train_command_var = tk.StringVar(value="")
        
        self.setup_custom_style()
        self.setup_ui()
        self.bind_events()
        self.root.protocol("WM_DELETE_WINDOW", self.on_app_close)
        self.load_session_state()
        self.root.after(120, self.show_startup_source_dialog)
    
    def setup_fonts(self):
        """Configure platform-specific font families and sizes."""
        import platform
        system = platform.system()
        
        if system == "Windows":
            self.font_primary = ("Segoe UI", 10)
            self.font_bold = ("Segoe UI", 10, "bold")
            self.font_title = ("Segoe UI", 14, "bold")
            self.font_mono = ("Consolas", 9)
        elif system == "Darwin":  # macOS
            self.font_primary = ("SF Pro Text", 10)
            self.font_bold = ("SF Pro Text", 10, "bold")
            self.font_title = ("SF Pro Display", 14, "bold")
            self.font_mono = ("SF Mono", 9)
        else:  # Linux
            self.font_primary = ("Ubuntu", 10)
            self.font_bold = ("Ubuntu", 10, "bold")
            self.font_title = ("Ubuntu", 14, "bold")
            self.font_mono = ("Ubuntu Mono", 9)
    
    def setup_custom_style(self):
        """?????? ttk ???"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # ?????? Combobox
        style.configure("TCombobox",
            fieldbackground=COLORS["bg_white"],
            background=COLORS["bg_light"],
            foreground=COLORS["text_primary"],
            borderwidth=0,
            relief="flat",
            arrowcolor=COLORS["text_secondary"]
        )
        
        style.map("TCombobox",
            fieldbackground=[('readonly', COLORS["bg_white"])],
            selectbackground=[('readonly', COLORS["primary_bg"])],
            selectforeground=[('readonly', COLORS["primary"])]
        )

    def _resolve_asset_path(self, relative_path: str) -> str | None:
        try:
            packaged = resources.files("ai_labeller").joinpath(relative_path)
            if packaged.is_file():
                return str(packaged)
        except Exception:
            pass
        local = os.path.join(os.path.dirname(__file__), relative_path)
        if os.path.isfile(local):
            return local
        return None

    def setup_app_icon(self) -> None:
        icon_path = self._resolve_asset_path("assets/app_icon.png")
        if not icon_path:
            return
        try:
            icon_img = Image.open(icon_path).convert("RGBA")
            win_icon = icon_img.resize((32, 32), Image.Resampling.LANCZOS)
            toolbar_icon = icon_img.resize((20, 20), Image.Resampling.LANCZOS)
            self.window_icon_tk = ImageTk.PhotoImage(win_icon)
            self.toolbar_logo_tk = ImageTk.PhotoImage(toolbar_icon)
            self.root.iconphoto(True, self.window_icon_tk)
            icon_img.close()
        except Exception:
            self.logger.exception("Failed to load app icon")
    
    def delete_selected(self, e=None):
        """?????????????"""
        focus_widget = self.root.focus_get()
        if focus_widget is not None:
            try:
                if focus_widget.winfo_toplevel() is not self.root:
                    return "break"
            except Exception:
                pass
        selected = self._get_selected_indices()
        if not selected:
            return "break"
        self.push_history()
        for idx in sorted(selected, reverse=True):
            self.rects.pop(idx)
        self._set_selected_indices([])
        self.render()
        return "break"

    def select_all_boxes(self, e=None):
        if not self.rects:
            return "break"
        all_indices = list(range(len(self.rects)))
        self._set_selected_indices(all_indices, primary_idx=all_indices[-1])
        self._sync_class_combo_with_selection()
        self.render()
        return "break"

    def _get_selected_indices(self) -> list[int]:
        valid = [idx for idx in self.selected_indices if 0 <= idx < len(self.rects)]
        if self.selected_idx is not None and 0 <= self.selected_idx < len(self.rects) and self.selected_idx not in valid:
            valid.append(self.selected_idx)
        valid.sort()
        return valid

    def _set_selected_indices(self, indices: list[int], primary_idx: int | None = None) -> None:
        valid = sorted({idx for idx in indices if 0 <= idx < len(self.rects)})
        self.selected_indices = set(valid)
        if not valid:
            self.selected_idx = None
            return

        if primary_idx is not None and primary_idx in self.selected_indices:
            self.selected_idx = primary_idx
        elif self.selected_idx in self.selected_indices:
            pass
        else:
            self.selected_idx = valid[-1]

    def _sync_class_combo_with_selection(self) -> None:
        selected = self._get_selected_indices()
        if not selected:
            return
        class_ids = {self.rects[idx][4] for idx in selected}
        if len(class_ids) != 1:
            return
        only_cid = int(next(iter(class_ids)))
        if 0 <= only_cid < len(self.class_names):
            self.combo_cls.current(only_cid)

    def _pick_box_at_point(self, ix: float, iy: float) -> int | None:
        candidates: list[tuple[float, int]] = []
        for idx, rect in enumerate(self.rects):
            if self._point_in_rotated_box(ix, iy, rect):
                x1 = min(rect[0], rect[2])
                y1 = min(rect[1], rect[3])
                x2 = max(rect[0], rect[2])
                y2 = max(rect[1], rect[3])
                area = max(1.0, (x2 - x1) * (y2 - y1))
                candidates.append((area, idx))
        if not candidates:
            return None
        # For nested/overlapping boxes, prioritize smaller area so inner box is easier to adjust.
        candidates.sort(key=lambda item: (item[0], -item[1]))
        return candidates[0][1]

    def _pick_boxes_in_img_rect(self, ix1: float, iy1: float, ix2: float, iy2: float) -> list[int]:
        sx1, sx2 = sorted((ix1, ix2))
        sy1, sy2 = sorted((iy1, iy2))
        hits: list[int] = []
        for idx, rect in enumerate(self.rects):
            corners = self.get_rotated_corners(rect)
            rx1 = min(px for px, _ in corners)
            ry1 = min(py for _, py in corners)
            rx2 = max(px for px, _ in corners)
            ry2 = max(py for _, py in corners)
            intersects = not (rx2 < sx1 or rx1 > sx2 or ry2 < sy1 or ry1 > sy2)
            if intersects:
                hits.append(idx)
        return hits
    
    def setup_ui(self):
        # ==================== ??????====================
        self.setup_toolbar()
        
        # ==================== ?????====================
        self.setup_sidebar()
        
        # ==================== Canvas ====================
        self.canvas = tk.Canvas(
            self.root,
            bg=COLORS["bg_canvas"],
            cursor="none",
            highlightthickness=0,
            relief="flat"
        )
        self.canvas.pack(fill="both", expand=True)
    
    def setup_toolbar(self):
        """????????????"""
        toolbar = tk.Frame(self.root, bg=COLORS["bg_dark"], height=56)
        toolbar.pack(side="top", fill="x")
        toolbar.pack_propagate(False)
        
        # ???????
        left_frame = tk.Frame(toolbar, bg=COLORS["bg_dark"])
        left_frame.pack(side="left", fill="y", padx=16)
        
        # Logo ????
        title_frame = tk.Frame(left_frame, bg=COLORS["bg_dark"])
        title_frame.pack(side="left", pady=12)
        
        if self.toolbar_logo_tk is not None:
            logo = tk.Label(
                title_frame,
                image=self.toolbar_logo_tk,
                bg=COLORS["bg_dark"]
            )
        else:
            logo = tk.Label(
                title_frame,
                text="AI",
                font=("Arial", 20),
                fg=COLORS["primary"],
                bg=COLORS["bg_dark"]
            )
        logo.pack(side="left", padx=(0, 8))
        logo.bind("<Button-1>", self.return_to_source_select)
        logo.bind("<Enter>", lambda _e: logo.config(cursor="hand2"))
        
        tk.Label(
            title_frame,
            text=LANG_MAP[self.lang]["title"],
            font=self.font_title,
            fg=self.toolbar_text_color(COLORS["bg_dark"]),
            bg=COLORS["bg_dark"]
        ).pack(side="left")
        
        # Draw a new box?
        tk.Frame(
            left_frame,
            width=1,
            bg=COLORS["divider"]
        ).pack(side="left", fill="y", padx=16)
        
        # ??????
        self.create_toolbar_button(
            left_frame,
            text=LANG_MAP[self.lang]["load_proj"],
            command=lambda: self.show_startup_source_dialog(force=True),
            bg=COLORS["primary"]
        ).pack(side="left", padx=4)
        
        # Dataset ?????
        dataset_frame = tk.Frame(left_frame, bg=COLORS["bg_dark"])
        dataset_frame.pack(side="left", padx=12)
        
        tk.Label(
            dataset_frame,
            text=LANG_MAP[self.lang]["dataset"],
            font=self.font_primary,
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_dark"]
        ).pack(side="left", padx=(0, 8))
        
        self.combo_split = ttk.Combobox(
            dataset_frame,
            values=["train", "val", "test"],
            width=10,
            state="readonly",
            font=self.font_primary
        )
        self.combo_split.current(0)
        self.combo_split.pack(side="left")
        self.combo_split.bind("<<ComboboxSelected>>", self.on_split_change)
        
        # ????????- ??????????
        center_frame = tk.Frame(toolbar, bg=COLORS["bg_dark"])
        center_frame.pack(side="left", fill="y", padx=16)
        
        # Draw a new box??
        self.create_toolbar_icon_button(
            center_frame,
            text="U",
            command=self.undo,
            tooltip=LANG_MAP[self.lang]["undo"]
        ).pack(side="left", padx=2)
        
        self.create_toolbar_icon_button(
            center_frame,
            text="R",
            command=self.redo,
            tooltip=LANG_MAP[self.lang]["redo"]
        ).pack(side="left", padx=2)
        
        tk.Frame(center_frame, width=1, bg=COLORS["divider"]).pack(
            side="left", fill="y", padx=8
        )
        
        # AI ???
        self.create_toolbar_icon_button(
            center_frame,
            text="?",
            command=self.autolabel_red,
            tooltip=LANG_MAP[self.lang]["autolabel"],
            bg=COLORS["danger"]
        ).pack(side="left", padx=2)
        
        # Draw a new box???
        right_frame = tk.Frame(toolbar, bg=COLORS["bg_dark"])
        right_frame.pack(side="right", fill="y", padx=16)

        # ????????????????????? 1 ????????????
        self.create_help_icon(right_frame).pack(side="right", padx=4, pady=12)

        # ??????
        self.create_toolbar_button(
            right_frame,
            text=self.get_theme_switch_label(),
            command=self.toggle_theme,
            bg=COLORS["bg_medium"]
        ).pack(side="right", padx=4, pady=12)

        # Draw a new box?
        self.create_toolbar_button(
            right_frame,
            text=LANG_MAP[self.lang]["lang_switch"],
            command=self.toggle_language,
            bg=COLORS["bg_medium"]
        ).pack(side="right", padx=4, pady=12)
    
    def create_toolbar_button(self, parent, text, command, bg=None):
        """??????????????"""
        bg_val = bg or COLORS["bg_medium"]
        fg_val = self.toolbar_text_color(bg_val)
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg_val,
            fg=fg_val,
            font=self.font_primary,
            relief="flat",
            padx=16,
            pady=8,
            cursor="hand2",
            borderwidth=0,
            highlightthickness=0
        )
        
        # ?????
        def on_enter(e):
            btn.config(bg=COLORS["primary_hover"] if bg == COLORS["primary"] else COLORS["bg_medium"])
        
        def on_leave(e):
            btn.config(bg=bg_val)
        
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        
        return btn

    def create_help_icon(self, parent):
        """?????????????????"""
        btn = tk.Label(
            parent,
            text="?",
            font=self.font_bold,
            fg=self.toolbar_text_color(COLORS["bg_medium"]),
            bg=COLORS["bg_medium"],
            width=3,
            height=1,
            relief="flat"
        )
        btn.bind("<Enter>", lambda e: self.show_shortcut_tooltip(btn))
        btn.bind("<Leave>", lambda e: self.hide_shortcut_tooltip())
        return btn

    def build_shortcut_text(self):
        items = [
            ("F", LANG_MAP[self.lang]["next"]),
            ("D", LANG_MAP[self.lang]["prev"]),
            ("A", LANG_MAP[self.lang]["autolabel"]),
            ("Q / E", "Rotate selected box"),
            ("Ctrl+Z", LANG_MAP[self.lang]["undo"]),
            ("Ctrl+Y", LANG_MAP[self.lang]["redo"]),
            ("Del", LANG_MAP[self.lang]["delete"]),
        ]
        lines = [LANG_MAP[self.lang]["shortcut_help"]]
        for key, desc in items:
            lines.append(f"{key} - {desc}")
        return "\n".join(lines)

    def show_shortcut_tooltip(self, widget):
        self.hide_shortcut_tooltip()
        self._tooltip_after_id = self.root.after(1000, lambda: self._show_tooltip_now(widget))

    def _show_tooltip_now(self, widget):
        if self._tooltip_win:
            return
        x = widget.winfo_rootx() + 10
        y = widget.winfo_rooty() + widget.winfo_height() + 6
        win = tk.Toplevel(self.root)
        win.wm_overrideredirect(True)
        win.configure(bg=COLORS["bg_white"])
        label = tk.Label(
            win,
            text=self.build_shortcut_text(),
            font=self.font_primary,
            fg=COLORS["text_primary"],
            bg=COLORS["bg_white"],
            justify="left",
            anchor="w",
            padx=10,
            pady=8
        )
        label.pack()
        win.update_idletasks()

        tooltip_w = win.winfo_reqwidth()
        tooltip_h = win.winfo_reqheight()
        left, top, right, bottom = self._get_widget_monitor_bounds(widget)
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
        self._tooltip_win = win

    def _get_widget_monitor_bounds(self, widget):
        # Fallback for non-Windows or API failure: virtual desktop bounds.
        left = widget.winfo_vrootx()
        top = widget.winfo_vrooty()
        right = left + widget.winfo_vrootwidth()
        bottom = top + widget.winfo_vrootheight()

        try:
            user32 = ctypes.windll.user32
            monitor = user32.MonitorFromPoint(
                ctypes.wintypes.POINT(widget.winfo_rootx(), widget.winfo_rooty()),
                2
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

    def hide_shortcut_tooltip(self):
        if self._tooltip_after_id:
            self.root.after_cancel(self._tooltip_after_id)
            self._tooltip_after_id = None
        if self._tooltip_win:
            self._tooltip_win.destroy()
            self._tooltip_win = None
    
    def create_toolbar_icon_button(self, parent, text, command, tooltip="", bg=None):
        """????????????????"""
        bg_val = bg or COLORS["bg_medium"]
        fg_val = self.toolbar_text_color(bg_val)
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg_val,
            fg=fg_val,
            font=("Arial", 14),
            relief="flat",
            width=3,
            height=1,
            cursor="hand2",
            borderwidth=0,
            highlightthickness=0
        )
        
        # ?????
        def on_enter(e):
            if bg:
                btn.config(bg=self.lighten_color(bg))
            else:
                btn.config(bg=COLORS["divider"])
        
        def on_leave(e):
            btn.config(bg=bg_val)
        
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        
        return btn
    
    def lighten_color(self, color):
        """??????????"""
        if color == COLORS["danger"]:
            return "#FF6B54"
        elif color == COLORS["warning"]:
            return "#FFBB33"
        return color
    
    def setup_sidebar(self):
        """???????????"""
        sidebar = tk.Frame(self.root, width=320, bg=COLORS["bg_light"])
        sidebar.pack(side="right", fill="y")
        sidebar.pack_propagate(False)
        
        # ?????????
        self.sidebar_canvas = tk.Canvas(
            sidebar,
            bg=COLORS["bg_light"],
            highlightthickness=0,
            relief="flat"
        )
        self.sidebar_scrollbar = ttk.Scrollbar(
            sidebar,
            orient="vertical",
            command=self.sidebar_canvas.yview
        )
        self.sidebar_scroll_frame = tk.Frame(self.sidebar_canvas, bg=COLORS["bg_light"])
        
        self.sidebar_window = self.sidebar_canvas.create_window(
            (0, 0),
            window=self.sidebar_scroll_frame,
            anchor="nw"
        )
        self.sidebar_canvas.configure(yscrollcommand=self.sidebar_scrollbar.set)
        self.sidebar_scroll_frame.bind("<Configure>", self._on_sidebar_frame_configure)
        self.sidebar_canvas.bind("<Configure>", self._on_sidebar_canvas_configure)
        self.sidebar_canvas.bind("<MouseWheel>", self._on_sidebar_mousewheel)
        self.sidebar_canvas.bind("<Button-4>", lambda e: self.sidebar_canvas.yview_scroll(-1, "units"))
        self.sidebar_canvas.bind("<Button-5>", lambda e: self.sidebar_canvas.yview_scroll(1, "units"))
        
        # ===== ??????? =====
        self.create_info_card(self.sidebar_scroll_frame)
        
        # ===== ???????? =====
        self.create_class_card(self.sidebar_scroll_frame)
        
        # ===== AI ?????? =====
        self.create_ai_card(self.sidebar_scroll_frame)
        
        # ===== ????????=====
        self.create_shortcut_card(self.sidebar_scroll_frame)
        
        # ===== ??? =====
        self.create_navigation(sidebar)
        self._bind_sidebar_mousewheel(self.sidebar_scroll_frame)
        
        self.sidebar_canvas.pack(side="left", fill="both", expand=True)
        self.sidebar_scrollbar.pack(side="right", fill="y")

    def _on_sidebar_frame_configure(self, e=None):
        if hasattr(self, "sidebar_canvas"):
            self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))

    def _on_sidebar_canvas_configure(self, e):
        if hasattr(self, "sidebar_canvas") and hasattr(self, "sidebar_window"):
            self.sidebar_canvas.itemconfigure(self.sidebar_window, width=e.width)

    def _on_sidebar_mousewheel(self, e):
        if e.delta:
            self.sidebar_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        return "break"

    def _bind_sidebar_mousewheel(self, widget):
        widget.bind("<MouseWheel>", self._on_sidebar_mousewheel, add="+")
        widget.bind("<Button-4>", lambda e: self.sidebar_canvas.yview_scroll(-1, "units"), add="+")
        widget.bind("<Button-5>", lambda e: self.sidebar_canvas.yview_scroll(1, "units"), add="+")
        for child in widget.winfo_children():
            self._bind_sidebar_mousewheel(child)

    def get_theme_switch_label(self):
        key = "theme_light" if self.theme == "dark" else "theme_dark"
        return LANG_MAP[self.lang][key]

    def is_accent_bg(self, bg):
        return bg in {
            COLORS["primary"],
            COLORS["warning"],
            COLORS["danger"],
            COLORS["success"],
            COLORS["info"],
            COLORS["primary_light"],
        }

    def toolbar_text_color(self, bg):
        if self.is_accent_bg(bg):
            return COLORS["text_white"]
        return COLORS["text_white"] if self.theme == "dark" else COLORS["text_primary"]

    def apply_theme(self, theme, rebuild=True):
        self.theme = theme
        palette = THEMES.get(theme, THEMES["dark"])
        for k, v in palette.items():
            COLORS[k] = v
        self.root.configure(bg=COLORS["bg_dark"])
        if rebuild:
            self.rebuild_ui()

    def toggle_theme(self):
        new_theme = "light" if self.theme == "dark" else "dark"
        self.apply_theme(new_theme)

    def rebuild_ui(self):
        self.hide_shortcut_tooltip()
        for child in self.root.winfo_children():
            child.destroy()
        self.setup_custom_style()
        self.setup_ui()
        self.bind_events()
        self.root.title(LANG_MAP[self.lang]["title"])
        self.update_info_text()
        self.render()
    
    def create_card(self, parent, title=None):
        """Create a styled card container and return its content frame."""
        card = tk.Frame(
            parent,
            bg=COLORS["bg_white"],
            relief="flat",
            borderwidth=0
        )
        card.pack(fill="x", padx=16, pady=8)
        
        # ?????????????????????????????
        card_border = tk.Frame(card, bg=COLORS["border"], height=1)
        card_border.pack(fill="x", side="bottom")
        
        content = tk.Frame(card, bg=COLORS["bg_white"])
        content.pack(fill="both", expand=True, padx=16, pady=16)
        
        if title:
            title_label = tk.Label(
                content,
                text=title,
                font=self.font_bold,
                fg=COLORS["text_primary"],
                bg=COLORS["bg_white"],
                anchor="w"
            )
            title_label.pack(fill="x", pady=(0, 12))
        
        return content
    
    def create_info_card(self, parent):
        """??????????????"""
        content = self.create_card(parent, LANG_MAP[self.lang]["file_info"])
        
        # Draw a new box??
        self.lbl_filename = tk.Label(
            content,
            text=LANG_MAP[self.lang]["no_img"],
            font=self.font_mono,
            fg=COLORS["text_primary"],
            bg=COLORS["bg_white"],
            anchor="w",
            wraplength=260
        )
        self.lbl_filename.pack(fill="x")

        tk.Label(
            content,
            text=LANG_MAP[self.lang].get("select_image", "Select Image"),
            font=self.font_primary,
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_white"],
            anchor="w"
        ).pack(fill="x", pady=(10, 6))

        self.combo_image = ttk.Combobox(
            content,
            values=[],
            state="readonly",
            font=self.font_primary
        )
        self.combo_image.pack(fill="x")
        self.combo_image.bind("<<ComboboxSelected>>", self.on_image_selected)
        
        # Draw a new box??????
        progress_frame = tk.Frame(content, bg=COLORS["bg_white"])
        progress_frame.pack(fill="x", pady=(8, 0))
        
        # Draw a new box???
        self.lbl_progress = tk.Label(
            progress_frame,
            text="0 / 0",
            font=self.font_primary,
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_white"]
        )
        self.lbl_progress.pack(side="left")
        
        # ??????
        self.lbl_box_count = tk.Label(
            progress_frame,
            text=f"{LANG_MAP[self.lang]['boxes']}: 0",
            font=self.font_primary,
            fg=COLORS["primary"],
            bg=COLORS["bg_white"]
        )
        self.lbl_box_count.pack(side="right")
    
    def create_class_card(self, parent):
        """???????????????"""
        content = self.create_card(parent, LANG_MAP[self.lang]["class_mgmt"])
        
        # Draw a new box?????
        tk.Label(
            content,
            text=LANG_MAP[self.lang]["current_class"],
            font=self.font_primary,
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_white"],
            anchor="w"
        ).pack(fill="x", pady=(0, 8))
        
        # ??????
        self.combo_cls = ttk.Combobox(
            content,
            values=self.class_names,
            state="readonly",
            font=self.font_primary
        )
        self.combo_cls.current(0)
        self.combo_cls.pack(fill="x", pady=(0, 12))
        self.combo_cls.bind("<<ComboboxSelected>>", self.on_class_change_request)
        
        # ????????????
        self.create_primary_button(
            content,
            text=LANG_MAP[self.lang]["edit_classes"],
            command=self.edit_classes_table
        ).pack(fill="x", pady=(0, 12))

        # Draw a new box?????
        self.create_secondary_button(
            content,
            text=LANG_MAP[self.lang]["clear_labels"],
            command=self.clear_current_labels
        ).pack(fill="x", pady=(0, 12))

        self.create_secondary_button(
            content,
            text=LANG_MAP[self.lang].get("remove_from_split", "Remove From Split"),
            command=self.remove_current_from_split
        ).pack(fill="x", pady=(0, 12))

        self.create_secondary_button(
            content,
            text=LANG_MAP[self.lang].get("restore_from_split", "Restore Deleted Frame"),
            command=self.open_restore_removed_dialog
        ).pack(fill="x", pady=(0, 12))
        
        # Export format
        tk.Label(
            content,
            text=LANG_MAP[self.lang]["export_format"],
            font=self.font_primary,
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_white"],
            anchor="w"
        ).pack(fill="x", pady=(0, 8))
        
        ttk.Combobox(
            content,
            textvariable=self.var_export_format,
            values=["YOLO (.txt)", "JSON"],
            state="readonly",
            font=self.font_primary
        ).pack(fill="x")

        self.create_primary_button(
            content,
            text=LANG_MAP[self.lang]["export"],
            command=self.export_all_by_selected_format,
            bg=COLORS["info"]
        ).pack(fill="x", pady=(10, 0))
    
    def create_ai_card(self, parent):
        """??????? AI ??????"""
        content = self.create_card(parent, LANG_MAP[self.lang]["ai_tools"])
        
        # ?????????
        checkbox_style = {
            "bg": COLORS["bg_white"],
            "fg": COLORS["text_primary"],
            "font": self.font_primary,
            "activebackground": COLORS["bg_white"],
            "selectcolor": COLORS["bg_white"],
            "anchor": "w"
        }
        
        tk.Checkbutton(
            content,
            text=LANG_MAP[self.lang]["auto_detect"],
            variable=self.var_auto_yolo,
            **checkbox_style
        ).pack(fill="x", pady=4)
        
        propagate_row = tk.Frame(content, bg=COLORS["bg_white"])
        propagate_row.pack(fill="x", pady=4)
        tk.Checkbutton(
            propagate_row,
            text=LANG_MAP[self.lang]["propagate"],
            variable=self.var_propagate,
            **checkbox_style
        ).pack(side="left")
        self.combo_propagate_mode = ttk.Combobox(
            propagate_row,
            state="readonly",
            width=18,
            font=self.font_primary,
        )
        self.combo_propagate_mode.pack(side="right")
        self.combo_propagate_mode.bind("<<ComboboxSelected>>", self.on_propagate_mode_changed)
        self._refresh_propagate_mode_combo()

        tk.Label(
            content,
            text=LANG_MAP[self.lang].get("detection_model", "Detection Model"),
            font=self.font_primary,
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_white"],
            anchor="w"
        ).pack(fill="x", pady=(10, 4))

        self.combo_det_model = ttk.Combobox(
            content,
            textvariable=self.det_model_mode,
            values=[
                "Official YOLO26m.pt (Bundled)",
                "Custom YOLO (v5/v7/v8/v9/v11/v26)",
                "Custom RF-DETR",
            ],
            state="readonly",
            font=self.font_primary
        )
        self.combo_det_model.pack(fill="x", pady=(0, 6))
        self.combo_det_model.bind("<<ComboboxSelected>>", self.on_detection_model_mode_changed)

        self.combo_model_path = ttk.Combobox(
            content,
            textvariable=self.yolo_path,
            values=self.model_library,
            state="readonly",
            font=self.font_primary
        )
        self.combo_model_path.pack(fill="x", pady=(0, 6))
        self._refresh_model_dropdown()

        picker_row = tk.Frame(content, bg=COLORS["bg_white"])
        picker_row.pack(fill="x", pady=(0, 6))

        self.create_secondary_button(
            picker_row,
            text=LANG_MAP[self.lang].get("browse_model", "Browse Model"),
            command=self.browse_detection_model
        ).pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.create_secondary_button(
            picker_row,
            text=LANG_MAP[self.lang].get("use_official_yolo26n", "Use Official yolo26m.pt"),
            command=self.use_official_yolo26n
        ).pack(side="left", fill="x", expand=True, padx=(4, 0))
        
        # Detection trigger
        self.create_primary_button(
            content,
            text=LANG_MAP[self.lang]["run_detection"],
            command=self.run_yolo_detection,
            bg=COLORS["success"]
        ).pack(fill="x", pady=(12, 0))

        self.create_primary_button(
            content,
            text=LANG_MAP[self.lang].get("train_from_labels", "Train From Labels"),
            command=self.start_training_from_labels,
            bg=COLORS["warning"],
        ).pack(fill="x", pady=(8, 0))

        tk.Label(
            content,
            text=LANG_MAP[self.lang].get("train_monitor", "Training Monitor"),
            font=self.font_bold,
            fg=COLORS["text_primary"],
            bg=COLORS["bg_white"],
            anchor="w",
        ).pack(fill="x", pady=(12, 6))

        self.lbl_train_status = tk.Label(
            content,
            text=f"{LANG_MAP[self.lang].get('train_status', 'Status')}: {LANG_MAP[self.lang].get('train_idle', 'Idle')}",
            font=self.font_primary,
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_white"],
            anchor="w",
        )
        self.lbl_train_status.pack(fill="x")

        self.lbl_train_progress = tk.Label(
            content,
            text=f"{LANG_MAP[self.lang].get('train_progress', 'Progress')}: -",
            font=self.font_primary,
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_white"],
            anchor="w",
        )
        self.lbl_train_progress.pack(fill="x")

        self.lbl_train_eta = tk.Label(
            content,
            text=f"{LANG_MAP[self.lang].get('train_eta', 'ETA')}: -",
            font=self.font_primary,
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_white"],
            anchor="w",
        )
        self.lbl_train_eta.pack(fill="x", pady=(0, 4))

        tk.Label(
            content,
            text=LANG_MAP[self.lang].get("train_command", "Command"),
            font=self.font_primary,
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_white"],
            anchor="w",
        ).pack(fill="x")

        self.entry_train_cmd = tk.Entry(
            content,
            textvariable=self.train_command_var,
            font=self.font_mono,
            state="readonly",
            readonlybackground=COLORS["bg_light"],
            fg=COLORS["text_primary"],
        )
        self.entry_train_cmd.pack(fill="x", pady=(0, 6))

        log_wrap = tk.Frame(content, bg=COLORS["bg_white"])
        log_wrap.pack(fill="both", expand=True)
        self.txt_train_log = tk.Text(
            log_wrap,
            height=8,
            wrap="word",
            font=self.font_mono,
            bg=COLORS["bg_light"],
            fg=COLORS["text_primary"],
            relief="flat",
        )
        self.txt_train_log.pack(side="left", fill="both", expand=True)
        sb_log = ttk.Scrollbar(log_wrap, orient="vertical", command=self.txt_train_log.yview)
        sb_log.pack(side="right", fill="y")
        self.txt_train_log.configure(yscrollcommand=sb_log.set)
    
    def create_shortcut_card(self, parent):
        """Render shortcut hint list."""
        content = self.create_card(parent, LANG_MAP[self.lang]["shortcuts"])
        
        shortcuts = [
            ("F", LANG_MAP[self.lang]["next"]),
            ("D", LANG_MAP[self.lang]["prev"]),
            ("A", LANG_MAP[self.lang]["autolabel"]),
            ("Q / E", "Rotate selected box"),
            ("Ctrl+Z", LANG_MAP[self.lang]["undo"]),
            ("Ctrl+Y", LANG_MAP[self.lang]["redo"]),
            ("Del", LANG_MAP[self.lang]["delete"])
        ]
        
        for key, desc in shortcuts:
            row = tk.Frame(content, bg=COLORS["bg_white"])
            row.pack(fill="x", pady=2)
            
            tk.Label(
                row,
                text=key,
                font=self.font_mono,
                fg=COLORS["primary"],
                bg=COLORS["bg_white"],
                width=12,
                anchor="w"
            ).pack(side="left")
            
            tk.Label(
                row,
                text=desc,
                font=self.font_primary,
                fg=COLORS["text_secondary"],
                bg=COLORS["bg_white"],
                anchor="w"
            ).pack(side="left")
    
    def create_navigation(self, parent):
        """??????????"""
        nav_frame = tk.Frame(parent, bg=COLORS["bg_light"], height=80)
        nav_frame.pack(side="bottom", fill="x")
        nav_frame.pack_propagate(False)
        
        btn_container = tk.Frame(nav_frame, bg=COLORS["bg_light"])
        btn_container.pack(fill="both", expand=True, padx=16, pady=16)
        
        # Draw a new box?
        self.create_nav_button(
            btn_container,
            text=LANG_MAP[self.lang]["prev"],
            command=self.prev_img,
            side="left"
        )
        
        # Draw a new box?
        self.create_nav_button(
            btn_container,
            text=f"{LANG_MAP[self.lang]['next']} >",
            command=self.save_and_next,
            side="right",
            primary=True
        )
    
    def create_primary_button(self, parent, text, command, bg=None):
        """?????????????"""
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg or COLORS["primary"],
            fg=COLORS["text_white"],
            font=self.font_primary,
            relief="flat",
            pady=10,
            cursor="hand2",
            borderwidth=0,
            highlightthickness=0
        )
        
        # ?????
        original_bg = bg or COLORS["primary"]
        hover_bg = COLORS["primary_hover"] if not bg else self.lighten_color(bg)
        
        btn.bind("<Enter>", lambda e: btn.config(bg=hover_bg))
        btn.bind("<Leave>", lambda e: btn.config(bg=original_bg))
        
        return btn

    def create_secondary_button(self, parent, text, command):
        """Create a secondary sidebar button."""
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=COLORS["bg_white"],
            fg=COLORS["text_primary"],
            font=self.font_primary,
            relief="flat",
            pady=10,
            cursor="hand2",
            borderwidth=1,
            highlightthickness=0
        )
        btn.config(highlightbackground=COLORS["border"], highlightcolor=COLORS["border"])
        btn.bind("<Enter>", lambda e: btn.config(bg=COLORS["bg_light"]))
        btn.bind("<Leave>", lambda e: btn.config(bg=COLORS["bg_white"]))
        return btn
    
    def create_nav_button(self, parent, text, command, side, primary=False):
        """??????????"""
        bg = COLORS["primary"] if primary else COLORS["bg_white"]
        fg = COLORS["text_white"] if primary else COLORS["text_primary"]
        
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            font=self.font_bold,
            relief="flat",
            pady=12,
            cursor="hand2",
            borderwidth=0 if primary else 1,
            highlightthickness=0
        )
        
        if not primary:
            btn.config(highlightbackground=COLORS["border"], highlightcolor=COLORS["border"])
        
        # ?????
        if primary:
            btn.bind("<Enter>", lambda e: btn.config(bg=COLORS["primary_hover"]))
            btn.bind("<Leave>", lambda e: btn.config(bg=COLORS["primary"]))
        else:
            btn.bind("<Enter>", lambda e: btn.config(bg=COLORS["bg_light"]))
            btn.bind("<Leave>", lambda e: btn.config(bg=COLORS["bg_white"]))
        
        if side == "left":
            btn.pack(side="left", fill="both", expand=True, padx=(0, 4))
        else:
            btn.pack(side="right", fill="both", expand=True, padx=(4, 0))
    
    def toggle_language(self):
        """????"""
        self.lang = "en" if self.lang == "zh" else "zh"
        self.rebuild_ui()
    
    def update_info_text(self):
        """????????"""
        if not self.image_files:
            self.lbl_filename.config(text=LANG_MAP[self.lang]["no_img"])
            self.lbl_progress.config(text="0 / 0")
            if hasattr(self, "combo_image"):
                self.combo_image.configure(values=[])
                self.combo_image.set("")
        else:
            filename = os.path.basename(self.image_files[self.current_idx])
            self.lbl_filename.config(text=filename)
            self.lbl_progress.config(
                text=f"{self.current_idx + 1} / {len(self.image_files)}"
            )
            self.refresh_image_dropdown()
        
        self.lbl_box_count.config(
            text=f"{LANG_MAP[self.lang]['boxes']}: {len(self.rects)}"
        )

    def refresh_image_dropdown(self):
        if not hasattr(self, "combo_image"):
            return
        names = [os.path.basename(p) for p in self.image_files]
        self.combo_image.configure(values=names)
        if not names:
            self.combo_image.set("")
            return
        if 0 <= self.current_idx < len(names):
            self.combo_image.set(names[self.current_idx])

    def on_image_selected(self, e: Any = None) -> None:
        if not self.image_files:
            return
        idx = self.combo_image.current()
        if idx < 0 or idx == self.current_idx:
            return
        self.save_current()
        self.current_idx = idx
        self.load_img()

    def _register_model_path(self, model_path: str) -> None:
        path = model_path.strip()
        if not path:
            return
        if path not in self.model_library:
            self.model_library.append(path)
        self._refresh_model_dropdown()

    def _refresh_model_dropdown(self) -> None:
        if hasattr(self, "combo_model_path"):
            self.combo_model_path.configure(values=self.model_library)

    def _propagate_mode_choices(self) -> list[tuple[str, str]]:
        L = LANG_MAP[self.lang]
        return [
            ("if_missing", L.get("propagate_mode_if_missing", "No label only")),
            ("always", L.get("propagate_mode_always", "Always (overwrite existing)")),
            ("selected", L.get("propagate_mode_selected", "Selected labels only")),
        ]

    def _refresh_propagate_mode_combo(self) -> None:
        if not hasattr(self, "combo_propagate_mode"):
            return
        choices = self._propagate_mode_choices()
        self._propagate_label_to_code = {label: code for code, label in choices}
        self._propagate_code_to_label = {code: label for code, label in choices}
        self.combo_propagate_mode.configure(values=[label for _, label in choices])
        current_code = self.var_propagate_mode.get()
        if current_code not in self._propagate_code_to_label:
            current_code = "if_missing"
            self.var_propagate_mode.set(current_code)
        self.combo_propagate_mode.set(self._propagate_code_to_label[current_code])

    def on_propagate_mode_changed(self, e: Any = None) -> None:
        if not hasattr(self, "combo_propagate_mode"):
            return
        label = self.combo_propagate_mode.get()
        code = getattr(self, "_propagate_label_to_code", {}).get(label)
        if code:
            self.var_propagate_mode.set(code)

    def _refresh_class_dropdown(self, preferred_idx: int | None = None) -> None:
        if not hasattr(self, "combo_cls"):
            return
        self.combo_cls.configure(values=self.class_names)
        if not self.class_names:
            return
        current_idx = self.combo_cls.current() if preferred_idx is None else preferred_idx
        if current_idx < 0 or current_idx >= len(self.class_names):
            current_idx = 0
        self.combo_cls.current(current_idx)

    def _ensure_class_name(self, class_name: str, fallback_id: int | None = None) -> int:
        normalized_name = class_name.strip()
        if not normalized_name:
            if fallback_id is not None and 0 <= fallback_id < len(self.class_names):
                return fallback_id
            normalized_name = f"class_{fallback_id}" if fallback_id is not None else "object"

        if normalized_name in self.class_names:
            return self.class_names.index(normalized_name)

        previous_idx = self.combo_cls.current() if hasattr(self, "combo_cls") else 0
        self.class_names.append(normalized_name)
        self._refresh_class_dropdown(preferred_idx=previous_idx)
        return len(self.class_names) - 1

    def _resolve_detected_class_index(self, result: Any, det_idx: int, fallback_idx: int) -> int:
        model_class_id: int | None = None
        model_class_name: str | None = None

        boxes = getattr(result, "boxes", None)
        cls_values = getattr(boxes, "cls", None)
        if cls_values is not None and det_idx < len(cls_values):
            model_class_id = int(cls_values[det_idx].item())

        names = getattr(result, "names", None)
        if model_class_id is not None and names is not None:
            if isinstance(names, dict):
                model_class_name = names.get(model_class_id)
            elif isinstance(names, (list, tuple)) and 0 <= model_class_id < len(names):
                model_class_name = names[model_class_id]

        if model_class_name is None and model_class_id is not None and self.yolo_model is not None:
            model_names = getattr(self.yolo_model, "names", None)
            if isinstance(model_names, dict):
                model_class_name = model_names.get(model_class_id)
            elif isinstance(model_names, (list, tuple)) and 0 <= model_class_id < len(model_names):
                model_class_name = model_names[model_class_id]

        if isinstance(model_class_name, str) and model_class_name.strip():
            return self._ensure_class_name(model_class_name, fallback_id=model_class_id)

        if model_class_id is not None:
            if 0 <= model_class_id < len(self.class_names):
                return model_class_id
            return self._ensure_class_name("", fallback_id=model_class_id)

        return fallback_idx

    def _project_progress_yaml_path(self, project_root: str | None = None) -> str | None:
        root = (project_root or self.project_root or "").strip()
        if not root:
            return None
        return os.path.join(root, ".ai_labeller_progress.yaml")

    def _write_project_progress_yaml(self) -> None:
        yaml_path = self._project_progress_yaml_path()
        if not yaml_path:
            return
        image_name = ""
        image_index = self.current_idx
        if self.image_files and 0 <= self.current_idx < len(self.image_files):
            image_name = os.path.basename(self.image_files[self.current_idx])

        def q(value: str) -> str:
            return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'

        lines = [
            "# AI Labeller progress",
            f"project_root: {q(self.project_root)}",
            f"split: {q(self.current_split)}",
            f"image_name: {q(image_name)}",
            f"image_index: {image_index}",
            f"class_count: {len(self.class_names)}",
            f"updated_at: {q(datetime.datetime.now().isoformat(timespec='seconds'))}",
        ]
        for idx, class_name in enumerate(self.class_names):
            lines.append(f"class_{idx}: {q(class_name)}")
        try:
            atomic_write_text(yaml_path, "\n".join(lines) + "\n")
        except Exception:
            self.logger.exception("Failed to write project progress yaml: %s", yaml_path)

    def _read_project_progress_yaml(self, project_root: str) -> dict[str, str]:
        yaml_path = self._project_progress_yaml_path(project_root)
        if not yaml_path or not os.path.isfile(yaml_path):
            return {}
        data: dict[str, str] = {}
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#") or ":" not in line:
                        continue
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    if value.startswith('"') and value.endswith('"') and len(value) >= 2:
                        value = value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
                    data[key] = value
        except Exception:
            self.logger.exception("Failed to read project progress yaml: %s", yaml_path)
            return {}
        return data

    def _extract_class_names_from_progress(self, progress: dict[str, str]) -> list[str]:
        count_raw = progress.get("class_count", "")
        try:
            class_count = int(count_raw)
        except ValueError:
            class_count = 0
        if class_count <= 0:
            return []
        names: list[str] = []
        for idx in range(class_count):
            key = f"class_{idx}"
            name = progress.get(key, "").strip()
            if not name:
                return []
            names.append(name)
        return names

    def save_session_state(self) -> None:
        state = SessionState(
            project_root=self.project_root,
            split=self.current_split,
            image_name="",
            detection_model_mode=self.det_model_mode.get(),
            detection_model_path=self.yolo_path.get().strip(),
        )
        if self.image_files and 0 <= self.current_idx < len(self.image_files):
            state.image_name = os.path.basename(self.image_files[self.current_idx])
        try:
            atomic_write_json(self.session_path, state.__dict__)
        except Exception:
            self.logger.exception("Failed to save session state")
        self._write_project_progress_yaml()

    def load_session_state(self) -> None:
        if not os.path.exists(self.session_path):
            return
        try:
            with open(self.session_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            self.logger.exception("Failed to load session state")
            return

        project_root = data.get("project_root", "")
        split = data.get("split", "train")
        image_name = data.get("image_name", "")
        model_mode = data.get("detection_model_mode", "Official YOLO26m.pt (Bundled)")
        model_path = data.get("detection_model_path", self.config.yolo_model_path)

        if not project_root or not os.path.exists(project_root):
            return

        if split not in ["train", "val", "test"]:
            split = "train"
        if model_mode not in {
            "Official YOLO26m.pt (Bundled)",
            "Custom YOLO (v5/v7/v8/v9/v11/v26)",
            "Custom RF-DETR",
        }:
            model_mode = "Official YOLO26m.pt (Bundled)"
        self.det_model_mode.set(model_mode)
        if model_path:
            self.yolo_path.set(model_path)
            self._register_model_path(model_path)
        self.current_split = split
        self.combo_split.set(split)
        self.load_project_from_path(project_root, preferred_image=image_name, save_session=False)

    def on_detection_model_mode_changed(self, e: Any = None) -> None:
        if self.det_model_mode.get() == "Official YOLO26m.pt (Bundled)":
            self.yolo_path.set(self.config.yolo_model_path)
            self._register_model_path(self.config.yolo_model_path)
        self.yolo_model = None
        self._loaded_model_key = None

    def use_official_yolo26n(self) -> None:
        self.det_model_mode.set("Official YOLO26m.pt (Bundled)")
        self.yolo_path.set(self.config.yolo_model_path)
        self._register_model_path(self.config.yolo_model_path)
        self.yolo_model = None
        self._loaded_model_key = None

    def _resolve_official_model_path(self) -> str:
        candidates: list[str] = []
        try:
            packaged = resources.files("ai_labeller").joinpath("models", self.config.yolo_model_path)
            if packaged.is_file():
                return str(packaged)
        except Exception:
            self.logger.exception("Failed to resolve packaged official model path")

        candidates.append(self.config.yolo_model_path)
        candidates.append(os.path.join(os.path.dirname(__file__), "models", self.config.yolo_model_path))
        candidates.append(os.path.join(os.getcwd(), self.config.yolo_model_path))

        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                return candidate

        raise FileNotFoundError(
            f"Official model not found: {self.config.yolo_model_path}. "
            "Please reinstall package or choose a custom model."
        )

    def _resolve_custom_model_path(self, raw_path: str) -> str:
        path = (raw_path or "").strip().strip('"').strip("'")
        if not path:
            raise FileNotFoundError("Model file not found: empty path")

        normalized = os.path.abspath(os.path.expanduser(path))
        if os.path.isfile(normalized):
            return normalized

        candidates: list[str] = []
        lower_norm = normalized.lower()

        # If a run directory or weights directory is given, try common YOLO outputs.
        if os.path.isdir(normalized):
            candidates.extend(
                [
                    os.path.join(normalized, "weights", "best.pt"),
                    os.path.join(normalized, "weights", "last.pt"),
                    os.path.join(normalized, "best.pt"),
                    os.path.join(normalized, "last.pt"),
                ]
            )
        else:
            # If user entered a non-existing file path, try sibling fallback.
            parent = os.path.dirname(normalized)
            name = os.path.basename(normalized).lower()
            if name == "best.pt":
                candidates.append(os.path.join(parent, "last.pt"))
            elif name == "last.pt":
                candidates.append(os.path.join(parent, "best.pt"))

            # If path looks like a run folder string, infer weights files.
            root, ext = os.path.splitext(normalized)
            if not ext:
                candidates.extend(
                    [
                        os.path.join(normalized, "weights", "best.pt"),
                        os.path.join(normalized, "weights", "last.pt"),
                        os.path.join(normalized, "best.pt"),
                        os.path.join(normalized, "last.pt"),
                    ]
                )
            if lower_norm.endswith(os.path.join("weights", "best.pt").lower()):
                run_dir = os.path.dirname(os.path.dirname(normalized))
                candidates.append(os.path.join(run_dir, "weights", "last.pt"))
            if lower_norm.endswith(os.path.join("weights", "last.pt").lower()):
                run_dir = os.path.dirname(os.path.dirname(normalized))
                candidates.append(os.path.join(run_dir, "weights", "best.pt"))

        for candidate in candidates:
            if os.path.isfile(candidate):
                self.logger.warning("Model path repaired: %s -> %s", normalized, candidate)
                return os.path.abspath(candidate)

        raise FileNotFoundError(f"Model file not found:\n{normalized}")

    def browse_detection_model(self) -> None:
        self.pick_model_file()

    def pick_model_file(self, forced_mode: str | None = None) -> bool:
        self.logger.info("Opening model file dialog (mode=%s)", forced_mode or "auto")
        model_path = filedialog.askopenfilename(
            parent=self.root,
            title="Select model",
            filetypes=[
                ("Model files", "*.pt *.onnx"),
                ("PyTorch", "*.pt"),
                ("ONNX", "*.onnx"),
                ("All files", "*.*"),
            ],
        )
        if not model_path:
            self.logger.info("Model selection cancelled")
            return False
        model_path = os.path.abspath(model_path)
        if not os.path.isfile(model_path):
            self.logger.error("Selected model file not found: %s", model_path)
            messagebox.showerror("Model Error", f"Model file not found:\n{model_path}")
            return False
        if not model_path.lower().endswith((".pt", ".onnx")):
            proceed = messagebox.askyesno(
                "Model Warning",
                f"Selected file may not be a YOLO model:\n{os.path.basename(model_path)}\n\nContinue?",
            )
            if not proceed:
                self.logger.info("Model selection rejected by user due to extension warning")
                return False
        self.yolo_path.set(model_path)
        self._register_model_path(model_path)
        if forced_mode:
            self.det_model_mode.set(forced_mode)
        elif "rfdetr" in os.path.basename(model_path).lower():
            self.det_model_mode.set("Custom RF-DETR")
        else:
            self.det_model_mode.set("Custom YOLO (v5/v7/v8/v9/v11/v26)")
        self.yolo_model = None
        self._loaded_model_key = None
        self.save_session_state()
        self.logger.info("Model selected: %s (%s)", model_path, self.det_model_mode.get())
        return True

    def show_startup_source_dialog(self, force: bool = False, reason: str | None = None) -> None:
        if self._startup_dialog_open:
            return
        if not force and getattr(self, "_startup_dialog_shown", False):
            return
        self._startup_dialog_shown = True
        self._startup_dialog_open = True
        if reason:
            self.logger.info("Showing startup source dialog: %s", reason)

        win = tk.Toplevel(self.root)
        win.title(LANG_MAP[self.lang]["startup_choose_source"])
        win.geometry("420x240")
        win.resizable(False, False)
        win.configure(bg=COLORS["bg_light"])
        win.transient(self.root)
        win.grab_set()

        tk.Label(
            win,
            text=LANG_MAP[self.lang]["startup_prompt"],
            bg=COLORS["bg_light"],
            fg=COLORS["text_primary"],
            font=self.font_bold,
            anchor="w",
        ).pack(fill="x", padx=20, pady=(20, 14))

        def choose_images() -> None:
            self._close_startup_dialog(win)
            self.root.after(1, lambda: self.startup_choose_images_folder("images"))

        def choose_yolo() -> None:
            self._close_startup_dialog(win)
            self.det_model_mode.set("Custom YOLO (v5/v7/v8/v9/v11/v26)")
            self.root.after(1, lambda: self.startup_choose_images_folder("yolo"))

        def choose_rfdetr() -> None:
            self._close_startup_dialog(win)
            self.det_model_mode.set("Custom RF-DETR")
            self.root.after(1, lambda: self.startup_choose_images_folder("rfdetr"))

        self.create_primary_button(
            win,
            text=LANG_MAP[self.lang]["startup_images"],
            command=choose_images,
            bg=COLORS["primary"],
        ).pack(fill="x", padx=20, pady=(0, 10))

        self.create_primary_button(
            win,
            text=LANG_MAP[self.lang]["startup_yolo"],
            command=choose_yolo,
            bg=COLORS["success"],
        ).pack(fill="x", padx=20, pady=(0, 10))

        self.create_primary_button(
            win,
            text=LANG_MAP[self.lang]["startup_rfdetr"],
            command=choose_rfdetr,
            bg=COLORS["warning"],
        ).pack(fill="x", padx=20, pady=(0, 10))

        self.create_secondary_button(
            win,
            text=LANG_MAP[self.lang]["startup_skip"],
            command=lambda: self._close_startup_dialog(win),
        ).pack(fill="x", padx=20, pady=(0, 18))
        win.protocol("WM_DELETE_WINDOW", lambda: self._close_startup_dialog(win))

    def _close_startup_dialog(self, win: tk.Toplevel) -> None:
        try:
            win.grab_release()
        except Exception:
            pass
        try:
            win.destroy()
        except Exception:
            pass
        self._startup_dialog_open = False
        self.logger.info("Startup source dialog closed")

    def _choose_model_then_images(self, mode: str) -> None:
        try:
            ok = self.pick_model_file(mode)
            if not ok:
                use_images_only = messagebox.askyesno(
                    LANG_MAP[self.lang].get("startup_model_cancel_title", "Model Selection Cancelled"),
                    LANG_MAP[self.lang].get(
                        "startup_model_cancel_msg",
                        "No model selected. Continue with images folder only?",
                    ),
                    parent=self.root,
                )
                if use_images_only:
                    self.root.after(120, self.startup_choose_images_folder)
                else:
                    self.root.after(120, lambda: self.show_startup_source_dialog(force=True, reason="model selection cancelled"))
                return
            model_path = self.yolo_path.get().strip()
            try:
                model_path = self._resolve_custom_model_path(model_path)
            except FileNotFoundError:
                messagebox.showerror("Model Error", "Invalid model file selected. Please try again.")
                self.root.after(120, lambda: self.show_startup_source_dialog(force=True, reason="invalid model path"))
                return
            self.yolo_path.set(model_path)
            self._register_model_path(model_path)
            # Open on next idle tick + short delay to avoid native-dialog focus races on Windows.
            self.root.after_idle(lambda: self.root.after(180, self.startup_choose_images_folder))
        except Exception:
            self.logger.exception("Error while selecting model and folder")
            messagebox.showerror("Error", "Failed during model selection.")
            self.root.after(120, lambda: self.show_startup_source_dialog(force=True, reason="selection error"))

    def return_to_source_select(self, e: Any = None) -> None:
        if self.project_root and self.image_files:
            try:
                self.save_current()
            except Exception:
                self.logger.exception("Failed to save before returning to source selector")
        self.show_startup_source_dialog(force=True)

    def startup_choose_images_folder(self, source_mode: str = "images") -> None:
        if self._folder_dialog_open:
            return
        self._folder_dialog_open = True
        self.logger.info("=== startup_choose_images_folder START (%s) ===", source_mode)
        try:
            directory = filedialog.askdirectory(
                parent=self.root,
                title=LANG_MAP[self.lang].get("pick_folder_title", "Select Folder")
            )
            if not directory:
                self.logger.info("Folder selection cancelled")
                return
            directory = os.path.abspath(directory)
            self.logger.info("Selected directory: %s", directory)

            diag = self.diagnose_folder_structure(directory)
            self.logger.info("Folder diagnosis: %s", diag)
            if not diag["is_yolo_project"] and diag["flat_images"] == 0:
                self.show_folder_diagnosis(directory)
                return

            root_dir = self.normalize_project_root(directory)
            yolo_root = self.find_yolo_project_root(root_dir)

            try:
                if yolo_root:
                    self.load_project_from_path(yolo_root)
                    if self.image_files:
                        self.root.lift()
                        self.root.focus_force()
                        messagebox.showinfo(
                            LANG_MAP[self.lang]["title"],
                            LANG_MAP[self.lang].get(
                                "loaded_from",
                                "Loaded {count} images\nFrom: {path}\nSplit: {split}",
                            ).format(
                                count=len(self.image_files),
                                path=yolo_root,
                                split=self.current_split,
                            ),
                            parent=self.root,
                        )
                    else:
                        self.show_folder_diagnosis(yolo_root)
                    return

                self.load_images_folder_only(root_dir)
                if self.image_files:
                    self.root.lift()
                    self.root.focus_force()
                    messagebox.showinfo(
                        LANG_MAP[self.lang]["title"],
                        LANG_MAP[self.lang].get(
                            "loaded_from",
                            "Loaded {count} images\nFrom: {path}\nSplit: {split}",
                        ).format(
                            count=len(self.image_files),
                            path=root_dir,
                            split=self.current_split,
                        ),
                        parent=self.root,
                    )
                else:
                    self.show_folder_diagnosis(root_dir)
            except Exception as exc:
                self.logger.exception("Failed to load selected folder: %s", directory)
                self.root.lift()
                self.root.focus_force()
                messagebox.showerror(
                    "Error",
                    f"Failed to load folder:\n{directory}\n\nError: {exc}",
                    parent=self.root,
                )
        finally:
            self._folder_dialog_open = False
            self.logger.info("=== startup_choose_images_folder END ===")

    def normalize_project_root(self, directory: str) -> str:
        root_dir = directory.replace("\\", "/").rstrip("/")
        base = os.path.basename(root_dir).lower()
        parent = os.path.dirname(root_dir).replace("\\", "/")
        parent_base = os.path.basename(parent).lower()
        grandparent = os.path.dirname(parent).replace("\\", "/")

        if base in {"train", "val", "test"} and parent_base == "images":
            if os.path.exists(f"{grandparent}/labels"):
                return grandparent
        if base == "images" and os.path.exists(f"{parent}/labels"):
            return parent
        if base == "labels" and os.path.exists(f"{parent}/images"):
            return parent
        return root_dir

    def find_yolo_project_root(self, directory: str) -> str | None:
        root = directory.replace("\\", "/").rstrip("/")
        if not root:
            return None

        def is_yolo_root(candidate: str) -> bool:
            return any(
                os.path.exists(f"{candidate}/images/{split}")
                for split in ("train", "val", "test")
            )

        # Try selected folder and a few parent levels first.
        candidates = [root]
        parent = os.path.dirname(root).replace("\\", "/")
        if parent and parent not in candidates:
            candidates.append(parent)
        grandparent = os.path.dirname(parent).replace("\\", "/") if parent else ""
        if grandparent and grandparent not in candidates:
            candidates.append(grandparent)

        for candidate in candidates:
            if is_yolo_root(candidate):
                return candidate

        # If user selected an upper folder, scan one level of child directories.
        try:
            for child in os.listdir(root):
                child_path = os.path.join(root, child).replace("\\", "/")
                if os.path.isdir(child_path) and is_yolo_root(child_path):
                    return child_path
        except Exception:
            pass

        return None

    def _list_split_images_for_root(self, project_root: str, split: str) -> list[str]:
        img_path = f"{project_root}/images/{split}"
        return sorted([
            f for f in glob.glob(f"{img_path}/*.*")
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

    def _existing_image_splits(self, project_root: str) -> list[str]:
        splits: list[str] = []
        for split in ("train", "val", "test"):
            if os.path.isdir(f"{project_root}/images/{split}"):
                splits.append(split)
        return splits

    def ensure_yolo_label_dirs(self, project_root: str) -> None:
        """Ensure labels/<split> exists for every existing images/<split>."""
        splits = self._existing_image_splits(project_root)
        if not splits:
            return
        for split in splits:
            os.makedirs(f"{project_root}/labels/{split}", exist_ok=True)

    def diagnose_folder_structure(self, directory: str) -> dict[str, Any]:
        result: dict[str, Any] = {
            "is_yolo_project": False,
            "has_images_folder": False,
            "has_labels_folder": False,
            "splits_found": [],
            "total_images": 0,
            "images_by_split": {},
            "flat_images": 0,
            "errors": [],
        }
        try:
            images_path = os.path.join(directory, "images")
            labels_path = os.path.join(directory, "labels")
            result["has_images_folder"] = os.path.isdir(images_path)
            result["has_labels_folder"] = os.path.isdir(labels_path)

            if result["has_images_folder"]:
                for split in ("train", "val", "test"):
                    split_path = os.path.join(images_path, split)
                    if not os.path.isdir(split_path):
                        continue
                    files = [
                        f for f in os.listdir(split_path)
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))
                    ]
                    if files:
                        result["splits_found"].append(split)
                        result["images_by_split"][split] = len(files)
                        result["total_images"] += len(files)
                result["is_yolo_project"] = len(result["splits_found"]) > 0

            root_images = [
                f for f in os.listdir(directory)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            result["flat_images"] = len(root_images)
        except Exception as exc:
            result["errors"].append(str(exc))
        return result

    def show_folder_diagnosis(self, directory: str) -> None:
        diag = self.diagnose_folder_structure(directory)
        lines = [f"Folder: {directory}", ""]
        if diag["is_yolo_project"]:
            lines.append("YOLO project structure detected.")
            lines.append(f"Splits: {', '.join(diag['splits_found'])}")
            for split in ("train", "val", "test"):
                if split in diag["images_by_split"]:
                    lines.append(f"- {split}: {diag['images_by_split'][split]} images")
            lines.append(f"Total images: {diag['total_images']}")
        elif diag["flat_images"] > 0:
            lines.append("Flat image folder detected.")
            lines.append(f"Images found: {diag['flat_images']}")
        else:
            lines.append("No supported images found.")
            lines.append("Expected either:")
            lines.append("- folder/images/train|val|test/*.jpg")
            lines.append("- folder/*.jpg")
        if diag["errors"]:
            lines.append("")
            lines.append("Errors:")
            for err in diag["errors"]:
                lines.append(f"- {err}")
        self.root.lift()
        self.root.focus_force()
        messagebox.showwarning("Folder Diagnosis", "\n".join(lines), parent=self.root)

    def load_images_folder_only(self, directory: str) -> None:
        self.project_root = directory
        self.current_split = "train"
        self.combo_split.set(self.current_split)

        self.image_files = sorted([
            f for f in glob.glob(f"{self.project_root}/*.*")
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        os.makedirs(f"{self.project_root}/labels/{self.current_split}", exist_ok=True)

        if not self.image_files:
            messagebox.showinfo(
                LANG_MAP[self.lang]["title"],
                LANG_MAP[self.lang]["no_img"],
            )
            self.current_idx = 0
            self.img_pil = None
            self.img_tk = None
            self.rects = []
            self.update_info_text()
            self.render()
            self.save_session_state()
            return

        self.current_idx = 0
        self.load_img()

    def on_app_close(self) -> None:
        try:
            self.save_current()
        except Exception:
            self.logger.exception("Failed while saving on close")
        if self.training_process is not None and self.training_process.poll() is None:
            try:
                self.training_process.terminate()
            except Exception:
                self.logger.exception("Failed to terminate training process on close")
        if self.img_pil is not None:
            self.img_pil.close()
            self.img_pil = None
        self.save_session_state()
        self.root.destroy()

    def render(self) -> None:
        """Redraw canvas image, boxes, guides, and overlays."""
        self.canvas.delete("all")
        self._cursor_line_x = None
        self._cursor_line_y = None
        self._cursor_text_id = None
        self._cursor_bg_id = None

        if not self.img_pil:
            return
        
        # 1. ???????????
        w = int(self.img_pil.width * self.scale)
        h = int(self.img_pil.height * self.scale)
        self.img_tk = ImageTk.PhotoImage(
            self.img_pil.resize((w, h), Image.Resampling.NEAREST)
        )
        self.canvas.create_image(
            self.offset_x,
            self.offset_y,
            image=self.img_tk,
            anchor="nw"
        )
        
        # 2. ????????????
        box_colors = [
            COLORS["box_1"], COLORS["box_2"], COLORS["box_3"],
            COLORS["box_4"], COLORS["box_5"], COLORS["box_6"]
        ]
        
        selected_set = set(self._get_selected_indices())
        for i, rect in enumerate(self.rects):
            x1, y1 = self.img_to_canvas(rect[0], rect[1])
            x2, y2 = self.img_to_canvas(rect[2], rect[3])
            corners = self.get_rotated_corners(rect)
            canvas_points: list[float] = []
            for px, py in corners:
                cxp, cyp = self.img_to_canvas(px, py)
                canvas_points.extend([cxp, cyp])
            
            is_selected = i in selected_set
            color = COLORS["box_selected"] if is_selected else box_colors[rect[4] % len(box_colors)]
            width = 3 if is_selected else 2
            
            # ???????????????
            self.canvas.create_polygon(
                canvas_points,
                outline=color,
                width=width,
                fill=""
            )

            # Heading line helps visualizing orientation.
            angle_deg = self.get_rect_angle_deg(rect)
            if abs(angle_deg) > 1e-3:
                cx_mid = (rect[0] + rect[2]) / 2
                cy_mid = (rect[1] + rect[3]) / 2
                xh = (rect[0] + rect[2]) / 2
                yh = min(rect[1], rect[3])
                pxh, pyh = self.rotate_point_around_center(xh, yh, cx_mid, cy_mid, angle_deg)
                cxc, cyc = self.img_to_canvas(cx_mid, cy_mid)
                cxx, cyy = self.img_to_canvas(pxh, pyh)
                self.canvas.create_line(cxc, cyc, cxx, cyy, fill=color, width=2)
            
            # ????????????????????
            if is_selected and self.selected_idx == i and len(selected_set) == 1:
                for hx, hy in self.get_handles(rect):
                    cx, cy = self.img_to_canvas(hx, hy)
                    self.canvas.create_oval(
                        cx - self.HANDLE_SIZE,
                        cy - self.HANDLE_SIZE,
                        cx + self.HANDLE_SIZE,
                        cy + self.HANDLE_SIZE,
                        fill=COLORS["bg_white"],
                        outline=color,
                        width=2
                    )
                top_x, top_y, rot_x, rot_y = self.get_rotation_handle_points(rect)
                ctx, cty = self.img_to_canvas(top_x, top_y)
                crx, cry = self.img_to_canvas(rot_x, rot_y)
                self.canvas.create_line(ctx, cty, crx, cry, fill=color, width=2)
                knob_r = self.HANDLE_SIZE + 1
                self.canvas.create_oval(
                    crx - knob_r,
                    cry - knob_r,
                    crx + knob_r,
                    cry + knob_r,
                    fill=color,
                    outline=COLORS["bg_white"],
                    width=2,
                )
            
            if self.show_all_labels:
                # ?????????????????
                class_name = (
                    self.class_names[rect[4]]
                    if rect[4] < len(self.class_names)
                    else f"ID:{rect[4]}"
                )
                if abs(angle_deg) > 1e-3:
                    class_name = f"{class_name} ({angle_deg:.1f}°)"
                
                # Draw a new box???????????????????????
                min_canvas_y = min(canvas_points[1::2]) if canvas_points else y1
                min_canvas_x = min(canvas_points[0::2]) if canvas_points else x1
                label_y = max(min_canvas_y - 24, 8)  # Draw a new box???????
                
                # ???????????
                text_id = self.canvas.create_text(
                    min_canvas_x + 8,
                    label_y + 4,
                    text=class_name,
                    fill=COLORS["text_white"],
                    font=self.font_primary,
                    anchor="nw"
                )
                
                bbox = self.canvas.bbox(text_id)
                if bbox:
                    padding = 4
                    bg_id = self.canvas.create_rectangle(
                        bbox[0] - padding,
                        bbox[1] - padding,
                        bbox[2] + padding,
                        bbox[3] + padding,
                        fill=color,
                        outline=""
                    )
                    self.canvas.tag_lower(bg_id, text_id)
        
        # 3. Draw temporary rectangle while dragging
        if self.temp_rect_coords:
            cx, cy, ex, ey = self.temp_rect_coords
            self.canvas.create_rectangle(
                cx, cy, ex, ey,
                outline=COLORS["primary_light"],
                width=2,
                dash=(6, 4)
            )
        if self.select_rect_coords:
            cx, cy, ex, ey = self.select_rect_coords
            self.canvas.create_rectangle(
                cx, cy, ex, ey,
                outline=COLORS["box_selected"],
                width=2,
                dash=(4, 4)
            )
        
        # 4. Update cursor overlay
        self.update_cursor_overlay()
        
        # Refresh side info labels
        self.update_info_text()
    
    # ==================== Mouse Interaction ====================
    
    def on_mouse_move(self, e: Any) -> None:
        """Track cursor and refresh overlay."""
        self.mouse_pos = (e.x, e.y)
        self.update_cursor_overlay()
    
    def update_cursor_overlay(self) -> None:
        if not self.canvas:
            return

        mx, my = self.mouse_pos
        canvas_h = self.canvas.winfo_height()
        canvas_w = self.canvas.winfo_width()

        if self._cursor_line_x is None:
            self._cursor_line_x = self.canvas.create_line(
                mx, 0, mx, canvas_h, fill=COLORS["primary"], width=1, dash=(2, 4), tags="cursor_overlay"
            )
        else:
            self.canvas.coords(self._cursor_line_x, mx, 0, mx, canvas_h)

        if self._cursor_line_y is None:
            self._cursor_line_y = self.canvas.create_line(
                0, my, canvas_w, my, fill=COLORS["primary"], width=1, dash=(2, 4), tags="cursor_overlay"
            )
        else:
            self.canvas.coords(self._cursor_line_y, 0, my, canvas_w, my)

        coord_text = f"{mx}, {my}"
        if self._cursor_text_id is None:
            self._cursor_text_id = self.canvas.create_text(
                mx + 12,
                my - 12,
                text=coord_text,
                fill=COLORS["text_primary"] if self.theme == "light" else COLORS["text_white"],
                font=self.font_mono,
                anchor="nw",
                tags="cursor_overlay",
            )
        else:
            self.canvas.coords(self._cursor_text_id, mx + 12, my - 12)
            self.canvas.itemconfig(self._cursor_text_id, text=coord_text)

        coord_bbox = self.canvas.bbox(self._cursor_text_id)
        if not coord_bbox:
            return

        padding = 4
        bx1 = coord_bbox[0] - padding
        by1 = coord_bbox[1] - padding
        bx2 = coord_bbox[2] + padding
        by2 = coord_bbox[3] + padding
        if self._cursor_bg_id is None:
            self._cursor_bg_id = self.canvas.create_rectangle(
                bx1,
                by1,
                bx2,
                by2,
                fill=COLORS["bg_dark"],
                outline=COLORS["primary"],
                width=1,
                tags="cursor_overlay",
            )
        else:
            self.canvas.coords(self._cursor_bg_id, bx1, by1, bx2, by2)
        self.canvas.tag_lower(self._cursor_bg_id, self._cursor_text_id)

    def on_mouse_down(self, e):
        """Handle mouse press for select, move, resize, or create box."""
        if not self.img_pil:
            return
        
        ix, iy = self.canvas_to_img(e.x, e.y)
        is_additive_select = bool(e.state & 0x0001) or bool(e.state & 0x0004)
        is_ctrl_select = bool(e.state & 0x0004)
        self.active_rotate_handle = False
        self.active_handle = None
        self.is_drag_selecting = False
        self.select_rect_coords = None
        
        # Check if one of the resize handles is selected
        if self.selected_idx is not None and len(self._get_selected_indices()) == 1 and not is_additive_select:
            active_rect = self.rects[self.selected_idx]
            _, _, rhx, rhy = self.get_rotation_handle_points(active_rect)
            rotate_dist = np.sqrt((ix - rhx) ** 2 + (iy - rhy) ** 2) * self.scale
            if rotate_dist < (self.config.mouse_handle_hit_radius_px + 3):
                cx = (active_rect[0] + active_rect[2]) / 2
                cy = (active_rect[1] + active_rect[3]) / 2
                pointer_deg = math.degrees(math.atan2(iy - cy, ix - cx))
                self.rotate_drag_offset_deg = pointer_deg - self.get_rect_angle_deg(active_rect)
                self.active_rotate_handle = True
                self.drag_start = (ix, iy)
                self.push_history()
                return
            for i, (hx, hy) in enumerate(self.get_handles(active_rect)):
                dist = np.sqrt((ix - hx) ** 2 + (iy - hy) ** 2) * self.scale
                if dist < self.config.mouse_handle_hit_radius_px:
                    self.active_handle = i
                    self.drag_start = (ix, iy)
                    self.push_history()
                    return
        
        # Check if pointer is inside an existing box
        clicked_idx = self._pick_box_at_point(ix, iy)
        
        if clicked_idx is not None:
            if is_additive_select:
                selected = self._get_selected_indices()
                if clicked_idx in selected:
                    selected = [idx for idx in selected if idx != clicked_idx]
                else:
                    selected.append(clicked_idx)
                self._set_selected_indices(selected, primary_idx=clicked_idx if clicked_idx in selected else None)
                self._sync_class_combo_with_selection()
                self.is_moving_box = False
                self.drag_start = None
            else:
                self._set_selected_indices([clicked_idx], primary_idx=clicked_idx)
                self.is_moving_box = True
                self.drag_start = (ix, iy)
                self._sync_class_combo_with_selection()
                self.push_history()
        else:
            if is_ctrl_select:
                self.drag_start = (ix, iy)
                self.is_drag_selecting = True
                self.select_rect_coords = (e.x, e.y, e.x, e.y)
            else:
                if not is_additive_select:
                    self._set_selected_indices([])
                self.drag_start = (ix, iy)
                self.temp_rect_coords = (e.x, e.y, e.x, e.y)
        
        self.render()
    
    def on_mouse_drag(self, e):
        """???"""
        self.mouse_pos = (e.x, e.y)
        
        if not self.img_pil or not self.drag_start:
            self.render()
            return
        
        ix, iy = self.canvas_to_img(e.x, e.y)
        W, H = self.img_pil.width, self.img_pil.height
        
        # Draw a new box????
        ix = max(0, min(W, ix))
        iy = max(0, min(H, iy))
        
        if self.selected_idx is not None and self.active_rotate_handle:
            rect = self.rects[self.selected_idx]
            cx = (rect[0] + rect[2]) / 2
            cy = (rect[1] + rect[3]) / 2
            pointer_deg = math.degrees(math.atan2(iy - cy, ix - cx))
            self.set_rect_angle_deg(rect, pointer_deg - self.rotate_drag_offset_deg)
        elif self.selected_idx is not None and self.active_handle is not None:
            # ???????????
            rect = self.rects[self.selected_idx]
            cx = (rect[0] + rect[2]) / 2
            cy = (rect[1] + rect[3]) / 2
            angle_deg = self.get_rect_angle_deg(rect)
            lx, ly = self.rotate_point_around_center(ix, iy, cx, cy, -angle_deg)
            if self.active_handle in [0, 6, 7]:
                rect[0] = lx
            if self.active_handle in [0, 1, 2]:
                rect[1] = ly
            if self.active_handle in [2, 3, 4]:
                rect[2] = lx
            if self.active_handle in [4, 5, 6]:
                rect[3] = ly
        
        elif self.is_moving_box:
            # Move selected box
            dx = ix - self.drag_start[0]
            dy = iy - self.drag_start[1]
            selected = self._get_selected_indices()
            if not selected and self.selected_idx is not None:
                selected = [self.selected_idx]
            if selected:
                min_dx = max(-self.rects[idx][0] for idx in selected)
                max_dx = min(W - self.rects[idx][2] for idx in selected)
                min_dy = max(-self.rects[idx][1] for idx in selected)
                max_dy = min(H - self.rects[idx][3] for idx in selected)
                clamped_dx = min(max(dx, min_dx), max_dx)
                clamped_dy = min(max(dy, min_dy), max_dy)
                for idx in selected:
                    rect = self.rects[idx]
                    rect[0] += clamped_dx
                    rect[1] += clamped_dy
                rect[2] += clamped_dx
                rect[3] += clamped_dy
            self.drag_start = (ix, iy)
        elif self.is_drag_selecting:
            if self.select_rect_coords:
                self.select_rect_coords = (
                    self.select_rect_coords[0],
                    self.select_rect_coords[1],
                    e.x,
                    e.y,
                )
        
        else:
            # Draw a new box
            if self.temp_rect_coords:
                self.temp_rect_coords = (
                    self.temp_rect_coords[0],
                    self.temp_rect_coords[1],
                    e.x,
                    e.y
                )
        
        self.render()
    
    def on_mouse_up(self, e):
        """?????"""
        if self.is_drag_selecting and self.select_rect_coords:
            sx, sy, ex, ey = self.select_rect_coords
            ix1, iy1 = self.canvas_to_img(sx, sy)
            ix2, iy2 = self.canvas_to_img(ex, ey)
            hits = self._pick_boxes_in_img_rect(ix1, iy1, ix2, iy2)
            merged = sorted(set(self._get_selected_indices() + hits))
            self._set_selected_indices(merged, primary_idx=hits[-1] if hits else self.selected_idx)
            self._sync_class_combo_with_selection()
            self.is_drag_selecting = False
            self.select_rect_coords = None
            self.drag_start = None
            self.render()
            return

        if self.temp_rect_coords:
            ix, iy = self.canvas_to_img(e.x, e.y)
            new_box = self.clamp_box([
                self.drag_start[0],
                self.drag_start[1],
                ix,
                iy,
                self.combo_cls.current()
            ])
            
            # Keep minimum box size threshold
            if (new_box[2] - new_box[0]) > 2 and (new_box[3] - new_box[1]) > 2:
                self.push_history()
                self.rects.append(new_box)
            
            self.temp_rect_coords = None
        
        for idx in self._get_selected_indices():
            self.rects[idx] = self.clamp_box(self.rects[idx])
        
        self.is_moving_box = False
        self.active_handle = None
        self.active_rotate_handle = False
        self.rotate_drag_offset_deg = 0.0
        self.is_drag_selecting = False
        self.select_rect_coords = None
        self.render()
    
    def on_zoom(self, e):
        """Zoom image around mouse pointer."""
        factor = self.config.zoom_in_factor if e.delta > 0 else self.config.zoom_out_factor
        
        self.offset_x = e.x - (e.x - self.offset_x) * factor
        self.offset_y = e.y - (e.y - self.offset_y) * factor
        self.scale *= factor
        
        self.render()
    
    # ==================== ???? ====================
    
    def on_class_change_request(self, e=None):
        """???"""
        selected = self._get_selected_indices()
        if not selected:
            return
        new_cid = self.combo_cls.current()
        if new_cid < 0:
            return
        if any(self.rects[idx][4] != new_cid for idx in selected):
            self.push_history()
            for idx in selected:
                self.rects[idx][4] = new_cid
            self.render()

    def rotate_selected_boxes(self, delta_deg: float) -> None:
        selected = self._get_selected_indices()
        if not selected:
            return
        self.push_history()
        for idx in selected:
            rect = self.rects[idx]
            self.set_rect_angle_deg(rect, self.get_rect_angle_deg(rect) + delta_deg)
        self.render()
    
    def edit_classes_table(self):
        """??????????????"""
        L = LANG_MAP[self.lang]
        
        win = tk.Toplevel(self.root)
        win.title(L["edit_classes"])
        win.geometry("500x600")
        win.configure(bg=COLORS["bg_light"])
        
        # TreeView
        tree = ttk.Treeview(
            win,
            columns=("id", "name"),
            show="headings",
            height=15
        )
        tree.heading("id", text="ID")
        tree.column("id", width=80, anchor="center")
        tree.heading("name", text=L["class_name"])
        tree.column("name", width=300)
        tree.pack(fill="both", expand=True, padx=20, pady=20)
        
        def refresh():
            for item in tree.get_children():
                tree.delete(item)
            for i, name in enumerate(self.class_names):
                tree.insert("", "end", values=(i, name))
        
        def rename():
            sel = tree.selection()
            if not sel:
                return
            idx = int(tree.item(sel[0])['values'][0])
            new_name = simpledialog.askstring(
                L["rename"],
                L["rename_prompt"].format(name=self.class_names[idx]),
                initialvalue=self.class_names[idx]
            )
            if new_name:
                self.class_names[idx] = new_name
                refresh()
        
        def add():
            new_name = simpledialog.askstring(L["add"], L["add_prompt"])
            if new_name:
                self.class_names.append(new_name)
                refresh()

        def delete_class():
            sel = tree.selection()
            if not sel:
                return
            if len(self.class_names) <= 1:
                messagebox.showinfo(L["class_mgmt"], L.get("delete_class_last", "Cannot delete the last class."))
                return
            del_idx = int(tree.item(sel[0])["values"][0])
            del_name = self.class_names[del_idx]
            if not messagebox.askyesno(
                L["class_mgmt"],
                L.get(
                    "delete_class_confirm",
                    "Delete class '{name}' (ID {idx})?\nLabels with this class in current image will be reassigned.",
                ).format(name=del_name, idx=del_idx),
                parent=win,
            ):
                return

            # Keep current-image boxes valid after class-id reindex.
            self.push_history()
            self._reindex_dataset_labels_after_class_delete(del_idx)
            remapped_rects: list[list[float]] = []
            for rect in self.rects:
                cid = int(rect[4])
                if cid == del_idx:
                    continue
                if cid > del_idx:
                    rect[4] = cid - 1
                remapped_rects.append(rect)
            self.rects = remapped_rects

            self.class_names.pop(del_idx)
            refresh()
            preferred_idx = min(max(0, del_idx), len(self.class_names) - 1)
            self._refresh_class_dropdown(preferred_idx=preferred_idx)
            self._set_selected_indices([])
            self._sync_class_combo_with_selection()
            self.render()

        def on_double_click(e):
            row = tree.identify_row(e.y)
            if row:
                tree.selection_set(row)
                rename()
        
        # ??
        btn_frame = tk.Frame(win, bg=COLORS["bg_light"])
        btn_frame.pack(fill="x", padx=20, pady=10)
        
        tk.Button(
            btn_frame,
            text=L["add"],
            command=add,
            bg=COLORS["success"],
            fg=COLORS["text_white"],
            font=self.font_primary,
            relief="flat",
            padx=20,
            pady=10
        ).pack(side="left", expand=True, fill="x", padx=(0, 5))
        
        tk.Button(
            btn_frame,
            text=L["rename"],
            command=rename,
            bg=COLORS["warning"],
            fg=COLORS["text_white"],
            font=self.font_primary,
            relief="flat",
            padx=20,
            pady=10
        ).pack(side="left", expand=True, fill="x", padx=(5, 0))

        tk.Button(
            btn_frame,
            text=L.get("delete_class", "Delete Class"),
            command=delete_class,
            bg=COLORS["danger"],
            fg=COLORS["text_white"],
            font=self.font_primary,
            relief="flat",
            padx=20,
            pady=10
        ).pack(side="left", expand=True, fill="x", padx=(5, 0))
        
        tk.Button(
            win,
            text=L["apply"],
            command=lambda: [
                self._refresh_class_dropdown(),
                self.render(),
                win.destroy()
            ],
            bg=COLORS["primary"],
            fg=COLORS["text_white"],
            font=self.font_bold,
            relief="flat",
            pady=12
        ).pack(fill="x", padx=20, pady=(0, 20))
        
        tree.bind("<Double-1>", on_double_click)
        refresh()

    def reassign_labeled_class(self):
        """????????????????"""
        selected = self._get_selected_indices()
        if not selected:
            messagebox.showinfo(LANG_MAP[self.lang]["class_mgmt"], LANG_MAP[self.lang]["no_label_selected"])
            return
        if not self.class_names:
            messagebox.showinfo(LANG_MAP[self.lang]["class_mgmt"], LANG_MAP[self.lang]["no_classes_available"])
            return

        win = tk.Toplevel(self.root)
        win.title(LANG_MAP[self.lang]["reassign_class"])
        win.geometry("420x220")
        win.configure(bg=COLORS["bg_light"])

        selected_class_ids = {self.rects[idx][4] for idx in selected}
        if len(selected_class_ids) == 1:
            current_idx = int(next(iter(selected_class_ids)))
            current_name = (
                self.class_names[current_idx]
                if current_idx < len(self.class_names)
                else str(current_idx)
            )
        else:
            current_idx = self.combo_cls.current()
            current_name = f"Multiple ({len(selected)} boxes)"

        tk.Label(
            win,
            text=f"{LANG_MAP[self.lang]['current']}: {current_name}",
            font=self.font_primary,
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_light"],
            anchor="w"
        ).pack(fill="x", padx=20, pady=(20, 6))

        tk.Label(
            win,
            text=LANG_MAP[self.lang]["to"],
            font=self.font_primary,
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_light"],
            anchor="w"
        ).pack(fill="x", padx=20, pady=(10, 6))

        to_default = self.class_names[current_idx] if 0 <= current_idx < len(self.class_names) else self.class_names[0]
        to_var = tk.StringVar(value=to_default)
        ttk.Combobox(
            win,
            values=self.class_names,
            textvariable=to_var,
            state="readonly",
            font=self.font_primary
        ).pack(fill="x", padx=20)

        def apply_change():
            to_name = to_var.get()
            try:
                to_idx = self.class_names.index(to_name)
            except ValueError:
                win.destroy()
                return

            if all(self.rects[idx][4] == to_idx for idx in selected):
                win.destroy()
                return

            self.push_history()
            for idx in selected:
                self.rects[idx][4] = to_idx
            self.combo_cls.current(to_idx)
            self.render()
            win.destroy()

        tk.Button(
            win,
            text=LANG_MAP[self.lang]["apply"],
            command=apply_change,
            bg=COLORS["primary"],
            fg=COLORS["text_white"],
            font=self.font_bold,
            relief="flat",
            pady=10
        ).pack(fill="x", padx=20, pady=(20, 16))

    def clear_current_labels(self):
        """???????????"""
        if not self.rects:
            return
        # Keep behavior identical to Ctrl+A then Delete.
        self.select_all_boxes()
        self.delete_selected()

    
    def _build_removed_path(self, kind, src_path):
        ext = os.path.splitext(src_path)[1]
        base = os.path.splitext(os.path.basename(src_path))[0]
        dst_dir = os.path.join(self.project_root, "removed", self.current_split, kind)
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, f"{base}{ext}")
        if not os.path.exists(dst_path):
            return dst_path

        i = 1
        while True:
            candidate = os.path.join(dst_dir, f"{base}_{i}{ext}")
            if not os.path.exists(candidate):
                return candidate
            i += 1

    def _unique_target_path(self, target_path):
        if not os.path.exists(target_path):
            return target_path
        folder = os.path.dirname(target_path)
        base, ext = os.path.splitext(os.path.basename(target_path))
        i = 1
        while True:
            candidate = os.path.join(folder, f"{base}_{i}{ext}")
            if not os.path.exists(candidate):
                return candidate
            i += 1

    def remove_current_from_split(self):
        if not self.image_files:
            messagebox.showinfo(
                LANG_MAP[self.lang]["title"],
                LANG_MAP[self.lang].get("remove_none", "No image to remove.")
            )
            return
        if not self.project_root:
            return

        confirm_msg = LANG_MAP[self.lang].get(
            "remove_confirm",
            "Remove current image from {split}?"
        ).format(split=self.current_split)
        if not messagebox.askyesno(LANG_MAP[self.lang]["title"], confirm_msg):
            return

        image_path = self.image_files[self.current_idx]
        image_name = os.path.basename(image_path)
        base = os.path.splitext(image_name)[0]

        try:
            moved_image_path = self._build_removed_path("images", image_path)
            shutil.move(image_path, moved_image_path)
            label_dir = os.path.join(self.project_root, "labels", self.current_split)
            for ext in (".txt", ".json"):
                label_path = os.path.join(label_dir, f"{base}{ext}")
                if os.path.exists(label_path):
                    moved_label_path = self._build_removed_path("labels", label_path)
                    shutil.move(label_path, moved_label_path)
                    if ext == ".txt":
                        rot_path = self._rotation_meta_path_for_label(label_path)
                        if os.path.exists(rot_path):
                            moved_rot_path = self._build_removed_path("labels", rot_path)
                            shutil.move(rot_path, moved_rot_path)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        del self.image_files[self.current_idx]
        if self.current_idx >= len(self.image_files):
            self.current_idx = max(0, len(self.image_files) - 1)

        if self.image_files:
            self.load_img()
        else:
            self.img_pil = None
            self.img_tk = None
            self.rects = []
            self.history_manager.clear()
            self.selected_idx = None
            self.selected_indices = set()
            self.active_handle = None
            self.active_rotate_handle = False
            self.rotate_drag_offset_deg = 0.0
            self.is_moving_box = False
            self.is_drag_selecting = False
            self.drag_start = None
            self.temp_rect_coords = None
            self.select_rect_coords = None
            self.update_info_text()
            self.render()

        self.save_session_state()
        done_msg = LANG_MAP[self.lang].get("remove_done", "Removed: {name}").format(name=image_name)
        messagebox.showinfo(LANG_MAP[self.lang]["title"], done_msg)

    def open_restore_removed_dialog(self):
        if not self.project_root:
            return

        removed_img_dir = os.path.join(self.project_root, "removed", self.current_split, "images")
        if not os.path.isdir(removed_img_dir):
            messagebox.showinfo(
                LANG_MAP[self.lang]["title"],
                LANG_MAP[self.lang].get("restore_none", "No removed frame found in this split.")
            )
            return

        removed_files = sorted([
            f for f in os.listdir(removed_img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        if not removed_files:
            messagebox.showinfo(
                LANG_MAP[self.lang]["title"],
                LANG_MAP[self.lang].get("restore_none", "No removed frame found in this split.")
            )
            return

        win = tk.Toplevel(self.root)
        win.title(LANG_MAP[self.lang].get("restore_title", "Restore Deleted Frame"))
        win.geometry("520x420")
        win.configure(bg=COLORS["bg_light"])

        tk.Label(
            win,
            text=LANG_MAP[self.lang].get("restore_select", "Select a frame to restore:"),
            font=self.font_primary,
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_light"],
            anchor="w"
        ).pack(fill="x", padx=16, pady=(16, 8))

        list_wrap = tk.Frame(win, bg=COLORS["bg_light"])
        list_wrap.pack(fill="both", expand=True, padx=16, pady=(0, 8))

        lb = tk.Listbox(
            list_wrap,
            font=self.font_mono,
            activestyle="none",
            selectmode="browse"
        )
        for name in removed_files:
            lb.insert("end", name)
        lb.pack(side="left", fill="both", expand=True)

        sb = ttk.Scrollbar(list_wrap, orient="vertical", command=lb.yview)
        sb.pack(side="right", fill="y")
        lb.config(yscrollcommand=sb.set)

        def do_restore():
            sel = lb.curselection()
            if not sel:
                return
            filename = lb.get(sel[0])
            self.restore_removed_file_by_name(filename)
            win.destroy()

        self.create_primary_button(
            win,
            text=LANG_MAP[self.lang].get("restore_from_split", "Restore Deleted Frame"),
            command=do_restore,
            bg=COLORS["success"]
        ).pack(fill="x", padx=16, pady=(0, 16))

    def restore_removed_file_by_name(self, filename):
        removed_img_path = os.path.join(self.project_root, "removed", self.current_split, "images", filename)
        if not os.path.exists(removed_img_path):
            return

        split_img_dir = os.path.join(self.project_root, "images", self.current_split)
        os.makedirs(split_img_dir, exist_ok=True)
        target_img_path = self._unique_target_path(os.path.join(split_img_dir, filename))

        base = os.path.splitext(filename)[0]
        removed_lbl_dir = os.path.join(self.project_root, "removed", self.current_split, "labels")
        split_lbl_dir = os.path.join(self.project_root, "labels", self.current_split)
        os.makedirs(split_lbl_dir, exist_ok=True)

        if self.image_files and self.img_pil:
            self.save_current()

        try:
            shutil.move(removed_img_path, target_img_path)
            for ext in (".txt", ".json"):
                removed_lbl_path = os.path.join(removed_lbl_dir, f"{base}{ext}")
                if os.path.exists(removed_lbl_path):
                    target_lbl_path = self._unique_target_path(
                        os.path.join(split_lbl_dir, f"{base}{ext}")
                    )
                    shutil.move(removed_lbl_path, target_lbl_path)
                    if ext == ".txt":
                        removed_rot_path = self._rotation_meta_path_for_label(removed_lbl_path)
                        if os.path.exists(removed_rot_path):
                            target_rot_path = self._rotation_meta_path_for_label(target_lbl_path)
                            target_rot_path = self._unique_target_path(target_rot_path)
                            shutil.move(removed_rot_path, target_rot_path)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self.load_split_data()
        if target_img_path in self.image_files:
            self.current_idx = self.image_files.index(target_img_path)
            self.load_img()

        msg = LANG_MAP[self.lang].get("restore_done", "Restored: {name}").format(name=os.path.basename(target_img_path))
        messagebox.showinfo(LANG_MAP[self.lang]["title"], msg)

    def load_img(self) -> None:
        """????????"""
        if not self.image_files:
            return
        
        path = self.image_files[self.current_idx]
        self.update_info_text()
        if self.img_pil is not None:
            self.img_pil.close()
            self.img_pil = None
        try:
            self.img_pil = Image.open(path)
        except Exception:
            self.logger.exception("Failed to load image: %s", path)
            messagebox.showerror("Error", f"Failed to open image:\n{path}")
            return
        
        prev_rects = copy.deepcopy(self.rects)
        prev_selected_indices = self._get_selected_indices()
        prev_selected_rects = [copy.deepcopy(self.rects[idx]) for idx in prev_selected_indices if 0 <= idx < len(self.rects)]
        self.rects = []
        self.history_manager.clear()
        self.selected_idx = None
        self.selected_indices = set()
        self.active_handle = None
        self.active_rotate_handle = False
        self.rotate_drag_offset_deg = 0.0
        self.is_moving_box = False
        self.is_drag_selecting = False
        self.drag_start = None
        self.temp_rect_coords = None
        self.select_rect_coords = None
        
        # ?????
        base = os.path.splitext(os.path.basename(path))[0]
        label_path = f"{self.project_root}/labels/{self.current_split}/{base}.txt"
        rot_meta_path = self._rotation_meta_path_for_label(label_path)
        
        label_exists = os.path.exists(label_path) and os.path.getsize(label_path) > 0
        propagate_mode = self.var_propagate_mode.get()
        should_propagate = False
        if self.var_propagate.get():
            if propagate_mode == "if_missing":
                should_propagate = not label_exists
            else:
                should_propagate = True

        if should_propagate:
            source_rects = prev_rects
            if propagate_mode == "selected":
                source_rects = prev_selected_rects
            self.rects = [self.clamp_box(copy.deepcopy(r)) for r in source_rects]

        if label_exists and not should_propagate:
            W, H = self.img_pil.width, self.img_pil.height
            has_inline_angle = False
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 5:
                            c, cx, cy, w, h = map(float, parts[:5])
                            angle_deg = float(parts[5]) if len(parts) >= 6 else 0.0
                            has_inline_angle = has_inline_angle or len(parts) >= 6
                            self.rects.append([
                                (cx - w / 2) * W,
                                (cy - h / 2) * H,
                                (cx + w / 2) * W,
                                (cy + h / 2) * H,
                                int(c),
                                self.normalize_angle_deg(angle_deg),
                            ])
                if self.rects and not has_inline_angle:
                    loaded_angles = self._read_rotation_meta_angles(rot_meta_path)
                    if loaded_angles and len(loaded_angles) == len(self.rects):
                        for rect, angle in zip(self.rects, loaded_angles):
                            self.set_rect_angle_deg(rect, angle)
            except Exception:
                self.logger.exception("Failed to parse label file: %s", label_path)
                messagebox.showerror("Error", f"Failed to read label file: {label_path}")
                self.rects = []
        elif not label_exists:
            if self.var_auto_yolo.get():
                self.run_yolo_detection()
        
        self.fit_image_to_canvas()
        self.save_session_state()
    
    def save_current(self) -> None:
        """?????????"""
        if not self.project_root or not self.img_pil:
            return
        
        path = self.image_files[self.current_idx]
        base = os.path.splitext(os.path.basename(path))[0]
        W, H = self.img_pil.width, self.img_pil.height
        
        label_path = f"{self.project_root}/labels/{self.current_split}/{base}.txt"
        rot_meta_path = self._rotation_meta_path_for_label(label_path)
        
        lines = []
        angles_deg = []
        for rect in self.rects:
            cx = (rect[0] + rect[2]) / 2 / W
            cy = (rect[1] + rect[3]) / 2 / H
            w = (rect[2] - rect[0]) / W
            h = (rect[3] - rect[1]) / H
            lines.append(f"{rect[4]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            angles_deg.append(self.get_rect_angle_deg(rect))
        if not lines:
            if os.path.exists(label_path):
                try:
                    os.remove(label_path)
                except OSError:
                    self.logger.exception("Failed to remove empty label file: %s", label_path)
            if os.path.exists(rot_meta_path):
                try:
                    os.remove(rot_meta_path)
                except OSError:
                    self.logger.exception("Failed to remove empty rotation meta file: %s", rot_meta_path)
            return
        try:
            atomic_write_text(label_path, "".join(lines))
            if any(abs(a) > 1e-3 for a in angles_deg):
                atomic_write_json(rot_meta_path, {"version": 1, "angles_deg": angles_deg})
            elif os.path.exists(rot_meta_path):
                os.remove(rot_meta_path)
        except Exception:
            self.logger.exception("Failed to save label file: %s", label_path)
            messagebox.showerror("Error", f"Failed to save label file:\n{label_path}")
            return

    def _reindex_dataset_labels_after_class_delete(self, deleted_idx: int) -> None:
        if not self.project_root:
            return

        label_files: list[str] = []
        split_roots = [s for s in ("train", "val", "test") if os.path.isdir(f"{self.project_root}/labels/{s}")]
        if split_roots:
            for split in split_roots:
                label_files.extend(glob.glob(f"{self.project_root}/labels/{split}/*.txt"))
        else:
            label_files.extend(glob.glob(f"{self.project_root}/labels/train/*.txt"))

        for lbl_path in label_files:
            try:
                with open(lbl_path, "r", encoding="utf-8") as f:
                    raw_lines = f.readlines()
            except Exception:
                self.logger.exception("Failed to read label file while deleting class: %s", lbl_path)
                continue

            updated_lines: list[str] = []
            for raw in raw_lines:
                line = raw.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    updated_lines.append(raw if raw.endswith("\n") else raw + "\n")
                    continue
                try:
                    cid = int(float(parts[0]))
                except ValueError:
                    updated_lines.append(raw if raw.endswith("\n") else raw + "\n")
                    continue

                if cid == deleted_idx:
                    continue
                if cid > deleted_idx:
                    parts[0] = str(cid - 1)
                    updated_lines.append(" ".join(parts) + "\n")
                else:
                    updated_lines.append(raw if raw.endswith("\n") else raw + "\n")

            if updated_lines:
                try:
                    atomic_write_text(lbl_path, "".join(updated_lines))
                except Exception:
                    self.logger.exception("Failed to write updated label file: %s", lbl_path)
            else:
                try:
                    if os.path.exists(lbl_path):
                        os.remove(lbl_path)
                except OSError:
                    self.logger.exception("Failed to remove empty label file: %s", lbl_path)
    
    def load_project_from_path(self, directory, preferred_image=None, save_session=True):
        self.project_root = directory.replace('\\', '/')
        self.ensure_yolo_label_dirs(self.project_root)

        progress = self._read_project_progress_yaml(self.project_root)
        progress_split = progress.get("split", "")
        progress_image = progress.get("image_name", "")
        progress_class_names = self._extract_class_names_from_progress(progress)

        # Always restore class names from project yaml when available.
        if progress_class_names:
            self.class_names[:] = progress_class_names
            self._refresh_class_dropdown()

        # For split/image resume, explicit session choice still has priority.
        if not preferred_image:
            if progress_split in {"train", "val", "test"}:
                self.current_split = progress_split
            if progress_image:
                preferred_image = progress_image

        split_files = {
            split: self._list_split_images(split)
            for split in ("train", "val", "test")
            if os.path.exists(f"{self.project_root}/images/{split}")
        }
        non_empty_splits = [s for s, files in split_files.items() if files]
        if non_empty_splits:
            if self.current_split not in non_empty_splits:
                self.current_split = "train" if "train" in non_empty_splits else non_empty_splits[0]
        elif split_files:
            if self.current_split not in split_files:
                self.current_split = "train" if "train" in split_files else next(iter(split_files))
        img_path = f"{self.project_root}/images/{self.current_split}"

        if not os.path.exists(img_path):
            os.makedirs(f"{self.project_root}/labels/{self.current_split}", exist_ok=True)

        self.combo_split.set(self.current_split)
        self.load_split_data(preferred_image=preferred_image)
        if save_session:
            self.save_session_state()

    def load_project_root(self):
        """????"""
        directory = filedialog.askdirectory()
        if not directory:
            return
        normalized = self.normalize_project_root(directory)
        yolo_root = self.find_yolo_project_root(normalized)
        self.load_project_from_path(yolo_root or normalized)
    
    def on_split_change(self, e=None):
        """??????"""
        if self.project_root:
            self.save_current()
            self.current_split = self.combo_split.get()
            self.load_split_data()
            self.save_session_state()
    
    def load_split_data(self, preferred_image=None):
        """??????"""
        img_path = f"{self.project_root}/images/{self.current_split}"

        if self.project_root and not os.path.exists(img_path):
            fallback = next(
                (s for s in ("train", "val", "test") if os.path.exists(f"{self.project_root}/images/{s}")),
                None,
            )
            if fallback:
                self.current_split = fallback
                self.combo_split.set(self.current_split)
                img_path = f"{self.project_root}/images/{self.current_split}"

        if self.project_root and os.path.exists(img_path):
            current_files = self._list_split_images(self.current_split)
            if not current_files:
                fallback_non_empty = next(
                    (
                        s for s in ("train", "val", "test")
                        if os.path.exists(f"{self.project_root}/images/{s}") and self._list_split_images(s)
                    ),
                    None,
                )
                if fallback_non_empty and fallback_non_empty != self.current_split:
                    self.current_split = fallback_non_empty
                    self.combo_split.set(self.current_split)
                    img_path = f"{self.project_root}/images/{self.current_split}"
        
        if os.path.exists(img_path):
            self.image_files = self._list_split_images(self.current_split)
            if self.image_files:
                if preferred_image:
                    name_to_idx = {
                        os.path.basename(path): i
                        for i, path in enumerate(self.image_files)
                    }
                    self.current_idx = name_to_idx.get(preferred_image, 0)
                else:
                    self.current_idx = 0
                self.load_img()
            else:
                self.current_idx = 0
                self.img_pil = None
                self.img_tk = None
                self.rects = []
                self.update_info_text()
        else:
            self.image_files = []
            self.current_idx = 0
            self.img_pil = None
            self.img_tk = None
            self.rects = []
            self.update_info_text()
        
        self.render()
        self.save_session_state()

    def _list_split_images(self, split: str) -> list[str]:
        img_path = f"{self.project_root}/images/{split}"
        return sorted([
            f for f in glob.glob(f"{img_path}/*.*")
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
    
    def autolabel_red(self) -> None:
        """Auto-label reddish regions using LAB + contour filtering."""
        if not HAS_CV2 or not self.img_pil:
            return
        try:
            self.push_history()

            img = cv2.cvtColor(np.array(self.img_pil), cv2.COLOR_RGB2BGR)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            _, a, _ = cv2.split(lab)
            _, red = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            k = max(1, int(self.config.red_detection_kernel_size))
            kernel = np.ones((k, k), np.uint8)
            red = cv2.dilate(
                red,
                kernel,
                iterations=max(1, int(self.config.red_detection_dilate_iterations)),
            )

            contours, _ = cv2.findContours(
                red,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w * h > self.config.auto_label_min_area:
                    self.rects.append(self.clamp_box([
                        x, y, x + w, y + h,
                        self.combo_cls.current()
                    ]))

            self.render()
        except Exception:
            self.logger.exception("Red auto-label failed")
            messagebox.showerror("Error", "Red auto-label failed. See logs for details.")
    
    def run_yolo_detection(self):
        """Run detection by the selected model and append boxes."""
        if not HAS_YOLO:
            messagebox.showwarning("YOLO Not Available", "Please install ultralytics first.")
            return
        if not self.img_pil:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        
        try:
            mode = self.det_model_mode.get()
            if mode == "Official YOLO26m.pt (Bundled)":
                model_path = self._resolve_official_model_path()
            else:
                model_path = self._resolve_custom_model_path(self.yolo_path.get().strip())

            if not model_path:
                messagebox.showwarning("Model", "Please choose a model file first.")
                return
            model_path = os.path.abspath(model_path)
            self.yolo_path.set(model_path)
            self._register_model_path(model_path)

            loaded_key = (mode, model_path)
            if self.yolo_model is None or self._loaded_model_key != loaded_key:
                self.root.config(cursor="watch")
                self.root.update_idletasks()
                if self.yolo_model is not None:
                    del self.yolo_model
                    self.yolo_model = None
                    gc.collect()
                try:
                    self.logger.info("Loading YOLO model: %s", model_path)
                    self.yolo_model = YOLO(model_path)
                    self._loaded_model_key = loaded_key
                except Exception as exc:
                    self.yolo_model = None
                    self._loaded_model_key = None
                    raise RuntimeError(f"Failed to load model: {exc}") from exc
                finally:
                    self.root.config(cursor="")
                    self.root.update_idletasks()

            preferred_device = "cpu"
            if not self._force_cpu_detection and self._can_use_cuda_runtime():
                preferred_device = 0

            try:
                results = self.yolo_model(
                    self.img_pil,
                    conf=self.var_yolo_conf.get(),
                    verbose=False,
                    device=preferred_device,
                )
            except RuntimeError as exc:
                if preferred_device != "cpu" and self._is_cuda_kernel_compat_error(exc):
                    self.logger.warning(
                        "CUDA kernel compatibility error detected; retrying YOLO detection on CPU. error=%s",
                        exc,
                    )
                    self._force_cpu_detection = True
                    results = self.yolo_model(
                        self.img_pil,
                        conf=self.var_yolo_conf.get(),
                        verbose=False,
                        device="cpu",
                    )
                else:
                    raise
            
            self.push_history()
            detection_count = 0
            fallback_class_idx = self.combo_cls.current()
            if fallback_class_idx < 0:
                fallback_class_idx = 0
            for result in results:
                if result.boxes is None:
                    continue
                for det_idx, box in enumerate(result.boxes.xyxy):
                    class_idx = self._resolve_detected_class_index(result, det_idx, fallback_class_idx)
                    self.rects.append(self.clamp_box([
                        box[0].item(),
                        box[1].item(),
                        box[2].item(),
                        box[3].item(),
                        class_idx
                    ]))
                    detection_count += 1
            
            self.render()
            self.logger.info("YOLO detection complete: %s boxes", detection_count)
        except FileNotFoundError as exc:
            self.logger.error("Model path error: %s", exc)
            messagebox.showerror("Model Error", str(exc))
        except MemoryError:
            self.logger.exception("Out of memory during YOLO detection")
            messagebox.showerror("Memory Error", "Not enough memory for detection.")
        except RuntimeError as exc:
            self.logger.exception("YOLO runtime error")
            messagebox.showerror("Detection Error", str(exc))
        except Exception as exc:
            self.logger.exception("YOLO detection failed")
            messagebox.showerror("Detection Error", f"YOLO detection failed:\n{exc}")

    def _is_cuda_kernel_compat_error(self, exc: BaseException) -> bool:
        msg = str(exc).lower()
        return (
            "no kernel image is available for execution on the device" in msg
            or "cudaerrornokernelimagefordevice" in msg
        )

    def _can_use_cuda_runtime(self) -> bool:
        """Best-effort CUDA runtime compatibility check for current torch build."""
        if "torch" not in globals():
            return False
        try:
            if not torch.cuda.is_available():
                return False
            cap = torch.cuda.get_device_capability(0)
            sm = f"sm_{cap[0]}{cap[1]}"
            arch_list = []
            if hasattr(torch.cuda, "get_arch_list"):
                arch_list = list(torch.cuda.get_arch_list() or [])
            if arch_list and sm not in arch_list:
                return False
            # Trigger a minimal CUDA op to fail fast on incompatible kernels.
            _ = torch.zeros((1,), device="cuda")
            return True
        except Exception:
            return False

    def _list_split_labeled_images_for_root(self, project_root: str, split: str) -> list[str]:
        labeled: list[str] = []
        for img_path in self._list_split_images_for_root(project_root, split):
            base = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = f"{project_root}/labels/{split}/{base}.txt"
            if os.path.isfile(lbl_path) and os.path.getsize(lbl_path) > 0:
                labeled.append(img_path)
        return labeled

    def _list_flat_labeled_images_for_root(self, project_root: str) -> list[str]:
        labeled: list[str] = []
        for img_path in sorted(
            f for f in glob.glob(f"{project_root}/*.*")
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ):
            base = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = f"{project_root}/labels/train/{base}.txt"
            if os.path.isfile(lbl_path) and os.path.getsize(lbl_path) > 0:
                labeled.append(img_path)
        return labeled

    def _write_training_dataset_files(
        self,
        out_dir: str,
        train_images: list[str],
        val_images: list[str],
        train_split: str,
        val_split: str,
        range_start: int,
        range_end: int,
    ) -> str:
        out_dir = out_dir.replace("\\", "/")
        train_txt = f"{out_dir}/train_images.txt"
        val_txt = f"{out_dir}/val_images.txt"
        dataset_yaml = f"{out_dir}/dataset.yaml"
        manifest_json = f"{out_dir}/training_manifest.json"

        atomic_write_text(train_txt, "".join(f"{p.replace('\\', '/')}\n" for p in train_images))
        atomic_write_text(val_txt, "".join(f"{p.replace('\\', '/')}\n" for p in val_images))

        yaml_lines = [
            f"train: {train_txt}",
            f"val: {val_txt}",
            f"nc: {len(self.class_names)}",
            "names:",
        ]
        for idx, cls_name in enumerate(self.class_names):
            safe_name = cls_name.replace("\"", "\\\"")
            yaml_lines.append(f"  {idx}: \"{safe_name}\"")
        atomic_write_text(dataset_yaml, "\n".join(yaml_lines) + "\n")

        atomic_write_json(
            manifest_json,
            {
                "project_root": self.project_root,
                "train_split": train_split,
                "val_split": val_split,
                "range_start_1based": range_start,
                "range_end_1based": range_end,
                "train_count": len(train_images),
                "val_count": len(val_images),
            },
        )
        return dataset_yaml

    def _append_training_log(self, line: str) -> None:
        if hasattr(self, "txt_train_log"):
            self.txt_train_log.insert("end", line.rstrip() + "\n")
            self.txt_train_log.see("end")

    def _set_training_status(self, running: bool) -> None:
        if not hasattr(self, "lbl_train_status"):
            return
        status_text = LANG_MAP[self.lang].get("train_running", "Running") if running else LANG_MAP[self.lang].get("train_idle", "Idle")
        self.lbl_train_status.config(
            text=f"{LANG_MAP[self.lang].get('train_status', 'Status')}: {status_text}"
        )

    def _set_training_progress(self, current_epoch: int, total_epochs: int) -> None:
        self.training_current_epoch = current_epoch
        self.training_total_epochs = total_epochs
        if hasattr(self, "lbl_train_progress"):
            self.lbl_train_progress.config(
                text=f"{LANG_MAP[self.lang].get('train_progress', 'Progress')}: {current_epoch}/{total_epochs}"
            )

    def _format_eta_seconds(self, seconds_left: float) -> str:
        seconds = max(0, int(seconds_left))
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _set_training_eta(self, eta_text: str) -> None:
        if hasattr(self, "lbl_train_eta"):
            self.lbl_train_eta.config(text=f"{LANG_MAP[self.lang].get('train_eta', 'ETA')}: {eta_text}")

    def _handle_training_output_line(self, line: str) -> None:
        self.training_queue.put(("log", line))
        if self.training_total_epochs <= 0:
            return
        match = re.search(r"(^|\s)(\d{1,4})/(\d{1,4})(\s|$)", line)
        if not match:
            return
        current = int(match.group(2))
        total = int(match.group(3))
        if total != self.training_total_epochs or current <= 0:
            return
        if self.training_start_time is not None:
            elapsed = max(1.0, time.time() - self.training_start_time)
            eta = (elapsed / current) * max(0, total - current)
            self.training_queue.put(("progress", current, total, self._format_eta_seconds(eta)))
        else:
            self.training_queue.put(("progress", current, total, "-"))

    def _run_training_subprocess(self, cmd: list[str], workdir: str) -> None:
        try:
            self.training_process = subprocess.Popen(
                cmd,
                cwd=workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            if self.training_process.stdout is not None:
                for line in self.training_process.stdout:
                    self._handle_training_output_line(line)
            rc = self.training_process.wait()
            if rc == 0:
                self.training_queue.put(("done",))
            else:
                self.training_queue.put(("error", f"Process exited with code {rc}"))
        except Exception as exc:
            self.training_queue.put(("error", str(exc)))
        finally:
            self.training_process = None

    def _poll_training_queue(self) -> None:
        keep_polling = self.training_running or not self.training_queue.empty()
        while not self.training_queue.empty():
            event = self.training_queue.get_nowait()
            kind = event[0]
            if kind == "log":
                self._append_training_log(event[1])
            elif kind == "progress":
                current, total, eta_text = event[1], event[2], event[3]
                self._set_training_progress(current, total)
                self._set_training_eta(eta_text)
            elif kind == "done":
                self.training_running = False
                self._set_training_status(False)
                self._set_training_eta("00:00")
                if self.training_thread is not None:
                    self.training_thread = None
                output_path = getattr(self, "_last_training_output_path", "")
                messagebox.showinfo(
                    LANG_MAP[self.lang]["title"],
                    LANG_MAP[self.lang].get("train_done", "Training finished.\nOutput: {path}").format(path=output_path),
                    parent=self.root,
                )
            elif kind == "error":
                self.training_running = False
                self._set_training_status(False)
                if self.training_thread is not None:
                    self.training_thread = None
                err = event[1]
                self.logger.error("Training process failed: %s", err)
                messagebox.showerror(
                    LANG_MAP[self.lang]["title"],
                    LANG_MAP[self.lang].get("train_failed", "Training failed: {err}").format(err=err),
                    parent=self.root,
                )
        if keep_polling:
            self.root.after(200, self._poll_training_queue)

    def _resolve_yolo_cli(self) -> str:
        py_dir = os.path.dirname(sys.executable)
        candidates = [
            os.path.join(py_dir, "Scripts", "yolo.exe"),
            os.path.join(py_dir, "Scripts", "yolo"),
            os.path.join(py_dir, "yolo.exe"),
            os.path.join(py_dir, "yolo"),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        found = shutil.which("yolo")
        if found:
            return found
        raise FileNotFoundError("YOLO CLI not found. Please ensure ultralytics is installed in this Python environment.")

    def _prompt_training_weight_source(self) -> tuple[str, str | None] | None:
        """Return ('official'|'custom'|'scratch', model_path_or_none). None means cancelled."""
        result: dict[str, str | None] = {"choice": None, "path": None}
        win = tk.Toplevel(self.root)
        win.title("Training Weights")
        win.geometry("440x210")
        win.resizable(False, False)
        win.configure(bg=COLORS["bg_light"])
        win.transient(self.root)
        win.grab_set()

        tk.Label(
            win,
            text="Choose training weight source:",
            font=self.font_bold,
            fg=COLORS["text_primary"],
            bg=COLORS["bg_light"],
            anchor="w",
        ).pack(fill="x", padx=18, pady=(18, 10))

        def choose_official() -> None:
            result["choice"] = "official"
            win.destroy()

        def choose_custom() -> None:
            model_path = filedialog.askopenfilename(
                parent=win,
                title="Select custom weight",
                filetypes=[
                    ("Model files", "*.pt *.onnx"),
                    ("PyTorch", "*.pt"),
                    ("ONNX", "*.onnx"),
                    ("All files", "*.*"),
                ],
            )
            if not model_path:
                return
            try:
                resolved = self._resolve_custom_model_path(model_path)
            except FileNotFoundError as exc:
                messagebox.showerror("Model Error", str(exc), parent=win)
                return
            result["choice"] = "custom"
            result["path"] = os.path.abspath(resolved)
            win.destroy()

        def choose_scratch() -> None:
            result["choice"] = "scratch"
            win.destroy()

        self.create_primary_button(
            win,
            text="Use Official yolo26m.pt",
            command=choose_official,
            bg=COLORS["primary"],
        ).pack(fill="x", padx=18, pady=(0, 8))
        self.create_primary_button(
            win,
            text="Choose Custom Weight",
            command=choose_custom,
            bg=COLORS["success"],
        ).pack(fill="x", padx=18, pady=(0, 8))
        self.create_secondary_button(
            win,
            text="From Scratch (skip pretrained weight)",
            command=choose_scratch,
        ).pack(fill="x", padx=18, pady=(0, 8))
        self.create_secondary_button(
            win,
            text="Cancel",
            command=win.destroy,
        ).pack(fill="x", padx=18, pady=(0, 12))

        win.protocol("WM_DELETE_WINDOW", win.destroy)
        self.root.wait_window(win)
        choice = result["choice"]
        if not choice:
            return None
        return str(choice), result["path"]

    def start_training_from_labels(self) -> None:
        if not HAS_YOLO:
            messagebox.showwarning("YOLO Not Available", "Please install ultralytics first.")
            return
        if self.training_running:
            messagebox.showinfo(
                LANG_MAP[self.lang]["title"],
                LANG_MAP[self.lang].get("train_already_running", "Training is already running."),
                parent=self.root,
            )
            return
        if not self.project_root:
            messagebox.showwarning(
                LANG_MAP[self.lang]["title"],
                LANG_MAP[self.lang].get("train_no_project", "No dataset loaded."),
                parent=self.root,
            )
            return
        if self.image_files and self.img_pil:
            self.save_current()

        split_roots = [s for s in ("train", "val", "test") if os.path.isdir(f"{self.project_root}/images/{s}")]
        if split_roots:
            train_split = self.current_split if self.current_split in split_roots else split_roots[0]
            train_candidates = self._list_split_labeled_images_for_root(self.project_root, train_split)
            if not train_candidates:
                messagebox.showwarning(
                    LANG_MAP[self.lang]["title"],
                    LANG_MAP[self.lang].get("train_no_labels", "No labeled images found for training."),
                    parent=self.root,
                )
                return
            val_split = "val" if "val" in split_roots else train_split
            val_candidates = self._list_split_labeled_images_for_root(self.project_root, val_split)
        else:
            train_split = "train"
            val_split = "train"
            train_candidates = self._list_flat_labeled_images_for_root(self.project_root)
            val_candidates = []
            if not train_candidates:
                messagebox.showwarning(
                    LANG_MAP[self.lang]["title"],
                    LANG_MAP[self.lang].get("train_no_labels", "No labeled images found for training."),
                    parent=self.root,
                )
                return

        max_idx = len(train_candidates)
        start_idx = simpledialog.askinteger(
            LANG_MAP[self.lang].get("train_from_labels", "Train From Labels"),
            f"{LANG_MAP[self.lang].get('train_range_start', 'Start Index (1-based)')}\n1 - {max_idx}",
            parent=self.root,
            minvalue=1,
            maxvalue=max_idx,
            initialvalue=1,
        )
        if start_idx is None:
            return
        end_idx = simpledialog.askinteger(
            LANG_MAP[self.lang].get("train_from_labels", "Train From Labels"),
            f"{LANG_MAP[self.lang].get('train_range_end', 'End Index (1-based)')}\n{start_idx} - {max_idx}",
            parent=self.root,
            minvalue=start_idx,
            maxvalue=max_idx,
            initialvalue=max_idx,
        )
        if end_idx is None:
            return
        if start_idx > end_idx:
            messagebox.showwarning(
                LANG_MAP[self.lang]["title"],
                LANG_MAP[self.lang].get("train_bad_range", "Invalid range. Please input valid start/end index."),
                parent=self.root,
            )
            return

        epochs = simpledialog.askinteger(
            LANG_MAP[self.lang].get("train_from_labels", "Train From Labels"),
            LANG_MAP[self.lang].get("train_epochs", "Epochs"),
            parent=self.root,
            minvalue=1,
            maxvalue=1000,
            initialvalue=50,
        )
        if epochs is None:
            return
        imgsz = simpledialog.askinteger(
            LANG_MAP[self.lang].get("train_from_labels", "Train From Labels"),
            LANG_MAP[self.lang].get("train_imgsz", "Image Size"),
            parent=self.root,
            minvalue=64,
            maxvalue=4096,
            initialvalue=640,
        )
        if imgsz is None:
            return

        out_dir = filedialog.askdirectory(
            parent=self.root,
            title=LANG_MAP[self.lang].get("select_train_output", "Select Training Output Folder"),
        )
        if not out_dir:
            return
        out_dir = out_dir.replace("\\", "/")

        weight_choice = self._prompt_training_weight_source()
        if weight_choice is None:
            return

        selected_train = train_candidates[start_idx - 1:end_idx]
        if not selected_train:
            messagebox.showwarning(
                LANG_MAP[self.lang]["title"],
                LANG_MAP[self.lang].get("train_no_labels", "No labeled images found for training."),
                parent=self.root,
            )
            return
        if val_candidates:
            selected_val = val_candidates
        else:
            val_count = max(1, int(len(selected_train) * 0.2))
            if len(selected_train) <= 1:
                selected_val = selected_train[:]
            else:
                selected_val = selected_train[-val_count:]
                selected_train = selected_train[:-val_count]
                if not selected_train:
                    selected_train = selected_val[:]

        try:
            choice, custom_path = weight_choice
            extra_train_args: list[str] = []
            if choice == "official":
                model_path = self._resolve_official_model_path()
            elif choice == "custom":
                if not custom_path:
                    messagebox.showerror("Model Error", "No custom model selected.", parent=self.root)
                    return
                model_path = self._resolve_custom_model_path(custom_path)
            else:
                # Scratch mode: use official architecture entry and disable pretrained weights.
                model_path = self._resolve_official_model_path()
                extra_train_args.append("pretrained=False")
            model_path = os.path.abspath(model_path)
            self.yolo_path.set(model_path)
            self._register_model_path(model_path)

            os.makedirs(out_dir, exist_ok=True)
            dataset_yaml = self._write_training_dataset_files(
                out_dir=out_dir,
                train_images=selected_train,
                val_images=selected_val,
                train_split=train_split,
                val_split=val_split,
                range_start=start_idx,
                range_end=end_idx,
            )
            run_name = f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            yolo_cli = self._resolve_yolo_cli()
            cmd = [
                yolo_cli,
                "train",
                f"model={model_path}",
                f"data={dataset_yaml}",
                f"epochs={epochs}",
                f"imgsz={imgsz}",
                f"project={out_dir}",
                f"name={run_name}",
                "exist_ok=True",
            ]
            train_device = "0" if self._can_use_cuda_runtime() else "cpu"
            cmd.append(f"device={train_device}")
            cmd.extend(extra_train_args)
            command_text = " ".join(f"\"{part}\"" if " " in part else part for part in cmd)
            self.train_command_var.set(command_text)
            self._append_training_log("=" * 60)
            self._append_training_log(command_text)
            self._append_training_log("=" * 60)
            self._set_training_status(True)
            self._set_training_progress(0, epochs)
            self._set_training_eta("-")
            self.training_running = True
            self.training_start_time = time.time()
            self.training_total_epochs = epochs
            self.training_current_epoch = 0
            self._last_training_output_path = f"{out_dir}/{run_name}"
            self.training_thread = threading.Thread(
                target=self._run_training_subprocess,
                args=(cmd, self.project_root),
                daemon=True,
            )
            self.training_thread.start()
            self._poll_training_queue()
        except Exception as exc:
            self.logger.exception("Training from labels failed")
            messagebox.showerror(
                LANG_MAP[self.lang]["title"],
                LANG_MAP[self.lang].get("train_failed", "Training failed: {err}").format(err=exc),
                parent=self.root,
            )

    def _iter_export_images(self) -> list[tuple[str, str, str]]:
        """Return (split, image_path, label_path) entries for full-dataset export."""
        entries: list[tuple[str, str, str]] = []
        if not self.project_root:
            return entries

        split_roots = [s for s in ("train", "val", "test") if os.path.isdir(f"{self.project_root}/images/{s}")]
        if split_roots:
            for split in split_roots:
                for img_path in self._list_split_images_for_root(self.project_root, split):
                    base = os.path.splitext(os.path.basename(img_path))[0]
                    lbl_path = f"{self.project_root}/labels/{split}/{base}.txt"
                    entries.append((split, img_path, lbl_path))
            return entries

        # Flat image folder mode
        for img_path in sorted(
            f for f in glob.glob(f"{self.project_root}/*.*")
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ):
            base = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = f"{self.project_root}/labels/train/{base}.txt"
            entries.append(("train", img_path, lbl_path))
        return entries

    def export_all_by_selected_format(self) -> None:
        if not self.project_root:
            messagebox.showwarning(LANG_MAP[self.lang]["title"], LANG_MAP[self.lang]["export_no_project"], parent=self.root)
            return
        if self.image_files and self.img_pil:
            self.save_current()

        out_dir = filedialog.askdirectory(
            parent=self.root,
            title=LANG_MAP[self.lang].get("select_export_folder", "Select Export Folder")
        )
        if not out_dir:
            return
        out_dir = out_dir.replace("\\", "/")

        fmt = self.var_export_format.get()
        try:
            if fmt == "YOLO (.txt)":
                count = self._export_all_yolo(out_dir)
            else:
                count = self._export_all_json(out_dir)
            messagebox.showinfo(
                LANG_MAP[self.lang]["title"],
                LANG_MAP[self.lang].get("export_done", "Export completed: {count} images\nOutput: {path}").format(
                    count=count,
                    path=out_dir,
                ),
                parent=self.root,
            )
        except Exception as exc:
            self.logger.exception("Export failed")
            messagebox.showerror(
                LANG_MAP[self.lang]["title"],
                LANG_MAP[self.lang].get("export_failed", "Export failed: {err}").format(err=exc),
                parent=self.root,
            )

    def _export_all_yolo(self, out_dir: str) -> int:
        count = 0
        dst_img_dir = f"{out_dir}/images/train"
        dst_lbl_dir = f"{out_dir}/labels/train"
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)

        def build_unique_stem(split: str, stem: str, ext: str) -> str:
            candidate = stem
            target_img = f"{dst_img_dir}/{candidate}{ext}"
            if not os.path.exists(target_img):
                return candidate
            candidate = f"{split}_{stem}"
            target_img = f"{dst_img_dir}/{candidate}{ext}"
            if not os.path.exists(target_img):
                return candidate
            i = 1
            while os.path.exists(f"{dst_img_dir}/{candidate}_{i}{ext}"):
                i += 1
            return f"{candidate}_{i}"

        for split, img_path, lbl_path in self._iter_export_images():
            img_name = os.path.basename(img_path)
            stem, ext = os.path.splitext(img_name)
            target_stem = build_unique_stem(split, stem, ext)
            shutil.copy2(img_path, f"{dst_img_dir}/{target_stem}{ext}")
            if os.path.isfile(lbl_path):
                shutil.copy2(lbl_path, f"{dst_lbl_dir}/{target_stem}.txt")
                rot_meta_path = self._rotation_meta_path_for_label(lbl_path)
                if os.path.isfile(rot_meta_path):
                    shutil.copy2(rot_meta_path, self._rotation_meta_path_for_label(f"{dst_lbl_dir}/{target_stem}.txt"))
            count += 1
        return count

    def _export_all_json(self, out_dir: str) -> int:
        count = 0
        for split, img_path, lbl_path in self._iter_export_images():
            dst_img_dir = f"{out_dir}/images/{split}"
            dst_ann_dir = f"{out_dir}/annotations/{split}"
            os.makedirs(dst_img_dir, exist_ok=True)
            os.makedirs(dst_ann_dir, exist_ok=True)

            img_name = os.path.basename(img_path)
            base = os.path.splitext(img_name)[0]
            shutil.copy2(img_path, f"{dst_img_dir}/{img_name}")

            with Image.open(img_path) as im:
                width, height = im.width, im.height

            anns: list[dict[str, Any]] = []
            angles_from_meta: list[float] = []
            if os.path.isfile(self._rotation_meta_path_for_label(lbl_path)):
                angles_from_meta = self._read_rotation_meta_angles(self._rotation_meta_path_for_label(lbl_path)) or []
            ann_idx = 0
            if os.path.isfile(lbl_path):
                with open(lbl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls_id = int(float(parts[0]))
                        cx, cy, w, h = map(float, parts[1:5])
                        angle_deg = float(parts[5]) if len(parts) >= 6 else 0.0
                        if len(parts) < 6 and ann_idx < len(angles_from_meta):
                            angle_deg = float(angles_from_meta[ann_idx])
                        x1 = (cx - w / 2) * width
                        y1 = (cy - h / 2) * height
                        x2 = (cx + w / 2) * width
                        y2 = (cy + h / 2) * height
                        cls_name = self.class_names[cls_id] if 0 <= cls_id < len(self.class_names) else str(cls_id)
                        anns.append({
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "bbox_xyxy": [x1, y1, x2, y2],
                            "bbox_yolo": [cx, cy, w, h],
                            "angle_deg": self.normalize_angle_deg(angle_deg),
                        })
                        ann_idx += 1

            payload = {
                "image": img_name,
                "split": split,
                "width": width,
                "height": height,
                "annotations": anns,
            }
            atomic_write_json(f"{dst_ann_dir}/{base}.json", payload)
            count += 1
        return count
    
    def export_full_coco(self):
        """Export dataset as COCO (placeholder)."""
        messagebox.showinfo("Export", "COCO export will be implemented.")
    
    # ==================== Geometry ====================
    
    def clamp_box(self, box):
        """Clamp and normalize box coordinates to image bounds."""
        if not self.img_pil:
            return box
        
        W, H = self.img_pil.width, self.img_pil.height
        x1, x2 = sorted([box[0], box[2]])
        y1, y2 = sorted([box[1], box[3]])
        
        out = [
            max(0, min(W, x1)),
            max(0, min(H, y1)),
            max(0, min(W, x2)),
            max(0, min(H, y2)),
            box[4],
        ]
        if len(box) >= 6:
            out.append(self.normalize_angle_deg(float(box[5])))
        else:
            out.append(0.0)
        return out

    def normalize_angle_deg(self, angle_deg: float) -> float:
        return ((angle_deg + 180.0) % 360.0) - 180.0

    def get_rect_angle_deg(self, rect: list[float]) -> float:
        if len(rect) >= 6:
            return self.normalize_angle_deg(float(rect[5]))
        return 0.0

    def set_rect_angle_deg(self, rect: list[float], angle_deg: float) -> None:
        normalized = self.normalize_angle_deg(float(angle_deg))
        if len(rect) >= 6:
            rect[5] = normalized
        else:
            rect.append(normalized)

    def rotate_point_around_center(
        self,
        x: float,
        y: float,
        cx: float,
        cy: float,
        angle_deg: float,
    ) -> tuple[float, float]:
        theta = math.radians(angle_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        dx = x - cx
        dy = y - cy
        return cx + dx * cos_t - dy * sin_t, cy + dx * sin_t + dy * cos_t

    def get_rotated_corners(self, rect: list[float]) -> list[tuple[float, float]]:
        x1 = min(rect[0], rect[2])
        y1 = min(rect[1], rect[3])
        x2 = max(rect[0], rect[2])
        y2 = max(rect[1], rect[3])
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        angle_deg = self.get_rect_angle_deg(rect)

        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        if abs(angle_deg) <= 1e-6:
            return corners
        return [self.rotate_point_around_center(px, py, cx, cy, angle_deg) for px, py in corners]

    def _point_in_rotated_box(self, x: float, y: float, rect: list[float]) -> bool:
        x1 = min(rect[0], rect[2])
        y1 = min(rect[1], rect[3])
        x2 = max(rect[0], rect[2])
        y2 = max(rect[1], rect[3])
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        angle_deg = self.get_rect_angle_deg(rect)

        # Rotate test point back into axis-aligned box local frame.
        px, py = self.rotate_point_around_center(x, y, cx, cy, -angle_deg)
        return x1 < px < x2 and y1 < py < y2

    def _rotation_meta_path_for_label(self, label_path: str) -> str:
        return f"{label_path}.rot.json"

    def _read_rotation_meta_angles(self, rot_meta_path: str) -> list[float] | None:
        if not os.path.isfile(rot_meta_path):
            return None
        try:
            with open(rot_meta_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            raw_angles = payload.get("angles_deg", [])
            if not isinstance(raw_angles, list):
                return None
            out: list[float] = []
            for angle in raw_angles:
                out.append(self.normalize_angle_deg(float(angle)))
            return out
        except Exception:
            self.logger.exception("Failed to read rotation meta: %s", rot_meta_path)
            return None
    
    def get_handles(self, rect):
        """??????????"""
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
        angle_deg = self.get_rect_angle_deg(rect)
        if abs(angle_deg) <= 1e-6:
            return local_handles
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return [self.rotate_point_around_center(px, py, cx, cy, angle_deg) for px, py in local_handles]

    def get_rotation_handle_points(self, rect: list[float]) -> tuple[float, float, float, float]:
        x1 = min(rect[0], rect[2])
        y1 = min(rect[1], rect[3])
        x2 = max(rect[0], rect[2])
        y2 = max(rect[1], rect[3])
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        angle_deg = self.get_rect_angle_deg(rect)

        top_center_local = ((x1 + x2) / 2, y1)
        top_x, top_y = self.rotate_point_around_center(
            top_center_local[0], top_center_local[1], cx, cy, angle_deg
        )
        vx = top_x - cx
        vy = top_y - cy
        vlen = math.hypot(vx, vy)
        if vlen <= 1e-6:
            vx, vy = 0.0, -1.0
            vlen = 1.0
        ux = vx / vlen
        uy = vy / vlen
        stem_len_img = max(10.0, 26.0 / max(self.scale, 1e-6))
        rot_x = top_x + ux * stem_len_img
        rot_y = top_y + uy * stem_len_img
        return top_x, top_y, rot_x, rot_y
    
    def canvas_to_img(self, x, y):
        """Canvas ???????????????"""
        return (x - self.offset_x) / self.scale, (y - self.offset_y) / self.scale
    
    def img_to_canvas(self, x, y):
        """????????????Canvas ??????"""
        return x * self.scale + self.offset_x, y * self.scale + self.offset_y
    
    def push_history(self) -> None:
        """??????"""
        self.history_manager.push_snapshot(self.rects)
    
    def undo(self) -> None:
        """Undo last edit."""
        if self.history_manager.undo():
            if self.project_root and self.img_pil:
                self.save_current()
            self.render()
    
    def redo(self) -> None:
        """??"""
        if self.history_manager.redo():
            if self.project_root and self.img_pil:
                self.save_current()
            self.render()
    
    def save_and_next(self):
        """??????????????????"""
        self.save_current()
        self.current_idx = min(len(self.image_files) - 1, self.current_idx + 1)
        self.load_img()
    
    def prev_img(self):
        """????"""
        self.save_current()
        self.current_idx = max(0, self.current_idx - 1)
        self.load_img()
    
    def bind_events(self):
        """Bind mouse and keyboard shortcuts."""
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<MouseWheel>", self.on_zoom)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        
        self.root.bind("<Key-f>", lambda e: self.save_and_next())
        self.root.bind("<Key-F>", lambda e: self.save_and_next())
        self.root.bind("<Key-d>", lambda e: self.prev_img())
        self.root.bind("<Key-D>", lambda e: self.prev_img())
        self.root.bind("<Key-a>", lambda e: self.autolabel_red())
        self.root.bind("<Key-A>", lambda e: self.autolabel_red())
        self.root.bind("<Key-q>", lambda e: self.rotate_selected_boxes(-5.0))
        self.root.bind("<Key-Q>", lambda e: self.rotate_selected_boxes(-15.0))
        self.root.bind("<Key-e>", lambda e: self.rotate_selected_boxes(5.0))
        self.root.bind("<Key-E>", lambda e: self.rotate_selected_boxes(15.0))
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-Z>", lambda e: self.undo())
        self.root.bind("<Control-y>", lambda e: self.redo())
        self.root.bind("<Control-a>", self.select_all_boxes)
        self.root.bind("<Control-A>", self.select_all_boxes)
        self.root.bind("<Delete>", self.delete_selected)
        self.root.bind("<KP_Delete>", self.delete_selected)
        self.canvas.bind("<Delete>", self.delete_selected)
        self.canvas.bind("<KP_Delete>", self.delete_selected)

    def on_canvas_resize(self, e):
        if not self.img_pil:
            return
        self.fit_image_to_canvas()

    def fit_image_to_canvas(self):
        if not self.img_pil:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return
        iw, ih = self.img_pil.width, self.img_pil.height
        if iw <= 0 or ih <= 0:
            self.logger.warning("Invalid image size: %sx%s", iw, ih)
            return
        scale = min(cw / iw, ch / ih)
        self.scale = scale
        self.offset_x = (cw - iw * scale) / 2
        self.offset_y = (ch - ih * scale) / 2
        self.render()

# ==================== Entrypoint ====================
def main():
    root = tk.Tk()
    app = UltimateLabeller(root)
    root.mainloop()

if __name__ == "__main__":
    main()
