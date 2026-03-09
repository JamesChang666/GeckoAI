if __package__ in {None, ""}:
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import datetime
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from collections import deque
from importlib import resources
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any

import numpy as np
from PIL import Image, ImageTk

from ai_labeller.ui import button_styles
from ai_labeller.ui.monitor_bounds import get_widget_monitor_bounds
from ai_labeller.ui import canvas_utils
from ai_labeller.ui import widget_styles
from ai_labeller.ui import window_pages
from ai_labeller.ui import detect_pages
from ai_labeller.ui import app_layout
from ai_labeller.ui import overlay_interaction
from ai_labeller.modes import detect as detect_mode
from ai_labeller.modes import label as label_mode
from ai_labeller.features import training_runner
from ai_labeller.features import training_threading
from ai_labeller.features import yolo_utils
from ai_labeller.features import image_load
from ai_labeller.ui import keybinds
from ai_labeller.features import camera_utils
from ai_labeller.features import report_utils
from ai_labeller.features import ocr_utils
from ai_labeller.features import golden_controller
from ai_labeller.features import detect_runtime
from ai_labeller.features import detect_controller
from ai_labeller.features import label_controller
from ai_labeller.features import project_utils
from ai_labeller.features import file_utils
from ai_labeller.features import export_utils
from ai_labeller.constants import COLORS, THEMES, LANG_MAP
from ai_labeller.core import (
    AppConfig,
    AppState,
    HistoryManager,
    atomic_write_json,
    atomic_write_text,
    setup_logging,
)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

HAS_YOLO = True


LOGGER = setup_logging()

WIN_NO_CONSOLE = 0
if os.name == "nt":
    WIN_NO_CONSOLE = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

class GeckoAI:
    def __init__(self, root: tk.Tk, startup_mode: str = "chooser"):
        self.root = root
        self.lang = "en"
        self.theme = "dark"
        self.config = AppConfig()
        self.state = AppState()
        self.history_manager = HistoryManager()
        self.logger = setup_logging(os.path.join(os.path.expanduser("~"), ".ai_labeller.log"))

        self.root.title(LANG_MAP[self.lang]["title"])
        # Expose constants on the app instance for UI modules expecting `app.COLORS` etc.
        self.COLORS = COLORS
        self.THEMES = THEMES
        self.LANG_MAP = LANG_MAP
        self.HAS_CV2 = HAS_CV2
        self.cv2 = cv2 if HAS_CV2 else None
        self.root.geometry(self.config.default_window_size)
        self.root.minsize(self.config.min_window_width, self.config.min_window_height)
        self.window_icon_tk = None
        self.toolbar_logo_tk = None
        
        self.setup_fonts()
        self.apply_theme(self.theme, rebuild=False)
        self.setup_app_icon()
        self._tooltip_after_id = None
        self._tooltip_win = None

        self.project_root = self.state.project_root
        self.current_split = self.state.current_split
        self.image_files = self.state.image_files
        self.current_idx = self.state.current_idx
        self.rects = self.state.rects  # [x1, y1, x2, y2, class_id, angle_deg]
        self.class_names = self.state.class_names
        self.learning_mem = deque(maxlen=self.config.max_learning_memory)

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
        self.var_show_prev_labels = tk.BooleanVar(value=False)
        self._prev_image_rects: list[list[float]] = []
        self._loaded_image_path: str | None = None
        self._cursor_line_x: int | None = None
        self._cursor_line_y: int | None = None
        self._cursor_text_id: int | None = None
        self._cursor_bg_id: int | None = None
        
        # --- AI/detection state ---
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
        self._app_mode_dialog_shown = False
        self._app_mode_dialog_open = False
        self._fullpage_overlay: tk.Frame | None = None
        self._detect_mode_active = False
        self._detect_workspace_frame: tk.Frame | None = None
        self._detect_image_label: tk.Label | None = None
        self._detect_class_listbox: tk.Listbox | None = None
        self._detect_status_var = tk.StringVar(value="")
        self._detect_verdict_var = tk.StringVar(value="")
        self._detect_verdict_label: tk.Label | None = None
        self._detect_photo: ImageTk.PhotoImage | None = None
        self._detect_last_plot_bgr = None
        self._detect_image_paths: list[str] = []
        self._detect_image_index = 0
        self._detect_video_cap = None
        self._detect_after_id: str | None = None
        self._detect_preferred_device: Any = "cpu"
        self._detect_conf_threshold = float(self.var_yolo_conf.get())
        self._detect_frame_interval_ms = 15
        self._detect_camera_max_fps = 0.0
        self._detect_report_csv_path: str | None = None
        self._detect_report_mode: str = "pure_detect"
        self._detect_video_frame_idx = 0
        self._detect_report_generated_paths: set[str] = set()
        self._detect_source_selected = False
        self.detect_camera_mode_var = tk.StringVar(value="auto")
        self.detect_camera_index_var = tk.StringVar(value="0")
        self.detect_manual_fps_var = tk.StringVar(value="10")
        self._detect_available_cameras: list[int] = []
        self.detect_run_mode_var = tk.StringVar(value="pure_detect")
        self.detect_golden_mode_var = tk.StringVar(value="both")
        self.detect_golden_iou_var = tk.DoubleVar(value=0.50)
        self.detect_golden_class_var = tk.StringVar(value="")
        self._detect_golden_sample: dict[str, Any] | None = None
        self._detect_bg_cut_bundle: Any = None
        self._detect_last_cut_piece_count: int = 0
        self._detect_last_piece_results: list[Any] = []
        self._detect_cut_piece_temp_root: str | None = None
        self._detect_cut_piece_last_dir: str | None = None
        self._detect_cut_piece_seq: int = 0
        self._detect_seen_cut_piece_hashes: set[str] = set()
        self._detect_last_piece_paths: list[str] = []
        self._detect_piece_index: int = 0
        self._detect_image_result_cache: dict[str, dict[str, Any]] = {}
        self._detect_report_logged_keys: set[str] = set()
        self._detect_last_ocr_id: str = ""
        self._detect_last_ocr_sub_id: str = ""
        self._detect_ocr_warning_shown = False
        self._easy_ocr_engine: Any = None
        self._golden_capture_active = False
        self._golden_capture_temp_root: str | None = None
        self._golden_capture_output_dir: str | None = None
        self._golden_capture_image_name: str | None = None
        self.training_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.training_thread: threading.Thread | None = None
        self.training_process: subprocess.Popen[str] | None = None
        self.training_running = False
        self.training_start_time: float | None = None
        self.training_total_epochs = 0
        self.training_current_epoch = 0
        self.train_command_var = tk.StringVar(value="")
        self._training_stop_requested = False
        self._training_monitor_win: tk.Toplevel | None = None
        self.lbl_train_status: tk.Label | None = None
        self.lbl_train_progress: tk.Label | None = None
        self.lbl_train_eta: tk.Label | None = None
        self.entry_train_cmd: tk.Entry | None = None
        self.txt_train_log: tk.Text | None = None
        self._training_log_lines: list[str] = []
        
        self.setup_custom_style()
        # bind button creators from ui/button_styles to reduce repetitive callers
        _creators = button_styles.bind_button_creators(self)
        self.create_toolbar_button = _creators["create_toolbar_button"]
        self.create_toolbar_icon_button = _creators["create_toolbar_icon_button"]
        self.create_primary_button = _creators["create_primary_button"]
        self.create_secondary_button = _creators["create_secondary_button"]
        self.create_nav_button = _creators["create_nav_button"]
        self.lighten_color = _creators["lighten_color"]
        self.is_accent_bg = _creators["is_accent_bg"]
        self.toolbar_text_color = _creators["toolbar_text_color"]

        # Bind app_layout helpers used by UI modules
        self.create_card = lambda parent, title=None: app_layout.create_card(self, parent, title)
        self.create_info_card = lambda parent: app_layout.create_info_card(self, parent)
        self.create_class_card = lambda parent: app_layout.create_class_card(self, parent)
        self.create_ai_card = lambda parent: app_layout.create_ai_card(self, parent)
        self.create_shortcut_card = lambda parent: app_layout.create_shortcut_card(self, parent)
        self.create_navigation = lambda parent: app_layout.create_navigation(self, parent)
        self._on_sidebar_frame_configure = lambda e=None: app_layout.on_sidebar_frame_configure(self, e)
        self._on_sidebar_canvas_configure = lambda e: app_layout.on_sidebar_canvas_configure(self, e)
        self._on_sidebar_mousewheel = lambda e: app_layout.on_sidebar_mousewheel(self, e)
        self._bind_sidebar_mousewheel = lambda widget: app_layout.bind_sidebar_mousewheel(self, widget)
        self._refresh_sidebar_scrollregion = lambda: app_layout.refresh_sidebar_scrollregion(self)
        self.get_theme_switch_label = lambda: app_layout.get_theme_switch_label(self)

        # Bind keybinds helpers
        self.create_help_icon = lambda parent: keybinds.create_help_icon(self, parent)
        self._shortcut_items = lambda: keybinds.shortcut_items(self)
        self.build_shortcut_text = lambda: keybinds.build_shortcut_text(self)
        self.show_shortcut_tooltip = lambda widget: keybinds.show_shortcut_tooltip(self, widget)
        self._show_tooltip_now = lambda widget: keybinds._show_tooltip_now(self, widget)
        self.hide_shortcut_tooltip = lambda: keybinds.hide_shortcut_tooltip(self)

        # Monitor bounds helper
        self._get_widget_monitor_bounds = get_widget_monitor_bounds

        # Bind common feature/controller helpers to instance to replace trivial passthroughs
        self.use_official_yolo26n = lambda *a, **k: yolo_utils.use_official_yolo26n(self, *a, **k)
        self._resolve_official_model_path = lambda *a, **k: yolo_utils._resolve_official_model_path(self, *a, **k)
        self._resolve_custom_model_path = lambda raw_path: yolo_utils._resolve_custom_model_path(self, raw_path)
        self.browse_detection_model = lambda *a, **k: yolo_utils.browse_detection_model(self, *a, **k)
        self.pick_model_file = lambda forced_mode=None: yolo_utils.pick_model_file(self, forced_mode=forced_mode)

        self._scan_available_cameras = lambda max_probe=6: camera_utils.scan_available_cameras(self, max_probe=max_probe)
        self._get_camera_max_fps = lambda camera_index=0: camera_utils.get_camera_max_fps(self, camera_index=camera_index)
        self._start_detect_video_stream = lambda source: camera_utils.start_detect_video_stream(self, source)
        self._detect_tick_video = lambda: camera_utils._detect_tick_video(self)
        self._stop_detect_stream = lambda: camera_utils.stop_detect_stream(self)

        self._init_detect_report_logger = lambda source_kind, source_value, output_dir=None: report_utils.init_detect_report_logger(self, source_kind, source_value, output_dir=output_dir)
        self._close_detect_report_logger = lambda: report_utils._close_detect_report_logger(self)
        self._trigger_detect_report_generation = lambda csv_path: report_utils._trigger_detect_report_generation(self, csv_path)
        self._append_detect_report_row = lambda image_name, result0, status, details: report_utils.append_detect_report_row(self, image_name, result0, status, details)
        self._append_detect_report_row_once = lambda image_name, result0, status, details: report_utils.append_detect_report_row_once(self, image_name, result0, status, details)

        self._configure_detect_golden_sample = lambda: golden_controller.configure_detect_golden_sample(self)
        self._load_detect_background_cut_bundle = lambda golden_dir: golden_controller.load_detect_background_cut_bundle(self, golden_dir)
        self._create_detect_golden_from_label_mode = lambda: golden_controller.create_detect_golden_from_label_mode(self)
        self._finalize_golden_from_label_mode = lambda: golden_controller.finalize_golden_from_label_mode(self)
        self._cancel_golden_capture_and_back_to_detect = lambda: golden_controller.cancel_golden_capture_and_back_to_detect(self)
        self._cleanup_golden_capture_temp = lambda: golden_controller._cleanup_golden_capture_temp(self)
        self._annotate_golden_image_label_style = lambda image_path, class_options: golden_controller.annotate_golden_image_label_style(self, image_path, class_options)
        self._parse_yolo_label_file = lambda label_path: golden_controller.parse_yolo_label_file(label_path)
        self._find_dataset_yaml_for_label = lambda label_path: golden_controller.find_dataset_yaml_for_label(label_path)
        self._find_dataset_yaml_in_folder = lambda folder: golden_controller.find_dataset_yaml_in_folder(folder)
        self._load_mapping_from_dataset_yaml = lambda yaml_path: golden_controller.load_mapping_from_dataset_yaml(yaml_path)
        self._find_golden_id_config_in_folder = lambda folder: golden_controller.find_golden_id_config_in_folder(folder)
        self._load_golden_id_config = lambda json_path: golden_controller.load_golden_id_config(json_path)
        self._prompt_golden_id_classes = lambda class_mapping, parent=None: golden_controller.prompt_golden_id_classes(self, class_mapping, parent=parent)
        self._write_golden_id_config = lambda folder, class_id, class_name, sub_id_class_id=None, sub_id_class_name=None: golden_controller.write_golden_id_config(folder, class_id, class_name, sub_id_class_id=sub_id_class_id, sub_id_class_name=sub_id_class_name)
        self._pick_golden_rect_on_image = lambda image_path: golden_controller.pick_golden_rect_on_image(self, image_path)

        self._should_use_background_cut_detection = lambda: detect_runtime.should_use_background_cut_detection(self)
        self._cleanup_detect_cut_piece_temp = lambda remove_root=False: detect_runtime.cleanup_detect_cut_piece_temp(self, remove_root=remove_root)
        self._ensure_detect_cut_piece_temp_root = lambda: detect_runtime.ensure_detect_cut_piece_temp_root(self)
        self._write_cut_pieces_to_temp_folder = lambda pieces: detect_runtime.write_cut_pieces_to_temp_folder(self, pieces)
        self._cut_piece_signature = lambda piece: detect_runtime.cut_piece_signature(self, piece)
        self._filter_unseen_cut_pieces = lambda pieces: detect_runtime.filter_unseen_cut_pieces(self, pieces)
        self._prepare_background_cut_detect_source = lambda source: detect_runtime.prepare_background_cut_detect_source(self, source)
        self._select_primary_result_index = lambda results: detect_runtime.select_primary_result_index(results)
        self._run_detect_inference = lambda source: detect_runtime.run_detect_inference(self, source)

        self._render_detect_current_piece_result = lambda source_path: detect_mode.render_current_piece_result(self, source_path)
        self._detect_render_image_index = lambda: detect_mode.detect_render_image_index(self)
        self._render_detect_current_piece_result = lambda source_path: detect_mode.render_current_piece_result(self, source_path)
        self._show_detect_plot = lambda plot_bgr: detect_mode.show_detect_plot(self, plot_bgr)
        self._refresh_detect_image = lambda: detect_mode.refresh_detect_image(self)
        self._detect_prev_image = lambda: detect_mode.detect_prev_image(self)
        self._detect_next_image = lambda: detect_mode.detect_next_image(self)

        self._get_easy_ocr_engine = lambda: ocr_utils.get_easy_ocr_engine(self)
        self._get_preferred_ocr_engine = lambda: ocr_utils.get_preferred_ocr_engine(self)
        self._extract_ocr_text_from_result = lambda result0, tgt_id, tgt_name: ocr_utils.extract_ocr_text_from_result(self, result0, tgt_id, tgt_name)
        self._extract_ocr_id_from_result = lambda result0: ocr_utils.extract_ocr_id_from_result(self, result0)
        self._extract_ocr_sub_id_from_result = lambda result0: ocr_utils.extract_ocr_sub_id_from_result(self, result0)

        # project utils (module functions that don't expect `self`)
        self.normalize_project_root = project_utils.normalize_project_root
        self.find_yolo_project_root = project_utils.find_yolo_project_root
        self._list_split_images_for_root = project_utils.list_split_images_for_root
        self._glob_image_files = project_utils._glob_image_files
        self._glob_label_files = project_utils._glob_label_files
        self._existing_image_splits = project_utils.existing_image_splits
        self.ensure_yolo_label_dirs = project_utils.ensure_yolo_label_dirs
        self.diagnose_folder_structure = project_utils.diagnose_folder_structure
        self.show_folder_diagnosis = lambda directory: project_utils.show_folder_diagnosis(self, directory)
        self.load_images_folder_only = lambda directory: project_utils.load_images_folder_only(self, directory)
        # Bind canvas geometry helpers
        self.clamp_box = lambda box: canvas_utils.clamp_box(self, box)
        self.normalize_angle_deg = canvas_utils.normalize_angle_deg
        self.get_rect_angle_deg = canvas_utils.get_rect_angle_deg
        self.set_rect_angle_deg = lambda rect, angle_deg: rect.__setitem__(slice(None), canvas_utils.set_rect_angle_deg(rect, angle_deg))
        # Additional bound delegators for controllers and helpers
        self._open_detect_workspace = lambda source_kind, source_value, output_dir=None: detect_mode.open_detect_workspace(self, source_kind, source_value, output_dir=output_dir)
        self.rotate_selected_boxes = lambda delta_deg: label_mode.rotate_selected_boxes(self, delta_deg)
        self._bbox_iou = lambda a, b: golden_controller.bbox_iou(a, b)
        self._evaluate_golden_match = lambda result0: golden_controller.evaluate_golden_match(self, result0)
        self._detect_class_counts = lambda result0: detect_runtime.detect_class_counts(result0)
        self.rotate_point_around_center = canvas_utils.rotate_point_around_center
        self.get_rotated_corners = canvas_utils.get_rotated_corners
        self.rect_to_obb_norm = canvas_utils.rect_to_obb_norm
        self.obb_norm_to_rect = lambda pts_norm, width, height, class_id: canvas_utils.obb_norm_to_rect(self, pts_norm, width, height, class_id)
        self._point_in_rotated_box = canvas_utils.point_in_rotated_box
        self.get_handles = lambda rect: canvas_utils.get_handles(self, rect)
        # Bind widget creators
        self.create_label = lambda parent, text="", **kw: widget_styles.create_label(self, parent, text, **kw)
        self.create_bold_label = lambda parent, text="", **kw: widget_styles.create_bold_label(self, parent, text, **kw)
        self.create_mono_label = lambda parent, text="", **kw: widget_styles.create_mono_label(self, parent, text, **kw)
        self.create_textbox = lambda parent, **kw: widget_styles.create_textbox(self, parent, **kw)
        self.create_combobox = lambda parent, **kw: widget_styles.create_combobox(self, parent, **kw)
        self.create_entry = lambda parent, **kw: widget_styles.create_entry(self, parent, **kw)
        # Bind file/path helpers
        self._build_removed_path = lambda kind, src_path: file_utils.build_removed_path(self, kind, src_path)
        self._unique_target_path = file_utils.unique_target_path
        self._rotation_meta_path_for_label = file_utils.rotation_meta_path_for_label
        self._read_rotation_meta_angles = lambda rot_meta_path: file_utils.read_rotation_meta_angles(self, rot_meta_path)
        # Bind UI/layout delegators
        self.setup_ui = lambda *a, **k: app_layout.setup_ui(self, *a, **k)
        self.setup_toolbar = lambda *a, **k: app_layout.setup_toolbar(self, *a, **k)
        self.setup_sidebar = lambda *a, **k: app_layout.setup_sidebar(self, *a, **k)
        self.create_help_icon = lambda parent: keybinds.create_help_icon(self, parent)
        self._shortcut_items = lambda: keybinds.shortcut_items(self)
        self.build_shortcut_text = lambda: keybinds.build_shortcut_text(self)
        self.show_shortcut_tooltip = lambda widget: keybinds.show_shortcut_tooltip(self, widget)
        self._show_tooltip_now = lambda widget: keybinds._show_tooltip_now(self, widget)
        self._get_widget_monitor_bounds = get_widget_monitor_bounds
        self.hide_shortcut_tooltip = lambda: keybinds.hide_shortcut_tooltip(self)
        self.create_card = lambda parent, title=None: app_layout.create_card(self, parent, title)
        self.create_info_card = lambda parent: app_layout.create_info_card(self, parent)
        self.create_class_card = lambda parent: app_layout.create_class_card(self, parent)
        self.create_ai_card = lambda parent: app_layout.create_ai_card(self, parent)
        self.create_shortcut_card = lambda parent: app_layout.create_shortcut_card(self, parent)
        self.create_navigation = lambda parent: app_layout.create_navigation(self, parent)
        # Bind overlay interaction delegators
        self.render = lambda: overlay_interaction.render(self)
        self.on_mouse_move = lambda e: overlay_interaction.on_mouse_move(self, e)
        self.update_cursor_overlay = lambda: overlay_interaction.update_cursor_overlay(self)
        self.on_mouse_down = lambda e: overlay_interaction.on_mouse_down(self, e)
        self.on_mouse_down_right = lambda e: overlay_interaction.on_mouse_down_right(self, e)
        self.on_mouse_drag = lambda e: overlay_interaction.on_mouse_drag(self, e)
        self.on_mouse_up = lambda e: overlay_interaction.on_mouse_up(self, e)
        self.on_mouse_up_right = lambda e: overlay_interaction.on_mouse_up_right(self, e)
        self.paste_previous_labels = lambda ix, iy: overlay_interaction.paste_previous_labels(self, ix, iy)
        self.on_zoom = lambda e: overlay_interaction.on_zoom(self, e)
        self.on_canvas_resize = lambda e: overlay_interaction.on_canvas_resize(self, e)
        self.fit_image_to_canvas = lambda: overlay_interaction.fit_image_to_canvas(self)
        # Bind detect/label/project/image/yolo delegators
        self.open_detect_workspace = lambda source_kind, source_value, output_dir=None: detect_mode.open_detect_workspace(self, source_kind, source_value, output_dir=output_dir)
        self.remove_current_from_split = lambda: label_mode.remove_current_from_split(self)
        self.restore_removed_file_by_name = lambda filename: label_mode.restore_removed_file_by_name(self, filename)
        self.load_img = lambda: image_load.load_image(self)
        self.save_current = lambda: label_mode.save_current(self)
        self._reindex_dataset_labels_after_class_delete = lambda deleted_idx: label_mode._reindex_dataset_labels_after_class_delete(self, deleted_idx)
        self.load_project_from_path = lambda directory, preferred_image=None, save_session=True: project_utils.load_project_from_path(self, directory, preferred_image=preferred_image, save_session=save_session)
        self.load_project_root = lambda: project_utils.load_project_root(self)
        self.load_split_data = lambda preferred_image=None: project_utils.load_split_data(self, preferred_image=preferred_image)
        self._list_split_images = lambda split: project_utils.list_split_images(self.project_root, split)
        self.autolabel_red = lambda: yolo_utils.autolabel_red(self)
        self.run_yolo_detection = lambda: yolo_utils.run_yolo_detection(self)
        self._is_cuda_kernel_compat_error = yolo_utils._is_cuda_kernel_compat_error
        self._can_use_cuda_runtime = yolo_utils._can_use_cuda_runtime
        self._auto_runtime_device = lambda allow_forced_cpu=False: yolo_utils._auto_runtime_device(self, allow_forced_cpu=allow_forced_cpu)
        # Bind training_threading helpers
        self.stop_training = lambda: training_threading.stop_training(self)
        self._force_kill_training_if_alive = lambda proc: training_threading.force_kill_training_if_alive(self, proc)
        self._append_training_log = lambda line: training_threading.append_training_log(self, line)
        self._set_training_status = lambda running: training_threading.set_training_status(self, running)
        self._set_training_progress = lambda current_epoch, total_epochs: training_threading.set_training_progress(self, current_epoch, total_epochs)
        self._format_eta_seconds = lambda seconds_left: training_threading.format_eta_seconds(self, seconds_left)
        self._set_training_eta = lambda eta_text: training_threading.set_training_eta(self, eta_text)
        self._handle_training_output_line = lambda line: training_threading.handle_training_output_line(self, line)
        self._run_training_subprocess = lambda cmd, workdir: training_threading.run_training_subprocess(self, cmd, workdir)
        self._poll_training_queue = lambda: training_threading.poll_training_queue(self)

        # Bind export utilities
        self._iter_export_images = lambda: export_utils._iter_export_images(self)
        self.export_all_by_selected_format = lambda: export_utils.export_all_by_selected_format(self)
        self._export_all_yolo = lambda out_dir: export_utils._export_all_yolo(self, out_dir)
        self._write_export_yolo_dataset_yaml = lambda out_dir, val_rel_path="images/train": export_utils._write_export_yolo_dataset_yaml(self, out_dir, val_rel_path=val_rel_path)
        self._export_val_with_aug_for_val = lambda out_dir: export_utils._export_val_with_aug_for_val(self, out_dir)
        self._export_all_json = lambda out_dir: export_utils._export_all_json(self, out_dir)

        self.setup_ui()
        self.bind_events()
        self.root.protocol("WM_DELETE_WINDOW", self.on_app_close)
        self.load_session_state()
        self._startup_mode = (startup_mode or "chooser").strip().lower()
        mode = self._startup_mode
        if mode == "detect":
            self.root.after(120, self.show_detect_mode_page)
        elif mode == "label":
            self.root.after(120, lambda: self.show_startup_source_dialog(force=True))
        else:
            self.root.after(120, self.show_app_mode_dialog)
    
    def setup_fonts(self):
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
        style = ttk.Style()
        style.theme_use('clam')
        
        # Combobox style
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

    def _pick_prev_box_at_point(self, ix: float, iy: float) -> int | None:
        candidates: list[tuple[float, int]] = []
        for idx, rect in enumerate(self._prev_image_rects):
            if self._point_in_rotated_box(ix, iy, rect):
                x1 = min(rect[0], rect[2])
                y1 = min(rect[1], rect[3])
                x2 = max(rect[0], rect[2])
                y2 = max(rect[1], rect[3])
                area = max(1.0, (x2 - x1) * (y2 - y1))
                candidates.append((area, idx))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], -item[1]))
        return candidates[0][1]
    
    

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

    def _open_fullpage_overlay(self) -> tk.Frame:
        self._close_fullpage_overlay()
        overlay = tk.Frame(self.root, bg=COLORS["bg_dark"])
        overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._fullpage_overlay = overlay
        return overlay

    def _close_fullpage_overlay(self) -> None:
        if self._fullpage_overlay is not None:
            try:
                self._fullpage_overlay.destroy()
            except Exception:
                pass
            self._fullpage_overlay = None
    
    def toggle_language(self):
        self.lang = "en"
    
    def update_info_text(self):
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
        if hasattr(self, "lbl_class_count"):
            frame_class_count = len({int(r[4]) for r in self.rects if len(r) >= 5})
            total_class_count = len(self.class_names)
            self.lbl_class_count.config(
                text=f"{LANG_MAP[self.lang]['class_mgmt']}: {frame_class_count} / {total_class_count}"
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
        try:
            if not self.combo_cls.winfo_exists():
                return
            self.combo_cls.configure(values=self.class_names)
        except tk.TclError:
            return
        if not self.class_names:
            return
        try:
            current_idx = self.combo_cls.current() if preferred_idx is None else preferred_idx
        except tk.TclError:
            return
        if current_idx < 0 or current_idx >= len(self.class_names):
            current_idx = 0
        try:
            self.combo_cls.current(current_idx)
        except tk.TclError:
            return

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
        from ai_labeller.features.io_utils import project_progress_yaml_path
        return project_progress_yaml_path(project_root or self.project_root)

    def _write_project_progress_yaml(self) -> None:
        from ai_labeller.features.io_utils import write_project_progress_yaml
        write_project_progress_yaml(self)

    def _read_project_progress_yaml(self, project_root: str) -> dict[str, str]:
        from ai_labeller.features.io_utils import read_project_progress_yaml
        return read_project_progress_yaml(project_root)

    def _extract_class_names_from_progress(self, progress: dict[str, str]) -> list[str]:
        from ai_labeller.features.io_utils import extract_class_names_from_progress
        return extract_class_names_from_progress(progress)

    def save_session_state(self) -> None:
        from ai_labeller.features.io_utils import save_session_state
        save_session_state(self)

    def load_session_state(self) -> None:
        from ai_labeller.features.io_utils import load_session_state
        load_session_state(self)

    def on_detection_model_mode_changed(self, e: Any = None) -> None:
        if self.det_model_mode.get() == "Official YOLO26m.pt (Bundled)":
            self.yolo_path.set(self.config.yolo_model_path)
            self._register_model_path(self.config.yolo_model_path)
        self.yolo_model = None
        self._loaded_model_key = None


    def show_app_mode_dialog(self, force: bool = False) -> None:
        window_pages.show_app_mode_dialog(self, COLORS, LANG_MAP, force=force)

    def _close_app_mode_dialog(self) -> None:
        window_pages.close_app_mode_dialog(self)

    def _reset_detect_setup_page(self) -> tk.Frame:
        self._detect_mode_active = True
        self._stop_detect_stream()
        self._detect_workspace_frame = None
        self.hide_shortcut_tooltip()
        for child in self.root.winfo_children():
            child.destroy()
        wrap = tk.Frame(self.root, bg=COLORS["bg_dark"])
        wrap.pack(fill="both", expand=True)
        return wrap

    def _create_detect_setup_card(
        self,
        wrap: tk.Frame,
        *,
        width: int,
        height: int,
        title: str,
        subtitle: str,
    ) -> tk.Frame:
        card = tk.Frame(wrap, bg=COLORS["bg_white"], bd=0, highlightthickness=0)
        card.place(relx=0.5, rely=0.5, anchor="center", width=width, height=height)
        self.create_bold_label(card, text=title, font=self.font_title, fg=COLORS["text_primary"], bg=COLORS["bg_white"], anchor="center").pack(fill="x", padx=24, pady=(28, 8))
        self.create_label(card, text=subtitle, font=self.font_primary, fg=COLORS["text_secondary"], bg=COLORS["bg_white"], anchor="center").pack(fill="x", padx=24, pady=(0, 14))
        return card

    def show_detect_mode_page(self) -> None:
        detect_pages.show_detect_mode_page(self)

    def show_detect_source_page(self) -> None:
        detect_pages.show_detect_source_page(self)

    def show_detect_camera_mode_page(self) -> None:
        detect_pages.show_detect_camera_mode_page(self)

    def show_detect_file_settings_page(self) -> None:
        detect_pages.show_detect_file_settings_page(self)

    def _on_detect_pick_model(self) -> None:
        model_path = filedialog.askopenfilename(
            parent=self.root,
            title="Select model for detect mode",
            filetypes=[
                ("Model files", "*.pt *.onnx"),
                ("PyTorch", "*.pt"),
                ("ONNX", "*.onnx"),
                ("All files", "*.*"),
            ],
        )
        if not model_path:
            return
        self.detect_model_path_var.set(os.path.abspath(model_path))
        self.show_detect_mode_page()

    def _go_detect_source_page(self) -> None:
        model_path = self.detect_model_path_var.get().strip()
        if not model_path:
            messagebox.showwarning("Detect Mode", "Please choose model before next step.", parent=self.root)
            return
        self.show_detect_source_page()

    def _on_detect_choose_camera(self) -> None:
        cams = self._scan_available_cameras()
        if not cams:
            messagebox.showwarning("Detect Mode", "No camera found.", parent=self.root)
            return
        self._detect_available_cameras = cams[:]
        if str(self.detect_camera_index_var.get().strip()) not in {str(c) for c in cams}:
            self.detect_camera_index_var.set(str(cams[0]))
        self.detect_source_mode_var.set("camera")
        self.detect_media_path_var.set(self.detect_camera_index_var.get().strip() or "0")
        self._detect_source_selected = True
        self.show_detect_camera_mode_page()

    def _on_detect_browse_media_file(self) -> None:
        src = filedialog.askdirectory(
            parent=self.root,
            title="Select image folder",
        )
        if not src:
            return
        src_abs = os.path.abspath(src)
        if not self._detect_folder_has_images(src_abs):
            messagebox.showwarning("Detect Mode", "No images found in selected folder.", parent=self.root)
            return
        self.detect_source_mode_var.set("file")
        self.detect_media_path_var.set(src_abs)
        self._detect_source_selected = True
        self.show_detect_file_settings_page()

    def _on_detect_choose_output_dir(self) -> None:
        out_dir = filedialog.askdirectory(
            parent=self.root,
            title="Select detect output folder",
        )
        if not out_dir:
            return
        self.detect_output_dir_var.set(os.path.abspath(out_dir))
        self._show_detect_settings_page_for_current_source()

    def _show_detect_settings_page_for_current_source(self) -> None:
        source = self.detect_source_mode_var.get().strip().lower()
        if source == "camera":
            self.show_detect_camera_mode_page()
            return
        if source == "file":
            self.show_detect_file_settings_page()
            return
        self.show_detect_source_page()


    def _start_detect_from_setup(self) -> None:
        if not self.detect_model_path_var.get().strip():
            messagebox.showwarning("Detect Mode", "Please choose model in Step 1.", parent=self.root)
            return
        if not self._detect_source_selected:
            messagebox.showwarning("Detect Mode", "Please choose source in Step 2.", parent=self.root)
            return
        source_kind = self.detect_source_mode_var.get().strip().lower()
        source_value: Any = self.detect_media_path_var.get().strip()
        if source_kind == "camera":
            try:
                source_value = int(str(source_value or self.detect_camera_index_var.get().strip() or "0"))
            except Exception:
                source_value = 0
            self.detect_camera_index_var.set(str(source_value))
            self.detect_media_path_var.set(str(source_value))
        output_dir = self.detect_output_dir_var.get().strip()
        if source_kind == "file":
            if not output_dir:
                messagebox.showwarning("Detect Mode", "Please choose output folder before start.", parent=self.root)
                return
            if not os.path.isdir(output_dir):
                messagebox.showerror("Detect Mode", f"Output folder not found:\n{output_dir}", parent=self.root)
                return
        else:
            if not output_dir:
                output_dir = os.path.abspath(os.getcwd())
                self.detect_output_dir_var.set(output_dir)
            if not os.path.isdir(output_dir):
                try:
                    os.makedirs(output_dir, exist_ok=True)
                except Exception as exc:
                    messagebox.showerror("Detect Mode", f"Failed to create output folder:\n{exc}", parent=self.root)
                    return

        self._detect_frame_interval_ms = 15
        if source_kind == "camera":
            max_fps = self._get_camera_max_fps(int(source_value))
            camera_mode = self.detect_camera_mode_var.get().strip().lower()
            if camera_mode == "manual":
                try:
                    preferred_fps = float(self.detect_manual_fps_var.get().strip())
                except Exception:
                    messagebox.showwarning("Detect Mode", "Manual FPS must be a number.", parent=self.root)
                    return
                if preferred_fps <= 0:
                    messagebox.showwarning("Detect Mode", "Manual FPS must be > 0.", parent=self.root)
                    return
                if max_fps > 0:
                    preferred_fps = min(preferred_fps, max_fps)
                self._detect_frame_interval_ms = max(1, int(round(1000.0 / max(0.1, preferred_fps))))
            else:
                auto_fps = max_fps if max_fps > 0 else 30.0
                self._detect_frame_interval_ms = max(1, int(round(1000.0 / max(0.1, auto_fps))))

        run_mode = self.detect_run_mode_var.get().strip().lower()
        if run_mode == "golden":
            if self._detect_golden_sample is None:
                messagebox.showwarning("Detect Mode", "Run Type is golden. Please import golden sample first.", parent=self.root)
                return
            targets = self._detect_golden_sample.get("targets") or []
            if not targets:
                messagebox.showwarning("Detect Mode", "Golden sample has no targets.", parent=self.root)
                return
            mode = self.detect_golden_mode_var.get().strip().lower()
            has_class = any((t.get("class_id") is not None or t.get("class_name")) for t in targets)
            if mode in {"class", "both"} and not has_class:
                messagebox.showwarning("Detect Mode", "Golden mode requires class info (ID or mapping name).", parent=self.root)
                return
            id_enabled = (
                self._detect_golden_sample.get("id_class_id") is not None
                or bool(str(self._detect_golden_sample.get("id_class_name", "")).strip())
                or self._detect_golden_sample.get("sub_id_class_id") is not None
                or bool(str(self._detect_golden_sample.get("sub_id_class_name", "")).strip())
            )
            if id_enabled and not HAS_EASY_OCR:
                messagebox.showwarning(
                    "Detect Mode",
                    "ID/Sub ID OCR is configured, but EasyOCR is not installed. Detection will run without OCR IDs.",
                    parent=self.root,
                )
        self.start_detect_mode(
            model_path=self.detect_model_path_var.get().strip(),
            source_kind=source_kind,
            source_value=source_value,
            output_dir=output_dir,
            conf_threshold=float(self.detect_conf_var.get()),
        )

    def _prompt_detect_source(self) -> tuple[str, Any] | None:
        result: dict[str, Any] = {"kind": None, "value": None}
        done = tk.BooleanVar(value=False)
        overlay = self._open_fullpage_overlay()
        card = tk.Frame(overlay, bg=COLORS["bg_white"], bd=0, highlightthickness=0)
        card.place(relx=0.5, rely=0.5, anchor="center", width=520, height=260)

        self.create_bold_label(card, text="Choose detection source", font=self.font_title, fg=COLORS["text_primary"], bg=COLORS["bg_white"], anchor="center").pack(fill="x", padx=20, pady=(24, 16))

        def use_camera() -> None:
            result["kind"] = "camera"
            result["value"] = 0
            self._close_fullpage_overlay()
            done.set(True)

        def choose_file() -> None:
            src = filedialog.askopenfilename(
                parent=self.root,
                title="Select image or video",
                filetypes=[
                    ("Media", "*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov *.mkv *.webm"),
                    ("Image", "*.jpg *.jpeg *.png *.bmp"),
                    ("Video", "*.mp4 *.avi *.mov *.mkv *.webm"),
                    ("All files", "*.*"),
                ],
            )
            if not src:
                return
            result["kind"] = "file"
            result["value"] = os.path.abspath(src)
            self._close_fullpage_overlay()
            done.set(True)

        def cancel() -> None:
            self._close_fullpage_overlay()
            done.set(True)

        self.create_primary_button(card, text="Camera (Realtime)", command=use_camera, bg=COLORS["primary"]).pack(
            fill="x", padx=28, pady=(0, 10)
        )
        self.create_primary_button(card, text="Image/Video File", command=choose_file, bg=COLORS["success"]).pack(
            fill="x", padx=28, pady=(0, 10)
        )
        self.create_secondary_button(card, text="Cancel", command=cancel).pack(fill="x", padx=28, pady=(0, 20))
        self.root.wait_variable(done)

        kind = result.get("kind")
        if not kind:
            return None
        return kind, result.get("value")

    def start_detect_mode(
        self,
        model_path: str | None = None,
        source_kind: str | None = None,
        source_value: Any = None,
        output_dir: str | None = None,
        conf_threshold: float | None = None,
    ) -> None:
        if not HAS_YOLO:
            messagebox.showwarning("YOLO Not Available", "Please install ultralytics first.")
            return
        if not HAS_CV2:
            messagebox.showwarning("OpenCV Not Available", "Please install opencv-python first.")
            return
        if not model_path:
            messagebox.showwarning("Detect Mode", "Please choose model in Step 1.")
            return
        try:
            model_path = self._resolve_custom_model_path(model_path)
        except FileNotFoundError as exc:
            messagebox.showerror("Model Error", str(exc), parent=self.root)
            return

        source_kind = (source_kind or "").strip().lower()
        if source_kind not in {"camera", "file"}:
            messagebox.showwarning("Detect Mode", "Please choose source in Step 2.", parent=self.root)
            return
        if source_kind == "file":
            if not source_value:
                messagebox.showwarning("Detect Mode", "Please choose image folder in Step 2.")
                return
            source_value = os.path.abspath(str(source_value))
            if not os.path.exists(source_value):
                messagebox.showerror("Detect Mode", f"Image folder not found:\n{source_value}")
                return
            if os.path.isdir(source_value) and not self._detect_folder_has_images(source_value):
                messagebox.showwarning("Detect Mode", "No images found in selected folder.", parent=self.root)
                return
        output_dir = (output_dir or "").strip()
        if not output_dir:
            messagebox.showwarning("Detect Mode", "Please choose output folder in setup page.", parent=self.root)
            return
        output_dir = os.path.abspath(output_dir)
        if not os.path.isdir(output_dir):
            messagebox.showerror("Detect Mode", f"Output folder not found:\n{output_dir}", parent=self.root)
            return
        try:
            conf_threshold = float(conf_threshold if conf_threshold is not None else self.detect_conf_var.get())
        except Exception:
            messagebox.showwarning("Detect Mode", "Invalid confidence threshold.", parent=self.root)
            return
        if conf_threshold < 0.01 or conf_threshold > 1.0:
            messagebox.showwarning("Detect Mode", "Conf threshold must be between 0.01 and 1.0.", parent=self.root)
            return

        try:
            loaded_key = ("detect_mode", os.path.abspath(model_path))
            # Delegate model loading to yolo_utils to keep AI runtime logic out of main.
            yolo_utils.ensure_yolo_model(self, loaded_key, model_path)

            preferred_device: Any = 0 if self._auto_runtime_device() == "0" else "cpu"
            self._detect_preferred_device = preferred_device
            self._detect_conf_threshold = conf_threshold
            self._open_detect_workspace(source_kind, source_value, output_dir=output_dir)
        except Exception as exc:
            self.logger.exception("Detect mode failed")
            messagebox.showerror("Detect Mode Error", str(exc), parent=self.root)

    def _detect_folder_has_images(self, folder: str) -> bool:
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        try:
            return any(
                os.path.isfile(os.path.join(folder, name)) and name.lower().endswith(exts)
                for name in os.listdir(folder)
            )
        except Exception:
            return False

    

    def _set_detect_verdict(self, status: str | None, details: str) -> None:
        if self.detect_run_mode_var.get().strip().lower() != "golden":
            self._detect_verdict_var.set("Pure Detect")
            if self._detect_verdict_label is not None:
                self._detect_verdict_label.config(fg=COLORS["text_secondary"])
            return
        if status == "PASS":
            self._detect_verdict_var.set(f"PASS {details}".strip())
            if self._detect_verdict_label is not None:
                self._detect_verdict_label.config(fg="#0FA958")
        elif status == "FAIL":
            self._detect_verdict_var.set(f"FAIL {details}".strip())
            if self._detect_verdict_label is not None:
                self._detect_verdict_label.config(fg="#F24822")
        else:
            self._detect_verdict_var.set(details or "No Golden Check")
            if self._detect_verdict_label is not None:
                self._detect_verdict_label.config(fg=COLORS["text_secondary"])

    
        
    def _exit_detect_workspace_to_source(self) -> None:
        self._stop_detect_stream()
        self._show_detect_settings_page_for_current_source()

    

    def _update_detect_class_panel(self, result0: Any) -> None:
        if self._detect_class_listbox is None:
            return
        self._detect_class_listbox.delete(0, tk.END)
        sample = self._detect_golden_sample or {}
        id_enabled = sample.get("id_class_id") is not None or bool(str(sample.get("id_class_name", "")).strip())
        sub_id_enabled = sample.get("sub_id_class_id") is not None or bool(str(sample.get("sub_id_class_name", "")).strip())
        if id_enabled:
            self._detect_class_listbox.insert(tk.END, f"[OCR ID] {self._detect_last_ocr_id or '(none)'}")
        if sub_id_enabled:
            self._detect_class_listbox.insert(tk.END, f"[OCR SUB ID] {self._detect_last_ocr_sub_id or '(none)'}")
        counts = self._detect_class_counts(result0)
        if not counts:
            self._detect_class_listbox.insert(tk.END, "No detections")
            return
        for cls_name in sorted(counts.keys()):
            self._detect_class_listbox.insert(tk.END, f"{cls_name} x{counts[cls_name]}")

    def show_startup_source_dialog(
        self,
        force: bool = False,
        reason: str | None = None,
        bypass_detect_lock: bool = False,
    ) -> None:
        window_pages.show_startup_source_dialog(
            self,
            COLORS,
            LANG_MAP,
            force=force,
            reason=reason,
            bypass_detect_lock=bypass_detect_lock,
        )

    def _close_startup_dialog(self) -> None:
        window_pages.close_startup_dialog(self)

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
        mode = getattr(self, "_startup_mode", "chooser")
        if mode == "detect":
            self.show_detect_mode_page()
            return
        if mode == "label":
            self.show_startup_source_dialog(force=True)
            return
        self.show_app_mode_dialog(force=True)

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

            if getattr(self, "_startup_mode", "chooser") != "detect":
                run_cut_bg = messagebox.askyesno(
                    "Cut Background",
                    "Do you want to cut background and detect all images first?\n\n"
                    "If Yes, one golden setup will be reused for all images.\n"
                    "Default match threshold is 0.3.",
                    parent=self.root,
                )
                if run_cut_bg:
                    if not HAS_CV2:
                        messagebox.showwarning(
                            "Cut Background",
                            "OpenCV is not available. Install opencv-python first.",
                            parent=self.root,
                        )
                    else:
                        try:
                            from ai_labeller.cut_background_detect import run_cut_background_batch

                            result = run_cut_background_batch(
                                root_dir=directory,
                                threshold=0.3,
                                parent=self.root,
                            )
                            if result is not None:
                                messagebox.showinfo(
                                    "Cut Background Complete",
                                    "Batch finished.\n\n"
                                    f"Golden folder:\n{result.golden_dir}\n\n"
                                    f"Output folder:\n{result.output_dir}\n\n"
                                    f"Images scanned: {result.total_images}\n"
                                    f"Boards detected: {result.processed_images}\n"
                                    f"Total cut pieces: {result.total_crops}",
                                    parent=self.root,
                                )
                        except Exception as exc:
                            self.logger.exception("Cut background batch failed")
                            messagebox.showerror(
                                "Cut Background Error",
                                f"Failed to run cut background batch:\n{exc}",
                                parent=self.root,
                            )

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
        
    def on_app_close(self) -> None:
        self._stop_detect_stream()
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

    # ==================== Mouse Interaction ====================
    
    # ==================== Class Operations ====================
    
    def on_class_change_request(self, e=None):
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

    
    
    def edit_classes_table(self):
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
        
        # Action buttons
        btn_frame = tk.Frame(win, bg=COLORS["bg_light"])
        btn_frame.pack(fill="x", padx=20, pady=10)
        
        self.create_primary_button(btn_frame, text=L["add"], command=add, bg=COLORS["success"]).pack(side="left", expand=True, fill="x", padx=(0, 5))
        
        self.create_primary_button(btn_frame, text=L["rename"], command=rename, bg=COLORS["warning"]).pack(side="left", expand=True, fill="x", padx=(5, 0))

        self.create_primary_button(btn_frame, text=L.get("delete_class", "Delete Class"), command=delete_class, bg=COLORS["danger"]).pack(side="left", expand=True, fill="x", padx=(5, 0))
        
        self.create_primary_button(win, text=L["apply"], command=lambda: [
            self._refresh_class_dropdown(),
            self.render(),
            win.destroy()
        ], bg=COLORS["primary"]).pack(fill="x", padx=20, pady=(0, 20))
        
        tree.bind("<Double-1>", on_double_click)
        refresh()

    def reassign_labeled_class(self):
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

        self.create_label(win, text=f"{LANG_MAP[self.lang]['current']}: {current_name}", font=self.font_primary, fg=COLORS["text_secondary"], bg=COLORS["bg_light"], anchor="w").pack(fill="x", padx=20, pady=(20, 6))

        self.create_label(win, text=LANG_MAP[self.lang]["to"], font=self.font_primary, fg=COLORS["text_secondary"], bg=COLORS["bg_light"], anchor="w").pack(fill="x", padx=20, pady=(10, 6))

        to_default = self.class_names[current_idx] if 0 <= current_idx < len(self.class_names) else self.class_names[0]
        to_var = tk.StringVar(value=to_default)
        self.create_combobox(win, values=self.class_names, textvariable=to_var, state="readonly", font=self.font_primary).pack(fill="x", padx=20)

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

        self.create_primary_button(win, text=LANG_MAP[self.lang]["apply"], command=apply_change, bg=COLORS["primary"]).pack(fill="x", padx=20, pady=(20, 16))

    def clear_current_labels(self):
        if not self.rects:
            return
        # Keep behavior identical to Ctrl+A then Delete.
        self.select_all_boxes()
        self.delete_selected()

    
    

    

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

        self.create_label(win, text=LANG_MAP[self.lang].get("restore_select", "Select a frame to restore:"), font=self.font_primary, fg=COLORS["text_secondary"], bg=COLORS["bg_light"], anchor="w").pack(fill="x", padx=16, pady=(16, 8))

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

    

    
    
    def on_split_change(self, e=None):
        if self.project_root:
            self.save_current()
            self.current_split = self.combo_split.get()
            self.load_split_data()
            self.save_session_state()
    
    

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

    def open_training_monitor_popup(self) -> None:
        if self._training_monitor_win is not None and self._training_monitor_win.winfo_exists():
            self._training_monitor_win.lift()
            self._training_monitor_win.focus_force()
            return

        win = tk.Toplevel(self.root)
        win.title(LANG_MAP[self.lang].get("train_monitor", "Training Monitor"))
        win.geometry("760x520")
        win.minsize(640, 420)
        win.configure(bg=COLORS["bg_white"])
        self._training_monitor_win = win

        outer = tk.Frame(win, bg=COLORS["bg_white"])
        outer.pack(fill="both", expand=True, padx=12, pady=12)

        _lbl = self.create_label(outer, text=f"{LANG_MAP[self.lang].get('train_status', 'Status')}: {LANG_MAP[self.lang].get('train_idle', 'Idle')}", font=self.font_primary, fg=COLORS["text_secondary"], bg=COLORS["bg_white"], anchor="w")
        _lbl.pack(fill="x")
        self.lbl_train_status = _lbl

        _lbl = self.create_label(outer, text=f"{LANG_MAP[self.lang].get('train_progress', 'Progress')}: -", font=self.font_primary, fg=COLORS["text_secondary"], bg=COLORS["bg_white"], anchor="w")
        _lbl.pack(fill="x")
        self.lbl_train_progress = _lbl

        _lbl = self.create_label(outer, text=f"{LANG_MAP[self.lang].get('train_eta', 'ETA')}: -", font=self.font_primary, fg=COLORS["text_secondary"], bg=COLORS["bg_white"], anchor="w")
        _lbl.pack(fill="x", pady=(0, 6))
        self.lbl_train_eta = _lbl

        self.create_label(outer, text=LANG_MAP[self.lang].get("train_command", "Command"), font=self.font_primary, fg=COLORS["text_secondary"], bg=COLORS["bg_white"], anchor="w").pack(fill="x")

        self.entry_train_cmd = self.create_entry(outer, textvariable=self.train_command_var, font=self.font_mono, state="readonly", readonlybackground=COLORS["bg_light"], fg=COLORS["text_primary"])
        self.entry_train_cmd.pack(fill="x", pady=(0, 6))

        stop_row = tk.Frame(outer, bg=COLORS["bg_white"])
        stop_row.pack(fill="x", pady=(0, 6))
        self.create_secondary_button(
            stop_row,
            text="Stop Training",
            command=self.stop_training,
        ).pack(side="right")

        log_wrap = tk.Frame(outer, bg=COLORS["bg_white"])
        log_wrap.pack(fill="both", expand=True)
        log_wrap.grid_rowconfigure(0, weight=1)
        log_wrap.grid_columnconfigure(0, weight=1)
        self.txt_train_log = tk.Text(
            log_wrap,
            wrap="none",
            font=self.font_mono,
            bg=COLORS["bg_light"],
            fg=COLORS["text_primary"],
            relief="flat",
        )
        self.txt_train_log.grid(row=0, column=0, sticky="nsew")
        sb_log_y = tk.Scrollbar(log_wrap, orient="vertical", command=self.txt_train_log.yview)
        sb_log_y.grid(row=0, column=1, sticky="ns")
        sb_log_x = tk.Scrollbar(log_wrap, orient="horizontal", command=self.txt_train_log.xview)
        sb_log_x.grid(row=1, column=0, sticky="ew")
        self.txt_train_log.configure(
            yscrollcommand=sb_log_y.set,
            xscrollcommand=sb_log_x.set,
        )

        if self._training_log_lines:
            self.txt_train_log.insert("end", "".join(self._training_log_lines))
            self.txt_train_log.see("end")

        self._set_training_status(self.training_running)
        if self.training_total_epochs > 0:
            self._set_training_progress(self.training_current_epoch, self.training_total_epochs)
        else:
            self._set_training_progress(0, 0)

        def on_close() -> None:
            if self._training_monitor_win is win:
                self._training_monitor_win = None
            self.lbl_train_status = None
            self.lbl_train_progress = None
            self.lbl_train_eta = None
            self.entry_train_cmd = None
            self.txt_train_log = None
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)

    

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
        result: dict[str, str | None] = {"choice": None, "path": None}
        done = tk.BooleanVar(value=False)
        overlay = self._open_fullpage_overlay()
        card = tk.Frame(overlay, bg=COLORS["bg_white"], bd=0, highlightthickness=0)
        card.place(relx=0.5, rely=0.5, anchor="center", width=560, height=320)

        self.create_bold_label(card, text="Choose training weight source", font=self.font_title, fg=COLORS["text_primary"], bg=COLORS["bg_white"], anchor="center").pack(fill="x", padx=20, pady=(24, 18))

        def choose_official() -> None:
            result["choice"] = "official"
            self._close_fullpage_overlay()
            done.set(True)

        def choose_custom() -> None:
            model_path = filedialog.askopenfilename(
                parent=self.root,
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
                messagebox.showerror("Model Error", str(exc), parent=self.root)
                return
            result["choice"] = "custom"
            result["path"] = os.path.abspath(resolved)
            self._close_fullpage_overlay()
            done.set(True)

        def choose_scratch() -> None:
            result["choice"] = "scratch"
            self._close_fullpage_overlay()
            done.set(True)

        def cancel() -> None:
            self._close_fullpage_overlay()
            done.set(True)

        self.create_primary_button(
            card,
            text="Use Official yolo26m.pt",
            command=choose_official,
            bg=COLORS["primary"],
        ).pack(fill="x", padx=28, pady=(0, 10))
        self.create_primary_button(
            card,
            text="Choose Custom Weight",
            command=choose_custom,
            bg=COLORS["success"],
        ).pack(fill="x", padx=28, pady=(0, 10))
        self.create_secondary_button(
            card,
            text="From Scratch (skip pretrained weight)",
            command=choose_scratch,
        ).pack(fill="x", padx=28, pady=(0, 10))
        self.create_secondary_button(
            card,
            text="Cancel",
            command=cancel,
        ).pack(fill="x", padx=28, pady=(0, 20))

        self.root.wait_variable(done)
        choice = result["choice"]
        if not choice:
            return None
        return str(choice), result["path"]

    def _prompt_training_runtime_settings(
        self,
        max_idx: int,
    ) -> tuple[int, int, int, int, int, str, str | None] | None:
        result: dict[str, Any] = {
            "start_idx": None,
            "end_idx": None,
            "epochs": None,
            "imgsz": None,
            "batch": None,
            "weight_mode": None,
            "custom_weight_path": None,
        }
        done = tk.BooleanVar(value=False)
        overlay = self._open_fullpage_overlay()
        card = tk.Frame(overlay, bg=COLORS["bg_white"], bd=0, highlightthickness=0)
        card.place(relx=0.5, rely=0.5, anchor="center", width=560, height=460)

        self.create_bold_label(card, text="Training Settings", font=self.font_title, fg=COLORS["text_primary"], bg=COLORS["bg_white"], anchor="center").pack(fill="x", padx=20, pady=(20, 14))

        form = tk.Frame(card, bg=COLORS["bg_white"])
        form.pack(fill="x", padx=28, pady=(0, 10))

        def add_combo(row: int, label: str, values: list[str], default: str) -> tk.StringVar:
            self.create_label(form, text=label, font=self.font_primary, fg=COLORS["text_secondary"], bg=COLORS["bg_white"], anchor="w").grid(row=row, column=0, sticky="w", pady=(0, 8))
            var = tk.StringVar(value=default)
            self.create_combobox(form, textvariable=var, values=values, state="readonly", font=self.font_primary).grid(row=row, column=1, sticky="ew", padx=(12, 0), pady=(0, 8))
            return var

        form.grid_columnconfigure(1, weight=1)
        idx_values = [str(i) for i in range(1, max_idx + 1)]
        start_var = add_combo(0, LANG_MAP[self.lang].get("train_range_start", "Start Index (1-based)"), idx_values, "1")
        end_var = add_combo(1, LANG_MAP[self.lang].get("train_range_end", "End Index (1-based)"), idx_values, str(max_idx))
        epochs_var = add_combo(2, LANG_MAP[self.lang].get("train_epochs", "Epochs"), ["10", "20", "50", "100", "200", "300"], "50")
        imgsz_var = add_combo(3, LANG_MAP[self.lang].get("train_imgsz", "Image Size"), ["320", "416", "512", "640", "768", "960", "1280"], "640")
        batch_var = add_combo(4, "Batch Size", ["-1 (Auto)", "1", "2", "4", "8", "16", "32", "64"], "-1 (Auto)")
        weight_var = add_combo(
            5,
            "Weight",
            [
                "Use Official yolo26m.pt",
                "Choose Custom Weight",
                "From Scratch",
            ],
            "Use Official yolo26m.pt",
        )
        custom_weight_var = tk.StringVar(value="")
        custom_row = tk.Frame(form, bg=COLORS["bg_white"])
        custom_row.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        custom_row.grid_columnconfigure(1, weight=1)
        self.create_label(custom_row, text="Custom Weight", font=self.font_primary, fg=COLORS["text_secondary"], bg=COLORS["bg_white"], anchor="w").grid(row=0, column=0, sticky="w")
        custom_entry = self.create_entry(custom_row, textvariable=custom_weight_var, font=self.font_mono, state="readonly", readonlybackground=COLORS["bg_light"], fg=COLORS["text_primary"])
        custom_entry.grid(row=0, column=1, sticky="ew", padx=(12, 8))

        def browse_custom_weight() -> None:
            model_path = filedialog.askopenfilename(
                parent=self.root,
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
                messagebox.showerror("Model Error", str(exc), parent=self.root)
                return
            custom_weight_var.set(os.path.abspath(resolved))

        browse_btn = self.create_secondary_button(custom_row, text="Browse Weight", command=browse_custom_weight)
        browse_btn.grid(row=0, column=2, sticky="e")

        def sync_custom_controls(*_args: Any) -> None:
            mode = str(weight_var.get()).strip()
            is_custom = mode.startswith("Choose Custom")
            custom_entry.configure(state="readonly" if is_custom else "disabled")
            browse_btn.configure(state="normal" if is_custom else "disabled")

        weight_var.trace_add("write", sync_custom_controls)
        sync_custom_controls()

        def confirm() -> None:
            try:
                start_idx = int(str(start_var.get()).strip())
                end_idx = int(str(end_var.get()).strip())
                epochs = int(str(epochs_var.get()).strip())
                imgsz = int(str(imgsz_var.get()).strip())
                batch_text = str(batch_var.get()).strip().split()[0]
                batch = int(batch_text)
                weight_text = str(weight_var.get()).strip()
                if weight_text.startswith("Use Official"):
                    weight_mode = "official"
                elif weight_text.startswith("Choose Custom"):
                    weight_mode = "custom"
                else:
                    weight_mode = "scratch"
                custom_weight_path = str(custom_weight_var.get()).strip() if weight_mode == "custom" else None
                if (
                    start_idx < 1
                    or end_idx < start_idx
                    or end_idx > max_idx
                    or epochs <= 0
                    or imgsz <= 0
                    or (batch == 0 or batch < -1)
                ):
                    raise ValueError("bad range")
                if weight_mode == "custom" and not custom_weight_path:
                    raise ValueError("custom weight missing")
            except Exception:
                messagebox.showwarning(LANG_MAP[self.lang]["title"], "Invalid training settings.", parent=self.root)
                return
            result["start_idx"] = start_idx
            result["end_idx"] = end_idx
            result["epochs"] = epochs
            result["imgsz"] = imgsz
            result["batch"] = batch
            result["weight_mode"] = weight_mode
            result["custom_weight_path"] = custom_weight_path
            self._close_fullpage_overlay()
            done.set(True)

        def cancel() -> None:
            self._close_fullpage_overlay()
            done.set(True)

        self.create_primary_button(card, text="Confirm", command=confirm, bg=COLORS["success"]).pack(fill="x", padx=28, pady=(6, 8))
        self.create_secondary_button(card, text="Cancel", command=cancel).pack(fill="x", padx=28, pady=(0, 18))
        self.root.wait_variable(done)
        if (
            result["start_idx"] is None
            or result["end_idx"] is None
            or result["epochs"] is None
            or result["imgsz"] is None
            or result["batch"] is None
            or result["weight_mode"] is None
        ):
            return None
        return (
            int(result["start_idx"]),
            int(result["end_idx"]),
            int(result["epochs"]),
            int(result["imgsz"]),
            int(result["batch"]),
            str(result["weight_mode"]),
            str(result["custom_weight_path"]) if result["custom_weight_path"] else None,
        )

    def start_training_from_labels(self) -> None:
        training_runner.start_training_from_labels(self, has_yolo=HAS_YOLO)

    def export_golden_folder(self) -> None:
        if not self.project_root:
            messagebox.showwarning(LANG_MAP[self.lang]["title"], LANG_MAP[self.lang]["export_no_project"], parent=self.root)
            return
        if not self.image_files or self.current_idx < 0 or self.current_idx >= len(self.image_files):
            messagebox.showwarning(
                LANG_MAP[self.lang]["title"],
                LANG_MAP[self.lang].get("golden_export_no_image", "No current image to export."),
                parent=self.root,
            )
            return
        if self.img_pil:
            self.save_current()

        out_dir = filedialog.askdirectory(
            parent=self.root,
            title=LANG_MAP[self.lang].get("select_golden_export_folder", "Select Golden Export Folder"),
        )
        if not out_dir:
            return
        out_dir = os.path.abspath(out_dir).replace("\\", "/")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        golden_dir = f"{out_dir}/golden_sample_{timestamp}"
        if os.path.exists(golden_dir):
            suffix = 1
            while os.path.exists(f"{golden_dir}_{suffix}"):
                suffix += 1
            golden_dir = f"{golden_dir}_{suffix}"

        try:
            image_path = self.image_files[self.current_idx]
            image_name = os.path.basename(image_path)
            stem, _ = os.path.splitext(image_name)
            label_path = f"{self.project_root}/labels/{self.current_split}/{stem}.txt"
            if not os.path.isfile(label_path):
                messagebox.showwarning(
                    LANG_MAP[self.lang]["title"],
                    LANG_MAP[self.lang].get(
                        "golden_export_no_label",
                        "Current image has no label txt. Please annotate and save first.",
                    ),
                    parent=self.root,
                )
                return

            os.makedirs(golden_dir, exist_ok=True)
            shutil.copy2(image_path, f"{golden_dir}/{image_name}")
            dst_lbl_path = f"{golden_dir}/{stem}.txt"
            shutil.copy2(label_path, dst_lbl_path)
            yaml_lines = [
                "nc: " + str(len(self.class_names)),
                "names:",
            ]
            for idx, cls_name in enumerate(self.class_names):
                safe_name = cls_name.replace('"', '\\"')
                yaml_lines.append(f'  {idx}: "{safe_name}"')
            atomic_write_text(f"{golden_dir}/dataset.yaml", "\n".join(yaml_lines) + "\n")
            id_choice, sub_id_choice = self._prompt_golden_id_classes(
                {i: n for i, n in enumerate(self.class_names)},
                parent=self.root,
            )
            if id_choice is not None or sub_id_choice is not None:
                id_class_id = id_choice[0] if id_choice is not None else None
                id_class_name = id_choice[1] if id_choice is not None else None
                sub_id_class_id = sub_id_choice[0] if sub_id_choice is not None else None
                sub_id_class_name = sub_id_choice[1] if sub_id_choice is not None else None
                self._write_golden_id_config(
                    golden_dir,
                    id_class_id,
                    id_class_name,
                    sub_id_class_id=sub_id_class_id,
                    sub_id_class_name=sub_id_class_name,
                )

            merged_cut_bg_files: list[str] = []
            if messagebox.askyesno(
                LANG_MAP[self.lang]["title"],
                "Do you have background-cut golden files to combine?\n\n"
                "If Yes, select that golden folder next.",
                parent=self.root,
            ):
                cut_bg_dir = filedialog.askdirectory(
                    parent=self.root,
                    title="Select Background-Cut Golden Folder",
                )
                if cut_bg_dir:
                    merged_cut_bg_files = self._merge_background_cut_golden_folder(cut_bg_dir, golden_dir)

            done_msg = LANG_MAP[self.lang].get("golden_export_done", "Golden folder exported.\nOutput: {path}").format(path=golden_dir)
            if merged_cut_bg_files:
                done_msg += f"\nMerged background-cut files: {len(merged_cut_bg_files)}"
            messagebox.showinfo(
                LANG_MAP[self.lang]["title"],
                done_msg,
                parent=self.root,
            )
        except Exception as exc:
            self.logger.exception("Golden export failed")
            messagebox.showerror(
                LANG_MAP[self.lang]["title"],
                LANG_MAP[self.lang].get("golden_export_failed", "Golden export failed: {err}").format(err=exc),
                parent=self.root,
            )

    def _merge_background_cut_golden_folder(self, source_dir: str, golden_dir: str) -> list[str]:
        src = os.path.abspath(source_dir).replace("\\", "/")
        dst_root = os.path.join(golden_dir, "background_cut_golden").replace("\\", "/")
        if not os.path.isdir(src):
            raise FileNotFoundError(f"Background-cut golden folder not found: {src}")
        os.makedirs(dst_root, exist_ok=True)

        merged_files: list[str] = []
        for base, _dirs, files in os.walk(src):
            rel = os.path.relpath(base, src)
            dst_dir = dst_root if rel in {".", ""} else os.path.join(dst_root, rel).replace("\\", "/")
            os.makedirs(dst_dir, exist_ok=True)
            for name in files:
                src_path = os.path.join(base, name).replace("\\", "/")
                dst_path = os.path.join(dst_dir, name).replace("\\", "/")
                if os.path.exists(dst_path):
                    stem, ext = os.path.splitext(name)
                    suffix = 1
                    while os.path.exists(os.path.join(dst_dir, f"{stem}_{suffix}{ext}").replace("\\", "/")):
                        suffix += 1
                    dst_path = os.path.join(dst_dir, f"{stem}_{suffix}{ext}").replace("\\", "/")
                shutil.copy2(src_path, dst_path)
                merged_files.append(dst_path)

        return merged_files
    
    def export_full_coco(self):
        messagebox.showinfo("Export", "COCO export will be implemented.")
    
    # ==================== Geometry ====================
    def normalize_angle_deg(self, angle_deg: float) -> float:
        return canvas_utils.normalize_angle_deg(angle_deg)    

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
        return (x - self.offset_x) / self.scale, (y - self.offset_y) / self.scale
    
    def img_to_canvas(self, x, y):
        return x * self.scale + self.offset_x, y * self.scale + self.offset_y
    
    def push_history(self) -> None:
        self.history_manager.push_snapshot(self.rects)
    
    def undo(self) -> None:
        if self.history_manager.undo():
            if self.project_root and self.img_pil:
                self.save_current()
            self.render()
    
    def redo(self) -> None:
        if self.history_manager.redo():
            if self.project_root and self.img_pil:
                self.save_current()
            self.render()
    
    def save_and_next(self):
        if self._detect_mode_active and self._detect_workspace_frame is not None:
            self._detect_next_image()
            return
        self.save_current()
        self.current_idx = min(len(self.image_files) - 1, self.current_idx + 1)
        self.load_img()
    
    def prev_img(self):
        if self._detect_mode_active and self._detect_workspace_frame is not None:
            self._detect_prev_image()
            return
        self.save_current()
        self.current_idx = max(0, self.current_idx - 1)
        self.load_img()
    
    def bind_events(self):
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<ButtonPress-3>", self.on_mouse_down_right)
        self.canvas.bind("<B3-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_mouse_up_right)
        self.canvas.bind("<MouseWheel>", self.on_zoom)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        
        self.root.bind("<Key-f>", lambda e: self.save_and_next())
        self.root.bind("<Key-F>", lambda e: self.save_and_next())
        self.root.bind("<Key-d>", lambda e: self.prev_img())
        self.root.bind("<Key-D>", lambda e: self.prev_img())
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

    

# ==================== Entrypoint ====================

def main():
    root = tk.Tk()
    app = GeckoAI(root, startup_mode="chooser")
    root.mainloop()


def main_label():
    root = tk.Tk()
    app = GeckoAI(root, startup_mode="label")
    root.mainloop()


def main_detect():
    root = tk.Tk()
    app = GeckoAI(root, startup_mode="detect")
    root.mainloop()


# Backward-compat alias for older entry modules/imports.
GeckoAILabeller = GeckoAI

if __name__ == "__main__":
    main()

