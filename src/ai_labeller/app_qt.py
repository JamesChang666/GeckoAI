from __future__ import annotations

import argparse
import copy
import csv
import datetime
import math
import os
import shutil
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from typing import Any


def _build_main_window(startup_mode: str):
    _session_progress_cache: dict[str, dict[str, str]] = {}
    _global_theme_mode = "dark"

    def _get_global_theme_mode() -> str:
        return _global_theme_mode

    def _set_global_theme_mode(mode: str) -> None:
        nonlocal _global_theme_mode
        _global_theme_mode = "light" if str(mode).strip().lower() == "light" else "dark"

    if os.name == "nt":
        platform_name = str(os.environ.get("QT_QPA_PLATFORM", "")).strip().lower()
        if platform_name in {"offscreen", "minimal", "headless"}:
            os.environ.pop("QT_QPA_PLATFORM", None)
        if not os.environ.get("QT_QPA_PLATFORM"):
            os.environ["QT_QPA_PLATFORM"] = "windows"

    def _detect_available_camera_indices(max_index: int = 12) -> list[int]:
        try:
            import cv2  # type: ignore
        except Exception:
            return []
        found: list[int] = []
        for idx in range(max(1, int(max_index))):
            cap = None
            try:
                cap = cv2.VideoCapture(int(idx))
                if cap is not None and cap.isOpened():
                    found.append(int(idx))
            except Exception:
                pass
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
        return found
    try:
        from ai_labeller import cut_background_detect
        from ai_labeller.features import golden as golden_core
        from ai_labeller.features import ocr_utils
        from PySide6.QtCore import Qt, QTimer, QPointF, Signal
        from PySide6.QtGui import QImage, QPixmap, QCloseEvent, QPainter, QPen, QColor, QPolygonF, QCursor, QIcon
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QColorDialog,
            QComboBox,
            QDialog,
            QDialogButtonBox,
            QDoubleSpinBox,
            QFileDialog,
            QFormLayout,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QInputDialog,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QListWidget,
            QScrollArea,
            QSlider,
            QSpinBox,
            QStackedWidget,
            QSizePolicy,
            QTextEdit,
            QToolTip,
            QVBoxLayout,
            QWidget,
        )
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        raise RuntimeError(
            "PySide6 is required for geckoai-qt. Install with: pip install -e \".[qt]\""
        ) from exc

    class _SimpleVar:
        def __init__(self, value: Any):
            self._value = value

        def get(self) -> Any:
            return self._value

        def set(self, value: Any) -> None:
            self._value = value

    class ClickableLabel(QLabel):
        clicked = Signal()

        def mousePressEvent(self, event) -> None:
            if event.button() == Qt.MouseButton.LeftButton:
                self.clicked.emit()
            super().mousePressEvent(event)

    def _resolve_app_icon_path() -> str:
        candidates = [
            os.path.join(os.path.dirname(__file__), "assets", "app_icon.png"),
            os.path.join(os.getcwd(), "src", "ai_labeller", "assets", "app_icon.png"),
            os.path.join(os.getcwd(), "ai_labeller", "assets", "app_icon.png"),
            os.path.join(os.getcwd(), "assets", "app_icon.png"),
        ]
        for p in candidates:
            if p and os.path.isfile(p):
                return os.path.abspath(p)
        return ""

    def _load_logo_pixmap(size: int = 18) -> QPixmap:
        icon_path = _resolve_app_icon_path()
        if not icon_path:
            return QPixmap()
        px = QPixmap(icon_path)
        if px.isNull():
            return QPixmap()
        return px.scaled(
            max(1, int(size)),
            max(1, int(size)),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    class PathPickerField(QWidget):
        def __init__(
            self,
            button_text: str,
            dialog_title: str,
            *,
            select_mode: str = "dir",
            file_filter: str = "All files (*.*)",
            parent: QWidget | None = None,
        ):
            super().__init__(parent)
            self._select_mode = "file" if str(select_mode).strip().lower() == "file" else "dir"
            self._dialog_title = str(dialog_title or "Select Path")
            self._file_filter = str(file_filter or "All files (*.*)")
            self._value = ""
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)
            self.btn_pick = QPushButton(button_text or "Select", self)
            self.btn_pick.setMinimumHeight(36)
            self.btn_pick.clicked.connect(self._pick)
            self.lbl_path = QLabel("Not selected", self)
            self.lbl_path.setWordWrap(True)
            self.lbl_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            layout.addWidget(self.btn_pick)
            layout.addWidget(self.lbl_path)

        def set_button_text(self, text: str) -> None:
            self.btn_pick.setText(str(text or "Select"))

        def set_dialog_title(self, title: str) -> None:
            self._dialog_title = str(title or "Select Path")

        def text(self) -> str:
            return str(self._value or "")

        def setText(self, value: str) -> None:
            raw = str(value or "").strip()
            self._value = os.path.abspath(raw) if raw else ""
            self.lbl_path.setText(self._value or "Not selected")

        def setEnabled(self, enabled: bool) -> None:
            super().setEnabled(enabled)
            self.btn_pick.setEnabled(enabled)
            self.lbl_path.setEnabled(enabled)

        def _pick(self) -> None:
            start = self.text().strip()
            start_dir = start if os.path.isdir(start) else (os.path.dirname(start) if os.path.isfile(start) else "")
            if self._select_mode == "file":
                path, _ = QFileDialog.getOpenFileName(
                    self,
                    self._dialog_title,
                    start_dir,
                    self._file_filter,
                )
            else:
                path = QFileDialog.getExistingDirectory(self, self._dialog_title, start_dir)
            if path:
                self.setText(path)

    class LoadProjectDialog(QDialog):
        def __init__(self, parent: QWidget | None = None):
            super().__init__(parent)
            self.setWindowTitle("Load Project")
            self.resize(620, 180)
            self._theme_mode = _get_global_theme_mode()
            self._setup_ui()

        def _setup_ui(self) -> None:
            layout = QVBoxLayout(self)
            top_row = QHBoxLayout()
            top_row.addStretch(1)
            self.btn_theme = QPushButton("Light Mode", self)
            self.btn_theme.clicked.connect(self._toggle_theme)
            top_row.addWidget(self.btn_theme)
            layout.addLayout(top_row)
            self.kind_combo = QComboBox(self)
            self.kind_combo.addItem("Image Folder", "image_folder")
            self.kind_combo.addItem("YOLO Dataset Root", "yolo_dataset")
            self.path_edit = PathPickerField(
                "Select Dataset Folder",
                "Select Folder",
                select_mode="dir",
                parent=self,
            )
            self.kind_combo.currentIndexChanged.connect(self._on_kind_changed)
            form = QFormLayout()
            form.addRow("Project Type", self.kind_combo)
            form.addRow("Path", self.path_edit)
            layout.addLayout(form)
            ok_btn = QPushButton("Next", self)
            cancel_btn = QPushButton("Back", self)
            ok_btn.clicked.connect(self._accept_if_valid)
            cancel_btn.clicked.connect(self.reject)
            ok_btn.setMinimumHeight(42)
            cancel_btn.setMinimumHeight(42)
            ok_btn.setMinimumWidth(140)
            cancel_btn.setMinimumWidth(140)
            ok_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            cancel_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            ok_btn.setStyleSheet(
                "QPushButton{background:#5551FF;color:#FFFFFF;border:none;border-radius:6px;padding:10px 14px;font-size:14px;}"
                "QPushButton:hover{background:#4845E4;}"
            )
            cancel_btn.setStyleSheet(
                "QPushButton{background:#F24822;color:#FFFFFF;border:none;border-radius:6px;padding:10px 14px;font-size:14px;}"
                "QPushButton:hover{background:#D63F1D;}"
            )
            btn_col = QVBoxLayout()
            btn_col.setSpacing(8)
            btn_col.addWidget(ok_btn)
            btn_col.addWidget(cancel_btn)
            layout.addLayout(btn_col)
            self._on_kind_changed()
            self._apply_theme_styles()

        def _on_kind_changed(self) -> None:
            kind = str(self.kind_combo.currentData() or "image_folder")
            if kind == "yolo_dataset":
                self.path_edit.set_button_text("Select YOLO Dataset Folder")
                self.path_edit.set_dialog_title("Select YOLO Dataset Root")
            else:
                self.path_edit.set_button_text("Select Image Folder")
                self.path_edit.set_dialog_title("Select Image Folder")

        def _toggle_theme(self) -> None:
            self._theme_mode = "light" if self._theme_mode == "dark" else "dark"
            _set_global_theme_mode(self._theme_mode)
            self._apply_theme_styles()

        def _apply_theme_styles(self) -> None:
            if self._theme_mode == "light":
                self.setStyleSheet(
                    "QDialog{background:#FFFFFF;color:#111111;}"
                    "QLabel{color:#111111;}"
                    "QPushButton{background:#F2F2F5;color:#111111;border:1px solid #C8C8CD;border-radius:6px;padding:6px 10px;}"
                    "QPushButton:hover{background:#E8E8ED;}"
                    "QComboBox{background:#FFFFFF;color:#111111;border:1px solid #D1D1D6;border-radius:6px;padding:4px;}"
                )
                self.btn_theme.setText("Dark Mode")
            else:
                self.setStyleSheet(
                    "QDialog{background:#1F1F1F;color:#F2F2F2;}"
                    "QLabel{color:#F2F2F2;}"
                    "QPushButton{background:#2F2F2F;color:#F2F2F2;border:1px solid #595959;border-radius:6px;padding:6px 10px;}"
                    "QPushButton:hover{background:#3A3A3A;}"
                    "QComboBox{background:#2A2A2A;color:#F2F2F2;border:1px solid #595959;border-radius:6px;padding:4px;}"
                )
                self.btn_theme.setText("Light Mode")

        def _accept_if_valid(self) -> None:
            path = self.path_edit.text().strip()
            if not path:
                QMessageBox.warning(self, "Load Project", "Please select a folder.")
                return
            if not os.path.isdir(path):
                QMessageBox.warning(self, "Load Project", "Folder does not exist.")
                return
            self.accept()

        def payload(self) -> dict[str, str]:
            return {
                "kind": str(self.kind_combo.currentData() or "image_folder"),
                "path": os.path.abspath(self.path_edit.text().strip()),
            }

    class TrainSettingsDialog(QDialog):
        def __init__(self, parent: QWidget | None = None, default_model: str = ""):
            super().__init__(parent)
            self.setWindowTitle("Train Settings")
            self.resize(760, 320)
            self._default_model = default_model
            self._theme_mode = _get_global_theme_mode()
            self._setup_ui()

        def _setup_ui(self) -> None:
            layout = QVBoxLayout(self)
            top_row = QHBoxLayout()
            top_row.addStretch(1)
            self.btn_theme = QPushButton("Light Mode", self)
            self.btn_theme.clicked.connect(self._toggle_theme)
            top_row.addWidget(self.btn_theme)
            layout.addLayout(top_row)
            self.output_dir = PathPickerField(
                "Select Output Folder",
                "Select Training Output Folder",
                select_mode="dir",
                parent=self,
            )

            self.model_path = PathPickerField(
                "Select Model File",
                "Select model",
                select_mode="file",
                file_filter="Model files (*.pt *.onnx);;All files (*.*)",
                parent=self,
            )
            self.model_path.setText(self._default_model or "")

            self.run_name = QLineEdit(self)
            self.run_name.setText(f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
            self.epochs = QSpinBox(self)
            self.epochs.setRange(1, 2000)
            self.epochs.setValue(50)
            self.imgsz = QSpinBox(self)
            self.imgsz.setRange(64, 4096)
            self.imgsz.setSingleStep(32)
            self.imgsz.setValue(640)
            self.batch = QSpinBox(self)
            self.batch.setRange(-1, 512)
            self.batch.setValue(-1)
            self.batch.setSpecialValueText("Auto (-1)")

            form = QFormLayout()
            form.addRow("Output Folder", self.output_dir)
            form.addRow("Model Path (.pt/.onnx)", self.model_path)
            form.addRow("Run Name", self.run_name)
            form.addRow("Epochs", self.epochs)
            form.addRow("Image Size", self.imgsz)
            form.addRow("Batch Size", self.batch)
            layout.addLayout(form)

            ok_btn = QPushButton("Next", self)
            cancel_btn = QPushButton("Back", self)
            ok_btn.clicked.connect(self._accept_if_valid)
            cancel_btn.clicked.connect(self.reject)
            ok_btn.setMinimumHeight(42)
            cancel_btn.setMinimumHeight(42)
            ok_btn.setMinimumWidth(140)
            cancel_btn.setMinimumWidth(140)
            ok_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            cancel_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            ok_btn.setStyleSheet(
                "QPushButton{background:#5551FF;color:#FFFFFF;border:none;border-radius:6px;padding:10px 14px;font-size:14px;}"
                "QPushButton:hover{background:#4845E4;}"
            )
            cancel_btn.setStyleSheet(
                "QPushButton{background:#F24822;color:#FFFFFF;border:none;border-radius:6px;padding:10px 14px;font-size:14px;}"
                "QPushButton:hover{background:#D63F1D;}"
            )
            btn_col = QVBoxLayout()
            btn_col.setSpacing(8)
            btn_col.addWidget(ok_btn)
            btn_col.addWidget(cancel_btn)
            layout.addLayout(btn_col)
            self._apply_theme_styles()

        def _toggle_theme(self) -> None:
            self._theme_mode = "light" if self._theme_mode == "dark" else "dark"
            _set_global_theme_mode(self._theme_mode)
            self._apply_theme_styles()

        def _apply_theme_styles(self) -> None:
            if self._theme_mode == "light":
                self.setStyleSheet(
                    "QDialog{background:#FFFFFF;color:#111111;}"
                    "QLabel{color:#111111;}"
                    "QLineEdit,QSpinBox{background:#FFFFFF;color:#111111;border:1px solid #D1D1D6;border-radius:6px;padding:4px;}"
                    "QPushButton{background:#F2F2F5;color:#111111;border:1px solid #C8C8CD;border-radius:6px;padding:6px 10px;}"
                    "QPushButton:hover{background:#E8E8ED;}"
                )
                self.btn_theme.setText("Dark Mode")
            else:
                self.setStyleSheet(
                    "QDialog{background:#1F1F1F;color:#F2F2F2;}"
                    "QLabel{color:#F2F2F2;}"
                    "QLineEdit,QSpinBox{background:#2A2A2A;color:#F2F2F2;border:1px solid #595959;border-radius:6px;padding:4px;}"
                    "QPushButton{background:#2F2F2F;color:#F2F2F2;border:1px solid #595959;border-radius:6px;padding:6px 10px;}"
                    "QPushButton:hover{background:#3A3A3A;}"
                )
                self.btn_theme.setText("Light Mode")

        def _accept_if_valid(self) -> None:
            out_dir = self.output_dir.text().strip()
            model = self.model_path.text().strip()
            run_name = self.run_name.text().strip()
            if not out_dir:
                QMessageBox.warning(self, "Train Settings", "Output folder is required.")
                return
            if not os.path.isdir(out_dir):
                QMessageBox.warning(self, "Train Settings", "Output folder does not exist.")
                return
            if not model:
                QMessageBox.warning(self, "Train Settings", "Model path is required.")
                return
            if not os.path.isfile(model):
                QMessageBox.warning(self, "Train Settings", "Model file does not exist.")
                return
            if not run_name:
                QMessageBox.warning(self, "Train Settings", "Run name is required.")
                return
            self.accept()

        def payload(self) -> dict[str, Any]:
            return {
                "out_dir": os.path.abspath(self.output_dir.text().strip()),
                "model_path": os.path.abspath(self.model_path.text().strip()),
                "run_name": self.run_name.text().strip(),
                "epochs": int(self.epochs.value()),
                "imgsz": int(self.imgsz.value()),
                "batch": int(self.batch.value()),
            }

    class TrainingMonitorDialog(QDialog):
        def __init__(self, parent: QWidget | None = None, on_stop=None):
            super().__init__(parent)
            self.setWindowTitle("Training Monitor")
            self.resize(980, 640)
            self._on_stop = on_stop
            self._theme_mode = _get_global_theme_mode()
            layout = QVBoxLayout(self)
            top_row = QHBoxLayout()
            top_row.addStretch(1)
            self.btn_theme = QPushButton("Light Mode", self)
            self.btn_theme.clicked.connect(self._toggle_theme)
            top_row.addWidget(self.btn_theme)
            layout.addLayout(top_row)
            self.lbl_command = QLabel(self)
            self.lbl_command.setWordWrap(True)
            self.txt_log = QTextEdit(self)
            self.txt_log.setReadOnly(True)
            btn_stop = QPushButton("Stop", self)
            btn_stop.clicked.connect(self._request_stop)
            btn_save = QPushButton("Save Log", self)
            btn_save.clicked.connect(self._save_log)
            btn_clear = QPushButton("Clear", self)
            btn_clear.clicked.connect(self.txt_log.clear)
            btn_row = QHBoxLayout()
            btn_row.addWidget(btn_stop)
            btn_row.addWidget(btn_save)
            btn_row.addWidget(btn_clear)
            layout.addWidget(self.lbl_command)
            layout.addWidget(self.txt_log, 1)
            layout.addLayout(btn_row)
            self._apply_theme_styles()

        def set_command(self, command_text: str) -> None:
            self.lbl_command.setText(command_text or "")

        def _request_stop(self) -> None:
            try:
                if callable(self._on_stop):
                    self._on_stop()
            except Exception:
                pass

        def _save_log(self) -> None:
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Training Log",
                os.path.join(os.getcwd(), "training.log"),
                "Log Files (*.log);;Text Files (*.txt);;All Files (*.*)",
            )
            if not path:
                return
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self.txt_log.toPlainText())
            except Exception as exc:
                QMessageBox.critical(self, "Save Log", f"Failed to save log:\n{exc}")

        def _toggle_theme(self) -> None:
            self._theme_mode = "light" if self._theme_mode == "dark" else "dark"
            _set_global_theme_mode(self._theme_mode)
            self._apply_theme_styles()

        def _apply_theme_styles(self) -> None:
            if self._theme_mode == "light":
                self.setStyleSheet(
                    "QDialog{background:#FFFFFF;color:#111111;}"
                    "QLabel{color:#111111;}"
                    "QTextEdit{background:#FFFFFF;color:#111111;border:1px solid #D1D1D6;border-radius:6px;}"
                    "QPushButton{background:#F2F2F5;color:#111111;border:1px solid #C8C8CD;border-radius:6px;padding:6px 10px;}"
                    "QPushButton:hover{background:#E8E8ED;}"
                )
                self.btn_theme.setText("Dark Mode")
            else:
                self.setStyleSheet(
                    "QDialog{background:#1F1F1F;color:#F2F2F2;}"
                    "QLabel{color:#F2F2F2;}"
                    "QTextEdit{background:#24262A;color:#F2F2F2;border:1px solid #4A4D52;border-radius:6px;}"
                    "QPushButton{background:#2F2F2F;color:#F2F2F2;border:1px solid #595959;border-radius:6px;padding:6px 10px;}"
                    "QPushButton:hover{background:#3A3A3A;}"
                )
                self.btn_theme.setText("Light Mode")

    class HoverGuideButton(QPushButton):
        def __init__(self, text: str, guide_text: str, parent: QWidget | None = None):
            super().__init__(text, parent)
            self._guide_text = str(guide_text or "")
            self._theme_mode = _get_global_theme_mode()

        def set_theme_mode(self, mode: str) -> None:
            self._theme_mode = "light" if str(mode).strip().lower() == "light" else "dark"

        def enterEvent(self, event) -> None:
            super().enterEvent(event)
            if self._guide_text:
                if self._theme_mode == "light":
                    tip = (
                        '<div style="background:#FFFFFF;color:#111111;'
                        'border:1px solid #222222;padding:8px;white-space:pre-wrap;">'
                        f"{self._guide_text}</div>"
                    )
                else:
                    tip = (
                        '<div style="background:#1F1F1F;color:#F2F2F2;'
                        'border:1px solid #E5E5E5;padding:8px;white-space:pre-wrap;">'
                        f"{self._guide_text}</div>"
                    )
                QToolTip.showText(QCursor.pos(), tip, self)

        def leaveEvent(self, event) -> None:
            QToolTip.hideText()
            super().leaveEvent(event)

    class DetectSetupDialog(QDialog):
        def __init__(self, parent: QWidget | None = None):
            super().__init__(parent)
            self.setWindowTitle("Detect Setup")
            self.resize(760, 480)
            self._theme_mode = _get_global_theme_mode()
            self._setup_ui()

        def _setup_ui(self) -> None:
            layout = QVBoxLayout(self)
            layout.setContentsMargins(16, 14, 16, 14)
            layout.setSpacing(10)
            top_row = QHBoxLayout()
            top_row.addStretch(1)
            self.btn_theme = QPushButton("Light Mode", self)
            self.btn_theme.clicked.connect(self._toggle_theme)
            top_row.addWidget(self.btn_theme)
            layout.addLayout(top_row)

            intro = QLabel(
                "Compatibility mode: all detect/golden/OCR logic will run in original app functions.",
                self,
            )
            intro.setWordWrap(True)
            intro.setStyleSheet("color:#555;")
            layout.addWidget(intro)

            self._step_titles = ["Model", "Source", "Output", "Golden", "Confidence"]
            self._step_label = QLabel("", self)
            self._step_label.setStyleSheet("font-size:14px;font-weight:600;")
            layout.addWidget(self._step_label)

            self._steps = QStackedWidget(self)
            layout.addWidget(self._steps, 1)

            self.model_path = PathPickerField(
                "Select Model File",
                "Select model",
                select_mode="file",
                file_filter="Model files (*.pt *.onnx);;All files (*.*)",
                parent=self,
            )
            page_model = QWidget(self)
            page_model_layout = QVBoxLayout(page_model)
            page_model_layout.addWidget(QLabel("Select YOLO model (.pt/.onnx)", page_model))
            page_model_layout.addWidget(self.model_path)
            page_model_layout.addStretch(1)
            self._steps.addWidget(page_model)

            page_source = QWidget(self)
            page_source_layout = QVBoxLayout(page_source)
            self._source_kind = ""
            self._source_stack = QStackedWidget(page_source)

            page_source_pick = QWidget(page_source)
            pick_layout = QVBoxLayout(page_source_pick)
            self.btn_use_camera = QPushButton("Use Camera", page_source_pick)
            self.btn_choose_folder = QPushButton("Choose Image Folder", page_source_pick)
            self.btn_use_camera.setMinimumHeight(44)
            self.btn_choose_folder.setMinimumHeight(44)
            self.btn_use_camera.clicked.connect(self._select_source_camera)
            self.btn_choose_folder.clicked.connect(self._select_source_folder)
            pick_layout.addWidget(self.btn_use_camera)
            pick_layout.addWidget(self.btn_choose_folder)
            pick_layout.addStretch(1)
            self._source_stack.addWidget(page_source_pick)

            page_source_camera = QWidget(page_source)
            cam_layout = QGridLayout(page_source_camera)
            self.camera_index_combo = QComboBox(page_source_camera)
            self.camera_index_combo.setEditable(False)
            self.btn_refresh_cameras = QPushButton("Refresh Cameras", page_source_camera)
            self.btn_refresh_cameras.clicked.connect(self._refresh_camera_list)
            self.lbl_camera_status = QLabel("", page_source_camera)
            self.camera_mode_combo = QComboBox(page_source_camera)
            self.camera_mode_combo.addItem("Camera Detect by Frame", "frame")
            self.camera_mode_combo.addItem("Operator Triggered Capture", "triggered")
            cam_layout.addWidget(QLabel("Camera", page_source_camera), 0, 0)
            cam_layout.addWidget(self.camera_index_combo, 0, 1)
            cam_layout.addWidget(self.btn_refresh_cameras, 0, 2)
            cam_layout.addWidget(self.lbl_camera_status, 1, 0, 1, 3)
            cam_layout.addWidget(QLabel("Camera Mode", page_source_camera), 2, 0)
            cam_layout.addWidget(self.camera_mode_combo, 2, 1, 1, 2)
            self.btn_source_back_from_camera = QPushButton("Back: Choose Source", page_source_camera)
            self.btn_source_back_from_camera.clicked.connect(self._back_to_source_choice)
            cam_layout.addWidget(self.btn_source_back_from_camera, 3, 0, 1, 3)
            self._source_stack.addWidget(page_source_camera)

            page_source_folder = QWidget(page_source)
            folder_layout = QVBoxLayout(page_source_folder)
            self.folder_path = PathPickerField(
                "Select Dataset Folder",
                "Select image folder",
                select_mode="dir",
                parent=page_source_folder,
            )
            folder_layout.addWidget(QLabel("Image Folder", page_source_folder))
            folder_layout.addWidget(self.folder_path)
            self.btn_source_back_from_folder = QPushButton("Back: Choose Source", page_source_folder)
            self.btn_source_back_from_folder.clicked.connect(self._back_to_source_choice)
            folder_layout.addWidget(self.btn_source_back_from_folder)
            folder_layout.addStretch(1)
            self._source_stack.addWidget(page_source_folder)

            page_source_layout.addWidget(self._source_stack)
            page_source_layout.addStretch(1)
            self._steps.addWidget(page_source)

            self.output_dir = PathPickerField(
                "Select Output Folder",
                "Select output folder",
                select_mode="dir",
                parent=self,
            )
            page_output = QWidget(self)
            page_output_layout = QVBoxLayout(page_output)
            page_output_layout.addWidget(QLabel("Select output folder", page_output))
            page_output_layout.addWidget(self.output_dir)
            page_output_layout.addStretch(1)
            self._steps.addWidget(page_output)

            self.enable_golden = QCheckBox("Enable Golden Match", self)
            self.enable_golden.toggled.connect(self._refresh_golden_state)
            self.enable_golden.setChecked(True)
            self.golden_dir = PathPickerField(
                "Select Golden Folder",
                "Select golden folder",
                select_mode="dir",
                parent=self,
            )
            golden_row = QHBoxLayout()
            golden_row.addWidget(self.golden_dir, 1)

            self.golden_mode = QComboBox(self)
            self.golden_mode.addItems(["both", "class", "position"])
            self.golden_iou = QDoubleSpinBox(self)
            self.golden_iou.setRange(0.01, 1.0)
            self.golden_iou.setSingleStep(0.01)
            self.golden_iou.setValue(0.50)
            self.include_id_regions_in_match = QCheckBox("Include ID/Sub-ID regions in golden match", self)
            self.include_id_regions_in_match.setChecked(False)
            page_golden = QWidget(self)
            page_golden_layout = QVBoxLayout(page_golden)
            page_golden_layout.addWidget(self.enable_golden)
            page_golden_layout.addLayout(golden_row)
            golden_form = QFormLayout()
            golden_form.addRow("Golden Mode", self.golden_mode)
            golden_form.addRow("Golden IoU", self.golden_iou)
            page_golden_layout.addLayout(golden_form)
            page_golden_layout.addWidget(self.include_id_regions_in_match)
            page_golden_layout.addStretch(1)
            self._steps.addWidget(page_golden)

            conf_box = QGroupBox("Confidence", self)
            conf_layout = QHBoxLayout(conf_box)
            self.conf_slider = QSlider(Qt.Orientation.Horizontal, conf_box)
            self.conf_slider.setRange(1, 100)
            self.conf_slider.setValue(50)
            self.conf_value = QLabel("0.50", conf_box)
            self.conf_slider.valueChanged.connect(
                lambda value: self.conf_value.setText(f"{value / 100.0:.2f}")
            )
            conf_layout.addWidget(self.conf_slider)
            conf_layout.addWidget(self.conf_value)
            page_conf = QWidget(self)
            page_conf_layout = QVBoxLayout(page_conf)
            page_conf_layout.addWidget(conf_box)
            page_conf_layout.addStretch(1)
            self._steps.addWidget(page_conf)

            btn_row = QVBoxLayout()
            self.btn_cancel = QPushButton("Back", self)
            self.btn_ok = QPushButton("Next", self)
            self.btn_cancel.clicked.connect(self._step_back)
            self.btn_ok.clicked.connect(self._step_next)
            self.btn_ok.setMinimumHeight(44)
            self.btn_cancel.setMinimumHeight(44)
            self.btn_ok.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.btn_cancel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.btn_ok.setStyleSheet(
                "QPushButton{background:#5551FF;color:#FFFFFF;border:none;border-radius:6px;padding:10px 14px;font-size:14px;}"
                "QPushButton:hover{background:#4845E4;}"
            )
            self.btn_cancel.setStyleSheet(
                "QPushButton{background:#F24822;color:#FFFFFF;border:none;border-radius:6px;padding:10px 14px;font-size:14px;}"
                "QPushButton:hover{background:#D63F1D;}"
            )
            btn_row.addWidget(self.btn_ok)
            btn_row.addWidget(self.btn_cancel)
            layout.addLayout(btn_row)

            self._refresh_source_state()
            self._refresh_camera_list()
            self._refresh_golden_state()
            self._steps.setCurrentIndex(0)
            self._update_step_nav()
            self._apply_readable_styles()

        def _toggle_theme(self) -> None:
            self._theme_mode = "light" if self._theme_mode == "dark" else "dark"
            _set_global_theme_mode(self._theme_mode)
            self._apply_readable_styles()

        def _update_step_nav(self) -> None:
            idx = int(self._steps.currentIndex())
            total = int(self._steps.count())
            self._step_label.setText(f"Step {idx + 1}/{total}: {self._step_titles[idx]}")
            self.btn_cancel.setText("Back")
            self.btn_ok.setText("Start Detect" if idx == total - 1 else "Next")

        def _select_source_camera(self) -> None:
            self._source_kind = "camera"
            self._refresh_source_state()

        def _select_source_folder(self) -> None:
            self._source_kind = "file"
            self._refresh_source_state()

        def _back_to_source_choice(self) -> None:
            self._source_kind = ""
            self._refresh_source_state()

        def _validate_current_step(self) -> bool:
            idx = int(self._steps.currentIndex())
            if idx == 0:
                if not self.model_path.text().strip():
                    QMessageBox.warning(self, "Detect Setup", "Model path is required.")
                    return False
                if not os.path.isfile(self.model_path.text().strip()):
                    QMessageBox.warning(self, "Detect Setup", "Model file does not exist.")
                    return False
                return True
            if idx == 1:
                if self._source_kind not in {"camera", "file"}:
                    QMessageBox.warning(self, "Detect Setup", "Please choose source type first.")
                    return False
                if self._source_kind == "camera" and self.camera_index_combo.count() <= 0:
                    QMessageBox.warning(self, "Detect Setup", "No camera detected. Click Refresh Cameras or check connection.")
                    return False
                if self._source_kind == "file" and not self.folder_path.text().strip():
                    QMessageBox.warning(self, "Detect Setup", "Image folder is required for folder source.")
                    return False
                if self._source_kind == "file" and not os.path.isdir(self.folder_path.text().strip()):
                    QMessageBox.warning(self, "Detect Setup", "Image folder does not exist.")
                    return False
                return True
            if idx == 2:
                if not self.output_dir.text().strip():
                    QMessageBox.warning(self, "Detect Setup", "Output folder is required.")
                    return False
                if not os.path.isdir(self.output_dir.text().strip()):
                    QMessageBox.warning(self, "Detect Setup", "Output folder does not exist.")
                    return False
                return True
            if idx == 3 and self.enable_golden.isChecked():
                folder = self.golden_dir.text().strip()
                if not folder:
                    QMessageBox.warning(self, "Detect Setup", "Golden folder is required when golden mode is enabled.")
                    return False
                if not os.path.isdir(folder):
                    QMessageBox.warning(self, "Detect Setup", "Golden folder does not exist.")
                    return False
                resolved = golden_core.resolve_golden_project_folder(folder)
                if not resolved:
                    QMessageBox.warning(
                        self,
                        "Detect Setup",
                        "Cannot resolve golden bundle.\nSelect golden export root (has dataset.yaml + label .txt) "
                        "or its background_cut_golden subfolder.",
                    )
                    return False
                self.golden_dir.setText(str(resolved.get("project_root", folder)))
                return True
            return True

        def _step_back(self) -> None:
            idx = int(self._steps.currentIndex())
            if idx <= 0:
                self.reject()
                return
            self._steps.setCurrentIndex(idx - 1)
            self._update_step_nav()

        def _step_next(self) -> None:
            if not self._validate_current_step():
                return
            idx = int(self._steps.currentIndex())
            last = int(self._steps.count()) - 1
            if idx >= last:
                self._confirm()
                return
            self._steps.setCurrentIndex(idx + 1)
            self._update_step_nav()

        def _apply_readable_styles(self) -> None:
            if self._theme_mode == "light":
                self.setStyleSheet(
                    "QDialog{background:#FFFFFF;color:#111111;}"
                    "QLabel{color:#111111;}"
                    "QGroupBox{color:#111111;border:1px solid #D1D1D6;border-radius:8px;margin-top:8px;}"
                    "QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 4px;color:#111111;}"
                    "QLineEdit,QComboBox,QSpinBox,QDoubleSpinBox{background:#FFFFFF;color:#111111;border:1px solid #D1D1D6;border-radius:6px;padding:4px;}"
                    "QPushButton{background:#F2F2F5;color:#111111;border:1px solid #C8C8CD;border-radius:6px;padding:6px 10px;}"
                    "QPushButton:hover{background:#E8E8ED;}"
                    "QCheckBox{color:#111111;}"
                )
                self.btn_theme.setText("Dark Mode")
                path_color = "#333333"
            else:
                self.setStyleSheet(
                    "QDialog{background:#1F1F1F;color:#F2F2F2;}"
                    "QLabel{color:#F2F2F2;}"
                    "QGroupBox{color:#F2F2F2;border:1px solid #4A4A4A;border-radius:8px;margin-top:8px;}"
                    "QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 4px;color:#F2F2F2;}"
                    "QLineEdit,QComboBox,QSpinBox,QDoubleSpinBox{background:#2A2A2A;color:#F2F2F2;border:1px solid #595959;border-radius:6px;padding:4px;}"
                    "QPushButton{background:#2F2F2F;color:#F2F2F2;border:1px solid #595959;border-radius:6px;padding:6px 10px;}"
                    "QPushButton:hover{background:#3A3A3A;}"
                    "QCheckBox{color:#F2F2F2;}"
                )
                self.btn_theme.setText("Light Mode")
                path_color = "#E0E0E0"
            self.btn_use_camera.setStyleSheet(
                "QPushButton{background:#5551FF;color:#FFFFFF;border:none;border-radius:6px;padding:10px 14px;font-size:14px;}"
                "QPushButton:hover{background:#4845E4;}"
            )
            self.btn_choose_folder.setStyleSheet(
                "QPushButton{background:#0FA958;color:#FFFFFF;border:none;border-radius:6px;padding:10px 14px;font-size:14px;}"
                "QPushButton:hover{background:#0C8A48;}"
            )
            green_picker_style = (
                "QPushButton{background:#0FA958;color:#FFFFFF;border:none;border-radius:6px;padding:10px 14px;font-size:14px;}"
                "QPushButton:hover{background:#0C8A48;}"
            )
            if hasattr(self, "model_path") and self.model_path is not None:
                self.model_path.btn_pick.setStyleSheet(green_picker_style)
            if hasattr(self, "folder_path") and self.folder_path is not None:
                self.folder_path.btn_pick.setStyleSheet(green_picker_style)
            if hasattr(self, "output_dir") and self.output_dir is not None:
                self.output_dir.btn_pick.setStyleSheet(green_picker_style)
            if hasattr(self, "golden_dir") and self.golden_dir is not None:
                self.golden_dir.btn_pick.setStyleSheet(green_picker_style)
            self.btn_source_back_from_camera.setStyleSheet(
                "QPushButton{background:#F24822;color:#FFFFFF;border:none;border-radius:6px;padding:10px 14px;font-size:14px;}"
                "QPushButton:hover{background:#D63F1D;}"
            )
            self.btn_source_back_from_folder.setStyleSheet(
                "QPushButton{background:#F24822;color:#FFFFFF;border:none;border-radius:6px;padding:10px 14px;font-size:14px;}"
                "QPushButton:hover{background:#D63F1D;}"
            )
            if hasattr(self, "model_path") and self.model_path is not None:
                self.model_path.lbl_path.setStyleSheet(f"color:{path_color};")
            if hasattr(self, "folder_path") and self.folder_path is not None:
                self.folder_path.lbl_path.setStyleSheet(f"color:{path_color};")
            if hasattr(self, "output_dir") and self.output_dir is not None:
                self.output_dir.lbl_path.setStyleSheet(f"color:{path_color};")
            if hasattr(self, "golden_dir") and self.golden_dir is not None:
                self.golden_dir.lbl_path.setStyleSheet(f"color:{path_color};")

        def _refresh_source_state(self) -> None:
            kind = str(self._source_kind or "").strip().lower()
            if kind == "camera":
                self._source_stack.setCurrentIndex(1)
            elif kind == "file":
                self._source_stack.setCurrentIndex(2)
            else:
                self._source_stack.setCurrentIndex(0)

        def _refresh_camera_list(self) -> None:
            cams = _detect_available_camera_indices()
            self.camera_index_combo.blockSignals(True)
            self.camera_index_combo.clear()
            for idx in cams:
                self.camera_index_combo.addItem(f"Camera {idx}", int(idx))
            self.camera_index_combo.blockSignals(False)
            if cams:
                self.lbl_camera_status.setText(f"Detected cameras: {len(cams)}")
                self.camera_index_combo.setCurrentIndex(0)
            else:
                self.lbl_camera_status.setText("Detected cameras: 0")

        def _selected_camera_index(self) -> int:
            if self.camera_index_combo.count() <= 0:
                return 0
            try:
                return int(self.camera_index_combo.currentData())
            except Exception:
                return 0

        def _refresh_golden_state(self) -> None:
            enabled = self.enable_golden.isChecked()
            self.golden_dir.setEnabled(enabled)
            self.golden_mode.setEnabled(enabled)
            self.golden_iou.setEnabled(enabled)
            self.include_id_regions_in_match.setEnabled(enabled)

        def _adjust_golden_template(self) -> None:
            folder = os.path.abspath(self.golden_dir.text().strip())
            if not folder or not os.path.isdir(folder):
                QMessageBox.warning(self, "Golden Template", "Please select a valid golden folder first.")
                return
            candidates = [os.path.join(folder, "background_cut_golden"), folder]
            bundle_dir = ""
            for c in candidates:
                if not os.path.isdir(c):
                    continue
                if os.path.isfile(os.path.join(c, "golden_template.png")):
                    bundle_dir = c
                    break
            if not bundle_dir:
                QMessageBox.warning(
                    self,
                    "Golden Template",
                    "No golden_template.png found in selected folder.",
                )
                return
            board_path = os.path.join(bundle_dir, "golden_board.png")
            if not os.path.isfile(board_path):
                board_path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Select board image for template adjustment",
                    bundle_dir,
                    "Image files (*.png *.jpg *.jpeg *.bmp)",
                )
                if not board_path:
                    return
            try:
                import cv2  # type: ignore

                board = cv2.imread(board_path)
                if board is None or getattr(board, "size", 0) == 0:
                    QMessageBox.warning(self, "Golden Template", f"Failed to read image:\n{board_path}")
                    return
                roi = cut_background_detect.select_adjustable_template_roi(
                    board,
                    "Adjust Golden Template",
                )
                if roi is None:
                    QMessageBox.information(self, "Golden Template", "Template update cancelled.")
                    return
                x, y, w, h = map(int, roi)
                if w <= 1 or h <= 1:
                    QMessageBox.information(self, "Golden Template", "Template update cancelled.")
                    return
                crop = board[y : y + h, x : x + w]
                if crop is None or getattr(crop, "size", 0) == 0:
                    QMessageBox.warning(self, "Golden Template", "Invalid template area.")
                    return
                template_path = os.path.join(bundle_dir, "golden_template.png")
                if not cv2.imwrite(template_path, crop):
                    QMessageBox.warning(self, "Golden Template", f"Failed to save template:\n{template_path}")
                    return
            except Exception as exc:
                QMessageBox.critical(self, "Golden Template", f"Adjust template failed:\n{exc}")
                return
            QMessageBox.information(self, "Golden Template", f"Template updated:\n{os.path.join(bundle_dir, 'golden_template.png')}")

        def _confirm(self) -> None:
            if not self.model_path.text().strip():
                QMessageBox.warning(self, "Detect Setup", "Model path is required.")
                return
            if not os.path.isfile(self.model_path.text().strip()):
                QMessageBox.warning(self, "Detect Setup", "Model file does not exist.")
                return
            if str(self._source_kind or "").strip().lower() not in {"camera", "file"}:
                QMessageBox.warning(self, "Detect Setup", "Please choose source type first.")
                return
            if str(self._source_kind).strip().lower() == "file" and not self.folder_path.text().strip():
                QMessageBox.warning(self, "Detect Setup", "Image folder is required for folder source.")
                return
            if str(self._source_kind).strip().lower() == "file" and not os.path.isdir(self.folder_path.text().strip()):
                QMessageBox.warning(self, "Detect Setup", "Image folder does not exist.")
                return
            if not self.output_dir.text().strip():
                QMessageBox.warning(self, "Detect Setup", "Output folder is required.")
                return
            if not os.path.isdir(self.output_dir.text().strip()):
                QMessageBox.warning(self, "Detect Setup", "Output folder does not exist.")
                return
            if self.enable_golden.isChecked():
                folder = self.golden_dir.text().strip()
                if not folder:
                    QMessageBox.warning(self, "Detect Setup", "Golden folder is required when golden mode is enabled.")
                    return
                if not os.path.isdir(folder):
                    QMessageBox.warning(self, "Detect Setup", "Golden folder does not exist.")
                    return
                resolved = golden_core.resolve_golden_project_folder(folder)
                if not resolved:
                    QMessageBox.warning(
                        self,
                        "Detect Setup",
                        "Cannot resolve golden bundle.\nSelect golden export root (has dataset.yaml + label .txt) "
                        "or its background_cut_golden subfolder.",
                    )
                    return
                self.golden_dir.setText(str(resolved.get("project_root", folder)))
            self.accept()

        def payload(self) -> dict[str, Any]:
            source_kind = "camera" if str(self._source_kind) == "camera" else "file"
            source_value: int | str
            if source_kind == "camera":
                source_value = int(self._selected_camera_index())
            else:
                source_value = self.folder_path.text().strip()
            payload: dict[str, Any] = {
                "kind": "detect_start",
                "model_path": self.model_path.text().strip(),
                "source_kind": source_kind,
                "source_value": source_value,
                "output_dir": self.output_dir.text().strip(),
                "conf_threshold": float(self.conf_slider.value()) / 100.0,
                "camera_mode": str(self.camera_mode_combo.currentData() or "frame").strip().lower(),
                "run_mode": "golden" if self.enable_golden.isChecked() else "pure_detect",
                "golden_mode": self.golden_mode.currentText().strip().lower() or "both",
                "golden_iou": float(self.golden_iou.value()),
                "golden_dir": self.golden_dir.text().strip(),
                "golden_include_id_regions_in_match": bool(self.include_id_regions_in_match.isChecked()),
            }
            if self.enable_golden.isChecked():
                folder = os.path.abspath(self.golden_dir.text().strip())
                resolved = golden_core.resolve_golden_project_folder(folder)
                root = os.path.abspath(str((resolved or {}).get("project_root", folder)))
                payload["golden_dir"] = root
                payload["golden_label_path"] = str((resolved or {}).get("label_path", ""))
                payload["golden_mapping_path"] = str((resolved or {}).get("mapping_path", ""))
                id_cfg_path = golden_core.find_golden_id_config_in_folder(root)
                id_cfg = golden_core.load_golden_id_config(id_cfg_path)
                if id_cfg:
                    payload["golden_id_class_id"] = id_cfg.get("id_class_id")
                    payload["golden_id_class_name"] = id_cfg.get("id_class_name")
                    payload["golden_sub_id_class_id"] = id_cfg.get("sub_id_class_id")
                    payload["golden_sub_id_class_name"] = id_cfg.get("sub_id_class_name")
            return payload

    class DetectWorkspaceWindow(QMainWindow):
        def __init__(self, payload: dict[str, Any], on_back):
            super().__init__()
            self.payload = dict(payload)
            self._on_back = on_back
            self._model = None
            self._cap = None
            self._timer = QTimer(self)
            self._timer.timeout.connect(self._tick_camera)
            self._source_refresh_timer = QTimer(self)
            self._source_refresh_timer.setInterval(2000)
            self._source_refresh_timer.timeout.connect(self._auto_refresh_source_tick)
            self._auto_batch_running = False
            self._workspace_ready = True
            self._image_paths: list[str] = []
            self._image_idx = 0
            self._known_source_images: set[str] = set()
            self._cache: dict[str, tuple[Any, dict[str, int], str | None, str, str, str]] = {}
            self._logged: set[str] = set()
            self._frame_idx = 0
            self._camera_capture_count = 0
            self._last_plot_bgr = None
            self._camera_trigger_capture_mode = False
            self._camera_manual_folder = ""
            self._last_camera_frame = None
            self._csv_path = ""
            self._saved_image_dir = ""
            self._easy_ocr_engine = None
            self._detect_ocr_warning_shown = False
            self._detect_last_ocr_id = ""
            self._detect_last_ocr_sub_id = ""
            self._detect_last_cut_piece_count = 0
            self._detect_cut_piece_count_by_path: dict[str, int] = {}
            self._detect_spatial_mismatch_rects_norm: list[tuple[float, float, float, float]] = []
            self._detect_last_fail_reason = ""
            self._detect_class_name_map: dict[int, str] = {}
            self._detect_class_color_overrides: dict[int, tuple[int, int, int]] = {}
            self._detect_golden_sample: dict[str, Any] | None = None
            self._detect_bg_cut_bundle = None
            self._theme_mode = _get_global_theme_mode()
            self.detect_run_mode_var = _SimpleVar(str(self.payload.get("run_mode", "pure_detect")).strip().lower() or "pure_detect")
            self.detect_golden_mode_var = _SimpleVar(str(self.payload.get("golden_mode", "both")).strip().lower() or "both")
            self.detect_golden_iou_var = _SimpleVar(float(self.payload.get("golden_iou", 0.5)))
            self.setWindowTitle("GeckoAI Detect Workspace")
            self.resize(1280, 760)
            self._setup_ui()
            self._load_golden_sample_if_needed()
            if not self._workspace_ready:
                return
            self._init_report()
            self._load_model()
            if self._model is not None:
                self._start_source()

        def _setup_ui(self) -> None:
            root = QWidget(self)
            layout = QVBoxLayout(root)
            layout.setContentsMargins(10, 10, 10, 10)
            layout.setSpacing(8)

            top_row = QHBoxLayout()
            self.lbl_logo = ClickableLabel(root)
            self.lbl_logo.setCursor(Qt.CursorShape.PointingHandCursor)
            self.lbl_logo.setToolTip("Back to Home")
            logo_px = _load_logo_pixmap(24)
            if not logo_px.isNull():
                self.lbl_logo.setPixmap(logo_px)
            self.lbl_logo.clicked.connect(self.close)
            self.lbl_status = QLabel("Ready", root)
            self.lbl_verdict = QLabel("Pure Detect", root)
            self.lbl_verdict.setStyleSheet("font-weight:600;color:#C9CED6;")
            self.btn_theme = QPushButton("Light Mode", root)
            self.btn_theme.clicked.connect(self._toggle_theme)
            self.btn_back = QPushButton("Back to Launcher", root)
            self.btn_back.clicked.connect(self.close)
            self.lbl_title = ClickableLabel("GeckoAI", root)
            self.lbl_title.setCursor(Qt.CursorShape.PointingHandCursor)
            self.lbl_title.setToolTip("Back to Home")
            self.lbl_title.clicked.connect(self.close)
            top_row.addWidget(self.lbl_logo)
            top_row.addWidget(self.lbl_title)
            top_row.addWidget(self.lbl_status, 1)
            top_row.addWidget(self.lbl_verdict)
            top_row.addWidget(self.btn_theme)
            top_row.addWidget(self.btn_back)
            layout.addLayout(top_row)

            body = QHBoxLayout()
            self.image_label = QLabel(root)
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.image_label.setStyleSheet("background:#101010;color:#bbb;")
            self.image_label.setText("No image")
            body.addWidget(self.image_label, 4)

            right = QWidget(root)
            right_layout = QVBoxLayout(right)
            right_layout.setContentsMargins(0, 0, 0, 0)
            right_layout.addWidget(QLabel("Detected Classes", right))
            self.class_list = QListWidget(right)
            right_layout.addWidget(self.class_list, 1)
            self.btn_class_color = QPushButton("Class Colors", right)
            self.btn_class_color.clicked.connect(self._configure_class_color)
            right_layout.addWidget(self.btn_class_color)
            nav = QHBoxLayout()
            self.btn_prev = QPushButton("Prev (D)", right)
            self.btn_next = QPushButton("Next (F)", right)
            self.btn_prev.clicked.connect(self._prev_image)
            self.btn_next.clicked.connect(self._next_image)
            nav.addWidget(self.btn_prev)
            nav.addWidget(self.btn_next)
            right_layout.addLayout(nav)
            body.addWidget(right, 1)
            layout.addLayout(body, 1)

            self.setCentralWidget(root)
            self._apply_readable_styles()

        def _toggle_theme(self) -> None:
            self._theme_mode = "light" if self._theme_mode == "dark" else "dark"
            _set_global_theme_mode(self._theme_mode)
            self._apply_readable_styles()

        def _apply_readable_styles(self) -> None:
            if self._theme_mode == "light":
                self.setStyleSheet("background:#F5F5F7;color:#111111;")
                self.lbl_title.setStyleSheet("font-size:16px;font-weight:600;color:#111111;")
                self.lbl_status.setStyleSheet("color:#333333;")
                self.class_list.setStyleSheet(
                    "QListWidget{background:#FFFFFF;color:#111111;border:1px solid #D1D1D6;border-radius:6px;}"
                )
                self.btn_back.setStyleSheet(
                    "QPushButton{background:#5F6368;color:#FFFFFF;border:none;border-radius:6px;padding:6px 10px;}"
                    "QPushButton:hover{background:#4F5357;}"
                )
                self.btn_prev.setStyleSheet(
                    "QPushButton{background:#FFFFFF;color:#111111;border:1px solid #D1D1D6;border-radius:6px;padding:6px 10px;}"
                    "QPushButton:hover{background:#EFEFF4;}"
                )
                self.btn_next.setStyleSheet(
                    "QPushButton{background:#FFFFFF;color:#111111;border:1px solid #D1D1D6;border-radius:6px;padding:6px 10px;}"
                    "QPushButton:hover{background:#EFEFF4;}"
                )
                self.btn_class_color.setStyleSheet(
                    "QPushButton{background:#FFFFFF;color:#111111;border:1px solid #D1D1D6;border-radius:6px;padding:6px 10px;}"
                    "QPushButton:hover{background:#EFEFF4;}"
                )
                self.btn_theme.setText("Dark Mode")
                self.btn_theme.setStyleSheet(
                    "QPushButton{background:#E5E5EA;color:#111111;border:1px solid #D1D1D6;border-radius:6px;padding:6px 10px;}"
                    "QPushButton:hover{background:#DCDCE1;}"
                )
                self.image_label.setStyleSheet("background:#E9EAF0;color:#555555;")
                return
            self.setStyleSheet("background:#1B1C1E;color:#F2F2F2;")
            self.lbl_title.setStyleSheet("font-size:16px;font-weight:600;color:#F2F2F2;")
            self.lbl_status.setStyleSheet("color:#D8DEE9;")
            self.class_list.setStyleSheet(
                "QListWidget{background:#24262A;color:#F2F2F2;border:1px solid #4A4D52;border-radius:6px;}"
            )
            self.btn_back.setStyleSheet(
                "QPushButton{background:#5F6368;color:#FFFFFF;border:none;border-radius:6px;padding:6px 10px;}"
                "QPushButton:hover{background:#4F5357;}"
            )
            self.btn_prev.setStyleSheet(
                "QPushButton{background:#2F2F2F;color:#F2F2F2;border:1px solid #595959;border-radius:6px;padding:6px 10px;}"
                "QPushButton:hover{background:#3A3A3A;}"
            )
            self.btn_next.setStyleSheet(
                "QPushButton{background:#2F2F2F;color:#F2F2F2;border:1px solid #595959;border-radius:6px;padding:6px 10px;}"
                "QPushButton:hover{background:#3A3A3A;}"
            )
            self.btn_class_color.setStyleSheet(
                "QPushButton{background:#2F2F2F;color:#F2F2F2;border:1px solid #595959;border-radius:6px;padding:6px 10px;}"
                "QPushButton:hover{background:#3A3A3A;}"
            )
            self.btn_theme.setText("Light Mode")
            self.btn_theme.setStyleSheet(
                "QPushButton{background:#2F2F2F;color:#F2F2F2;border:1px solid #595959;border-radius:6px;padding:6px 10px;}"
                "QPushButton:hover{background:#3A3A3A;}"
            )
            self.image_label.setStyleSheet("background:#101010;color:#bbb;")

        def _load_golden_sample_if_needed(self) -> None:
            if self.detect_run_mode_var.get() != "golden":
                self._detect_golden_sample = None
                self._detect_bg_cut_bundle = None
                return
            golden_dir = os.path.abspath(str(self.payload.get("golden_dir", "")).strip())
            if not golden_dir or not os.path.isdir(golden_dir):
                QMessageBox.critical(self, "Detect Workspace", "Golden mode enabled but golden folder is invalid.")
                self._workspace_ready = False
                self.close()
                return
            resolved = golden_core.resolve_golden_project_folder(golden_dir)
            if resolved and resolved.get("project_root"):
                golden_dir = os.path.abspath(str(resolved.get("project_root")))
            mapping_path = str(self.payload.get("golden_mapping_path", "")).strip() or str((resolved or {}).get("mapping_path", "")).strip()
            if not mapping_path:
                mapping_path = golden_core.find_dataset_yaml_in_folder(golden_dir) or ""
            label_path = str(self.payload.get("golden_label_path", "")).strip() or str((resolved or {}).get("label_path", "")).strip()
            if not label_path:
                txt_files = sorted(str(p) for p in Path(golden_dir).glob("*.txt") if p.is_file())
                best = ""
                for p in txt_files:
                    if golden_core.parse_yolo_label_file(p):
                        best = p
                        break
                label_path = best or (txt_files[0] if txt_files else "")
            if not mapping_path or not label_path:
                QMessageBox.critical(self, "Detect Workspace", "Golden folder missing mapping or label file.")
                self._workspace_ready = False
                self.close()
                return
            candidates = golden_core.parse_yolo_label_file(label_path)
            class_mapping = golden_core.load_mapping_from_dataset_yaml(mapping_path)
            targets: list[dict[str, Any]] = []
            for class_id, rect_norm in candidates:
                targets.append(
                    {
                        "class_id": int(class_id),
                        "class_name": class_mapping.get(int(class_id)) if class_mapping else None,
                        "rect_norm": rect_norm,
                    }
                )
            bg_cut_bundle_meta = golden_core.load_detect_background_cut_bundle(self, golden_dir)
            self._detect_bg_cut_bundle = bg_cut_bundle_meta.get("bundle") if bg_cut_bundle_meta else None
            if not targets:
                QMessageBox.critical(self, "Detect Workspace", "Golden label has no valid targets.")
                self._workspace_ready = False
                self.close()
                return
            self._detect_golden_sample = {
                "label_path": os.path.abspath(label_path),
                "targets": targets,
                "mapping_path": os.path.abspath(mapping_path),
                "id_class_id": self.payload.get("golden_id_class_id"),
                "id_class_name": self.payload.get("golden_id_class_name"),
                "sub_id_class_id": self.payload.get("golden_sub_id_class_id"),
                "sub_id_class_name": self.payload.get("golden_sub_id_class_name"),
                "include_id_in_match": bool(self.payload.get("golden_include_id_regions_in_match", False)),
                "id_config_path": "",
                "background_cut_root": bg_cut_bundle_meta.get("root") if bg_cut_bundle_meta else None,
                "background_cut_rules": bg_cut_bundle_meta.get("rules_path") if bg_cut_bundle_meta else None,
                "background_cut_template": bg_cut_bundle_meta.get("template_path") if bg_cut_bundle_meta else None,
            }

        def _init_report(self) -> None:
            base = os.path.abspath(str(self.payload.get("output_dir", "")).strip() or os.getcwd())
            os.makedirs(base, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(base, f"detect_results_{ts}")
            os.makedirs(run_dir, exist_ok=True)
            self._saved_image_dir = os.path.join(run_dir, "detected_images")
            os.makedirs(self._saved_image_dir, exist_ok=True)
            self._csv_path = os.path.join(run_dir, f"detect_results_{ts}.csv")
            with open(self._csv_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "id",
                        "sub_id",
                        "image_name",
                        "status",
                        "detected_classes",
                        "reason",
                        "golden_mode",
                        "iou_threshold",
                        "path",
                        "detect_image_path",
                        "details",
                        "golden_image_path",
                        "golden_label_path",
                    ]
                )

        def _resolve_golden_image_path_for_report(self) -> str:
            sample = self._detect_golden_sample or {}
            label_path = str(sample.get("label_path", "")).strip()
            if not label_path:
                return ""
            base_dir = os.path.dirname(os.path.abspath(label_path))
            stem = os.path.splitext(os.path.basename(label_path))[0]
            for ext in (".png", ".jpg", ".jpeg", ".bmp"):
                p = os.path.join(base_dir, stem + ext)
                if os.path.isfile(p):
                    return os.path.abspath(p)
            return ""

        def _should_use_background_cut_detection(self) -> bool:
            if self.detect_run_mode_var.get().strip().lower() != "golden":
                return False
            if self._detect_bg_cut_bundle is None:
                return False
            source_kind = str(self.payload.get("source_kind", "file")).strip().lower()
            return source_kind == "file"

        def _build_cut_background_source_images(self, image_paths: list[str], cv2_module) -> list[str]:
            bundle = self._detect_bg_cut_bundle
            if bundle is None or not image_paths:
                return []
            run_root = os.path.dirname(self._saved_image_dir) if self._saved_image_dir else os.getcwd()
            cut_root = os.path.join(run_root, "cut_input_images")
            os.makedirs(cut_root, exist_ok=True)
            out_paths: list[str] = []
            total_pieces = int(getattr(self, "_detect_last_cut_piece_count", 0) or 0)
            for src in image_paths:
                try:
                    bgr = cv2_module.imread(src)
                except Exception:
                    bgr = None
                if bgr is None or getattr(bgr, "size", 0) == 0:
                    continue
                try:
                    pieces = cut_background_detect.extract_cut_pieces_from_bgr(bgr, bundle)
                except Exception:
                    pieces = []
                if not pieces:
                    continue
                stem = os.path.splitext(os.path.basename(src))[0]
                for i, piece in enumerate(pieces, 1):
                    out_name = f"{stem}_cut_{i:03d}.png"
                    out_path = os.path.join(cut_root, out_name)
                    if os.path.exists(out_path):
                        k = 1
                        while os.path.exists(os.path.join(cut_root, f"{stem}_cut_{i:03d}_{k}.png")):
                            k += 1
                        out_path = os.path.join(cut_root, f"{stem}_cut_{i:03d}_{k}.png")
                    try:
                        ok = bool(cv2_module.imwrite(out_path, piece))
                    except Exception:
                        ok = False
                    if not ok:
                        continue
                    abs_out = os.path.abspath(out_path)
                    out_paths.append(abs_out)
                    self._detect_cut_piece_count_by_path[abs_out] = 1
                    total_pieces += 1
            self._detect_last_cut_piece_count = int(total_pieces)
            return out_paths

        def _scan_source_images(self) -> list[str]:
            source_kind = str(self.payload.get("source_kind", "file")).strip().lower()
            if source_kind != "file":
                return []
            src = os.path.abspath(str(self.payload.get("source_value", "")).strip())
            if not src or not os.path.isdir(src):
                return []
            exts = {".jpg", ".jpeg", ".png", ".bmp"}
            return [str(p) for p in sorted(Path(src).iterdir()) if p.is_file() and p.suffix.lower() in exts]

        def _auto_refresh_source_tick(self) -> None:
            if self._auto_batch_running:
                return
            source_kind = str(self.payload.get("source_kind", "file")).strip().lower()
            if source_kind != "file":
                return
            try:
                import cv2 as cv2_module
            except Exception:
                return
            source_images = [os.path.abspath(p) for p in self._scan_source_images()]
            if not source_images:
                return
            old_known = set(self._known_source_images)
            added_sources = [p for p in source_images if p not in old_known]
            if not added_sources:
                return
            self._known_source_images = set(source_images)
            if self._should_use_background_cut_detection():
                new_detect_paths = self._build_cut_background_source_images(added_sources, cv2_module)
            else:
                new_detect_paths = added_sources
            if not new_detect_paths:
                return
            current_path = self._image_paths[self._image_idx] if self._image_paths and 0 <= self._image_idx < len(self._image_paths) else ""
            merged = sorted(set(self._image_paths).union(os.path.abspath(p) for p in new_detect_paths))
            self._image_paths = list(merged)
            if current_path and current_path in self._image_paths:
                self._image_idx = self._image_paths.index(current_path)
            else:
                self._image_idx = max(0, min(self._image_idx, len(self._image_paths) - 1))
            self.lbl_status.setText(f"Auto refresh: +{len(new_detect_paths)} image(s)")
            QTimer.singleShot(0, lambda: self._run_detect_batch_all_images(cv2_module, paths=new_detect_paths, regenerate_report=True))

        def _extract_ocr_id_from_result(self, result0: Any) -> str:
            return ocr_utils.extract_ocr_id_from_result(self, result0)

        def _extract_ocr_sub_id_from_result(self, result0: Any) -> str:
            return ocr_utils.extract_ocr_sub_id_from_result(self, result0)

        def _evaluate_golden_match(self, result0: Any) -> tuple[str | None, str]:
            return golden_core.evaluate_golden_match(self, result0)

        def _load_model(self) -> None:
            model_path = str(self.payload.get("model_path", "")).strip()
            try:
                from ultralytics import YOLO
            except Exception:
                QMessageBox.critical(
                    self,
                    "Detect Workspace",
                    "ultralytics is required for detect mode.\nInstall with: pip install ultralytics",
                )
                self.close()
                return
            try:
                self._model = YOLO(model_path)
            except Exception as exc:
                QMessageBox.critical(self, "Detect Workspace", f"Failed to load model:\n{exc}")
                self.close()

        def _start_source(self) -> None:
            try:
                import cv2 as cv2_module
            except Exception:
                QMessageBox.critical(
                    self,
                    "Detect Workspace",
                    "OpenCV is required for detect mode.\nInstall with: pip install opencv-python",
                )
                self.close()
                return
            source_kind = str(self.payload.get("source_kind", "file")).strip().lower()
            if source_kind == "camera":
                if self._source_refresh_timer.isActive():
                    self._source_refresh_timer.stop()
                cam_idx = int(self.payload.get("source_value", 0))
                self._cap = cv2_module.VideoCapture(cam_idx)
                if not self._cap.isOpened():
                    QMessageBox.critical(self, "Detect Workspace", f"Cannot open camera index {cam_idx}.")
                    self.close()
                    return
                camera_mode = str(self.payload.get("camera_mode", "frame")).strip().lower() or "frame"
                self._camera_trigger_capture_mode = camera_mode == "triggered"
                self._last_camera_frame = None
                if self._camera_trigger_capture_mode:
                    base_out = os.path.abspath(str(self.payload.get("output_dir", "")).strip() or os.getcwd())
                    os.makedirs(base_out, exist_ok=True)
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    self._camera_manual_folder = os.path.join(base_out, f"camera_manual_capture_{ts}")
                    os.makedirs(self._camera_manual_folder, exist_ok=True)
                    self._camera_capture_count = 0
                    self.btn_prev.setEnabled(False)
                    self.btn_next.setEnabled(True)
                    self.btn_next.setText("OK Capture")
                    self.lbl_status.setText("camera preview running | click OK Capture")
                    fps = 10.0
                else:
                    self.btn_prev.setEnabled(False)
                    self.btn_next.setEnabled(False)
                    self.btn_next.setText("Next (F)")
                    self.lbl_status.setText(f"camera {cam_idx} running")
                    fps = 30.0
                interval_ms = max(1, int(round(1000.0 / fps)))
                self._timer.start(interval_ms)
                return

            src = os.path.abspath(str(self.payload.get("source_value", "")).strip())
            self._camera_trigger_capture_mode = False
            self._last_camera_frame = None
            self.btn_next.setText("Next (F)")
            self.btn_prev.setEnabled(True)
            self.btn_next.setEnabled(True)
            source_images = self._scan_source_images()
            self._known_source_images = set(os.path.abspath(p) for p in source_images)
            if self._should_use_background_cut_detection():
                cut_paths = self._build_cut_background_source_images(source_images, cv2_module)
                if cut_paths:
                    self._image_paths = cut_paths
                    self.lbl_status.setText(f"Background cut auto-applied: {len(cut_paths)} cut images")
                else:
                    self._image_paths = source_images
                    self.lbl_status.setText("Background cut bundle loaded, but no cut pieces found. Using original images.")
            else:
                self._image_paths = source_images
            if not self._image_paths:
                QMessageBox.warning(self, "Detect Workspace", "No images found in selected folder.")
                self.close()
                return
            self._image_idx = 0
            self._source_refresh_timer.start()
            self._render_current_image(cv2_module)
            QTimer.singleShot(0, self._refit_plot_after_layout)
            QTimer.singleShot(80, self._refit_plot_after_layout)
            QTimer.singleShot(120, lambda: self._run_detect_batch_all_images(cv2_module, regenerate_report=True))

        def _run_cut_background_batch_for_source(self) -> None:
            def _check_cut_bg_cv2_ready() -> tuple[bool, str]:
                try:
                    import cv2  # type: ignore
                except Exception as exc:
                    return False, f"OpenCV import failed: {exc}"
                required = ["imread", "namedWindow", "selectROI", "destroyWindow"]
                missing = [name for name in required if not hasattr(cv2, name)]
                if missing:
                    return False, f"OpenCV missing APIs: {', '.join(missing)}"
                return True, ""

            source_kind = str(self.payload.get("source_kind", "file")).strip().lower()
            if source_kind != "file":
                QMessageBox.information(self, "Cut Background", "Cut Background is only available for image-folder source.")
                return
            src = os.path.abspath(str(self.payload.get("source_value", "")).strip())
            if not os.path.isdir(src):
                QMessageBox.warning(self, "Cut Background", f"Source folder not found:\n{src}")
                return
            ok_cv2, why = _check_cut_bg_cv2_ready()
            if not ok_cv2:
                QMessageBox.critical(
                    self,
                    "Cut Background",
                    "Cut background requires full desktop OpenCV.\n"
                    f"{why}\n\n"
                    "Fix:\n"
                    "pip uninstall opencv-python-headless\n"
                    "pip install opencv-python",
                )
                return
            result = cut_background_detect.run_cut_background_batch(src, parent=self)
            if result is None:
                return
            msg = (
                f"Cut Background done.\n"
                f"Processed images: {result.processed_images}/{result.total_images}\n"
                f"Total crops: {result.total_crops}\n"
                f"Output: {result.output_dir}\n\n"
                f"Switch detect source to output folder now?"
            )
            ans = QMessageBox.question(
                self,
                "Cut Background",
                msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ans != QMessageBox.StandardButton.Yes:
                return
            self.payload["source_kind"] = "file"
            self.payload["source_value"] = os.path.abspath(result.output_dir)
            self._cache.clear()
            self._logged.clear()
            self._image_paths = []
            self._image_idx = 0
            self._known_source_images = set()
            try:
                import cv2
            except Exception:
                QMessageBox.warning(self, "Cut Background", "OpenCV unavailable. Please restart Detect mode.")
                return
            self._start_source()

        def _class_color_bgr(self, class_id: int) -> tuple[int, int, int]:
            cid = int(class_id)
            if cid in self._detect_class_color_overrides:
                return self._detect_class_color_overrides[cid]
            # High-contrast default palette (BGR)
            palette = [
                (255, 85, 85),
                (85, 255, 85),
                (85, 170, 255),
                (255, 170, 85),
                (170, 85, 255),
                (255, 255, 85),
                (85, 255, 255),
                (255, 85, 255),
                (255, 255, 255),
                (0, 165, 255),
                (255, 0, 128),
                (128, 255, 0),
            ]
            return palette[cid % len(palette)]

        def _configure_class_color(self) -> None:
            if not self._detect_class_name_map:
                QMessageBox.information(self, "Class Colors", "No detected classes yet. Run detect first.")
                return
            items = [f"{cid}: {name}" for cid, name in sorted(self._detect_class_name_map.items())]
            item, ok = QInputDialog.getItem(self, "Class Colors", "Select class", items, 0, False)
            if not ok or not str(item).strip():
                return
            try:
                cid = int(str(item).split(":", 1)[0].strip())
            except Exception:
                return
            b, g, r = self._class_color_bgr(cid)
            q = QColor(int(r), int(g), int(b))
            picked = QColorDialog.getColor(q, self, f"Select color for class {cid}")
            if not picked.isValid():
                return
            self._detect_class_color_overrides[cid] = (int(picked.blue()), int(picked.green()), int(picked.red()))
            self._cache.clear()
            try:
                import cv2 as cv2_module
                if self.payload.get("source_kind", "file") == "file":
                    self._render_current_image(cv2_module)
            except Exception:
                pass

        def _draw_detection_plot(self, result0: Any, cv2_module) -> Any:
            img = getattr(result0, "orig_img", None)
            if img is None:
                return result0.plot(line_width=1)
            out = img.copy()
            boxes = getattr(result0, "boxes", None)
            if boxes is None or getattr(boxes, "xyxy", None) is None:
                return out
            det_xyxy = boxes.xyxy.tolist()
            det_cls = boxes.cls.tolist() if getattr(boxes, "cls", None) is not None else []
            det_conf = boxes.conf.tolist() if getattr(boxes, "conf", None) is not None else []
            names_obj = getattr(result0, "names", None)
            idx_to_name: dict[int, str] = {}
            if isinstance(names_obj, dict):
                for key, value in names_obj.items():
                    try:
                        idx_to_name[int(key)] = str(value)
                    except Exception:
                        continue
            elif isinstance(names_obj, (list, tuple)):
                for i, value in enumerate(names_obj):
                    idx_to_name[i] = str(value)
            self._detect_class_name_map = dict(idx_to_name)
            for i, box in enumerate(det_xyxy):
                try:
                    cid = int(det_cls[i]) if i < len(det_cls) else -1
                except Exception:
                    cid = -1
                color = self._class_color_bgr(cid)
                x1, y1, x2, y2 = map(int, [box[0], box[1], box[2], box[3]])
                cv2_module.rectangle(out, (x1, y1), (x2, y2), color, 2)
                cname = idx_to_name.get(cid, str(cid))
                conf = float(det_conf[i]) if i < len(det_conf) else 0.0
                text = f"{cname} {conf:.2f}"
                (tw, th), _ = cv2_module.getTextSize(text, cv2_module.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                ty1 = max(0, y1 - th - 6)
                ty2 = y1
                tx2 = x1 + tw + 6
                cv2_module.rectangle(out, (x1, ty1), (tx2, ty2), color, -1)
                cv2_module.putText(out, text, (x1 + 3, max(12, y1 - 4)), cv2_module.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2_module.LINE_AA)
            return out

        def _draw_mismatch_overlay(self, plot_bgr: Any, cv2_module, result0: Any, status: str | None) -> Any:
            if str(status or "").strip().upper() != "FAIL":
                return plot_bgr
            rects = list(getattr(self, "_detect_spatial_mismatch_rects_norm", []) or [])
            if not rects:
                return plot_bgr
            out = plot_bgr.copy()
            h, w = getattr(result0, "orig_shape", (0, 0))
            if h <= 0 or w <= 0:
                return out
            for r in rects:
                try:
                    x1 = int(float(r[0]) * float(w))
                    y1 = int(float(r[1]) * float(h))
                    x2 = int(float(r[2]) * float(w))
                    y2 = int(float(r[3]) * float(h))
                except Exception:
                    continue
                if x2 <= x1 or y2 <= y1:
                    continue
                cv2_module.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2_module.line(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2_module.line(out, (x2, y1), (x1, y2), (0, 0, 255), 2)
            return out

        def _infer(self, source: Any, cv2_module) -> tuple[Any, Any, dict[str, int]]:
            conf = float(self.payload.get("conf_threshold", 0.5))
            try:
                results = self._model.predict(source=source, conf=conf, verbose=False, device=0)
            except Exception as gpu_exc:
                msg = str(gpu_exc).lower()
                if (
                    "cuda" in msg
                    or "cudart" in msg
                    or "no kernel image" in msg
                    or "no kernel image is available" in msg
                    or "acceleratorerror" in msg
                ):
                    self.lbl_status.setText("GPU unsupported, fallback to CPU")
                    results = self._model.predict(source=source, conf=conf, verbose=False, device="cpu")
                else:
                    raise
            result0 = results[0]
            plot_bgr = self._draw_detection_plot(result0, cv2_module)
            names_obj = getattr(result0, "names", None)
            idx_to_name: dict[int, str] = {}
            if isinstance(names_obj, dict):
                for key, value in names_obj.items():
                    try:
                        idx_to_name[int(key)] = str(value)
                    except Exception:
                        continue
            elif isinstance(names_obj, (list, tuple)):
                for i, value in enumerate(names_obj):
                    idx_to_name[i] = str(value)
            counts: dict[str, int] = {}
            boxes = getattr(result0, "boxes", None)
            if boxes is not None and getattr(boxes, "cls", None) is not None:
                try:
                    cls_values = boxes.cls.tolist()
                except Exception:
                    cls_values = []
                for cls_val in cls_values:
                    try:
                        cls_idx = int(cls_val)
                    except Exception:
                        continue
                    cls_name = idx_to_name.get(cls_idx, str(cls_idx))
                    counts[cls_name] = counts.get(cls_name, 0) + 1
            return result0, plot_bgr, counts

        def _show_plot(self, plot_bgr: Any, cv2_module) -> None:
            self._last_plot_bgr = plot_bgr
            rgb = cv2_module.cvtColor(plot_bgr, cv2_module.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
            pix = QPixmap.fromImage(qimg)
            scaled = pix.scaled(
                max(1, self.image_label.width()),
                max(1, self.image_label.height()),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.image_label.setPixmap(scaled)

        def _update_class_list(self, counts: dict[str, int]) -> None:
            self.class_list.clear()
            if self._detect_last_ocr_id:
                self.class_list.addItem(f"[ID] {self._detect_last_ocr_id}")
            if self._detect_last_ocr_sub_id:
                self.class_list.addItem(f"[SUB_ID] {self._detect_last_ocr_sub_id}")
            if not counts:
                self.class_list.addItem("No detections")
                return
            for name in sorted(counts.keys()):
                self.class_list.addItem(f"{name} x{counts[name]}")

        def _set_verdict(self, status: str | None, detail: str) -> None:
            if status is None:
                self.lbl_verdict.setText("Pure Detect")
                self.lbl_verdict.setStyleSheet("font-weight:600;color:#C9CED6;")
                return
            if status == "PASS":
                self.lbl_verdict.setText(f"PASS {detail}".strip())
                self.lbl_verdict.setStyleSheet("font-weight:600;color:#0FA958;")
                return
            if status == "FAIL":
                self.lbl_verdict.setText(f"FAIL {detail}".strip())
                self.lbl_verdict.setStyleSheet("font-weight:600;color:#F24822;")
                return
            self.lbl_verdict.setText(detail or "No Golden Check")
            self.lbl_verdict.setStyleSheet("font-weight:600;color:#C9CED6;")

        def _append_report_once(
            self,
            image_name: str,
            image_path: str,
            counts: dict[str, int],
            plot_bgr: Any,
            cv2_module,
            status: str | None,
            details: str,
        ) -> None:
            key = image_name.strip()
            if key in self._logged:
                return
            class_text = "; ".join(f"{k} x{v}" for k, v in sorted(counts.items())) if counts else "No detections"
            id_text = str(self._detect_last_ocr_id or "").strip()
            sub_id_text = str(self._detect_last_ocr_sub_id or "").strip() or "none"
            mode_text = self.detect_golden_mode_var.get() if self.detect_run_mode_var.get() == "golden" else ""
            iou_text = f"{float(self.detect_golden_iou_var.get()):.2f}" if self.detect_run_mode_var.get() == "golden" else ""
            reason_text = str(getattr(self, "_detect_last_fail_reason", "") or "").strip()
            status_text = str(status or "")
            path_text = os.path.abspath(str(image_path or "").strip()) if str(image_path or "").strip() else ""
            details_text = str(details or "")
            golden_image_path = self._resolve_golden_image_path_for_report()
            golden_label_path = ""
            if self._detect_golden_sample is not None:
                golden_label_path = str(self._detect_golden_sample.get("label_path", "")).strip()
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            def _safe_token(raw: str) -> str:
                token = str(raw or "").strip()
                bad = "\\/:*?\"<>|"
                for ch in bad:
                    token = token.replace(ch, "_")
                token = token.replace(" ", "_")
                while "__" in token:
                    token = token.replace("__", "_")
                return token.strip("._-")

            invalid_id_tokens = {"", "no_id", "unreadable_id", "none", "null", "n/a", "na"}
            invalid_sub_tokens = {"", "no_sub_id", "unreadable_sub_id", "none", "null", "n/a", "na"}

            id_token_raw = str(self._detect_last_ocr_id or "").strip()
            sub_id_token_raw = str(self._detect_last_ocr_sub_id or "").strip()
            image_base_raw = os.path.splitext(os.path.basename(key))[0]

            id_token = _safe_token(id_token_raw)
            sub_id_token = _safe_token(sub_id_token_raw)
            image_base = _safe_token(image_base_raw) or "detect_result"

            use_id = id_token and id_token_raw.lower() not in invalid_id_tokens
            use_sub = sub_id_token and sub_id_token_raw.lower() not in invalid_sub_tokens
            base_root = id_token if use_id else image_base

            # Keep original piece sequence when sub_id is unavailable.
            seq_token = ""
            marker = "_cut_"
            if marker in image_base_raw:
                tail = image_base_raw.rsplit(marker, 1)[1]
                head = tail.split("_", 1)[0].strip()
                if head.isdigit():
                    seq_token = head

            if use_sub:
                root = f"{base_root}_{sub_id_token}"
            elif seq_token:
                root = f"{base_root}_{seq_token}"
            else:
                root = base_root

            ext = os.path.splitext(os.path.basename(key))[1] or ".jpg"
            out_path = os.path.join(self._saved_image_dir, f"{root}{ext}")
            suffix = 1
            while os.path.exists(out_path):
                out_path = os.path.join(self._saved_image_dir, f"{root}_{suffix}{ext}")
                suffix += 1
            cv2_module.imwrite(out_path, plot_bgr)
            with open(self._csv_path, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        ts,
                        id_text,
                        sub_id_text,
                        image_name,
                        status_text,
                        class_text,
                        reason_text,
                        mode_text,
                        iou_text,
                        path_text,
                        os.path.abspath(out_path),
                        details_text,
                        golden_image_path,
                        golden_label_path,
                    ]
                )
            self._logged.add(key)

        def _render_current_image(self, cv2_module) -> None:
            if not self._image_paths:
                return
            self._image_idx = max(0, min(self._image_idx, len(self._image_paths) - 1))
            path = self._image_paths[self._image_idx]
            basename = os.path.basename(path)
            self.lbl_status.setText(f"{basename} ({self._image_idx + 1}/{len(self._image_paths)})")
            self._detect_last_cut_piece_count = int(self._detect_cut_piece_count_by_path.get(os.path.abspath(path), 0))
            cached = self._cache.get(path)
            if cached is None:
                try:
                    result0, plot_bgr, counts = self._infer(path, cv2_module)
                except Exception as exc:
                    QMessageBox.critical(self, "Detect Workspace", f"Detection failed:\n{exc}")
                    return
                status, detail = self._evaluate_golden_match(result0)
                plot_bgr = self._draw_mismatch_overlay(plot_bgr, cv2_module, result0, status)
                ocr_id = self._detect_last_ocr_id
                ocr_sub_id = self._detect_last_ocr_sub_id
                self._cache[path] = (plot_bgr, counts, status, detail, ocr_id, ocr_sub_id)
            else:
                plot_bgr, counts, status, detail, ocr_id, ocr_sub_id = cached
                self._detect_last_ocr_id = ocr_id
                self._detect_last_ocr_sub_id = ocr_sub_id
            self._set_verdict(status, detail)
            self._show_plot(plot_bgr, cv2_module)
            self._update_class_list(counts)
            self._append_report_once(basename, path, counts, plot_bgr, cv2_module, status, detail)
            self._cache[path] = (
                plot_bgr,
                counts,
                status,
                detail,
                self._detect_last_ocr_id,
                self._detect_last_ocr_sub_id,
            )

        def _prev_image(self) -> None:
            if not self._image_paths:
                return
            import cv2

            self._image_idx = max(0, self._image_idx - 1)
            self._render_current_image(cv2)

        def _next_image(self) -> None:
            if self._camera_trigger_capture_mode:
                self._capture_camera_to_folder_and_switch()
                return
            if not self._image_paths:
                return
            import cv2

            self._image_idx = min(len(self._image_paths) - 1, self._image_idx + 1)
            self._render_current_image(cv2)

        def _capture_camera_to_folder_and_switch(self) -> None:
            if self._cap is None:
                QMessageBox.warning(self, "Detect Workspace", "Camera is not ready.")
                return
            import cv2

            frame = self._last_camera_frame
            if frame is None:
                ok, latest = self._cap.read()
                if not ok or latest is None:
                    QMessageBox.warning(self, "Detect Workspace", "Failed to capture camera frame.")
                    return
                frame = latest
            fallback_base = os.path.abspath(str(self.payload.get("output_dir", "")).strip() or os.getcwd())
            target_dir = self._camera_manual_folder.strip() or os.path.join(fallback_base, "camera_manual_capture")
            os.makedirs(target_dir, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out_path = os.path.join(target_dir, f"camera_capture_{ts}.png")
            ok = bool(cv2.imwrite(out_path, frame))
            if not ok:
                QMessageBox.warning(self, "Detect Workspace", f"Failed to save captured image:\n{out_path}")
                return
            self._camera_capture_count += 1
            abs_out = os.path.abspath(out_path)
            try:
                detect_units: list[tuple[str, Any, str]] = []
                bundle = self._detect_bg_cut_bundle
                if bundle is not None:
                    pieces = cut_background_detect.extract_cut_pieces_from_bgr(frame, bundle)
                    if pieces:
                        piece_root = os.path.join(target_dir, "cut_pieces")
                        os.makedirs(piece_root, exist_ok=True)
                        capture_stem = os.path.splitext(os.path.basename(abs_out))[0]
                        for i, piece in enumerate(pieces, 1):
                            piece_name = f"{capture_stem}_piece_{i:03d}.png"
                            piece_path = os.path.abspath(os.path.join(piece_root, piece_name))
                            try:
                                cv2.imwrite(piece_path, piece)
                            except Exception:
                                piece_path = ""
                            detect_units.append((piece_name, piece, piece_path))
                        self._detect_last_cut_piece_count = len(detect_units)
                    else:
                        self._detect_last_cut_piece_count = 0
                if not detect_units:
                    detect_units.append((os.path.basename(abs_out), abs_out, abs_out))

                shown = False
                for unit_name, unit_source, unit_path in detect_units:
                    result0, plot_bgr, counts = self._infer(unit_source, cv2)
                    status, detail = self._evaluate_golden_match(result0)
                    plot_bgr = self._draw_mismatch_overlay(plot_bgr, cv2, result0, status)
                    self._set_verdict(status, detail)
                    self._show_plot(plot_bgr, cv2)
                    self._update_class_list(counts)
                    self._append_report_once(unit_name, unit_path, counts, plot_bgr, cv2, status, detail)
                    cache_key = os.path.abspath(unit_path) if unit_path else unit_name
                    self._cache[cache_key] = (
                        plot_bgr,
                        counts,
                        status,
                        detail,
                        self._detect_last_ocr_id,
                        self._detect_last_ocr_sub_id,
                    )
                    if unit_path:
                        abs_unit = os.path.abspath(unit_path)
                        if abs_unit not in self._image_paths:
                            self._image_paths.append(abs_unit)
                        self._known_source_images.add(abs_unit)
                        self._image_idx = max(0, len(self._image_paths) - 1)
                    shown = True
                if shown:
                    self._generate_detect_reports()
                piece_text = f", pieces={len(detect_units)}" if len(detect_units) > 1 else ""
                self.lbl_status.setText(
                    f"Captured & detected: {self._camera_capture_count}{piece_text} | {os.path.basename(abs_out)}"
                )
            except Exception as exc:
                self.lbl_status.setText(f"capture detect failed: {exc}")

        def _tick_camera(self) -> None:
            if self._cap is None:
                return
            import cv2

            ok, frame = self._cap.read()
            if not ok:
                self.lbl_status.setText("camera frame read failed")
                return
            if self._camera_trigger_capture_mode:
                self._last_camera_frame = frame.copy()
                self._show_plot(frame, cv2)
                self.lbl_status.setText("camera preview running | click OK Capture")
                return
            try:
                result0, plot_bgr, counts = self._infer(frame, cv2)
            except Exception as exc:
                self.lbl_status.setText(f"camera detect failed: {exc}")
                return
            self._frame_idx += 1
            frame_name = f"frame_{self._frame_idx:06d}.jpg"
            self.lbl_status.setText(f"camera running | frame {self._frame_idx}")
            status, detail = self._evaluate_golden_match(result0)
            plot_bgr = self._draw_mismatch_overlay(plot_bgr, cv2, result0, status)
            self._set_verdict(status, detail)
            self._show_plot(plot_bgr, cv2)
            self._update_class_list(counts)
            self._append_report_once(frame_name, "", counts, plot_bgr, cv2, status, detail)

        def _run_detect_batch_all_images(
            self,
            cv2_module,
            paths: list[str] | None = None,
            regenerate_report: bool = True,
        ) -> None:
            if self._auto_batch_running:
                return
            if str(self.payload.get("source_kind", "file")).strip().lower() != "file":
                return
            run_paths = [os.path.abspath(p) for p in (paths if paths is not None else self._image_paths)]
            run_paths = [p for p in run_paths if p]
            if not run_paths:
                return
            self._auto_batch_running = True
            prev_index = int(self._image_idx)
            try:
                try:
                    self._source_refresh_timer.stop()
                except Exception:
                    pass
                total = len(run_paths)
                for i, p in enumerate(run_paths, 1):
                    if p not in self._image_paths:
                        continue
                    self._image_idx = self._image_paths.index(p)
                    self.lbl_status.setText(f"Auto Detect {i}/{total}: {os.path.basename(p)}")
                    QApplication.processEvents()
                    self._render_current_image(cv2_module)
                    QApplication.processEvents()
                if regenerate_report:
                    self.lbl_status.setText("Auto Detect done. Generating report...")
                    QApplication.processEvents()
                    self._generate_detect_reports()
            finally:
                if self._image_paths:
                    self._image_idx = max(0, min(prev_index, len(self._image_paths) - 1))
                    try:
                        self._render_current_image(cv2_module)
                    except Exception:
                        pass
                self._auto_batch_running = False
                try:
                    self._source_refresh_timer.start()
                except Exception:
                    pass

        def _refit_plot_after_layout(self) -> None:
            if self._last_plot_bgr is None:
                return
            try:
                import cv2
                self._show_plot(self._last_plot_bgr, cv2)
            except Exception:
                return

        def _generate_detect_reports(self) -> None:
            if not self._csv_path or not os.path.isfile(self._csv_path):
                return
            try:
                from ai_labeller import detection_report_generator as drg
                loaded = drg.load_data(self._csv_path)
                if isinstance(loaded, tuple):
                    records, has_golden = loaded
                else:
                    records, has_golden = loaded, True
                agg = drg.aggregate(records)
                sorted_classes, class_img_count, prefix_stats, status_counts, iou_values = agg
                base = os.path.splitext(self._csv_path)[0]
                drg.build_excel(records, sorted_classes, class_img_count, prefix_stats, status_counts, iou_values, has_golden, base + "_report.xlsx")
                drg.build_html(records, sorted_classes, class_img_count, prefix_stats, status_counts, iou_values, has_golden, base + "_dashboard.html")
                try:
                    drg.build_pdf(records, sorted_classes, class_img_count, prefix_stats, status_counts, iou_values, has_golden, base + "_dashboard.pdf")
                except Exception:
                    pass
            except Exception:
                pass

        def keyPressEvent(self, event) -> None:
            key = event.key()
            if key in {Qt.Key.Key_D, Qt.Key.Key_Left}:
                self._prev_image()
                return
            if key in {Qt.Key.Key_F, Qt.Key.Key_Right}:
                self._next_image()
                return
            super().keyPressEvent(event)

        def showEvent(self, event) -> None:
            super().showEvent(event)
            QTimer.singleShot(0, self._refit_plot_after_layout)
            QTimer.singleShot(80, self._refit_plot_after_layout)

        def resizeEvent(self, event) -> None:
            super().resizeEvent(event)
            if self._last_plot_bgr is None:
                return
            import cv2

            self._show_plot(self._last_plot_bgr, cv2)

        def closeEvent(self, event: QCloseEvent) -> None:
            try:
                self._timer.stop()
            except Exception:
                pass
            try:
                self._source_refresh_timer.stop()
            except Exception:
                pass
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
            try:
                self._generate_detect_reports()
            except Exception:
                pass
            try:
                if callable(self._on_back):
                    self._on_back()
            finally:
                super().closeEvent(event)

    class LabelCanvas(QWidget):
        def __init__(self, parent: QWidget | None = None):
            super().__init__(parent)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.setMouseTracking(True)
            self._pixmap: QPixmap | None = None
            self._img_w = 0
            self._img_h = 0
            self._scale = 1.0
            self._offset_x = 0.0
            self._offset_y = 0.0
            self.rects: list[list[float]] = []
            self._drag_start: tuple[float, float] | None = None
            self._temp_rect: list[float] | None = None
            self._default_class_id = 0
            self.selected_idx: int | None = None
            self.selected_indices: set[int] = set()
            self._mode = ""
            self._move_origin: tuple[float, float] | None = None
            self._move_rect_origin: list[float] | None = None
            self._resize_handle: str | None = None
            self._resize_origin: tuple[float, float] | None = None
            self._resize_rect_origin: list[float] | None = None
            self._rotate_active = False
            self._select_start: tuple[float, float] | None = None
            self._select_rect: list[float] | None = None
            self._history: list[list[list[float]]] = []
            self._history_idx = -1
            self.on_rects_changed = None
            self.on_selection_changed = None
            self.ghost_rects: list[list[float]] = []
            self.resolve_class_name = None
            self.on_paste_prev_box = None

        def set_image(self, pixmap: QPixmap, img_w: int, img_h: int, rects: list[list[float]]) -> None:
            self._pixmap = pixmap
            self._img_w = max(1, int(img_w))
            self._img_h = max(1, int(img_h))
            self.rects = [list(r) for r in rects]
            self._temp_rect = None
            self.selected_idx = None
            self.selected_indices = set()
            self._mode = ""
            self._history = [[list(r) for r in self.rects]]
            self._history_idx = 0
            self.update()
            self._emit_selection_changed()

        def set_default_class_id(self, class_id: int) -> None:
            self._default_class_id = max(0, int(class_id))

        def set_ghost_rects(self, rects: list[list[float]]) -> None:
            self.ghost_rects = [list(r) for r in rects]
            self.update()

        def _emit_rects_changed(self) -> None:
            if callable(self.on_rects_changed):
                self.on_rects_changed()

        def _emit_selection_changed(self) -> None:
            if callable(self.on_selection_changed):
                self.on_selection_changed()

        def _snapshot(self) -> list[list[float]]:
            return [list(r) for r in self.rects]

        def _push_history(self) -> None:
            snap = self._snapshot()
            if self._history_idx >= 0 and self._history[self._history_idx] == snap:
                return
            self._history = self._history[: self._history_idx + 1]
            self._history.append(snap)
            self._history_idx = len(self._history) - 1

        def undo(self) -> None:
            if self._history_idx <= 0:
                return
            self._history_idx -= 1
            self.rects = [list(r) for r in self._history[self._history_idx]]
            if self.selected_idx is not None and self.selected_idx >= len(self.rects):
                self.selected_idx = len(self.rects) - 1 if self.rects else None
            self.selected_indices = {i for i in self.selected_indices if 0 <= i < len(self.rects)}
            if self.selected_idx is not None:
                self.selected_indices.add(self.selected_idx)
            self.update()
            self._emit_rects_changed()
            self._emit_selection_changed()

        def redo(self) -> None:
            if self._history_idx >= len(self._history) - 1:
                return
            self._history_idx += 1
            self.rects = [list(r) for r in self._history[self._history_idx]]
            if self.selected_idx is not None and self.selected_idx >= len(self.rects):
                self.selected_idx = len(self.rects) - 1 if self.rects else None
            self.selected_indices = {i for i in self.selected_indices if 0 <= i < len(self.rects)}
            if self.selected_idx is not None:
                self.selected_indices.add(self.selected_idx)
            self.update()
            self._emit_rects_changed()
            self._emit_selection_changed()

        def set_selected_class_id(self, class_id: int) -> None:
            targets = sorted(i for i in self.selected_indices if 0 <= i < len(self.rects))
            if not targets and self.selected_idx is not None and 0 <= self.selected_idx < len(self.rects):
                targets = [self.selected_idx]
            if not targets:
                return
            cid = max(0, int(class_id))
            for idx in targets:
                self.rects[idx][4] = cid
            self._push_history()
            self.update()
            self._emit_rects_changed()

        def _hit_test(self, ix: float, iy: float) -> int | None:
            return self._hit_test_in_rects(ix, iy, self.rects)

        def _hit_test_in_rects(self, ix: float, iy: float, rects: list[list[float]]) -> int | None:
            hit_idx = None
            hit_area = None
            for idx, rect in enumerate(rects):
                x1, y1, x2, y2 = self._rect_bounds(rect)
                lx, rx = x1, x2
                ty, by = y1, y2
                if lx <= ix <= rx and ty <= iy <= by:
                    area = (rx - lx) * (by - ty)
                    if hit_area is None or area < hit_area:
                        hit_area = area
                        hit_idx = idx
            return hit_idx

        def _rect_angle(self, rect: list[float]) -> float:
            if len(rect) >= 6:
                try:
                    return float(rect[5])
                except Exception:
                    return 0.0
            return 0.0

        def _rect_center_size(self, rect: list[float]) -> tuple[float, float, float, float, float]:
            x1, y1, x2, y2 = rect[:4]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            a = self._rect_angle(rect)
            return cx, cy, w, h, a

        def _rot(self, x: float, y: float, cx: float, cy: float, angle_deg: float) -> tuple[float, float]:
            r = math.radians(angle_deg)
            c = math.cos(r)
            s = math.sin(r)
            dx = x - cx
            dy = y - cy
            return cx + dx * c - dy * s, cy + dx * s + dy * c

        def _rect_corners(self, rect: list[float]) -> list[tuple[float, float]]:
            cx, cy, w, h, a = self._rect_center_size(rect)
            hw = w / 2.0
            hh = h / 2.0
            base = [
                (cx - hw, cy - hh),
                (cx + hw, cy - hh),
                (cx + hw, cy + hh),
                (cx - hw, cy + hh),
            ]
            if abs(a) < 1e-6:
                return base
            return [self._rot(x, y, cx, cy, a) for x, y in base]

        def _rect_bounds(self, rect: list[float]) -> tuple[float, float, float, float]:
            corners = self._rect_corners(rect)
            xs = [p[0] for p in corners]
            ys = [p[1] for p in corners]
            return min(xs), min(ys), max(xs), max(ys)

        def _selected_rect(self) -> list[float] | None:
            if len(self.selected_indices) != 1:
                return None
            if self.selected_idx is None or not (0 <= self.selected_idx < len(self.rects)):
                return None
            return self.rects[self.selected_idx]

        def _selected_handle_points(self) -> dict[str, tuple[float, float]]:
            rect = self._selected_rect()
            if rect is None:
                return {}
            x1, y1, x2, y2 = self._rect_bounds(rect)
            lx, rx = min(x1, x2), max(x1, x2)
            ty, by = min(y1, y2), max(y1, y2)
            mx = (lx + rx) / 2.0
            my = (ty + by) / 2.0
            return {
                "nw": (lx, ty),
                "n": (mx, ty),
                "ne": (rx, ty),
                "e": (rx, my),
                "se": (rx, by),
                "s": (mx, by),
                "sw": (lx, by),
                "w": (lx, my),
            }

        def _hit_handle(self, ix: float, iy: float) -> str | None:
            points = self._selected_handle_points()
            if not points:
                return None
            tol = max(2.0, 8.0 / max(self._scale, 1e-6))
            best_name = None
            best_d = None
            for name, (hx, hy) in points.items():
                d = ((ix - hx) ** 2 + (iy - hy) ** 2) ** 0.5
                if d <= tol and (best_d is None or d < best_d):
                    best_d = d
                    best_name = name
            return best_name

        def _rotate_handle_point(self) -> tuple[float, float] | None:
            rect = self._selected_rect()
            if rect is None:
                return None
            cx, cy, w, h, a = self._rect_center_size(rect)
            top_x, top_y = self._rot(cx, cy - h / 2.0, cx, cy, a)
            vx = top_x - cx
            vy = top_y - cy
            vlen = (vx * vx + vy * vy) ** 0.5
            if vlen <= 1e-6:
                ux, uy = 0.0, -1.0
            else:
                ux, uy = vx / vlen, vy / vlen
            stem_len = max(10.0, 24.0 / max(self._scale, 1e-6))
            return top_x + ux * stem_len, top_y + uy * stem_len

        def _hit_rotate_handle(self, ix: float, iy: float) -> bool:
            p = self._rotate_handle_point()
            if p is None:
                return False
            hx, hy = p
            tol = max(3.0, 10.0 / max(self._scale, 1e-6))
            d = ((ix - hx) ** 2 + (iy - hy) ** 2) ** 0.5
            return d <= tol

        def _calc_viewport(self) -> None:
            if self._pixmap is None:
                return
            cw = max(1, self.width())
            ch = max(1, self.height())
            self._scale = min(cw / max(1, self._img_w), ch / max(1, self._img_h))
            self._offset_x = (cw - self._img_w * self._scale) / 2.0
            self._offset_y = (ch - self._img_h * self._scale) / 2.0

        def _img_to_canvas(self, x: float, y: float) -> tuple[float, float]:
            return x * self._scale + self._offset_x, y * self._scale + self._offset_y

        def _canvas_to_img(self, x: float, y: float) -> tuple[float, float]:
            ix = (x - self._offset_x) / max(self._scale, 1e-6)
            iy = (y - self._offset_y) / max(self._scale, 1e-6)
            ix = max(0.0, min(float(self._img_w), ix))
            iy = max(0.0, min(float(self._img_h), iy))
            return ix, iy

        def paintEvent(self, event) -> None:
            painter = QPainter(self)
            try:
                painter.fillRect(self.rect(), QColor("#101010"))
                if self._pixmap is None:
                    painter.setPen(QColor("#bbbbbb"))
                    painter.drawText(self.rect(), int(Qt.AlignmentFlag.AlignCenter), "Open a folder to start")
                    return
                self._calc_viewport()
                draw_w = int(round(self._img_w * self._scale))
                draw_h = int(round(self._img_h * self._scale))
                target = self.rect().adjusted(int(self._offset_x), int(self._offset_y), 0, 0)
                target.setWidth(draw_w)
                target.setHeight(draw_h)
                painter.drawPixmap(target, self._pixmap)
                # Keep all annotation overlays inside image frame.
                painter.save()
                painter.setClipRect(target)
                if self.ghost_rects:
                    gpen = QPen(QColor("#7E8A9A"))
                    gpen.setWidth(1)
                    gpen.setStyle(Qt.PenStyle.CustomDashLine)
                    gpen.setDashPattern([4.0, 6.0])
                    painter.setPen(gpen)
                    for rect in self.ghost_rects:
                        corners = self._rect_corners(rect)
                        poly = QPolygonF()
                        for px, py in corners:
                            cx, cy = self._img_to_canvas(px, py)
                            poly.append(QPointF(float(cx), float(cy)))
                        painter.drawPolygon(poly)
                pen = QPen(QColor("#00E676"))
                pen.setWidth(2)
                painter.setPen(pen)
                for idx, rect in enumerate(self.rects):
                    corners = self._rect_corners(rect)
                    poly = QPolygonF()
                    for px, py in corners:
                        cx, cy = self._img_to_canvas(px, py)
                        poly.append(QPointF(float(cx), float(cy)))
                    painter.drawPolygon(poly)
                    class_id = int(rect[4]) if len(rect) >= 5 else 0
                    class_name = str(class_id)
                    if callable(self.resolve_class_name):
                        try:
                            class_name = str(self.resolve_class_name(class_id))
                        except Exception:
                            class_name = str(class_id)
                    angle_deg = self._rect_angle(rect)
                    label_text = class_name if abs(angle_deg) <= 1e-3 else f"{class_name} ({angle_deg:.1f}deg)"
                    xs = [pt.x() for pt in poly]
                    ys = [pt.y() for pt in poly]
                    if xs and ys:
                        tx = int(min(xs)) + 6
                        ty = max(8, int(min(ys)) - 22)
                        fm = painter.fontMetrics()
                        tw = fm.horizontalAdvance(label_text) + 8
                        th = fm.height() + 4
                        painter.fillRect(tx - 2, ty - 2, tw, th, QColor(20, 25, 30, 190))
                        painter.setPen(QColor("#FFFFFF"))
                        painter.drawText(tx + 2, ty + fm.ascent(), label_text)
                        painter.setPen(pen)
                    if idx in self.selected_indices or self.selected_idx == idx:
                        spen = QPen(QColor("#FFD84D"))
                        spen.setWidth(2)
                        painter.setPen(spen)
                        bx1, by1, bx2, by2 = self._rect_bounds(rect)
                        cx1, cy1 = self._img_to_canvas(bx1, by1)
                        cx2, cy2 = self._img_to_canvas(bx2, by2)
                        painter.drawRect(int(cx1) - 2, int(cy1) - 2, int(cx2 - cx1) + 4, int(cy2 - cy1) + 4)
                        handles = self._selected_handle_points()
                        for _hname, (hx, hy) in handles.items():
                            chx, chy = self._img_to_canvas(hx, hy)
                            painter.fillRect(int(chx) - 3, int(chy) - 3, 6, 6, QColor("#FFD84D"))
                        rp = self._rotate_handle_point()
                        if rp is not None:
                            rhx, rhy = rp
                            ccx, ccy, _w, h, a = self._rect_center_size(rect)
                            topx, topy = self._rot(ccx, ccy - h / 2.0, ccx, ccy, a)
                            ctx, cty = self._img_to_canvas(topx, topy)
                            crx, cry = self._img_to_canvas(rhx, rhy)
                            painter.drawLine(int(ctx), int(cty), int(crx), int(cry))
                            painter.setBrush(QColor("#FFD84D"))
                            painter.drawEllipse(int(crx) - 4, int(cry) - 4, 8, 8)
                            painter.setBrush(Qt.BrushStyle.NoBrush)
                        painter.setPen(pen)
                if self._temp_rect is not None:
                    tpen = QPen(QColor("#18A0FB"))
                    tpen.setWidth(2)
                    painter.setPen(tpen)
                    x1, y1, x2, y2 = self._temp_rect
                    cx1, cy1 = self._img_to_canvas(min(x1, x2), min(y1, y2))
                    cx2, cy2 = self._img_to_canvas(max(x1, x2), max(y1, y2))
                    painter.drawRect(int(cx1), int(cy1), int(cx2 - cx1), int(cy2 - cy1))
                if self._select_rect is not None:
                    sp = QPen(QColor("#8BC3FF"))
                    sp.setWidth(1)
                    painter.setPen(sp)
                    sx1, sy1, sx2, sy2 = self._select_rect
                    cx1, cy1 = self._img_to_canvas(min(sx1, sx2), min(sy1, sy2))
                    cx2, cy2 = self._img_to_canvas(max(sx1, sx2), max(sy1, sy2))
                    painter.drawRect(int(cx1), int(cy1), int(cx2 - cx1), int(cy2 - cy1))
                painter.restore()
            finally:
                painter.end()

        def mousePressEvent(self, event) -> None:
            if self._pixmap is None:
                return
            if event.button() == Qt.MouseButton.RightButton:
                ix, iy = self._canvas_to_img(float(event.position().x()), float(event.position().y()))
                handled = False
                if callable(self.on_paste_prev_box):
                    try:
                        handled = bool(self.on_paste_prev_box(ix, iy))
                    except Exception:
                        handled = False
                if handled:
                    self.update()
                    return
                self.selected_idx = None
                self.selected_indices = set()
                self._mode = "draw"
                self._drag_start = (ix, iy)
                self._temp_rect = [ix, iy, ix, iy]
                self._emit_selection_changed()
                self.update()
                return
            if event.button() != Qt.MouseButton.LeftButton:
                return
            ix, iy = self._canvas_to_img(float(event.position().x()), float(event.position().y()))
            if self._hit_rotate_handle(ix, iy):
                self._mode = "rotate"
                self._rotate_active = True
                self._temp_rect = None
                self.update()
                return
            mods = event.modifiers()
            if mods & Qt.KeyboardModifier.ControlModifier and self._hit_handle(ix, iy) is None:
                self._mode = "select"
                self._select_start = (ix, iy)
                self._select_rect = [ix, iy, ix, iy]
                self._temp_rect = None
                self.update()
                return
            hit_handle = self._hit_handle(ix, iy)
            if hit_handle is not None:
                self._mode = "resize"
                self._resize_handle = hit_handle
                self._resize_origin = (ix, iy)
                cur = self._selected_rect()
                self._resize_rect_origin = list(cur[:4]) if cur is not None else None
                self._temp_rect = None
                self.update()
                return
            hit_idx = self._hit_test(ix, iy)
            if hit_idx is not None:
                toggle_mode = bool(mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier))
                if toggle_mode:
                    if hit_idx in self.selected_indices:
                        self.selected_indices.discard(hit_idx)
                    else:
                        self.selected_indices.add(hit_idx)
                    self.selected_idx = hit_idx if hit_idx in self.selected_indices else (next(iter(self.selected_indices)) if self.selected_indices else None)
                else:
                    self.selected_idx = hit_idx
                    self.selected_indices = {hit_idx}
                self._mode = "move"
                self._move_origin = (ix, iy)
                self._move_rect_origin = list(self.rects[hit_idx][:4])
                self._temp_rect = None
            else:
                self.selected_idx = None
                self.selected_indices = set()
                self._mode = "draw"
                self._drag_start = (ix, iy)
                self._temp_rect = [ix, iy, ix, iy]
            self._emit_selection_changed()
            self.update()

        def mouseMoveEvent(self, event) -> None:
            if self._pixmap is None:
                return
            ix, iy = self._canvas_to_img(float(event.position().x()), float(event.position().y()))
            if self._mode == "draw" and self._drag_start is not None:
                sx, sy = self._drag_start
                self._temp_rect = [sx, sy, ix, iy]
                self.update()
                return
            if self._mode == "select" and self._select_start is not None:
                sx, sy = self._select_start
                self._select_rect = [sx, sy, ix, iy]
                self.update()
                return
            if (
                self._mode == "move"
                and self.selected_idx is not None
                and self._move_origin is not None
                and self._move_rect_origin is not None
                and 0 <= self.selected_idx < len(self.rects)
            ):
                ox, oy = self._move_origin
                dx = ix - ox
                dy = iy - oy
                x1, y1, x2, y2 = self._move_rect_origin
                nx1 = max(0.0, min(float(self._img_w), x1 + dx))
                ny1 = max(0.0, min(float(self._img_h), y1 + dy))
                nx2 = max(0.0, min(float(self._img_w), x2 + dx))
                ny2 = max(0.0, min(float(self._img_h), y2 + dy))
                self.rects[self.selected_idx][:4] = [nx1, ny1, nx2, ny2]
                self._emit_rects_changed()
                self.update()
                return
            if (
                self._mode == "resize"
                and self.selected_idx is not None
                and self._resize_handle is not None
                and self._resize_rect_origin is not None
                and 0 <= self.selected_idx < len(self.rects)
            ):
                x1, y1, x2, y2 = self._resize_rect_origin
                h = self._resize_handle
                nx1, ny1, nx2, ny2 = x1, y1, x2, y2
                if "w" in h:
                    nx1 = ix
                if "e" in h:
                    nx2 = ix
                if "n" in h:
                    ny1 = iy
                if "s" in h:
                    ny2 = iy
                nx1 = max(0.0, min(float(self._img_w), nx1))
                nx2 = max(0.0, min(float(self._img_w), nx2))
                ny1 = max(0.0, min(float(self._img_h), ny1))
                ny2 = max(0.0, min(float(self._img_h), ny2))
                self.rects[self.selected_idx][:4] = [nx1, ny1, nx2, ny2]
                self._emit_rects_changed()
                self.update()
                return
            if (
                self._mode == "rotate"
                and self._rotate_active
                and self.selected_idx is not None
                and 0 <= self.selected_idx < len(self.rects)
            ):
                rect = self.rects[self.selected_idx]
                cx, cy, _w, _h, _a = self._rect_center_size(rect)
                angle = math.degrees(math.atan2(iy - cy, ix - cx)) + 90.0
                while angle <= -180.0:
                    angle += 360.0
                while angle > 180.0:
                    angle -= 360.0
                if len(rect) >= 6:
                    rect[5] = angle
                else:
                    rect.append(angle)
                self._emit_rects_changed()
                self.update()

        def mouseReleaseEvent(self, event) -> None:
            if self._pixmap is None:
                return
            ix, iy = self._canvas_to_img(float(event.position().x()), float(event.position().y()))
            if self._mode == "draw" and self._drag_start is not None:
                sx, sy = self._drag_start
                x1, y1 = min(sx, ix), min(sy, iy)
                x2, y2 = max(sx, ix), max(sy, iy)
                if (x2 - x1) >= 2 and (y2 - y1) >= 2:
                    self.rects.append([x1, y1, x2, y2, self._default_class_id, 0.0])
                    self.selected_idx = len(self.rects) - 1
                    self.selected_indices = {self.selected_idx}
                    self._emit_rects_changed()
                    self._emit_selection_changed()
                    self._push_history()
            elif self._mode == "select" and self._select_start is not None and self._select_rect is not None:
                sx1, sy1, sx2, sy2 = self._select_rect
                lx, rx = min(sx1, sx2), max(sx1, sx2)
                ty, by = min(sy1, sy2), max(sy1, sy2)
                selected: set[int] = set()
                for idx, rect in enumerate(self.rects):
                    bx1, by1, bx2, by2 = self._rect_bounds(rect)
                    intersects = not (bx2 < lx or bx1 > rx or by2 < ty or by1 > by)
                    if intersects:
                        selected.add(idx)
                self.selected_indices = selected
                self.selected_idx = next(iter(sorted(selected))) if selected else None
                self._emit_selection_changed()
            elif self._mode == "move" and self.selected_idx is not None:
                self._push_history()
            elif self._mode == "resize" and self.selected_idx is not None:
                rect = self.rects[self.selected_idx]
                x1, y1, x2, y2 = rect[:4]
                rect[:4] = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                if (rect[2] - rect[0]) < 2 or (rect[3] - rect[1]) < 2:
                    self.rects.pop(self.selected_idx)
                    self.selected_idx = len(self.rects) - 1 if self.rects else None
                    self._emit_selection_changed()
                self._emit_rects_changed()
                self._push_history()
            elif self._mode == "rotate" and self.selected_idx is not None:
                self._push_history()
            self._drag_start = None
            self._move_origin = None
            self._move_rect_origin = None
            self._resize_handle = None
            self._resize_origin = None
            self._resize_rect_origin = None
            self._rotate_active = False
            self._select_start = None
            self._select_rect = None
            self._mode = ""
            self._temp_rect = None
            self.update()

        def keyPressEvent(self, event) -> None:
            key = event.key()
            mods = event.modifiers()
            if mods & Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_A:
                if self.rects:
                    self.selected_indices = set(range(len(self.rects)))
                    self.selected_idx = 0
                    self._emit_selection_changed()
                    self.update()
                return
            if mods & Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_Z:
                self.undo()
                return
            if mods & Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_Y:
                self.redo()
                return
            if key in {Qt.Key.Key_Q, Qt.Key.Key_E}:
                if self.selected_idx is not None and 0 <= self.selected_idx < len(self.rects):
                    delta = -5.0 if key == Qt.Key.Key_Q else 5.0
                    if mods & Qt.KeyboardModifier.ShiftModifier:
                        delta = -15.0 if key == Qt.Key.Key_Q else 15.0
                    rect = self.rects[self.selected_idx]
                    angle = self._rect_angle(rect) + delta
                    while angle <= -180.0:
                        angle += 360.0
                    while angle > 180.0:
                        angle -= 360.0
                    if len(rect) >= 6:
                        rect[5] = angle
                    else:
                        rect.append(angle)
                    self._push_history()
                    self._emit_rects_changed()
                    self.update()
                return
            if key in {Qt.Key.Key_Delete, Qt.Key.Key_Backspace} and self.rects:
                targets = sorted(i for i in self.selected_indices if 0 <= i < len(self.rects))
                if targets:
                    for idx in reversed(targets):
                        self.rects.pop(idx)
                    self.selected_indices = set()
                    self.selected_idx = None
                elif self.selected_idx is not None and 0 <= self.selected_idx < len(self.rects):
                    self.rects.pop(self.selected_idx)
                    self.selected_indices = set()
                    if self.selected_idx >= len(self.rects):
                        self.selected_idx = len(self.rects) - 1 if self.rects else None
                    if self.selected_idx is not None:
                        self.selected_indices.add(self.selected_idx)
                else:
                    self.rects.pop()
                self.update()
                self._emit_rects_changed()
                self._emit_selection_changed()
                self._push_history()
                return
            super().keyPressEvent(event)

    class CameraCaptureDialog(QDialog):
        def __init__(self, save_dir: str, parent: QWidget | None = None, camera_index: int = 0):
            super().__init__(parent)
            self._save_dir = os.path.abspath(save_dir)
            self._camera_index = int(camera_index)
            self._cap = None
            self._timer = QTimer(self)
            self._timer.setInterval(33)
            self._timer.timeout.connect(self._tick)
            self._last_frame = None
            self._captured_count = 0
            self.setWindowTitle("Load Images from Camera")
            self.resize(980, 660)

            lay = QVBoxLayout(self)
            lay.setContentsMargins(12, 12, 12, 12)
            lay.setSpacing(8)
            self.lbl_info = QLabel(f"Save Folder: {self._save_dir}", self)
            self.lbl_info.setWordWrap(True)
            lay.addWidget(self.lbl_info)
            self.preview = QLabel(self)
            self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.preview.setStyleSheet("background:#101010;color:#CCCCCC;border:1px solid #333333;")
            self.preview.setText("Starting camera...")
            self.preview.setMinimumSize(760, 460)
            lay.addWidget(self.preview, 1)
            row = QHBoxLayout()
            self.lbl_count = QLabel("Captured: 0", self)
            row.addWidget(self.lbl_count)
            row.addStretch(1)
            self.btn_capture = QPushButton("Capture", self)
            self.btn_finish = QPushButton("Finish", self)
            self.btn_cancel = QPushButton("Cancel", self)
            self.btn_capture.clicked.connect(self._capture)
            self.btn_finish.clicked.connect(self._finish)
            self.btn_cancel.clicked.connect(self.reject)
            row.addWidget(self.btn_capture)
            row.addWidget(self.btn_finish)
            row.addWidget(self.btn_cancel)
            lay.addLayout(row)

            try:
                import cv2

                self._cap = cv2.VideoCapture(self._camera_index)
                if self._cap is None or not self._cap.isOpened():
                    QMessageBox.critical(self, "Camera Capture", f"Cannot open camera index {self._camera_index}.")
                    QTimer.singleShot(0, self.reject)
                    return
                self._timer.start()
            except Exception as exc:
                QMessageBox.critical(self, "Camera Capture", f"Failed to open camera:\n{exc}")
                QTimer.singleShot(0, self.reject)

        def captured_count(self) -> int:
            return int(self._captured_count)

        def _tick(self) -> None:
            if self._cap is None:
                return
            try:
                import cv2
            except Exception:
                return
            ok, frame = self._cap.read()
            if not ok or frame is None:
                return
            self._last_frame = frame
            self._show_frame(frame, cv2)

        def _show_frame(self, frame, cv2_module) -> None:
            try:
                rgb = cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
                pix = QPixmap.fromImage(qimg)
                scaled = pix.scaled(
                    max(1, self.preview.width()),
                    max(1, self.preview.height()),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.preview.setPixmap(scaled)
            except Exception:
                pass

        def _capture(self) -> None:
            frame = self._last_frame
            if frame is None:
                QMessageBox.warning(self, "Camera Capture", "No camera frame yet.")
                return
            try:
                import cv2

                os.makedirs(self._save_dir, exist_ok=True)
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                out_path = os.path.join(self._save_dir, f"camera_label_{ts}.png")
                ok = bool(cv2.imwrite(out_path, frame))
                if not ok:
                    QMessageBox.warning(self, "Camera Capture", f"Failed to save image:\n{out_path}")
                    return
                self._captured_count += 1
                self.lbl_count.setText(f"Captured: {self._captured_count}")
            except Exception as exc:
                QMessageBox.warning(self, "Camera Capture", f"Capture failed:\n{exc}")

        def _finish(self) -> None:
            if self._captured_count <= 0:
                QMessageBox.warning(self, "Camera Capture", "Please capture at least one image before finish.")
                return
            self.accept()

        def resizeEvent(self, event) -> None:
            super().resizeEvent(event)
            if self._last_frame is None:
                return
            try:
                import cv2

                self._show_frame(self._last_frame, cv2)
            except Exception:
                pass

        def closeEvent(self, event) -> None:
            try:
                self._timer.stop()
            except Exception:
                pass
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
            super().closeEvent(event)

    class LabelWorkspaceWindow(QMainWindow):
        training_log_signal = Signal(str)
        training_command_signal = Signal(str)
        training_status_signal = Signal(str)
        training_done_signal = Signal(str)
        training_failed_signal = Signal(str)
        training_finalize_signal = Signal()

        def __init__(self, on_back, allow_detect_bridge: bool = True):
            super().__init__()
            self._on_back = on_back
            self._allow_detect_bridge = bool(allow_detect_bridge)
            self._image_paths: list[str] = []
            self._image_idx = 0
            self._class_names: list[str] = ["class0"]
            self._last_bgr = None
            self._labels_by_path: dict[str, list[list[float]]] = {}
            self._project_dir = ""
            self._project_root = ""
            self._is_yolo_project = False
            self._yolo_use_split_layout = False
            self._current_split = "train"
            self._training_running = False
            self._detect_model_default_path = os.path.join(os.path.dirname(__file__), "models", "yolo26m.pt")
            self._detect_model_custom_path = ""
            self._detect_model_path = self._detect_model_default_path
            self._show_prev_labels = False
            self._auto_detect_enabled = False
            self._propagate_enabled = False
            self._propagate_mode = "if_missing"
            self._prev_image_rects: list[list[float]] = []
            self._prev_image_selected_rects: list[list[float]] = []
            self._autosave_enabled = True
            self._autosave_dirty = False
            self._autosave_timer = QTimer(self)
            self._autosave_timer.setSingleShot(True)
            self._autosave_timer.timeout.connect(self._autosave_flush)
            self._auto_refresh_enabled = True
            self._folder_refresh_timer = QTimer(self)
            self._folder_refresh_timer.setInterval(2000)
            self._folder_refresh_timer.timeout.connect(self._auto_refresh_tick)
            self._folder_refresh_timer.start()
            self._progress_state: dict[str, str] = {}
            self._theme_mode = "dark"
            self._training_process: subprocess.Popen[str] | None = None
            self._training_stop_requested = False
            self._train_monitor_dialog: TrainingMonitorDialog | None = None
            self.setWindowTitle("GeckoAI Label Workspace")
            self.resize(1320, 780)
            self.setMinimumSize(980, 680)
            self._setup_ui()
            self.training_log_signal.connect(self._append_training_log)
            self.training_command_signal.connect(self._set_training_command)
            self.training_status_signal.connect(self._on_training_status)
            self.training_done_signal.connect(self._on_training_done)
            self.training_failed_signal.connect(self._on_training_failed)
            self.training_finalize_signal.connect(self._on_training_finalize)
            self._apply_adaptive_window_size()

        def _apply_adaptive_window_size(self) -> None:
            try:
                screen = self.screen()
                if screen is None:
                    return
                geo = screen.availableGeometry()
                w = max(1080, int(geo.width() * 0.86))
                h = max(680, int(geo.height() * 0.86))
                self.resize(min(w, geo.width()), min(h, geo.height()))
            except Exception:
                pass

        def _theme_colors(self) -> dict[str, str]:
            if self._theme_mode == "light":
                return {
                    "primary": "#5551FF",
                    "primary_hover": "#4845E4",
                    "success": "#0FA958",
                    "danger": "#F24822",
                    "bg_dark": "#F5F5F7",
                    "bg_medium": "#E5E5EA",
                    "bg_light": "#FFFFFF",
                    "bg_white": "#FFFFFF",
                    "bg_canvas": "#EFEFF4",
                    "text_primary": "#000000",
                    "text_secondary": "#5C5C5C",
                    "text_white": "#FFFFFF",
                    "toolbar_text": "#000000",
                    "toolbar_border": "#D1D1D6",
                    "border": "#D1D1D6",
                }
            return {
                "primary": "#5551FF",
                "primary_hover": "#4845E4",
                "success": "#0FA958",
                "danger": "#F24822",
                "bg_dark": "#1E1E1E",
                "bg_medium": "#2C2C2C",
                "bg_light": "#F5F5F5",
                "bg_white": "#FFFFFF",
                "bg_canvas": "#18191B",
                "text_primary": "#000000",
                "text_secondary": "#8E8E93",
                "text_white": "#FFFFFF",
                "toolbar_text": "#FFFFFF",
                "toolbar_border": "#38383A",
                "border": "#E5E5EA",
            }

        def _toggle_theme(self) -> None:
            self._theme_mode = "light" if self._theme_mode == "dark" else "dark"
            _set_global_theme_mode(self._theme_mode)
            self._apply_theme_styles()

        def _apply_popup_theme_styles(self) -> None:
            app = QApplication.instance()
            if app is None:
                return
            marker_start = "/*__GECKO_POPUP_THEME_START__*/"
            marker_end = "/*__GECKO_POPUP_THEME_END__*/"
            current = str(app.styleSheet() or "")
            if marker_start in current and marker_end in current:
                try:
                    head, tail = current.split(marker_start, 1)
                    _, tail2 = tail.split(marker_end, 1)
                    current = (head + tail2).strip()
                except Exception:
                    pass
            if self._theme_mode == "light":
                popup = (
                    "QDialog{background:#FFFFFF;color:#111111;}"
                    "QDialog QLabel{color:#111111;}"
                    "QDialog QLineEdit,QDialog QComboBox,QDialog QSpinBox,QDialog QDoubleSpinBox,"
                    "QDialog QListWidget,QDialog QTextEdit{background:#FFFFFF;color:#111111;border:1px solid #CFCFD4;border-radius:6px;padding:4px;}"
                    "QDialog QPushButton{background:#F2F2F5;color:#111111;border:1px solid #C8C8CD;border-radius:6px;padding:6px 10px;}"
                    "QDialog QPushButton:hover{background:#E8E8ED;}"
                    "QMessageBox{background:#FFFFFF;color:#111111;}"
                    "QMessageBox QLabel{color:#111111;}"
                    "QMessageBox QPushButton{background:#F2F2F5;color:#111111;border:1px solid #C8C8CD;border-radius:6px;padding:6px 10px;min-width:72px;}"
                    "QMessageBox QPushButton:hover{background:#E8E8ED;}"
                )
            else:
                popup = (
                    "QDialog{background:#1F1F1F;color:#F2F2F2;}"
                    "QDialog QLabel{color:#F2F2F2;}"
                    "QDialog QLineEdit,QDialog QComboBox,QDialog QSpinBox,QDialog QDoubleSpinBox,"
                    "QDialog QListWidget,QDialog QTextEdit{background:#2A2A2A;color:#F2F2F2;border:1px solid #4A4A4A;border-radius:6px;padding:4px;}"
                    "QDialog QPushButton{background:#2F2F2F;color:#F2F2F2;border:1px solid #595959;border-radius:6px;padding:6px 10px;}"
                    "QDialog QPushButton:hover{background:#3A3A3A;}"
                    "QMessageBox{background:#1F1F1F;color:#F2F2F2;}"
                    "QMessageBox QLabel{color:#F2F2F2;}"
                    "QMessageBox QPushButton{background:#2F2F2F;color:#F2F2F2;border:1px solid #595959;border-radius:6px;padding:6px 10px;min-width:72px;}"
                    "QMessageBox QPushButton:hover{background:#3A3A3A;}"
                )
            merged = "\n".join(
                [
                    current.strip(),
                    marker_start,
                    popup,
                    marker_end,
                ]
            ).strip()
            app.setStyleSheet(merged)

        def _apply_theme_styles(self) -> None:
            c = self._theme_colors()
            self._root_widget.setStyleSheet(f"background:{c['bg_dark']};")
            self._toolbar.setStyleSheet(f"background:{c['bg_dark']};")
            self._title_label.setStyleSheet(f"color:{c['toolbar_text']};font-size:18px;font-weight:600;")
            self.canvas.setStyleSheet(f"background:{c['bg_canvas']};")
            self._right_panel.setStyleSheet(f"background:{c['bg_light']};border-radius:8px;")
            self._right_scroll.setStyleSheet(f"QScrollArea{{background:{c['bg_light']};border:none;}}")
            self._right_scroll.verticalScrollBar().setStyleSheet(
                f"QScrollBar:vertical{{background:{c['bg_light']};width:10px;margin:0px;}}"
                f"QScrollBar::handle:vertical{{background:{c['border']};min-height:24px;border-radius:5px;}}"
                "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0px;}"
                "QScrollBar::add-page:vertical,QScrollBar::sub-page:vertical{background:transparent;}"
            )
            self._right_scroll_panel.setStyleSheet(f"background:{c['bg_light']};")
            self._nav_widget.setStyleSheet(f"background:{c['bg_light']};border-top:1px solid {c['border']};")

            top_btn_style = (
                f"QPushButton{{background:{c['bg_medium']};color:{c['toolbar_text']};border:1px solid {c['toolbar_border']};border-radius:6px;padding:6px 10px;}}"
                "QPushButton:hover{background:#3A3A3A;}"
            )
            for btn in [
                self.btn_back,
                self.btn_open,
                self.btn_open_camera,
                self.btn_save,
                self.btn_export,
                self.btn_golden,
                self.btn_train,
                self.btn_undo,
                self.btn_redo,
            ]:
                btn.setStyleSheet(top_btn_style)
            self.btn_back.setStyleSheet(
                "QPushButton{background:#5F6368;color:#FFFFFF;border:none;border-radius:6px;padding:6px 10px;}"
                "QPushButton:hover{background:#4F5357;}"
            )
            self.btn_open.setStyleSheet(
                f"QPushButton{{background:{c['primary']};color:{c['text_white']};border:none;border-radius:6px;padding:6px 10px;}}"
                f"QPushButton:hover{{background:{c['primary_hover']};}}"
            )
            self.btn_open_camera.setStyleSheet(
                f"QPushButton{{background:{c['success']};color:{c['text_white']};border:none;border-radius:6px;padding:6px 10px;}}"
                "QPushButton:hover{background:#0C8A48;}"
            )
            self.btn_save.setStyleSheet(
                f"QPushButton{{background:{c['success']};color:{c['text_white']};border:none;border-radius:6px;padding:6px 10px;}}"
                "QPushButton:hover{background:#0C8A48;}"
            )
            self.btn_undo.setStyleSheet(
                "QPushButton{background:#F39C12;color:#FFFFFF;border:none;border-radius:6px;padding:6px 10px;}"
                "QPushButton:hover{background:#D9890F;}"
            )
            self.btn_redo.setStyleSheet(
                "QPushButton{background:#F39C12;color:#FFFFFF;border:none;border-radius:6px;padding:6px 10px;}"
                "QPushButton:hover{background:#D9890F;}"
            )
            self.btn_export.setStyleSheet(
                f"QPushButton{{background:{c['success']};color:{c['text_white']};border:none;border-radius:6px;padding:6px 10px;}}"
                "QPushButton:hover{background:#0C8A48;}"
            )
            self.btn_golden.setStyleSheet(
                "QPushButton{background:#C89B2D;color:#FFFFFF;border:none;border-radius:6px;padding:6px 10px;}"
                "QPushButton:hover{background:#AF8727;}"
            )
            self.btn_train.setStyleSheet(
                f"QPushButton{{background:{c['danger']};color:{c['text_white']};border:none;border-radius:6px;padding:6px 10px;}}"
                "QPushButton:hover{background:#D63F1D;}"
            )
            combo_popup_style_top = (
                f"QComboBox QAbstractItemView{{background:{c['bg_medium']};color:{c['toolbar_text']};"
                f"selection-background-color:{c['primary']};selection-color:{c['text_white']};"
                f"border:1px solid {c['toolbar_border']};outline:none;}}"
            )
            combo_style = (
                f"QComboBox{{background:{c['bg_medium']};color:{c['toolbar_text']};border:1px solid {c['toolbar_border']};border-radius:6px;padding:4px;}}"
                + combo_popup_style_top
            )
            self.combo_split.setStyleSheet(combo_style)
            if hasattr(self, "combo_export_format") and self.combo_export_format is not None:
                self.combo_export_format.setStyleSheet(combo_style)
            self.lbl_status.setStyleSheet(f"color:{c['text_secondary']};")
            self.btn_theme.setText("Dark Mode" if self._theme_mode == "light" else "Light Mode")
            self.btn_theme.setStyleSheet(
                f"QPushButton{{background:{c['bg_medium']};color:{c['toolbar_text']};border:1px solid {c['toolbar_border']};border-radius:6px;padding:6px 10px;}}"
                "QPushButton:hover{background:#3A3A3A;}"
            )
            self.btn_guide.setStyleSheet(
                f"QPushButton{{background:{c['bg_medium']};color:{c['toolbar_text']};border:1px solid {c['toolbar_border']};border-radius:6px;padding:6px 10px;}}"
                "QPushButton:hover{background:#3A3A3A;}"
            )
            QToolTip.setFont(self.font())
            self.btn_guide.set_theme_mode(self._theme_mode)

            card_style = (
                f"QGroupBox{{color:{c['text_primary']};background:{c['bg_white']};border:1px solid {c['border']};border-radius:8px;margin-top:8px;}}"
                f"QGroupBox::title{{subcontrol-origin: margin;left:8px;padding:0 4px;color:{c['text_primary']};}}"
            )
            self._info_box.setStyleSheet(card_style)
            self._class_box.setStyleSheet(card_style)
            self._ai_box.setStyleSheet(card_style)

            self.lbl_filename.setStyleSheet(f"color:{c['text_primary']};")
            self._lbl_select_image.setStyleSheet(f"color:{c['text_secondary']};")
            combo_popup_style_panel = (
                f"QComboBox QAbstractItemView{{background:{c['bg_white']};color:{c['text_primary']};"
                f"selection-background-color:{c['primary']};selection-color:{c['text_white']};"
                f"border:1px solid {c['border']};outline:none;}}"
            )
            self.combo_image.setStyleSheet(
                f"QComboBox{{background:{c['bg_white']};color:{c['text_primary']};border:1px solid {c['border']};border-radius:6px;padding:4px;}}"
                + combo_popup_style_panel
            )
            self.lbl_progress.setStyleSheet(f"color:{c['text_secondary']};")
            self.lbl_box_count.setStyleSheet(f"color:{c['primary']};")
            self.lbl_class_count.setStyleSheet(f"color:{c['primary']};")
            self.btn_remove.setStyleSheet(
                f"QPushButton{{background:{c['danger']};color:{c['text_white']};border:none;border-radius:6px;padding:6px 10px;}}"
                "QPushButton:hover{background:#D63F1D;}"
            )
            self.btn_restore.setStyleSheet(
                f"QPushButton{{background:{c['success']};color:{c['text_white']};border:none;border-radius:6px;padding:6px 10px;}}"
                "QPushButton:hover{background:#0C8A48;}"
            )

            self._lbl_current_class.setStyleSheet(f"color:{c['text_secondary']};")
            self.class_combo.setStyleSheet(
                f"QComboBox{{background:{c['bg_white']};color:{c['text_primary']};border:1px solid {c['border']};border-radius:6px;padding:4px;}}"
                + combo_popup_style_panel
            )
            self.btn_add_class.setStyleSheet(
                f"QPushButton{{background:{c['primary']};color:{c['text_white']};border:none;border-radius:6px;padding:6px 10px;}}"
                f"QPushButton:hover{{background:{c['primary_hover']};}}"
            )
            self.btn_apply_class.setStyleSheet(
                f"QPushButton{{background:{c['success']};color:{c['text_white']};border:none;border-radius:6px;padding:6px 10px;}}"
                "QPushButton:hover{background:#0C8A48;}"
            )
            self.btn_edit_classes.setStyleSheet(
                f"QPushButton{{background:{c['bg_white']};color:{c['text_primary']};border:1px solid {c['border']};border-radius:6px;padding:6px 10px;}}"
                "QPushButton:hover{background:#F3F3F6;}"
            )
            self.btn_clear_labels.setStyleSheet(
                f"QPushButton{{background:{c['danger']};color:{c['text_white']};border:none;border-radius:6px;padding:6px 10px;}}"
                "QPushButton:hover{background:#D63F1D;}"
            )
            cb_border = "#FFFFFF" if self._theme_mode == "dark" else "#000000"
            cb_bg = "#1E1E1E" if self._theme_mode == "dark" else "#FFFFFF"
            cb_tick_path = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "assets",
                    "checkbox_check_white.svg" if self._theme_mode == "dark" else "checkbox_check_black.svg",
                )
            ).replace("\\", "/")
            checkbox_style = (
                f"QCheckBox{{color:{c['text_primary']};}}"
                f"QCheckBox::indicator{{width:14px;height:14px;border-radius:3px;border:1px solid {cb_border};background:{cb_bg};}}"
                f"QCheckBox::indicator:checked{{border:1px solid {cb_border};background:{cb_bg};image:url({cb_tick_path});}}"
            )
            self.chk_show_prev.setStyleSheet(checkbox_style)
            self.chk_auto_refresh_folder.setStyleSheet(checkbox_style)
            if hasattr(self, "class_list") and self.class_list is not None:
                self.class_list.setStyleSheet(
                    f"QListWidget{{background:{c['bg_white']};color:{c['text_primary']};border:1px solid {c['border']};border-radius:6px;}}"
                )
            self.lbl_selected_detail.setStyleSheet(f"color:{c['text_secondary']};")

            self.chk_auto_detect.setStyleSheet(checkbox_style)
            self.chk_propagate.setStyleSheet(checkbox_style)
            self.combo_propagate_mode.setStyleSheet(
                f"QComboBox{{background:{c['bg_white']};color:{c['text_primary']};border:1px solid {c['border']};border-radius:6px;padding:4px;}}"
                + combo_popup_style_panel
            )
            self.combo_model.setStyleSheet(
                f"QComboBox{{background:{c['bg_white']};color:{c['text_primary']};border:1px solid {c['border']};border-radius:6px;padding:4px;}}"
                + combo_popup_style_panel
            )
            self.btn_import_model.setStyleSheet(
                f"QPushButton{{background:{c['primary']};color:{c['text_white']};border:none;border-radius:6px;padding:6px 10px;}}"
                f"QPushButton:hover{{background:{c['primary_hover']};}}"
            )
            self.btn_detect.setStyleSheet(
                f"QPushButton{{background:{c['success']};color:{c['text_white']};border:none;border-radius:6px;padding:6px 10px;}}"
                "QPushButton:hover{background:#0C8A48;}"
            )

            self.btn_prev.setStyleSheet(
                f"QPushButton{{background:{c['bg_white']};color:{c['text_primary']};border:1px solid {c['border']};border-radius:6px;padding:8px 12px;}}"
                "QPushButton:hover{background:#F0F0FF;}"
            )
            self.btn_next.setStyleSheet(
                f"QPushButton{{background:{c['primary']};color:{c['text_white']};border:none;border-radius:6px;padding:8px 12px;}}"
                f"QPushButton:hover{{background:{c['primary_hover']};}}"
            )
            self._apply_popup_theme_styles()

        def _setup_ui(self) -> None:
            root = QWidget(self)
            self._root_widget = root
            layout = QVBoxLayout(root)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

            toolbar = QWidget(root)
            self._toolbar = toolbar
            toolbar.setFixedHeight(56)
            top = QHBoxLayout(toolbar)
            top.setContentsMargins(14, 8, 14, 8)
            top.setSpacing(6)

            self._logo_label = ClickableLabel(toolbar)
            self._logo_label.setCursor(Qt.CursorShape.PointingHandCursor)
            self._logo_label.setToolTip("Back to Home")
            logo_px = _load_logo_pixmap(24)
            if not logo_px.isNull():
                self._logo_label.setPixmap(logo_px)
            self._logo_label.clicked.connect(self.close)

            title = ClickableLabel("GeckoAI", toolbar)
            title.setCursor(Qt.CursorShape.PointingHandCursor)
            title.setToolTip("Back to Home")
            title.clicked.connect(self.close)
            self._title_label = title
            top.addWidget(self._logo_label)
            top.addWidget(title)

            self.btn_back = QPushButton("Back", toolbar)
            self.btn_open = QPushButton("Load Project", toolbar)
            self.btn_open_camera = QPushButton("Load from Camera", toolbar)
            self.combo_split = QComboBox(toolbar)
            self.combo_split.addItems(["train", "val", "test"])
            self.combo_split.currentTextChanged.connect(self._on_split_changed)
            self.btn_save = QPushButton("Save", toolbar)
            self._last_export_format = "YOLO (.txt)"
            self.btn_export = QPushButton("Export", toolbar)
            self.btn_golden = QPushButton("Export Golden", toolbar)
            self.btn_train = QPushButton("Train From Labels", toolbar)
            self.btn_undo = QPushButton("Undo", toolbar)
            self.btn_redo = QPushButton("Redo", toolbar)
            guide_text = (
                "Mouse:\n"
                "- Left drag: draw bbox\n"
                "- Left on box: select and move\n"
                "- Drag handle: resize\n"
                "- Drag top rotate handle: rotate\n"
                "- Ctrl + Left drag (empty): box-select multiple\n"
                "- Ctrl/Shift + Left click: multi-select toggle\n"
                "- Right click: paste previous box at cursor; if none, quick draw\n"
                "\n"
                "Keyboard:\n"
                "- D or Left Arrow: previous image\n"
                "- F or Right Arrow: next image\n"
                "- Delete/Backspace: delete selected box(es)\n"
                "- Q / E: rotate selected box -5 / +5 deg\n"
                "- Shift + Q / E: rotate -15 / +15 deg\n"
                "- Ctrl + Z / Ctrl + Y: undo / redo"
            )
            self.btn_guide = HoverGuideButton("Quick Guide", guide_text, toolbar)
            self.btn_back.clicked.connect(self.close)
            self.btn_open.clicked.connect(self._open_folder)
            self.btn_open_camera.clicked.connect(self._open_from_camera_capture)
            self.btn_save.clicked.connect(self._save_current_label)
            self.btn_export.clicked.connect(self._export_with_picker)
            self.btn_golden.clicked.connect(self._export_golden_current)
            self.btn_train.clicked.connect(self._train_from_labels)
            self.btn_undo.clicked.connect(self._undo_canvas)
            self.btn_redo.clicked.connect(self._redo_canvas)

            top.addSpacing(10)
            top.addWidget(self.btn_back)
            top.addWidget(self.btn_open)
            top.addWidget(self.btn_open_camera)
            top.addWidget(self.combo_split)
            top.addWidget(self.btn_save)
            top.addWidget(self.btn_undo)
            top.addWidget(self.btn_redo)
            top.addStretch(1)
            top.addWidget(self.btn_export)
            top.addWidget(self.btn_golden)
            top.addWidget(self.btn_train)
            self.lbl_status = QLabel("No project loaded", toolbar)
            self.lbl_status.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
            self.lbl_status.setMinimumWidth(0)
            top.addSpacing(10)
            top.addWidget(self.lbl_status)
            top.addWidget(self.btn_guide)
            self.btn_theme = QPushButton("Light Mode", toolbar)
            self.btn_theme.clicked.connect(self._toggle_theme)
            top.addWidget(self.btn_theme)
            layout.addWidget(toolbar)

            body = QHBoxLayout()
            body.setContentsMargins(10, 10, 10, 10)
            body.setSpacing(10)
            self.canvas = LabelCanvas(root)
            self.canvas.on_rects_changed = self._on_canvas_rects_changed
            self.canvas.on_selection_changed = self._on_canvas_selection_changed
            self.canvas.resolve_class_name = self._class_name_by_id
            self.canvas.on_paste_prev_box = self._paste_prev_label_at

            right = QWidget(root)
            self._right_panel = right
            right.setFixedWidth(320)
            right.setMinimumWidth(320)
            right.setMaximumWidth(320)
            right.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
            right_layout = QVBoxLayout(right)
            right_layout.setContentsMargins(0, 0, 0, 0)
            right_layout.setSpacing(0)
            right_layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetDefaultConstraint)

            scroll = QScrollArea(right)
            self._right_scroll = scroll
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            # Keep scrollbar lane stable so tool panel does not jitter left/right.
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
            panel = QWidget(scroll)
            self._right_scroll_panel = panel
            panel_layout = QVBoxLayout(panel)
            panel_layout.setContentsMargins(12, 12, 12, 12)
            panel_layout.setSpacing(10)
            panel_layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetDefaultConstraint)

            info_box = QGroupBox("File Info", panel)
            self._info_box = info_box
            info_layout = QVBoxLayout(info_box)
            self.lbl_filename = QLabel("No image", info_box)
            self.lbl_filename.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            self.lbl_filename.setMinimumWidth(0)
            info_layout.addWidget(self.lbl_filename)
            lbl_select_image = QLabel("Select Image", info_box)
            self._lbl_select_image = lbl_select_image
            info_layout.addWidget(lbl_select_image)
            row_image_select = QHBoxLayout()
            self.combo_image = QComboBox(info_box)
            self.combo_image.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.combo_image.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            self.combo_image.setMinimumContentsLength(1)
            self.combo_image.setMinimumWidth(0)
            try:
                self.combo_image.view().setTextElideMode(Qt.TextElideMode.ElideRight)
            except Exception:
                pass
            self.combo_image.currentIndexChanged.connect(self._jump_to_index)
            self.btn_remove = QPushButton("Remove", info_box)
            self.btn_restore = QPushButton("Restore", info_box)
            self.btn_remove.setFixedWidth(34)
            self.btn_restore.setFixedWidth(34)
            self.btn_remove.setText("X")
            self.btn_restore.setText("R")
            self.btn_remove.setToolTip("Remove current image")
            self.btn_restore.setToolTip("Restore removed image")
            self.btn_remove.clicked.connect(self._remove_current_image)
            self.btn_restore.clicked.connect(self._restore_removed_image)
            row_image_select.addWidget(self.combo_image, 1)
            row_image_select.addWidget(self.btn_remove)
            row_image_select.addWidget(self.btn_restore)
            info_layout.addLayout(row_image_select)
            row_counts = QHBoxLayout()
            self.lbl_progress = QLabel("0 / 0", info_box)
            self.lbl_box_count = QLabel("Boxes: 0", info_box)
            row_counts.addWidget(self.lbl_progress)
            row_counts.addStretch(1)
            row_counts.addWidget(self.lbl_box_count)
            info_layout.addLayout(row_counts)
            self.chk_auto_refresh_folder = QCheckBox("Auto Refresh Folder", info_box)
            self.chk_auto_refresh_folder.setChecked(True)
            self.chk_auto_refresh_folder.toggled.connect(self._on_auto_refresh_toggled)
            info_layout.addWidget(self.chk_auto_refresh_folder)
            row_class_counts = QHBoxLayout()
            self.lbl_class_count = QLabel("Classes: 0 / 0", info_box)
            row_class_counts.addStretch(1)
            row_class_counts.addWidget(self.lbl_class_count)
            info_layout.addLayout(row_class_counts)
            panel_layout.addWidget(info_box)

            class_box = QGroupBox("Class Management", panel)
            self._class_box = class_box
            class_layout = QVBoxLayout(class_box)
            lbl_current_class = QLabel("Current Class", class_box)
            self._lbl_current_class = lbl_current_class
            class_layout.addWidget(lbl_current_class)
            row_class_picker = QHBoxLayout()
            self.class_combo = QComboBox(class_box)
            self.class_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.class_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            self.class_combo.setMinimumContentsLength(1)
            self.class_combo.currentIndexChanged.connect(self._on_class_combo_changed)
            self.btn_add_class = QPushButton("Add", class_box)
            self.btn_add_class.setFixedWidth(56)
            self.btn_add_class.clicked.connect(self._add_class)
            self.btn_apply_class = QPushButton("Apply", class_box)
            self.btn_apply_class.setFixedWidth(64)
            self.btn_apply_class.clicked.connect(self._set_selected_box_class)
            self.btn_edit_classes = QPushButton("Edit", class_box)
            self.btn_clear_labels = QPushButton("Clear Labels", class_box)
            self.btn_edit_classes.clicked.connect(self._edit_classes_table)
            self.btn_clear_labels.clicked.connect(self._clear_current_labels)
            row_class_picker.addWidget(self.class_combo, 1)
            row_class_picker.addWidget(self.btn_add_class, 0)
            row_class_picker.addWidget(self.btn_apply_class, 0)
            class_layout.addLayout(row_class_picker)
            row_class_actions = QHBoxLayout()
            row_class_actions.addWidget(self.btn_edit_classes)
            row_class_actions.addWidget(self.btn_clear_labels)
            class_layout.addLayout(row_class_actions)
            self.chk_show_prev = QCheckBox("Show Last Photo Labels", class_box)
            self.chk_show_prev.toggled.connect(self._on_show_prev_toggled)
            class_layout.addWidget(self.chk_show_prev)
            self.lbl_selected_detail = QLabel("Selected: none", class_box)
            self.lbl_selected_detail.setWordWrap(True)
            self.lbl_selected_detail.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            self.lbl_selected_detail.setMinimumWidth(0)
            class_layout.addWidget(self.lbl_selected_detail)
            panel_layout.addWidget(class_box)

            ai_box = QGroupBox("AI Tools", panel)
            self._ai_box = ai_box
            ai_layout = QVBoxLayout(ai_box)
            self.chk_auto_detect = QCheckBox("Auto Detect", ai_box)
            self.chk_auto_detect.toggled.connect(self._on_auto_detect_toggled)
            self.chk_propagate = QCheckBox("Propagate Labels", ai_box)
            self.chk_propagate.toggled.connect(self._on_propagate_toggled)
            self.combo_propagate_mode = QComboBox(ai_box)
            self.combo_propagate_mode.addItems(["if_missing", "always", "selected_only"])
            self.combo_propagate_mode.currentTextChanged.connect(self._on_propagate_mode_changed)
            self.chk_propagate.setChecked(False)
            self.combo_propagate_mode.setEnabled(False)
            row_model = QHBoxLayout()
            self.combo_model = QComboBox(ai_box)
            self.combo_model.addItem("yolo26m (default)", "default_yolo26m")
            self.combo_model.addItem("Custom Imported", "custom_model")
            self.combo_model.currentIndexChanged.connect(self._on_model_combo_changed)
            self.btn_import_model = QPushButton("Import Model", ai_box)
            self.btn_import_model.clicked.connect(self._choose_detect_model)
            self.btn_detect = QPushButton("Run Auto Detect", ai_box)
            self.btn_detect.clicked.connect(self._run_auto_detect_current)
            row_model.addWidget(self.combo_model, 1)
            row_model.addWidget(self.btn_import_model, 0)
            ai_layout.addLayout(row_model)
            ai_layout.addWidget(self.btn_detect)
            ai_layout.addWidget(self.chk_auto_detect)
            ai_layout.addWidget(self.chk_propagate)
            ai_layout.addWidget(self.combo_propagate_mode)
            panel_layout.addWidget(ai_box)

            scroll.setWidget(panel)
            right_layout.addWidget(scroll, 1)

            nav = QWidget(right)
            self._nav_widget = nav
            nav_lay = QHBoxLayout(nav)
            nav_lay.setContentsMargins(12, 8, 12, 12)
            nav_lay.setSpacing(8)
            self.btn_prev = QPushButton("< Prev (D)", nav)
            self.btn_next = QPushButton("Next (F) >", nav)
            self.btn_prev.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.btn_next.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.btn_prev.setMinimumHeight(40)
            self.btn_next.setMinimumHeight(40)
            self.btn_prev.clicked.connect(self._prev_image)
            self.btn_next.clicked.connect(self._next_image)
            nav_lay.addWidget(self.btn_prev, 1)
            nav_lay.addWidget(self.btn_next, 1)
            right_layout.addWidget(nav, 0)
            body.addWidget(self.canvas, 1)
            body.addWidget(right, 0)
            layout.addLayout(body, 1)
            self.setCentralWidget(root)
            self._apply_theme_styles()
            self._refresh_class_widgets()
            self._on_model_combo_changed(self.combo_model.currentIndex())

        def _load_label_project(self, path: str, kind: str) -> None:
            self._project_dir = os.path.abspath(path)
            self._progress_state = self._read_progress_yaml(self._project_dir)
            images_root = os.path.join(path, "images")
            labels_root = os.path.join(path, "labels")
            has_split = any(os.path.isdir(os.path.join(images_root, s)) for s in ("train", "val", "test"))
            has_flat = os.path.isdir(images_root) and os.path.isdir(labels_root)
            if kind == "yolo_dataset":
                self._is_yolo_project = bool(has_split or has_flat)
                if not self._is_yolo_project:
                    QMessageBox.warning(
                        self,
                        "Load Project",
                        "Selected folder is not a YOLO dataset root.\nExpected images/ and labels/ (with or without train|val|test splits).",
                    )
                    return
            else:
                self._is_yolo_project = bool(has_split or has_flat)
            self._yolo_use_split_layout = bool(self._is_yolo_project and has_split)
            if self._is_yolo_project:
                self._project_root = path
                available = [s for s in ("train", "val", "test") if os.path.isdir(os.path.join(images_root, s))]
                self._load_class_names_from_dataset_yaml()
                progress_split = str(self._progress_state.get("split", "")).strip().lower()
                if progress_split in {"train", "val", "test"} and progress_split in available:
                    self._current_split = progress_split
                progress_class_names = self._extract_class_names_from_progress(self._progress_state)
                if progress_class_names:
                    self._class_names = progress_class_names
                self.combo_split.setEnabled(self._yolo_use_split_layout)
                self.combo_split.blockSignals(True)
                self.combo_split.clear()
                split_items = available if self._yolo_use_split_layout else ["all"]
                self.combo_split.addItems(split_items)
                if self._current_split not in split_items:
                    self._current_split = "train" if self._yolo_use_split_layout and "train" in available else split_items[0]
                self.combo_split.setCurrentText(self._current_split)
                self.combo_split.blockSignals(False)
                self._reload_images_for_current_source(reset_classes=False)
                self._restore_progress_position()
                self._save_progress_yaml()
                return
            self._project_root = ""
            self._class_names = ["class0"]
            progress_class_names = self._extract_class_names_from_progress(self._progress_state)
            if progress_class_names:
                self._class_names = progress_class_names
            self.combo_split.setEnabled(False)
            self._reload_images_for_current_source(reset_classes=False)
            self._restore_progress_position()
            self._save_progress_yaml()

        def _open_from_camera_capture(self) -> None:
            def _iter_images_recursive(root_dir: str) -> list[str]:
                exts = {".jpg", ".jpeg", ".png", ".bmp"}
                out: list[str] = []
                for base, _dirs, files in os.walk(root_dir):
                    for name in files:
                        p = os.path.join(base, name)
                        if os.path.splitext(name)[1].lower() in exts and os.path.isfile(p):
                            out.append(os.path.abspath(p))
                out.sort()
                return out

            def _prepare_cut_output_for_label(cut_output_dir: str) -> str | None:
                src_images = _iter_images_recursive(cut_output_dir)
                if not src_images:
                    return None
                ready_dir = os.path.join(cut_output_dir, "label_ready_images")
                os.makedirs(ready_dir, exist_ok=True)
                copied = 0
                for src in src_images:
                    name = os.path.basename(src)
                    stem, ext = os.path.splitext(name)
                    rel_parent = os.path.basename(os.path.dirname(src))
                    base_stem = f"{rel_parent}_{stem}".replace(" ", "_")
                    dst = os.path.join(ready_dir, f"{base_stem}{ext}")
                    i = 1
                    while os.path.exists(dst):
                        dst = os.path.join(ready_dir, f"{base_stem}_{i}{ext}")
                        i += 1
                    try:
                        shutil.copy2(src, dst)
                        copied += 1
                    except Exception:
                        continue
                if copied <= 0:
                    return None
                return ready_dir

            default_dir = os.path.abspath(self._project_dir or self._project_root or os.getcwd())
            save_dir = QFileDialog.getExistingDirectory(
                self,
                "Select parent folder for camera captures",
                default_dir,
            )
            if not save_dir:
                return
            capture_root = os.path.abspath(save_dir)
            capture_dir = os.path.join(
                capture_root,
                f"camera_label_capture_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
            os.makedirs(capture_dir, exist_ok=True)
            cams = _detect_available_camera_indices()
            if not cams:
                QMessageBox.warning(self, "Load from Camera", "No camera detected.")
                return
            cam_options = [f"Camera {idx}" for idx in cams]
            chosen, ok = QInputDialog.getItem(
                self,
                "Select Camera",
                "Choose camera:",
                cam_options,
                0,
                False,
            )
            if not ok or not chosen:
                return
            cam_idx = cams[0]
            try:
                cam_idx = int(str(chosen).split()[-1])
            except Exception:
                pass
            dlg = CameraCaptureDialog(save_dir=capture_dir, parent=self, camera_index=cam_idx)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            if dlg.captured_count() <= 0:
                return
            path = os.path.abspath(capture_dir)
            kind = "image_folder"
            ask_cut = QMessageBox.question(
                self,
                "Cut Background",
                "Run cut background before labeling?\n(After cut, label will open on cut results directly.)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ask_cut == QMessageBox.StandardButton.Yes:
                try:
                    import cv2  # type: ignore
                except Exception as exc:
                    QMessageBox.critical(
                        self,
                        "Cut Background",
                        "Cut background requires full desktop OpenCV.\n"
                        f"OpenCV import failed: {exc}\n\n"
                        "Fix:\n"
                        "pip uninstall opencv-python-headless\n"
                        "pip install opencv-python",
                    )
                    return
                missing = [name for name in ["imread", "namedWindow", "selectROI", "destroyWindow"] if not hasattr(cv2, name)]
                if missing:
                    QMessageBox.critical(
                        self,
                        "Cut Background",
                        "Cut background requires full desktop OpenCV.\n"
                        f"OpenCV missing APIs: {', '.join(missing)}\n\n"
                        "Fix:\n"
                        "pip uninstall opencv-python-headless\n"
                        "pip install opencv-python",
                    )
                    return
                golden_image_path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Select one golden image in this folder",
                    path,
                    "Image files (*.png *.jpg *.jpeg *.bmp)",
                )
                if not golden_image_path:
                    return
                try:
                    if os.path.commonpath([path, os.path.abspath(golden_image_path)]) != path:
                        QMessageBox.warning(self, "Cut Background", "Please select a golden image inside the selected folder.")
                        return
                except Exception:
                    pass
                try:
                    result = cut_background_detect.run_cut_background_batch_with_golden(
                        path,
                        golden_image_path=golden_image_path,
                        parent=self,
                    )
                except Exception as exc:
                    msg = str(exc)
                    if "cvNamedWindow" in msg or "The function is not implemented" in msg:
                        QMessageBox.critical(
                            self,
                            "Cut Background",
                            "Cut background requires OpenCV GUI backend.\n"
                            "Please install desktop OpenCV:\n"
                            "pip uninstall opencv-python-headless\n"
                            "pip install opencv-python",
                        )
                        return
                    QMessageBox.critical(self, "Cut Background", f"Cut background failed:\n{exc}")
                    return
                if result is None:
                    return
                ready = _prepare_cut_output_for_label(result.output_dir)
                if not ready or not os.path.isdir(ready):
                    QMessageBox.warning(
                        self,
                        "Cut Background",
                        "Cut background finished but no label-ready images were produced.",
                    )
                    return
                path = os.path.abspath(ready)
                kind = "image_folder"
            self._load_label_project(path, kind)

        def _open_folder(self) -> None:
            def _iter_images_recursive(root_dir: str) -> list[str]:
                exts = {".jpg", ".jpeg", ".png", ".bmp"}
                out: list[str] = []
                for base, _dirs, files in os.walk(root_dir):
                    for name in files:
                        p = os.path.join(base, name)
                        if os.path.splitext(name)[1].lower() in exts and os.path.isfile(p):
                            out.append(os.path.abspath(p))
                out.sort()
                return out

            def _prepare_cut_output_for_label(cut_output_dir: str) -> str | None:
                src_images = _iter_images_recursive(cut_output_dir)
                if not src_images:
                    return None
                ready_dir = os.path.join(cut_output_dir, "label_ready_images")
                os.makedirs(ready_dir, exist_ok=True)
                copied = 0
                for src in src_images:
                    name = os.path.basename(src)
                    stem, ext = os.path.splitext(name)
                    rel_parent = os.path.basename(os.path.dirname(src))
                    base_stem = f"{rel_parent}_{stem}".replace(" ", "_")
                    dst = os.path.join(ready_dir, f"{base_stem}{ext}")
                    i = 1
                    while os.path.exists(dst):
                        dst = os.path.join(ready_dir, f"{base_stem}_{i}{ext}")
                        i += 1
                    try:
                        shutil.copy2(src, dst)
                        copied += 1
                    except Exception:
                        continue
                if copied <= 0:
                    return None
                return ready_dir

            dialog = LoadProjectDialog(self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            payload = dialog.payload()
            path = payload["path"]
            kind = payload["kind"]
            if kind == "image_folder":
                ask_cut = QMessageBox.question(
                    self,
                    "Cut Background",
                    "Run cut background before labeling?\n(After cut, label will open on cut results directly.)",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if ask_cut == QMessageBox.StandardButton.Yes:
                    try:
                        import cv2  # type: ignore
                    except Exception as exc:
                        QMessageBox.critical(
                            self,
                            "Cut Background",
                            "Cut background requires full desktop OpenCV.\n"
                            f"OpenCV import failed: {exc}\n\n"
                            "Fix:\n"
                            "pip uninstall opencv-python-headless\n"
                            "pip install opencv-python",
                        )
                        return
                    missing = [name for name in ["imread", "namedWindow", "selectROI", "destroyWindow"] if not hasattr(cv2, name)]
                    if missing:
                        QMessageBox.critical(
                            self,
                            "Cut Background",
                            "Cut background requires full desktop OpenCV.\n"
                            f"OpenCV missing APIs: {', '.join(missing)}\n\n"
                            "Fix:\n"
                            "pip uninstall opencv-python-headless\n"
                            "pip install opencv-python",
                        )
                        return
                    golden_image_path, _ = QFileDialog.getOpenFileName(
                        self,
                        "Select one golden image in this folder",
                        os.path.abspath(path),
                        "Image files (*.png *.jpg *.jpeg *.bmp)",
                    )
                    if not golden_image_path:
                        return
                    try:
                        if os.path.commonpath([os.path.abspath(path), os.path.abspath(golden_image_path)]) != os.path.abspath(path):
                            QMessageBox.warning(self, "Cut Background", "Please select a golden image inside the selected folder.")
                            return
                    except Exception:
                        pass
                    try:
                        result = cut_background_detect.run_cut_background_batch_with_golden(
                            path,
                            golden_image_path=golden_image_path,
                            parent=self,
                        )
                    except Exception as exc:
                        msg = str(exc)
                        if "cvNamedWindow" in msg or "The function is not implemented" in msg:
                            QMessageBox.critical(
                                self,
                                "Cut Background",
                                "Cut background requires OpenCV GUI backend.\n"
                                "Please install desktop OpenCV:\n"
                                "pip uninstall opencv-python-headless\n"
                                "pip install opencv-python",
                            )
                            return
                        QMessageBox.critical(self, "Cut Background", f"Cut background failed:\n{exc}")
                        return
                    if result is None:
                        return
                    ready = _prepare_cut_output_for_label(result.output_dir)
                    if not ready or not os.path.isdir(ready):
                        QMessageBox.warning(
                            self,
                            "Cut Background",
                            "Cut background finished but no label-ready images were produced.",
                        )
                        return
                    path = ready
                    kind = "image_folder"
            self._project_dir = os.path.abspath(path)
            self._progress_state = self._read_progress_yaml(self._project_dir)
            images_root = os.path.join(path, "images")
            labels_root = os.path.join(path, "labels")
            has_split = any(os.path.isdir(os.path.join(images_root, s)) for s in ("train", "val", "test"))
            has_flat = os.path.isdir(images_root) and os.path.isdir(labels_root)
            if kind == "yolo_dataset":
                self._is_yolo_project = bool(has_split or has_flat)
                if not self._is_yolo_project:
                    QMessageBox.warning(
                        self,
                        "Load Project",
                        "Selected folder is not a YOLO dataset root.\nExpected images/ and labels/ (with or without train|val|test splits).",
                    )
                    return
            else:
                self._is_yolo_project = bool(has_split or has_flat)
            self._yolo_use_split_layout = bool(self._is_yolo_project and has_split)
            if self._is_yolo_project:
                self._project_root = path
                available = [s for s in ("train", "val", "test") if os.path.isdir(os.path.join(images_root, s))]
                self._load_class_names_from_dataset_yaml()
                progress_split = str(self._progress_state.get("split", "")).strip().lower()
                if progress_split in {"train", "val", "test"} and progress_split in available:
                    self._current_split = progress_split
                progress_class_names = self._extract_class_names_from_progress(self._progress_state)
                if progress_class_names:
                    self._class_names = progress_class_names
                self.combo_split.setEnabled(self._yolo_use_split_layout)
                self.combo_split.blockSignals(True)
                self.combo_split.clear()
                split_items = available if self._yolo_use_split_layout else ["all"]
                self.combo_split.addItems(split_items)
                if self._current_split not in split_items:
                    self._current_split = "train" if self._yolo_use_split_layout and "train" in available else split_items[0]
                self.combo_split.setCurrentText(self._current_split)
                self.combo_split.blockSignals(False)
                self._reload_images_for_current_source(reset_classes=False)
                self._restore_progress_position()
                self._save_progress_yaml()
                return
            self._project_root = ""
            self._class_names = ["class0"]
            progress_class_names = self._extract_class_names_from_progress(self._progress_state)
            if progress_class_names:
                self._class_names = progress_class_names
            self.combo_split.setEnabled(False)
            self._reload_images_for_current_source(reset_classes=False)
            self._restore_progress_position()
            self._save_progress_yaml()

        def _progress_yaml_path(self) -> str:
            base = self._project_root if self._is_yolo_project and self._project_root else self._project_dir
            base = os.path.abspath(base or os.getcwd())
            return os.path.join(base, ".ai_labeller_progress.yaml")

        def _read_progress_yaml(self, project_root: str) -> dict[str, str]:
            key = os.path.abspath(project_root or self._project_root or self._project_dir or os.getcwd())
            data = _session_progress_cache.get(key, {})
            return dict(data) if isinstance(data, dict) else {}

        def _extract_class_names_from_progress(self, progress: dict[str, str]) -> list[str]:
            try:
                class_count = int(str(progress.get("class_count", "0")).strip())
            except Exception:
                class_count = 0
            if class_count <= 0:
                return []
            out: list[str] = []
            for idx in range(class_count):
                name = str(progress.get(f"class_{idx}", "")).strip()
                if not name:
                    return []
                out.append(name)
            return out

        def _yaml_q(self, value: str) -> str:
            return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'

        def _save_progress_yaml(self) -> None:
            if not self._project_dir and not self._project_root:
                return
            image_name = ""
            image_index = int(self._image_idx)
            if self._image_paths and 0 <= self._image_idx < len(self._image_paths):
                image_name = os.path.basename(self._image_paths[self._image_idx])
            project_root = os.path.abspath(self._project_root or self._project_dir)
            data: dict[str, str] = {
                "project_root": project_root,
                "split": str(self._current_split),
                "image_name": str(image_name),
                "image_index": str(image_index),
                "class_count": str(len(self._class_names)),
                "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            }
            for idx, class_name in enumerate(self._class_names):
                data[f"class_{idx}"] = str(class_name)
            _session_progress_cache[project_root] = data

        def _restore_progress_position(self) -> None:
            if not self._image_paths:
                return
            progress = self._progress_state or {}
            target_name = str(progress.get("image_name", "")).strip()
            if target_name:
                for i, p in enumerate(self._image_paths):
                    if os.path.basename(p) == target_name:
                        self._image_idx = i
                        self._show_current_image()
                        return
            try:
                idx = int(str(progress.get("image_index", "0")).strip())
            except Exception:
                idx = 0
            if 0 <= idx < len(self._image_paths):
                self._image_idx = idx
                self._show_current_image()

        def _schedule_autosave(self) -> None:
            if not self._autosave_enabled:
                return
            self._autosave_dirty = True
            self._autosave_timer.start(350)

        def _autosave_flush(self) -> None:
            if not self._autosave_dirty:
                return
            self._autosave_dirty = False
            self._save_current_label(silent=True)
            self._save_progress_yaml()

        def _load_class_names_from_dataset_yaml(self) -> None:
            if not self._project_root:
                return
            yaml_path = golden_core.find_dataset_yaml_in_folder(self._project_root)
            if not yaml_path:
                return
            mapping = golden_core.load_mapping_from_dataset_yaml(yaml_path)
            if not mapping:
                return
            max_id = max(mapping.keys())
            classes = [f"class{i}" for i in range(max_id + 1)]
            for idx, name in mapping.items():
                if 0 <= idx < len(classes):
                    classes[idx] = str(name)
            if classes:
                self._class_names = classes

        def _scan_image_paths_for_current_source(self) -> list[str]:
            exts = {".jpg", ".jpeg", ".png", ".bmp"}
            if self._is_yolo_project and self._project_root:
                if self._yolo_use_split_layout:
                    split_dir = os.path.join(self._project_root, "images", self._current_split)
                    os.makedirs(os.path.join(self._project_root, "labels", self._current_split), exist_ok=True)
                    if not os.path.isdir(split_dir):
                        return []
                    return [str(p) for p in sorted(Path(split_dir).iterdir()) if p.is_file() and p.suffix.lower() in exts]
                else:
                    image_dir = os.path.join(self._project_root, "images")
                    os.makedirs(os.path.join(self._project_root, "labels"), exist_ok=True)
                    if not os.path.isdir(image_dir):
                        return []
                    return [str(p) for p in sorted(Path(image_dir).iterdir()) if p.is_file() and p.suffix.lower() in exts]
            else:
                if not self._project_dir or not os.path.isdir(self._project_dir):
                    return []
                return [str(p) for p in sorted(Path(self._project_dir).iterdir()) if p.is_file() and p.suffix.lower() in exts]

        def _refresh_combo_image_items(self) -> None:
            self.combo_image.blockSignals(True)
            self.combo_image.clear()
            for p in self._image_paths:
                base = os.path.basename(p)
                display = self._compact_name(base)
                self.combo_image.addItem(display, p)
                try:
                    self.combo_image.setItemData(self.combo_image.count() - 1, base, Qt.ItemDataRole.ToolTipRole)
                except Exception:
                    pass
            self.combo_image.blockSignals(False)

        def _reload_images_for_current_source(self, reset_classes: bool = False) -> None:
            self._image_paths = self._scan_image_paths_for_current_source()
            self._refresh_combo_image_items()
            if not self._image_paths:
                QMessageBox.warning(self, "Label Workspace", "No images found in selected folder.")
                self._image_idx = 0
                self._labels_by_path = {}
                self._refresh_info_labels()
                return
            if reset_classes:
                self._class_names = ["class0"]
            self._labels_by_path = {}
            self._image_idx = 0
            self._show_current_image()

        def _on_auto_refresh_toggled(self, checked: bool) -> None:
            self._auto_refresh_enabled = bool(checked)

        def _auto_refresh_tick(self) -> None:
            if not self._auto_refresh_enabled:
                return
            if not self._project_dir and not self._project_root:
                return
            try:
                new_paths = self._scan_image_paths_for_current_source()
            except Exception:
                return
            if new_paths == self._image_paths:
                return
            current_path = ""
            if self._image_paths and 0 <= self._image_idx < len(self._image_paths):
                current_path = self._image_paths[self._image_idx]
            self._sync_canvas_rects_to_current_image()
            old_paths = set(self._image_paths)
            self._image_paths = new_paths
            self._refresh_combo_image_items()
            self._labels_by_path = {k: v for k, v in self._labels_by_path.items() if k in set(new_paths)}
            if not self._image_paths:
                self._image_idx = 0
                self.lbl_status.setText("No images found in selected folder.")
                self._refresh_info_labels()
                return
            if current_path and current_path in self._image_paths:
                self._image_idx = self._image_paths.index(current_path)
            else:
                self._image_idx = max(0, min(self._image_idx, len(self._image_paths) - 1))
            added = len([p for p in self._image_paths if p not in old_paths])
            if added > 0:
                self.lbl_status.setText(f"Auto refresh: +{added} image(s)")
            self._show_current_image()

        def _on_split_changed(self, split: str) -> None:
            split = str(split or "").strip().lower()
            if not self._is_yolo_project:
                return
            if not self._yolo_use_split_layout:
                return
            if split not in {"train", "val", "test"}:
                return
            self._sync_canvas_rects_to_current_image()
            self._current_split = split
            self._reload_images_for_current_source(reset_classes=False)
            self._save_progress_yaml()

        def _load_image(self, image_path: str) -> None:
            import cv2

            bgr = cv2.imread(image_path)
            if bgr is None:
                raise RuntimeError(f"failed to read image: {image_path}")
            self._last_bgr = bgr
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
            pix = QPixmap.fromImage(qimg)
            rects = self._labels_by_path.get(image_path)
            if rects is None:
                rects = self._load_label_file_for_image(image_path, w, h)
                self._labels_by_path[image_path] = [list(r) for r in rects]
            self._ensure_class_count_from_rects(rects)
            self._refresh_class_widgets()
            self.canvas.set_image(pix, w, h, rects)
            self.canvas.set_default_class_id(self.class_combo.currentIndex())
            self.canvas.setFocus()

        def _refresh_image_preview(self) -> None:
            self.canvas.update()

        def _show_current_image(self) -> None:
            if not self._image_paths:
                return
            prev_rects = [list(r) for r in getattr(self.canvas, "rects", [])]
            self._prev_image_rects = prev_rects
            selected_rects: list[list[float]] = []
            try:
                selected_indices = sorted(
                    i for i in getattr(self.canvas, "selected_indices", set()) if 0 <= int(i) < len(prev_rects)
                )
                if not selected_indices:
                    sel_idx = getattr(self.canvas, "selected_idx", None)
                    if sel_idx is not None and 0 <= int(sel_idx) < len(prev_rects):
                        selected_indices = [int(sel_idx)]
                selected_rects = [list(prev_rects[i]) for i in selected_indices]
            except Exception:
                selected_rects = []
            self._prev_image_selected_rects = selected_rects
            self._image_idx = max(0, min(self._image_idx, len(self._image_paths) - 1))
            path = self._image_paths[self._image_idx]
            try:
                self._load_image(path)
            except Exception as exc:
                QMessageBox.critical(self, "Label Workspace", str(exc))
                return
            if self.chk_propagate.isChecked():
                self._apply_propagate_on_current_image()
            self._apply_auto_detect_on_current_image()
            self.canvas.set_ghost_rects(self._prev_image_rects if self._show_prev_labels else [])
            status_name = self._compact_name(os.path.basename(path), head=24, tail=16)
            self.lbl_status.setText(f"{status_name} ({self._image_idx + 1}/{len(self._image_paths)})")
            self.lbl_status.setToolTip(path)
            self.combo_image.blockSignals(True)
            self.combo_image.setCurrentIndex(self._image_idx)
            self.combo_image.blockSignals(False)
            self._refresh_info_labels()

        def _refresh_info_labels(self) -> None:
            total = len(self._image_paths)
            idx = self._image_idx + 1 if total else 0
            name = os.path.basename(self._image_paths[self._image_idx]) if total and 0 <= self._image_idx < total else "No image"
            self.lbl_filename.setText(self._compact_name(name, head=26, tail=18))
            self.lbl_filename.setToolTip(name if name != "No image" else "")
            self.lbl_progress.setText(f"{idx} / {total}")
            if self._is_yolo_project:
                self.lbl_progress.setText(f"{self._current_split}: {idx} / {total}")
            box_count = len(self.canvas.rects) if hasattr(self, "canvas") else 0
            self.lbl_box_count.setText(f"Boxes: {box_count}")
            class_count = len(self._class_names)
            used_ids = {int(r[4]) for r in self.canvas.rects if len(r) >= 5} if hasattr(self, "canvas") else set()
            self.lbl_class_count.setText(f"Classes: {len(used_ids)} / {class_count}")

        def _compact_name(self, text: str, head: int = 22, tail: int = 14) -> str:
            s = str(text or "")
            if len(s) <= head + tail + 3:
                return s
            return f"{s[:head]}...{s[-tail:]}"

        def _label_path_for_image(self, image_path: str) -> str:
            stem_name = os.path.splitext(os.path.basename(image_path))[0]
            if self._is_yolo_project and self._project_root:
                if self._yolo_use_split_layout:
                    label_dir = os.path.join(self._project_root, "labels", self._current_split)
                else:
                    label_dir = os.path.join(self._project_root, "labels")
                os.makedirs(label_dir, exist_ok=True)
                return os.path.join(label_dir, f"{stem_name}.txt")
            stem = os.path.splitext(image_path)[0]
            return f"{stem}.txt"

        def _load_label_file_for_image(self, image_path: str, img_w: int, img_h: int) -> list[list[float]]:
            label_path = self._label_path_for_image(image_path)
            if not os.path.isfile(label_path):
                return []
            rects: list[list[float]] = []
            try:
                with open(label_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except Exception:
                return []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    class_id = int(float(parts[0]))
                except Exception:
                    class_id = 0
                if len(parts) >= 9:
                    try:
                        pts = list(map(float, parts[1:9]))
                    except Exception:
                        continue
                    px = [pts[0] * img_w, pts[2] * img_w, pts[4] * img_w, pts[6] * img_w]
                    py = [pts[1] * img_h, pts[3] * img_h, pts[5] * img_h, pts[7] * img_h]
                    cx = sum(px) / 4.0
                    cy = sum(py) / 4.0
                    w = max(2.0, ((px[1] - px[0]) ** 2 + (py[1] - py[0]) ** 2) ** 0.5)
                    h = max(2.0, ((px[2] - px[1]) ** 2 + (py[2] - py[1]) ** 2) ** 0.5)
                    angle = math.degrees(math.atan2(py[1] - py[0], px[1] - px[0]))
                    x1 = cx - w / 2.0
                    y1 = cy - h / 2.0
                    x2 = cx + w / 2.0
                    y2 = cy + h / 2.0
                else:
                    try:
                        cx, cy, w, h = map(float, parts[1:5])
                    except Exception:
                        continue
                    x1 = (cx - w / 2) * img_w
                    y1 = (cy - h / 2) * img_h
                    x2 = (cx + w / 2) * img_w
                    y2 = (cy + h / 2) * img_h
                    angle = 0.0
                rects.append([x1, y1, x2, y2, class_id, angle])
            return rects

        def _ensure_class_count_from_rects(self, rects: list[list[float]]) -> None:
            max_id = -1
            for rect in rects:
                if len(rect) >= 5:
                    try:
                        max_id = max(max_id, int(rect[4]))
                    except Exception:
                        pass
            while len(self._class_names) <= max_id:
                self._class_names.append(f"class{len(self._class_names)}")

        def _refresh_class_widgets(self) -> None:
            counts: dict[int, int] = {}
            if hasattr(self, "canvas") and self.canvas is not None:
                for rect in self.canvas.rects:
                    cid = int(rect[4]) if len(rect) >= 5 else 0
                    counts[cid] = counts.get(cid, 0) + 1
            self.class_combo.blockSignals(True)
            cur = self.class_combo.currentIndex()
            self.class_combo.clear()
            for idx, name in enumerate(self._class_names):
                self.class_combo.addItem(f"{name} ({counts.get(idx, 0)})")
            if self.class_combo.count() > 0:
                if cur < 0 or cur >= self.class_combo.count():
                    cur = 0
                self.class_combo.setCurrentIndex(cur)
            self.class_combo.blockSignals(False)
            self._rebuild_class_list_from_canvas()

        def _rebuild_class_list_from_canvas(self) -> None:
            counts: dict[int, int] = {}
            for rect in self.canvas.rects:
                cid = int(rect[4]) if len(rect) >= 5 else 0
                counts[cid] = counts.get(cid, 0) + 1
            if self.class_combo.count() == len(self._class_names):
                cur_idx = self.class_combo.currentIndex()
                self.class_combo.blockSignals(True)
                for idx, name in enumerate(self._class_names):
                    self.class_combo.setItemText(idx, f"{name} ({counts.get(idx, 0)})")
                if 0 <= cur_idx < self.class_combo.count():
                    self.class_combo.setCurrentIndex(cur_idx)
                self.class_combo.blockSignals(False)
            self._refresh_info_labels()
            self._refresh_selected_detail()

        def _class_name_by_id(self, cid: int) -> str:
            if 0 <= int(cid) < len(self._class_names):
                return str(self._class_names[int(cid)])
            return f"class{int(cid)}"

        def _refresh_selected_detail(self) -> None:
            if not hasattr(self, "lbl_selected_detail"):
                return
            idx = self.canvas.selected_idx
            if idx is None or not (0 <= idx < len(self.canvas.rects)):
                self.lbl_selected_detail.setText("Selected: none")
                self.lbl_selected_detail.setToolTip("")
                return
            rect = self.canvas.rects[idx]
            x1, y1, x2, y2 = map(float, rect[:4])
            cid = int(rect[4]) if len(rect) >= 5 else 0
            ang = float(rect[5]) if len(rect) >= 6 else 0.0
            cname = self._class_name_by_id(cid)
            short_text = f"Selected: {idx+1} | {cname} ({cid}) | ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f}) | a={ang:.1f}"
            full_text = f"Selected: {idx+1} | {cname} ({cid}) | xyxy=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) | angle={ang:.1f}"
            self.lbl_selected_detail.setText(short_text)
            self.lbl_selected_detail.setToolTip(full_text)

        def _on_class_combo_changed(self, idx: int) -> None:
            self.canvas.set_default_class_id(max(0, idx))

        def _add_class(self) -> None:
            text, ok = QInputDialog.getText(self, "Add Class", "Class name:")
            if not ok:
                return
            name = str(text).strip()
            if not name:
                return
            self._class_names.append(name)
            self._refresh_class_widgets()
            self.class_combo.setCurrentIndex(len(self._class_names) - 1)
            self._save_progress_yaml()

        def _edit_classes_table(self) -> None:
            win = QDialog(self)
            win.setWindowTitle("Edit Classes")
            win.resize(420, 360)
            lay = QVBoxLayout(win)
            lst = QListWidget(win)
            lay.addWidget(lst, 1)
            row = QHBoxLayout()
            btn_add = QPushButton("Add", win)
            btn_rename = QPushButton("Rename", win)
            btn_delete = QPushButton("Delete", win)
            row.addWidget(btn_add)
            row.addWidget(btn_rename)
            row.addWidget(btn_delete)
            lay.addLayout(row)
            btn_close = QPushButton("Close", win)
            lay.addWidget(btn_close)

            def _refresh() -> None:
                lst.clear()
                for i, n in enumerate(self._class_names):
                    lst.addItem(f"{i}: {n}")

            def _add() -> None:
                text, ok = QInputDialog.getText(win, "Add Class", "Class name:")
                if ok and str(text).strip():
                    self._class_names.append(str(text).strip())
                    self._refresh_class_widgets()
                    self._save_progress_yaml()
                    _refresh()

            def _rename() -> None:
                idx = lst.currentRow()
                if idx < 0 or idx >= len(self._class_names):
                    return
                text, ok = QInputDialog.getText(win, "Rename Class", "Class name:", text=self._class_names[idx])
                if ok and str(text).strip():
                    self._class_names[idx] = str(text).strip()
                    self._refresh_class_widgets()
                    self._save_progress_yaml()
                    _refresh()

            def _delete() -> None:
                idx = lst.currentRow()
                if idx < 0 or idx >= len(self._class_names):
                    return
                if len(self._class_names) <= 1:
                    QMessageBox.information(win, "Edit Classes", "Cannot delete the last class.")
                    return
                ans = QMessageBox.question(
                    win,
                    "Edit Classes",
                    f"Delete class '{self._class_names[idx]}' and related labels?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if ans != QMessageBox.StandardButton.Yes:
                    return
                self._class_names.pop(idx)
                for p, rects in list(self._labels_by_path.items()):
                    new_rects: list[list[float]] = []
                    for rect in rects:
                        cid = int(rect[4]) if len(rect) >= 5 else 0
                        if cid == idx:
                            continue
                        rr = list(rect)
                        if cid > idx:
                            rr[4] = cid - 1
                        new_rects.append(rr)
                    self._labels_by_path[p] = new_rects
                if self._image_paths:
                    cur = self._image_paths[self._image_idx]
                    self.canvas.rects = [list(r) for r in self._labels_by_path.get(cur, [])]
                self.canvas._push_history()
                self._refresh_class_widgets()
                self._save_progress_yaml()
                self.canvas.update()
                _refresh()

            btn_add.clicked.connect(_add)
            btn_rename.clicked.connect(_rename)
            btn_delete.clicked.connect(_delete)
            btn_close.clicked.connect(win.accept)
            _refresh()
            win.exec()

        def _clear_current_labels(self) -> None:
            if not self._image_paths:
                return
            ans = QMessageBox.question(
                self,
                "Clear Labels",
                "Clear all labels on current image?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ans != QMessageBox.StandardButton.Yes:
                return
            self.canvas.rects = []
            self.canvas._push_history()
            self._sync_canvas_rects_to_current_image()
            self._rebuild_class_list_from_canvas()
            self.canvas.update()
            self._schedule_autosave()

        def _on_show_prev_toggled(self, checked: bool) -> None:
            self._show_prev_labels = bool(checked)
            self.canvas.set_ghost_rects(self._prev_image_rects if self._show_prev_labels else [])

        def _on_auto_detect_toggled(self, checked: bool) -> None:
            self._auto_detect_enabled = bool(checked)

        def _on_propagate_toggled(self, checked: bool) -> None:
            self._propagate_enabled = bool(checked)
            self.combo_propagate_mode.setEnabled(self._propagate_enabled)

        def _on_propagate_mode_changed(self, mode: str) -> None:
            mode = str(mode).strip().lower()
            self._propagate_mode = mode if mode in {"if_missing", "always", "selected_only"} else "if_missing"

        def _apply_propagate_on_current_image(self) -> None:
            if not self._propagate_enabled or not self._image_paths:
                return
            cur_path = self._image_paths[self._image_idx]
            cur_rects = self._labels_by_path.get(cur_path, [])
            if self._propagate_mode == "if_missing" and cur_rects:
                return
            def _rect_key(rect: list[float]) -> tuple[float, float, float, float, int, float]:
                x1, y1, x2, y2 = (float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]))
                cid = int(rect[4]) if len(rect) >= 5 else 0
                ang = float(rect[5]) if len(rect) >= 6 else 0.0
                return (round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3), cid, round(ang, 2))
            if self._propagate_mode == "selected_only":
                if not self._prev_image_selected_rects:
                    return
                existing = [list(r) for r in self.canvas.rects]
                existing_keys = {_rect_key(r) for r in existing if len(r) >= 4}
                for r in self._prev_image_selected_rects:
                    rr = list(r)
                    if len(rr) < 4:
                        continue
                    k = _rect_key(rr)
                    if k in existing_keys:
                        continue
                    existing.append(rr)
                    existing_keys.add(k)
                self.canvas.rects = existing
            elif self._propagate_mode == "always":
                # Always keeps current labels and appends all labels from previous image.
                existing = [list(r) for r in self.canvas.rects]
                existing_keys = {_rect_key(r) for r in existing if len(r) >= 4}
                for r in self._prev_image_rects:
                    rr = list(r)
                    if len(rr) < 4:
                        continue
                    k = _rect_key(rr)
                    if k in existing_keys:
                        continue
                    existing.append(rr)
                    existing_keys.add(k)
                self.canvas.rects = existing
            else:
                if not self._prev_image_rects:
                    return
                self.canvas.rects = [list(r) for r in self._prev_image_rects]
            self._sync_canvas_rects_to_current_image()
            self._rebuild_class_list_from_canvas()
            self.canvas.update()

        def _apply_auto_detect_on_current_image(self) -> None:
            if not self._auto_detect_enabled or not self._image_paths:
                return
            cur_path = self._image_paths[self._image_idx]
            cur_rects = self._labels_by_path.get(cur_path, [])
            if cur_rects:
                return
            self._run_auto_detect_current(silent=True)

        def _paste_prev_label_at(self, ix: float, iy: float) -> bool:
            if not self._show_prev_labels:
                return False
            if not self._prev_image_rects:
                return False
            idx = self.canvas._hit_test_in_rects(ix, iy, self._prev_image_rects)
            if idx is None:
                return False
            rect = copy.deepcopy(self._prev_image_rects[idx])
            cid = int(rect[4]) if len(rect) >= 5 else 0
            while len(self._class_names) <= cid:
                self._class_names.append(f"class{len(self._class_names)}")
            self.canvas.rects.append(rect)
            self.canvas.selected_idx = len(self.canvas.rects) - 1
            self.canvas.selected_indices = {self.canvas.selected_idx}
            self.canvas._push_history()
            self._sync_canvas_rects_to_current_image()
            self._refresh_class_widgets()
            self.canvas.update()
            return True

        def _set_selected_box_class(self) -> None:
            idx = self.class_combo.currentIndex()
            if idx < 0:
                return
            self.canvas.set_selected_class_id(idx)
            self._rebuild_class_list_from_canvas()
            self._schedule_autosave()

        def _on_canvas_rects_changed(self) -> None:
            self._rebuild_class_list_from_canvas()
            self._schedule_autosave()

        def _on_canvas_selection_changed(self) -> None:
            idx = self.canvas.selected_idx
            if idx is None or not (0 <= idx < len(self.canvas.rects)):
                self._refresh_selected_detail()
                return
            try:
                cid = int(self.canvas.rects[idx][4])
            except Exception:
                cid = 0
            if 0 <= cid < self.class_combo.count():
                self.class_combo.setCurrentIndex(cid)
            self._refresh_selected_detail()

        def _undo_canvas(self) -> None:
            self.canvas.undo()

        def _redo_canvas(self) -> None:
            self.canvas.redo()

        def _sync_canvas_rects_to_current_image(self) -> None:
            if not self._image_paths:
                return
            cur_path = self._image_paths[self._image_idx]
            if hasattr(self, "canvas") and self.canvas is not None:
                self._labels_by_path[cur_path] = [list(r) for r in self.canvas.rects]

        def _save_current_label(self, silent: bool = False) -> None:
            if not self._image_paths:
                return
            self._sync_canvas_rects_to_current_image()
            img_path = self._image_paths[self._image_idx]
            rects = self._labels_by_path.get(img_path, [])
            if self._last_bgr is None:
                return
            h, w = self._last_bgr.shape[:2]
            label_path = self._label_path_for_image(img_path)
            lines: list[str] = []
            for rect in rects:
                x1, y1, x2, y2 = rect[:4]
                class_id = int(rect[4]) if len(rect) >= 5 else 0
                angle = float(rect[5]) if len(rect) >= 6 else 0.0
                x1, y1 = max(0.0, min(w, x1)), max(0.0, min(h, y1))
                x2, y2 = max(0.0, min(w, x2)), max(0.0, min(h, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                rect_norm = [x1, y1, x2, y2, class_id, angle]
                corners = self.canvas._rect_corners(rect_norm)
                xy: list[str] = []
                for px, py in corners:
                    nx = max(0.0, min(1.0, px / w))
                    ny = max(0.0, min(1.0, py / h))
                    xy.append(f"{nx:.6f}")
                    xy.append(f"{ny:.6f}")
                lines.append(f"{int(class_id)} " + " ".join(xy))
            try:
                with open(label_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines) + ("\n" if lines else ""))
                if not silent:
                    self.lbl_status.setText(f"Saved: {os.path.basename(label_path)}")
                else:
                    self.lbl_status.setText(f"Auto-saved: {os.path.basename(label_path)}")
                self._save_progress_yaml()
            except Exception as exc:
                QMessageBox.critical(self, "Label Workspace", f"Failed to save label:\n{exc}")

        def _iter_labeled_images(self) -> list[str]:
            images: list[str] = []
            for p in self._image_paths:
                lp = self._label_path_for_image(p)
                if os.path.isfile(lp):
                    images.append(p)
            return images

        def _ask_include_unlabeled_for_export(self) -> bool:
            ans = QMessageBox.question(
                self,
                "Export",
                "Include unlabeled images as background samples?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            return ans == QMessageBox.StandardButton.Yes

        def _export_with_picker(self) -> None:
            options = ["YOLO (.txt)", "JSON", "COCO"]
            current = str(getattr(self, "_last_export_format", options[0]) or options[0]).strip()
            if current not in options:
                current = options[0]
            fmt, ok = QInputDialog.getItem(
                self,
                "Export Format",
                "Choose export format:",
                options,
                options.index(current),
                False,
            )
            if not ok:
                return
            fmt = str(fmt).strip()
            if not fmt:
                return
            self._last_export_format = fmt
            self._export_by_selected_format(fmt)

        def _export_by_selected_format(self, fmt_override: str | None = None) -> None:
            fmt = str(fmt_override or getattr(self, "_last_export_format", "YOLO (.txt)") or "").strip()
            if fmt == "YOLO (.txt)":
                self._export_yolo_dataset()
                return
            if fmt == "JSON":
                self._export_json_dataset()
                return
            QMessageBox.information(self, "Export", "COCO export will be implemented.")

        def _export_yolo_dataset(self) -> None:
            if not self._image_paths:
                QMessageBox.warning(self, "Export", "Please load an image folder first.")
                return
            self._save_current_label()
            include_unlabeled = self._ask_include_unlabeled_for_export()
            labeled_images = self._iter_labeled_images()
            export_images = list(self._image_paths) if include_unlabeled else labeled_images
            if not export_images:
                QMessageBox.warning(self, "Export", "No labeled images found.")
                return
            out_root = QFileDialog.getExistingDirectory(self, "Select Export Parent Folder")
            if not out_root:
                return
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir = os.path.join(out_root, f"export_all_{ts}")
            os.makedirs(os.path.join(export_dir, "images", "train"), exist_ok=True)
            os.makedirs(os.path.join(export_dir, "labels", "train"), exist_ok=True)
            make_aug_val = (
                QMessageBox.question(
                    self,
                    "Export Validation",
                    "Use brightness augmentation to create validation set?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                == QMessageBox.StandardButton.Yes
            )
            val_count = 0
            cv2_mod = None
            if make_aug_val:
                os.makedirs(os.path.join(export_dir, "images", "val"), exist_ok=True)
                os.makedirs(os.path.join(export_dir, "labels", "val"), exist_ok=True)
                try:
                    import cv2 as _cv2  # type: ignore

                    cv2_mod = _cv2
                except Exception:
                    cv2_mod = None
            count = 0
            labeled_set = set(labeled_images)
            for i, img_path in enumerate(export_images):
                name = os.path.basename(img_path)
                stem = os.path.splitext(name)[0]
                src_lbl = self._label_path_for_image(img_path)
                dst_img = os.path.join(export_dir, "images", "train", name)
                dst_lbl = os.path.join(export_dir, "labels", "train", f"{stem}.txt")
                try:
                    shutil.copy2(img_path, dst_img)
                    if img_path in labeled_set and os.path.isfile(src_lbl):
                        shutil.copy2(src_lbl, dst_lbl)
                    else:
                        with open(dst_lbl, "w", encoding="utf-8") as f:
                            f.write("")
                    count += 1
                except Exception:
                    continue
                if make_aug_val:
                    val_name = f"{stem}_valb{os.path.splitext(name)[1]}"
                    dst_val_img = os.path.join(export_dir, "images", "val", val_name)
                    dst_val_lbl = os.path.join(export_dir, "labels", "val", f"{stem}_valb.txt")
                    ok_val = False
                    if cv2_mod is not None:
                        try:
                            src_bgr = cv2_mod.imread(img_path)
                            if src_bgr is not None and getattr(src_bgr, "size", 0) != 0:
                                beta = 28 if (i % 2 == 0) else -24
                                aug = cv2_mod.convertScaleAbs(src_bgr, alpha=1.0, beta=beta)
                                ok_val = bool(cv2_mod.imwrite(dst_val_img, aug))
                        except Exception:
                            ok_val = False
                    if not ok_val:
                        try:
                            shutil.copy2(img_path, dst_val_img)
                            ok_val = True
                        except Exception:
                            ok_val = False
                    if ok_val:
                        try:
                            if img_path in labeled_set and os.path.isfile(src_lbl):
                                shutil.copy2(src_lbl, dst_val_lbl)
                            else:
                                with open(dst_val_lbl, "w", encoding="utf-8") as f:
                                    f.write("")
                            val_count += 1
                        except Exception:
                            pass
            yaml_lines = [
                f"path: {export_dir.replace(os.sep, '/')}",
                "train: images/train",
                f"val: {'images/val' if make_aug_val else 'images/train'}",
                f"nc: {len(self._class_names)}",
                "names:",
            ]
            for i, name in enumerate(self._class_names):
                safe = str(name).replace('"', '\\"')
                yaml_lines.append(f'  {i}: "{safe}"')
            try:
                with open(os.path.join(export_dir, "dataset.yaml"), "w", encoding="utf-8") as f:
                    f.write("\n".join(yaml_lines) + "\n")
            except Exception as exc:
                QMessageBox.critical(self, "Export", f"Failed to write dataset.yaml:\n{exc}")
                return
            extra = f"\nVal (brightness aug): {val_count}" if make_aug_val else "\nVal: using train split"
            unlabeled_count = max(0, len(export_images) - len(labeled_images))
            QMessageBox.information(
                self,
                "Export",
                f"Export done.\nTrain images: {count}\nLabeled: {len(labeled_images)}\nUnlabeled included: {unlabeled_count}{extra}\nFolder:\n{export_dir}",
            )

        def _export_json_dataset(self) -> None:
            if not self._image_paths:
                QMessageBox.warning(self, "Export", "Please load an image folder first.")
                return
            self._save_current_label()
            include_unlabeled = self._ask_include_unlabeled_for_export()
            labeled_images = self._iter_labeled_images()
            export_images = list(self._image_paths) if include_unlabeled else labeled_images
            if not export_images:
                QMessageBox.warning(self, "Export", "No labeled images found.")
                return
            out_root = QFileDialog.getExistingDirectory(self, "Select Export Parent Folder")
            if not out_root:
                return
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir = os.path.join(out_root, f"export_all_{ts}")
            os.makedirs(os.path.join(export_dir, "images"), exist_ok=True)
            data: dict[str, Any] = {
                "classes": list(self._class_names),
                "images": [],
            }
            for img_path in export_images:
                name = os.path.basename(img_path)
                try:
                    bgr = self._last_bgr if self._image_paths and img_path == self._image_paths[self._image_idx] else None
                    if bgr is None:
                        import cv2

                        bgr = cv2.imread(img_path)
                    if bgr is None:
                        continue
                    h, w = bgr.shape[:2]
                except Exception:
                    continue
                rects = self._labels_by_path.get(img_path)
                if rects is None:
                    rects = self._load_label_file_for_image(img_path, w, h)
                ann = []
                for r in rects:
                    x1, y1, x2, y2 = map(float, r[:4])
                    cid = int(r[4]) if len(r) >= 5 else 0
                    ang = float(r[5]) if len(r) >= 6 else 0.0
                    ann.append({"class_id": cid, "bbox_xyxy": [x1, y1, x2, y2], "angle_deg": ang})
                data["images"].append({"file_name": name, "width": w, "height": h, "annotations": ann})
                try:
                    shutil.copy2(img_path, os.path.join(export_dir, "images", name))
                except Exception:
                    pass
            try:
                import json

                with open(os.path.join(export_dir, "annotations.json"), "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception as exc:
                QMessageBox.critical(self, "Export", f"Failed to write annotations.json:\n{exc}")
                return
            QMessageBox.information(self, "Export", f"JSON export done.\nFolder:\n{export_dir}")

        def _export_golden_current(self) -> None:
            if not self._image_paths:
                QMessageBox.warning(self, "Golden", "Please load an image folder first.")
                return
            self._save_current_label()
            img_path = self._image_paths[self._image_idx]
            label_path = self._label_path_for_image(img_path)
            if not os.path.isfile(label_path):
                QMessageBox.warning(self, "Golden", "Current image has no label.")
                return
            out_parent = QFileDialog.getExistingDirectory(self, "Select Golden Export Parent Folder")
            if not out_parent:
                return
            out_parent = os.path.abspath(out_parent)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(out_parent, f"golden_{ts}")
            if os.path.exists(out_dir):
                i = 1
                while os.path.exists(f"{out_dir}_{i}"):
                    i += 1
                out_dir = f"{out_dir}_{i}"
            os.makedirs(out_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(img_path))[0]
            dst_lbl = os.path.join(out_dir, f"{stem}.txt")
            dst_img = os.path.join(out_dir, os.path.basename(img_path))
            dst_yaml = os.path.join(out_dir, "dataset.yaml")

            def _copy_with_fallback(src: str, dst: str) -> str:
                src_abs = os.path.abspath(src)
                dst_abs = os.path.abspath(dst)
                if src_abs == dst_abs:
                    return dst_abs
                try:
                    shutil.copy2(src_abs, dst_abs)
                    return dst_abs
                except PermissionError:
                    pass
                except OSError as exc:
                    if getattr(exc, "winerror", None) != 32:
                        raise
                base, ext = os.path.splitext(dst_abs)
                alt = f"{base}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
                shutil.copy2(src_abs, alt)
                return alt

            def _copy_background_cut_bundle_from_root(root: str) -> str:
                bundle = cut_background_detect.load_background_cut_bundle(root)
                if bundle is None:
                    raise RuntimeError("No background-cut golden bundle found in selected folder.")
                dst_bg_root = os.path.join(out_dir, "background_cut_golden")
                os.makedirs(dst_bg_root, exist_ok=True)
                rules_dst = os.path.join(dst_bg_root, "golden_rules.json")
                templ_dst = os.path.join(dst_bg_root, "golden_template.png")
                _copy_with_fallback(bundle.rules_path, rules_dst)
                _copy_with_fallback(bundle.template_path, templ_dst)
                board_src = os.path.join(os.path.dirname(bundle.rules_path), "golden_board.png")
                if os.path.isfile(board_src):
                    _copy_with_fallback(board_src, os.path.join(dst_bg_root, "golden_board.png"))
                source_src = os.path.join(os.path.dirname(bundle.rules_path), "golden_source_path.txt")
                if os.path.isfile(source_src):
                    _copy_with_fallback(source_src, os.path.join(dst_bg_root, "golden_source_path.txt"))
                return dst_bg_root

            try:
                dst_img = _copy_with_fallback(img_path, dst_img)
                dst_lbl = _copy_with_fallback(label_path, dst_lbl)
                yaml_lines = [
                    f"path: {out_dir.replace(os.sep, '/')}",
                    "train: .",
                    "val: .",
                    f"nc: {len(self._class_names)}",
                    "names:",
                ]
                for i, name in enumerate(self._class_names):
                    safe = str(name).replace('"', '\\"')
                    yaml_lines.append(f'  {i}: "{safe}"')
                with open(dst_yaml, "w", encoding="utf-8") as f:
                    f.write("\n".join(yaml_lines) + "\n")
            except Exception as exc:
                QMessageBox.critical(self, "Golden", f"Failed to export golden files:\n{exc}")
                return

            bg_bundle_root = ""
            id_cfg_path = ""
            try:
                # Follow old detect behavior: prefer nested background_cut_golden, then root.
                candidate_roots: list[str] = []
                if self._project_root:
                    candidate_roots.append(os.path.join(self._project_root, "background_cut_golden"))
                    candidate_roots.append(self._project_root)
                if self._project_dir and self._project_dir not in candidate_roots:
                    candidate_roots.append(os.path.join(self._project_dir, "background_cut_golden"))
                    candidate_roots.append(self._project_dir)
                for root in candidate_roots:
                    if not root or not os.path.isdir(root):
                        continue
                    try:
                        bg_bundle_root = _copy_background_cut_bundle_from_root(root)
                        break
                    except Exception:
                        continue
            except Exception:
                bg_bundle_root = ""
            if not bg_bundle_root:
                pick_bg = QMessageBox.question(
                    self,
                    "Cut Background Golden",
                    "No golden_for_cut_background found in current project.\nSelect a background-cut folder now?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if pick_bg == QMessageBox.StandardButton.Yes:
                    selected_bg = QFileDialog.getExistingDirectory(
                        self,
                        "Select Background Cut Folder (contains golden_rules.json + golden_template.png)",
                    )
                    if selected_bg:
                        try:
                            bg_bundle_root = _copy_background_cut_bundle_from_root(selected_bg)
                        except Exception as exc:
                            QMessageBox.warning(self, "Cut Background Golden", f"Load bundle failed:\n{exc}")
                if not bg_bundle_root:
                    try:
                        self.lbl_status.setText("Golden exported (without cut-background bundle)")
                    except Exception:
                        pass

            if bg_bundle_root:
                desired_dir = os.path.join(out_parent, f"golden_bundle_{ts}")
                if os.path.abspath(desired_dir) != os.path.abspath(out_dir):
                    final_dir = desired_dir
                    if os.path.exists(final_dir):
                        j = 1
                        while os.path.exists(f"{final_dir}_{j}"):
                            j += 1
                        final_dir = f"{final_dir}_{j}"
                    try:
                        shutil.move(out_dir, final_dir)
                        old_root = os.path.abspath(out_dir)
                        new_root = os.path.abspath(final_dir)

                        def _retarget(path_val: str) -> str:
                            p = os.path.abspath(str(path_val or ""))
                            if not p:
                                return p
                            if p == old_root:
                                return new_root
                            prefix = old_root + os.sep
                            if p.startswith(prefix):
                                return os.path.join(new_root, p[len(prefix) :])
                            return p

                        out_dir = new_root
                        dst_img = _retarget(dst_img)
                        dst_lbl = _retarget(dst_lbl)
                        dst_yaml = _retarget(dst_yaml)
                        bg_bundle_root = _retarget(bg_bundle_root)
                        if id_cfg_path:
                            id_cfg_path = _retarget(id_cfg_path)
                    except Exception:
                        pass

            if self._class_names:
                max_idx = len(self._class_names) - 1
                id_class_id = None
                id_class_name = None
                sub_id_class_id = None
                sub_id_class_name = None
                class_items = [f"-1: Disable"] + [f"{i}: {str(name)}" for i, name in enumerate(self._class_names)]

                id_prompt = "Select class for OCR image ID extraction in detect mode."
                id_item, ok = QInputDialog.getItem(
                    self,
                    "Golden ID Class",
                    id_prompt,
                    class_items,
                    0,
                    False,
                )
                if ok and str(id_item).strip():
                    try:
                        id_val = int(str(id_item).split(":", 1)[0].strip())
                    except Exception:
                        id_val = -1
                    if 0 <= id_val <= max_idx:
                        id_class_id = int(id_val)
                        id_class_name = str(self._class_names[id_class_id])

                sub_prompt = "Select class for OCR sub ID extraction in detect mode."
                sub_item, ok2 = QInputDialog.getItem(
                    self,
                    "Golden Sub ID Class",
                    sub_prompt,
                    class_items,
                    0,
                    False,
                )
                if ok2 and str(sub_item).strip():
                    try:
                        sub_val = int(str(sub_item).split(":", 1)[0].strip())
                    except Exception:
                        sub_val = -1
                    if 0 <= sub_val <= max_idx:
                        sub_id_class_id = int(sub_val)
                        sub_id_class_name = str(self._class_names[sub_id_class_id])
                if id_class_id is not None or sub_id_class_id is not None:
                    try:
                        id_cfg_path = golden_core.write_golden_id_config(
                            out_dir,
                            id_class_id,
                            id_class_name,
                            sub_id_class_id=sub_id_class_id,
                            sub_id_class_name=sub_id_class_name,
                        )
                    except Exception:
                        id_cfg_path = ""
            QMessageBox.information(
                self,
                "Golden",
                (
                    f"Golden exported:\nImage: {dst_img}\nLabel: {dst_lbl}\nMapping YAML: {dst_yaml}"
                    + (f"\nID Config: {id_cfg_path}" if id_cfg_path else "")
                    + (f"\nBG Cut Bundle: {bg_bundle_root}" if bg_bundle_root else "")
                ),
            )

        def _removed_base(self) -> str:
            if self._is_yolo_project and self._project_root:
                if self._yolo_use_split_layout:
                    return os.path.join(self._project_root, "removed", self._current_split)
                return os.path.join(self._project_root, "removed")
            return os.path.join(self._project_dir or os.getcwd(), "removed")

        def _remove_current_image(self) -> None:
            if not self._image_paths:
                return
            img_path = self._image_paths[self._image_idx]
            name = os.path.basename(img_path)
            ok = QMessageBox.question(
                self,
                "Remove",
                f"Remove current image?\n{name}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ok != QMessageBox.StandardButton.Yes:
                return
            self._save_current_label()
            label_path = self._label_path_for_image(img_path)
            removed_root = self._removed_base()
            removed_img_dir = os.path.join(removed_root, "images")
            removed_lbl_dir = os.path.join(removed_root, "labels")
            os.makedirs(removed_img_dir, exist_ok=True)
            os.makedirs(removed_lbl_dir, exist_ok=True)
            try:
                shutil.move(img_path, os.path.join(removed_img_dir, name))
                if os.path.isfile(label_path):
                    shutil.move(label_path, os.path.join(removed_lbl_dir, os.path.basename(label_path)))
            except Exception as exc:
                QMessageBox.critical(self, "Remove", f"Remove failed:\n{exc}")
                return
            self._reload_images_for_current_source()

        def _train_from_labels(self) -> None:
            if self._training_running:
                QMessageBox.information(self, "Train", "Training is already running.")
                return
            if not self._image_paths:
                QMessageBox.warning(self, "Train", "Please load an image folder first.")
                return
            self._save_current_label()
            labeled_images = self._iter_labeled_images()
            if not labeled_images:
                QMessageBox.warning(self, "Train", "No labeled images found.")
                return
            default_model = self._detect_model_default_path if os.path.isfile(self._detect_model_default_path) else self._detect_model_path
            settings = TrainSettingsDialog(self, default_model=default_model)
            if settings.exec() != QDialog.DialogCode.Accepted:
                return
            payload = settings.payload()
            out_dir = payload["out_dir"]
            model_path = payload["model_path"]
            run_name = payload["run_name"]
            epochs = int(payload["epochs"])
            imgsz = int(payload["imgsz"])
            batch = int(payload["batch"])

            self._training_running = True
            self._training_stop_requested = False
            self.btn_train.setEnabled(False)
            self._open_training_monitor_popup()

            def _worker() -> None:
                tmp_root = ""
                try:
                    tmp_root = os.path.abspath(os.path.join(out_dir, f"_qt_train_tmp_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"))
                    train_img_dir = os.path.join(tmp_root, "images", "train")
                    train_lbl_dir = os.path.join(tmp_root, "labels", "train")
                    os.makedirs(train_img_dir, exist_ok=True)
                    os.makedirs(train_lbl_dir, exist_ok=True)
                    for img_path in labeled_images:
                        name = os.path.basename(img_path)
                        stem = os.path.splitext(name)[0]
                        try:
                            shutil.copy2(img_path, os.path.join(train_img_dir, name))
                            shutil.copy2(self._label_path_for_image(img_path), os.path.join(train_lbl_dir, f"{stem}.txt"))
                        except Exception:
                            continue
                    yaml_path = os.path.join(tmp_root, "dataset.yaml")
                    yaml_lines = [
                        f"path: {tmp_root.replace(os.sep, '/')}",
                        "train: images/train",
                        "val: images/train",
                        f"nc: {len(self._class_names)}",
                        "names:",
                    ]
                    for i, name in enumerate(self._class_names):
                        safe = str(name).replace('"', '\\"')
                        yaml_lines.append(f'  {i}: "{safe}"')
                    with open(yaml_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(yaml_lines) + "\n")

                    yolo_cli = shutil.which("yolo")
                    if not yolo_cli:
                        raise RuntimeError("yolo CLI not found. Please install ultralytics.")
                    cmd = [
                        yolo_cli,
                        "train",
                        f"model={model_path}",
                        f"data={yaml_path}",
                        f"epochs={epochs}",
                        f"imgsz={imgsz}",
                        f"batch={batch}",
                        f"project={os.path.abspath(out_dir)}",
                        f"name={run_name}",
                        "exist_ok=True",
                        "verbose=True",
                        f"device={self._auto_train_device()}",
                    ]
                    cmd_text = " ".join(f'"{p}"' if " " in p else p for p in cmd)
                    run_cwd = self._project_root or self._project_dir or os.getcwd()
                    print("=" * 80, flush=True)
                    print(f"{run_cwd}> {cmd_text}", flush=True)
                    print("=" * 80, flush=True)
                    self._append_training_log_async("=" * 80)
                    self._append_training_log_async(f"{run_cwd}> {cmd_text}")
                    self._append_training_log_async("=" * 80)
                    self.training_command_signal.emit(f"{run_cwd}> {cmd_text}")
                    self.training_status_signal.emit("Training running...")

                    proc = subprocess.Popen(
                        cmd,
                        cwd=run_cwd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        bufsize=1,
                    )
                    self._training_process = proc
                    if proc.stdout is not None:
                        for raw in proc.stdout:
                            if self._training_stop_requested:
                                break
                            line = str(raw).rstrip("\r\n")
                            if not line:
                                continue
                            print(line, flush=True)
                            self._append_training_log_async(line)
                    rc = proc.wait()
                    self._training_process = None
                    if self._training_stop_requested:
                        self._append_training_log_async("[stopped] Training stopped by user.")
                        self.training_status_signal.emit("Training stopped")
                        return
                    if rc != 0:
                        raise RuntimeError(f"Training process exited with code {rc}")
                    output_path = os.path.join(os.path.abspath(out_dir), run_name)
                    self._append_training_log_async(f"[done] output: {output_path}")
                    self.training_done_signal.emit(output_path)
                except Exception as exc:
                    self._append_training_log_async(f"[error] {exc}")
                    self.training_failed_signal.emit(str(exc))
                finally:
                    self._training_process = None
                    self.training_finalize_signal.emit()

            threading.Thread(target=_worker, daemon=True).start()

        def _open_training_monitor_popup(self) -> None:
            if self._train_monitor_dialog is None:
                self._train_monitor_dialog = TrainingMonitorDialog(self, on_stop=self._stop_training_process)
            dlg = self._train_monitor_dialog
            if dlg is None:
                return
            dlg.show()
            dlg.raise_()
            dlg.activateWindow()
            dlg.txt_log.clear()
            dlg.set_command("")

        def _stop_training_process(self) -> None:
            if not self._training_running:
                return
            self._training_stop_requested = True
            self._append_training_log_async("[user] stop requested")
            proc = self._training_process
            if proc is None:
                return
            try:
                proc.terminate()
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        def _set_training_command(self, command_text: str) -> None:
            if self._train_monitor_dialog is None:
                return
            self._train_monitor_dialog.set_command(command_text)

        def _append_training_log(self, message: str) -> None:
            if self._train_monitor_dialog is None:
                return
            self._train_monitor_dialog.txt_log.append(str(message))

        def _append_training_log_async(self, message: str) -> None:
            self.training_log_signal.emit(str(message))

        def _on_training_status(self, text: str) -> None:
            self.lbl_status.setText(str(text))

        def _on_training_done(self, output_path: str) -> None:
            self.lbl_status.setText("Training finished")
            QMessageBox.information(self, "Train", f"Training finished.\nOutput:\n{output_path}")

        def _on_training_failed(self, message: str) -> None:
            self.lbl_status.setText("Training failed")
            QMessageBox.critical(self, "Train", f"Training failed:\n{message}")

        def _on_training_finalize(self) -> None:
            self._training_running = False
            self.btn_train.setEnabled(True)

        def _auto_train_device(self) -> str:
            try:
                import torch  # type: ignore

                if not torch.cuda.is_available():
                    return "cpu"
                cap = torch.cuda.get_device_capability(0)
                sm_tag = f"sm_{int(cap[0])}{int(cap[1])}"
                try:
                    arch_list = set(torch.cuda.get_arch_list() or [])
                except Exception:
                    arch_list = set()
                if arch_list and sm_tag not in arch_list:
                    print(
                        f"[train] CUDA device arch {sm_tag} is not supported by current torch build. "
                        "Fallback to CPU.",
                        flush=True,
                    )
                    return "cpu"
                return "0"
            except Exception:
                pass
            return "cpu"

        def _restore_removed_image(self) -> None:
            removed_root = self._removed_base()
            removed_img_dir = os.path.join(removed_root, "images")
            if not os.path.isdir(removed_img_dir):
                QMessageBox.information(self, "Restore", "No removed images.")
                return
            names = sorted([p.name for p in Path(removed_img_dir).iterdir() if p.is_file()])
            if not names:
                QMessageBox.information(self, "Restore", "No removed images.")
                return
            name, ok = QInputDialog.getItem(self, "Restore", "Select image to restore", names, 0, False)
            if not ok or not name:
                return
            src_img = os.path.join(removed_img_dir, name)
            src_lbl = os.path.join(removed_root, "labels", f"{os.path.splitext(name)[0]}.txt")
            if self._is_yolo_project and self._project_root:
                if self._yolo_use_split_layout:
                    dst_img_dir = os.path.join(self._project_root, "images", self._current_split)
                else:
                    dst_img_dir = os.path.join(self._project_root, "images")
            else:
                dst_img_dir = self._project_dir or os.getcwd()
            os.makedirs(dst_img_dir, exist_ok=True)
            dst_img = os.path.join(dst_img_dir, name)
            try:
                shutil.move(src_img, dst_img)
                if os.path.isfile(src_lbl):
                    shutil.move(src_lbl, self._label_path_for_image(dst_img))
            except Exception as exc:
                QMessageBox.critical(self, "Restore", f"Restore failed:\n{exc}")
                return
            self._reload_images_for_current_source()

        def _choose_detect_model(self) -> None:
            model_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select YOLO model",
                "",
                "Model files (*.pt *.onnx);;All files (*.*)",
            )
            if not model_path:
                return
            model_path = os.path.abspath(model_path)
            if not os.path.isfile(model_path):
                QMessageBox.warning(self, "Model", f"File not found:\n{model_path}")
                return
            self._detect_model_custom_path = model_path
            self._detect_model_path = model_path
            if hasattr(self, "combo_model") and self.combo_model is not None:
                custom_idx = self.combo_model.findData("custom_model")
                if custom_idx >= 0:
                    self.combo_model.blockSignals(True)
                    self.combo_model.setItemText(custom_idx, f"Custom: {os.path.basename(model_path)}")
                    self.combo_model.setCurrentIndex(custom_idx)
                    self.combo_model.blockSignals(False)
            self.lbl_status.setText(f"Model: {os.path.basename(model_path)}")

        def _on_model_combo_changed(self, _idx: int) -> None:
            key = ""
            try:
                key = str(self.combo_model.currentData() or "").strip()
            except Exception:
                key = ""
            if key == "default_yolo26m":
                self._detect_model_path = self._detect_model_default_path
                self.lbl_status.setText(f"Model: {os.path.basename(self._detect_model_path)} (default)")
                return
            if key == "custom_model":
                if self._detect_model_custom_path and os.path.isfile(self._detect_model_custom_path):
                    self._detect_model_path = self._detect_model_custom_path
                    self.lbl_status.setText(f"Model: {os.path.basename(self._detect_model_path)}")
                else:
                    self._detect_model_path = ""
                    self.lbl_status.setText("Model: custom not set (click Import Model)")

        def _run_auto_detect_current(self, silent: bool = False) -> None:
            if not self._image_paths:
                return
            img_path = self._image_paths[self._image_idx]
            model_path = self._detect_model_path
            if not model_path:
                if not silent:
                    QMessageBox.warning(self, "Auto Detect", "Please choose/import a model first.")
                return
            if not os.path.isfile(model_path):
                if not silent:
                    QMessageBox.warning(self, "Auto Detect", f"Model file not found:\n{model_path}")
                return
            try:
                from ultralytics import YOLO
            except Exception:
                if not silent:
                    QMessageBox.warning(self, "Auto Detect", "ultralytics is not available.")
                return
            try:
                model = YOLO(model_path)
                try:
                    results = model.predict(source=img_path, conf=0.25, verbose=False, device=0)
                except Exception as gpu_exc:
                    msg = str(gpu_exc).lower()
                    if "cuda" in msg or "cudart" in msg or "no kernel image" in msg:
                        # Fallback to CPU for unsupported CUDA runtime/GPU combos.
                        results = model.predict(source=img_path, conf=0.25, verbose=False, device="cpu")
                    else:
                        raise
                r0 = results[0]
                boxes = getattr(r0, "boxes", None)
                if boxes is None or getattr(boxes, "xyxy", None) is None:
                    if not silent:
                        QMessageBox.information(self, "Auto Detect", "No detections.")
                    return
                try:
                    xyxy = boxes.xyxy.tolist()
                except Exception:
                    xyxy = []
                try:
                    cls_vals = boxes.cls.tolist()
                except Exception:
                    cls_vals = []
                rects = self.canvas.rects
                for i, box in enumerate(xyxy):
                    if len(box) < 4:
                        continue
                    x1, y1, x2, y2 = map(float, box[:4])
                    raw_cid = int(cls_vals[i]) if i < len(cls_vals) else 0
                    cid = self._map_detect_class_id(raw_cid)
                    rects.append([x1, y1, x2, y2, int(cid), 0.0])
                self.canvas.rects = rects
                self.canvas._push_history()
                self._refresh_class_widgets()
                self.canvas.update()
                self._sync_canvas_rects_to_current_image()
                self.lbl_status.setText(f"Auto detect done: {len(xyxy)} boxes")
            except Exception as exc:
                if not silent:
                    QMessageBox.critical(self, "Auto Detect", f"Detection failed:\n{exc}")

        def _map_detect_class_id(self, raw_cid: int) -> int:
            try:
                cid = max(0, int(raw_cid))
            except Exception:
                cid = 0
            target_name = f"class{cid}"
            for idx, name in enumerate(self._class_names):
                if str(name).strip().lower() == target_name.lower():
                    return int(idx)
            self._class_names.append(target_name)
            return len(self._class_names) - 1

        def _jump_to_index(self, idx: int) -> None:
            if idx < 0 or idx >= len(self._image_paths):
                return
            self._sync_canvas_rects_to_current_image()
            self._image_idx = idx
            self._show_current_image()

        def _prev_image(self) -> None:
            if not self._image_paths:
                return
            self._sync_canvas_rects_to_current_image()
            self._image_idx = max(0, self._image_idx - 1)
            self._show_current_image()

        def _next_image(self) -> None:
            if not self._image_paths:
                return
            self._sync_canvas_rects_to_current_image()
            self._image_idx = min(len(self._image_paths) - 1, self._image_idx + 1)
            self._show_current_image()

        def keyPressEvent(self, event) -> None:
            key = event.key()
            mods = event.modifiers()
            if mods & Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_Z:
                self._undo_canvas()
                return
            if mods & Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_Y:
                self._redo_canvas()
                return
            if key in {Qt.Key.Key_D, Qt.Key.Key_Left}:
                self._prev_image()
                return
            if key in {Qt.Key.Key_F, Qt.Key.Key_Right}:
                self._next_image()
                return
            super().keyPressEvent(event)

        def resizeEvent(self, event) -> None:
            super().resizeEvent(event)
            self._refresh_image_preview()

        def closeEvent(self, event: QCloseEvent) -> None:
            try:
                if self._folder_refresh_timer.isActive():
                    self._folder_refresh_timer.stop()
                if self._autosave_timer.isActive():
                    self._autosave_timer.stop()
                self._autosave_flush()
                self._save_progress_yaml()
                if callable(self._on_back):
                    self._on_back()
            finally:
                super().closeEvent(event)

    class QtLauncherWindow(QMainWindow):
        def __init__(self, mode: str):
            super().__init__()
            self._startup_mode = mode
            self._allow_cross_mode = str(mode or "chooser").strip().lower() == "chooser"
            self._workspace = None
            self._label_workspace = None
            self._theme_mode = _get_global_theme_mode()
            self.setWindowTitle("GeckoAI")
            self.resize(560, 300)
            self._setup_ui()
            self._apply_theme_styles()
            if self._startup_mode == "label":
                QTimer.singleShot(0, self._launch_label_workspace)
            elif self._startup_mode == "detect":
                QTimer.singleShot(0, self._launch_detect_with_setup)

        def _setup_ui(self) -> None:
            root = QWidget(self)
            main_layout = QVBoxLayout(root)
            main_layout.setContentsMargins(24, 20, 24, 20)
            main_layout.setSpacing(14)
            self._launcher_root = root

            top = QHBoxLayout()
            top.addStretch(1)
            self.btn_theme = QPushButton("Light Mode", root)
            self.btn_theme.clicked.connect(self._toggle_theme)
            top.addWidget(self.btn_theme)
            main_layout.addLayout(top)

            title = QLabel("GeckoAI", root)
            self._launcher_title = title
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            main_layout.addWidget(title)

            subtitle = QLabel("Choose Label or Detect mode.", root)
            self._launcher_subtitle = subtitle
            subtitle.setWordWrap(True)
            subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
            main_layout.addWidget(subtitle)

            row = QHBoxLayout()
            row.setSpacing(10)
            self.btn_label = QPushButton("Label Mode", root)
            self.btn_detect = QPushButton("Detect Mode", root)
            self.btn_label.clicked.connect(self._launch_label_workspace)
            self.btn_detect.clicked.connect(self._launch_detect_with_setup)
            row.addWidget(self.btn_label)
            row.addWidget(self.btn_detect)
            main_layout.addLayout(row)
            mode_now = str(self._startup_mode or "chooser").strip().lower()
            if mode_now == "label" and not self._allow_cross_mode:
                self.btn_detect.hide()
            if mode_now == "detect" and not self._allow_cross_mode:
                self.btn_label.hide()

            self.btn_exit = QPushButton("Exit", root)
            self.btn_exit.clicked.connect(lambda: QApplication.instance().quit())
            main_layout.addWidget(self.btn_exit)

            self.setCentralWidget(root)

        def _toggle_theme(self) -> None:
            self._theme_mode = "light" if self._theme_mode == "dark" else "dark"
            _set_global_theme_mode(self._theme_mode)
            self._apply_theme_styles()

        def _apply_theme_styles(self) -> None:
            if self._theme_mode == "light":
                self._launcher_root.setStyleSheet("background:#FFFFFF;")
                self._launcher_title.setStyleSheet("font-size:24px;font-weight:700;color:#111111;")
                self._launcher_subtitle.setStyleSheet("color:#444444;")
                btn_style = (
                    "QPushButton{background:#F2F2F5;color:#111111;border:1px solid #C8C8CD;border-radius:6px;padding:8px 12px;}"
                    "QPushButton:hover{background:#E8E8ED;}"
                )
                self.btn_theme.setText("Dark Mode")
            else:
                self._launcher_root.setStyleSheet("background:#1F1F1F;")
                self._launcher_title.setStyleSheet("font-size:24px;font-weight:700;color:#F2F2F2;")
                self._launcher_subtitle.setStyleSheet("color:#C9CED6;")
                btn_style = (
                    "QPushButton{background:#2F2F2F;color:#F2F2F2;border:1px solid #595959;border-radius:6px;padding:8px 12px;}"
                    "QPushButton:hover{background:#3A3A3A;}"
                )
                self.btn_theme.setText("Light Mode")
            self.btn_label.setStyleSheet(btn_style)
            self.btn_detect.setStyleSheet(btn_style)
            self.btn_exit.setStyleSheet(btn_style)
            self.btn_theme.setStyleSheet(btn_style)

        def _set_busy(self, busy: bool) -> None:
            for btn in [self.btn_label, self.btn_detect, self.btn_exit]:
                btn.setEnabled(not busy)

        def _launch_detect_with_setup(self) -> None:
            dialog = DetectSetupDialog(self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            payload = dialog.payload()
            self._workspace = DetectWorkspaceWindow(payload, on_back=self._on_workspace_back)
            self._workspace.show()
            self.hide()

        def _on_workspace_back(self) -> None:
            self._workspace = None
            self.showNormal()
            self.raise_()
            self.activateWindow()

        def _launch_label_workspace(self) -> None:
            if self._label_workspace is None:
                self._label_workspace = LabelWorkspaceWindow(
                    on_back=self._on_label_workspace_back,
                    allow_detect_bridge=self._allow_cross_mode,
                )
            self._label_workspace.showNormal()
            self._label_workspace.raise_()
            self._label_workspace.activateWindow()
            self.hide()

        def _on_label_workspace_back(self) -> None:
            self.showNormal()
            self.raise_()
            self.activateWindow()

    app = QApplication.instance() or QApplication(sys.argv)
    try:
        icon_path = _resolve_app_icon_path()
        if icon_path:
            app.setWindowIcon(QIcon(icon_path))
    except Exception:
        pass
    win = QtLauncherWindow(startup_mode)
    app._gecko_main_window = win
    win.show()
    win.showNormal()
    win.raise_()
    win.activateWindow()
    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GeckoAI PySide6 launcher")
    parser.add_argument(
        "--mode",
        choices=["chooser", "label", "detect"],
        default="chooser",
        help="Startup mode",
    )
    return parser.parse_args(argv)


def _write_startup_error_log(exc: BaseException) -> None:
    try:
        log_path = os.path.join(os.path.expanduser("~"), ".ai_labeller_startup.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.datetime.now().isoformat(timespec='seconds')}] startup failure\n")
            f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
            f.write("\n")
    except Exception:
        pass


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        app = _build_main_window(args.mode)
        app.exec()
    except Exception as exc:
        _write_startup_error_log(exc)
        traceback.print_exc()
        raise


def run_qt_mode(startup_mode: str) -> None:
    try:
        app = _build_main_window((startup_mode or "chooser").strip().lower())
        app.exec()
    except Exception as exc:
        _write_startup_error_log(exc)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
