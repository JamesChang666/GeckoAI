from __future__ import annotations

from dataclasses import dataclass, field

from .types import Rect


@dataclass
class SessionState:
    project_root: str = ""
    split: str = "train"
    image_name: str = ""
    detection_model_mode: str = "Official YOLO26m.pt (Bundled)"
    detection_model_path: str = "yolo26m.pt"


@dataclass
class AppState:
    project_root: str = ""
    current_split: str = "train"
    image_files: list[str] = field(default_factory=list)
    current_idx: int = 0
    rects: list[Rect] = field(default_factory=list)
    class_names: list[str] = field(default_factory=lambda: ["0", "1", "2"])
