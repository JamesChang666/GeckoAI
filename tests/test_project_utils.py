import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

from ai_labeller.features import project_utils


class _Combo:
    def __init__(self):
        self.value = ""

    def set(self, value):
        self.value = value


class _DummyApp:
    def __init__(self, progress: dict[str, str]):
        self.project_root = ""
        self.current_split = ""
        self.combo_split = _Combo()
        self.class_names = ["0", "1", "2"]
        self.image_files = []
        self.current_idx = 0
        self.img_pil = None
        self.img_tk = None
        self.rects = []
        self.lang = "en"
        self.root = None
        self._progress = progress
        self.refresh_calls = 0
        self.saved = 0

    def _read_project_progress_yaml(self, _project_root: str):
        return self._progress

    def _extract_class_names_from_progress(self, progress: dict[str, str]):
        count = int(progress.get("class_count", "0") or "0")
        names = []
        for idx in range(count):
            name = progress.get(f"class_{idx}", "")
            if not name:
                return []
            names.append(name)
        return names

    def _refresh_class_dropdown(self, preferred_idx=0):
        self.refresh_calls += 1
        self._preferred_idx = preferred_idx

    def update_info_text(self):
        return None

    def render(self):
        return None

    def save_session_state(self):
        self.saved += 1


class ProjectUtilsTests(unittest.TestCase):
    def setUp(self):
        self.tmp_root = Path("tests/.tmp_project_utils")
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        (self.tmp_root / "a.png").write_bytes(b"png")
        (self.tmp_root / "b.png").write_bytes(b"png")
        (self.tmp_root / "labels" / "train").mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp_root, ignore_errors=True)

    def test_load_images_folder_restores_progress_classes_and_image(self):
        progress = {
            "image_name": "b.png",
            "image_index": "0",
            "class_count": "2",
            "class_0": "cat",
            "class_1": "dog",
        }
        app = _DummyApp(progress)

        with patch("ai_labeller.features.image_load.load_image") as mock_load_image:
            project_utils.load_images_folder_only(app, str(self.tmp_root))

        self.assertEqual(app.project_root, str(self.tmp_root).replace("\\", "/"))
        self.assertEqual(app.current_split, "train")
        self.assertEqual(app.combo_split.value, "train")
        self.assertEqual(app.class_names, ["cat", "dog"])
        self.assertEqual(app.current_idx, 1)
        self.assertEqual(app.refresh_calls, 1)
        mock_load_image.assert_called_once_with(app)


if __name__ == "__main__":
    unittest.main()
