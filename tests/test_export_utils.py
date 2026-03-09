import json
import logging
import shutil
import unittest
from pathlib import Path

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

from ai_labeller.features import export_utils

try:
    from PIL import Image
except Exception:
    Image = None


class _Var:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _Dialog:
    def __init__(self, out_dir: str):
        self._out_dir = out_dir

    def askdirectory(self, **_kwargs):
        return self._out_dir


class _DummyApp:
    def __init__(self, project_root: Path, out_dir: Path, fmt: str = "COCO"):
        self.project_root = str(project_root).replace("\\", "/")
        self.class_names = ["alpha", "beta"]
        self.root = None
        self.lang = "en"
        self.LANG_MAP = {"en": {"title": "GeckoAI"}}
        self.var_export_format = _Var(fmt)
        self.filedialog = _Dialog(str(out_dir).replace("\\", "/"))
        self.image_files = []
        self.img_pil = None
        self.logger = logging.getLogger("test-export-utils")

    def save_current(self):
        return None

    def _rotation_meta_path_for_label(self, lbl_path: str) -> str:
        return str(Path(lbl_path).with_suffix(".angles.txt"))

    def _read_rotation_meta_angles(self, meta_path: str):
        values = []
        if not Path(meta_path).is_file():
            return values
        for line in Path(meta_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            values.append(float(line))
        return values

    def _list_split_images_for_root(self, root: str, split: str):
        split_dir = Path(root) / "images" / split
        if not split_dir.is_dir():
            return []
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        return sorted(
            str(p).replace("\\", "/")
            for p in split_dir.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        )

    def _glob_image_files(self, root: str):
        root_dir = Path(root)
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        return sorted(
            str(p).replace("\\", "/")
            for p in root_dir.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        )


@unittest.skipUnless(Image is not None, "Pillow is required for export tests")
class ExportUtilsTests(unittest.TestCase):
    def setUp(self):
        self.tmp_root = Path("tests/.tmp_export")
        self.project_root = self.tmp_root / "project"
        self.out_parent = self.tmp_root / "output_parent"
        self.out_parent.mkdir(parents=True, exist_ok=True)
        (self.project_root / "images" / "train").mkdir(parents=True, exist_ok=True)
        (self.project_root / "images" / "val").mkdir(parents=True, exist_ok=True)
        (self.project_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (self.project_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

        self._make_image(self.project_root / "images" / "train" / "a.png", (100, 80))
        self._make_image(self.project_root / "images" / "val" / "b.png", (120, 60))

        # OBB format: class x1 y1 x2 y2 x3 y3 x4 y4 (normalized)
        (self.project_root / "labels" / "train" / "a.txt").write_text(
            "0 0.10 0.10 0.50 0.10 0.50 0.40 0.10 0.40\n",
            encoding="utf-8",
        )
        # axis-aligned format + angle from meta
        (self.project_root / "labels" / "val" / "b.txt").write_text(
            "2 0.5 0.5 0.4 0.6\n",
            encoding="utf-8",
        )
        (self.project_root / "labels" / "val" / "b.angles.txt").write_text(
            "30\n",
            encoding="utf-8",
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_root, ignore_errors=True)

    def _make_image(self, path: Path, size: tuple[int, int]) -> None:
        img = Image.new("RGB", size, color=(10, 20, 30))
        img.save(path)
        img.close()

    def test_export_all_coco_writes_expected_payloads(self):
        out_dir = self.tmp_root / "direct_export"
        app = _DummyApp(self.project_root, self.out_parent, fmt="COCO")

        count = export_utils._export_all_coco(app, str(out_dir).replace("\\", "/"))
        self.assertEqual(count, 2)

        all_json = out_dir / "annotations" / "instances_all.json"
        train_json = out_dir / "annotations" / "instances_train.json"
        val_json = out_dir / "annotations" / "instances_val.json"
        self.assertTrue(all_json.is_file())
        self.assertTrue(train_json.is_file())
        self.assertTrue(val_json.is_file())

        payload = json.loads(all_json.read_text(encoding="utf-8"))
        self.assertEqual(len(payload["images"]), 2)
        self.assertEqual(len(payload["annotations"]), 2)
        self.assertIn("categories", payload)
        category_ids = sorted(c["id"] for c in payload["categories"])
        self.assertEqual(category_ids, [0, 1, 2])

        anns = payload["annotations"]
        for ann in anns:
            self.assertEqual(ann["iscrowd"], 0)
            self.assertEqual(len(ann["segmentation"]), 1)
            self.assertEqual(len(ann["segmentation"][0]), 8)
            self.assertEqual(len(ann["bbox"]), 4)

    def test_export_all_by_selected_format_dispatches_coco(self):
        app = _DummyApp(self.project_root, self.out_parent, fmt="COCO")
        old_showinfo = export_utils.messagebox.showinfo
        old_showwarning = export_utils.messagebox.showwarning
        old_showerror = export_utils.messagebox.showerror
        old_askyesno = export_utils.messagebox.askyesno
        try:
            export_utils.messagebox.showinfo = lambda *args, **kwargs: None
            export_utils.messagebox.showwarning = lambda *args, **kwargs: None
            export_utils.messagebox.showerror = lambda *args, **kwargs: None
            export_utils.messagebox.askyesno = lambda *args, **kwargs: False
            export_utils.export_all_by_selected_format(app)
        finally:
            export_utils.messagebox.showinfo = old_showinfo
            export_utils.messagebox.showwarning = old_showwarning
            export_utils.messagebox.showerror = old_showerror
            export_utils.messagebox.askyesno = old_askyesno

        export_roots = sorted(self.out_parent.glob("export_all_*"))
        self.assertTrue(export_roots, "expected one export output folder")
        result = export_roots[-1]
        self.assertTrue((result / "annotations" / "instances_all.json").is_file())
        self.assertTrue((result / "images" / "train").is_dir())
        self.assertTrue((result / "images" / "val").is_dir())


if __name__ == "__main__":
    unittest.main()
