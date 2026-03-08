import glob
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Any

from ai_labeller.constants import LANG_MAP, COLORS


def normalize_project_root(directory: str) -> str:
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


def find_yolo_project_root(directory: str) -> str | None:
    root = directory.replace("\\", "/").rstrip("/")
    if not root:
        return None

    def is_yolo_root(candidate: str) -> bool:
        return any(
            os.path.exists(f"{candidate}/images/{split}")
            for split in ("train", "val", "test")
        )

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

    try:
        for child in os.listdir(root):
            child_path = os.path.join(root, child).replace("\\", "/")
            if os.path.isdir(child_path) and is_yolo_root(child_path):
                return child_path
    except Exception:
        pass

    return None


def _glob_image_files(folder_path: str, include_bmp: bool = False) -> list[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp") if include_bmp else (".png", ".jpg", ".jpeg")
    return sorted(
        f for f in glob.glob(f"{folder_path}/*.*")
        if f.lower().endswith(exts)
    )


def _glob_label_files(project_root: str) -> list[str]:
    split_roots = [s for s in ("train", "val", "test") if os.path.isdir(f"{project_root}/labels/{s}")]
    if split_roots:
        label_files: list[str] = []
        for split in split_roots:
            label_files.extend(glob.glob(f"{project_root}/labels/{split}/*.txt"))
        return label_files
    return glob.glob(f"{project_root}/labels/train/*.txt")


def list_split_images_for_root(project_root: str, split: str) -> list[str]:
    img_path = f"{project_root}/images/{split}"
    return _glob_image_files(img_path)


def existing_image_splits(project_root: str) -> list[str]:
    splits: list[str] = []
    for split in ("train", "val", "test"):
        if os.path.isdir(f"{project_root}/images/{split}"):
            splits.append(split)
    return splits


def ensure_yolo_label_dirs(project_root: str) -> None:
    splits = existing_image_splits(project_root)
    if not splits:
        return
    for split in splits:
        os.makedirs(f"{project_root}/labels/{split}", exist_ok=True)


def diagnose_folder_structure(directory: str) -> dict[str, Any]:
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


def show_folder_diagnosis(app, directory: str) -> None:
    diag = diagnose_folder_structure(directory)
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
    app.root.lift()
    app.root.focus_force()
    messagebox.showwarning("Folder Diagnosis", "\n".join(lines), parent=app.root)


def _list_split_labeled_images_for_root(project_root: str, split: str) -> list[str]:
    result: list[str] = []
    for img_path in list_split_images_for_root(project_root, split):
        base = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = f"{project_root}/labels/{split}/{base}.txt"
        if os.path.exists(lbl_path):
            result.append(img_path)
    return result


def _list_flat_labeled_images_for_root(project_root: str) -> list[str]:
    result: list[str] = []
    for img_path in _glob_image_files(project_root):
        base = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = f"{project_root}/labels/train/{base}.txt"
        if os.path.exists(lbl_path):
            result.append(img_path)
    return result


def list_split_images(project_root: str, split: str) -> list[str]:
    return list_split_images_for_root(project_root, split)


def load_images_folder_only(app, directory: str) -> None:
    app.project_root = directory
    app.current_split = "train"
    app.combo_split.set(app.current_split)

    app.image_files = _glob_image_files(app.project_root)

    os.makedirs(f"{app.project_root}/labels/{app.current_split}", exist_ok=True)

    if not app.image_files:
        messagebox.showinfo(
            LANG_MAP[app.lang]["title"],
            LANG_MAP[app.lang]["no_img"],
        )
        app.current_idx = 0
        app.img_pil = None
        app.img_tk = None
        app.rects = []
        app.update_info_text()
        app.render()
        app.save_session_state()
        return

    app.current_idx = 0
    from ai_labeller.features import image_load
    image_load.load_image(app)


def load_project_from_path(app, directory, preferred_image=None, save_session=True):
    app.project_root = directory.replace('\\', '/')
    ensure_yolo_label_dirs(app.project_root)

    progress = app._read_project_progress_yaml(app.project_root)
    progress_split = progress.get("split", "")
    progress_image = progress.get("image_name", "")
    progress_class_names = app._extract_class_names_from_progress(progress)

    if progress_class_names:
        app.class_names[:] = progress_class_names
        app._refresh_class_dropdown()

    if not preferred_image:
        if progress_split in {"train", "val", "test"}:
            app.current_split = progress_split
        if progress_image:
            preferred_image = progress_image

    split_files = {
        split: list_split_images_for_root(app.project_root, split)
        for split in ("train", "val", "test")
        if os.path.exists(f"{app.project_root}/images/{split}")
    }
    non_empty_splits = [s for s, files in split_files.items() if files]
    if non_empty_splits:
        if app.current_split not in non_empty_splits:
            app.current_split = "train" if "train" in non_empty_splits else non_empty_splits[0]
    elif split_files:
        if app.current_split not in split_files:
            app.current_split = "train" if "train" in split_files else next(iter(split_files))
    img_path = f"{app.project_root}/images/{app.current_split}"

    if not os.path.exists(img_path):
        os.makedirs(f"{app.project_root}/labels/{app.current_split}", exist_ok=True)

    if hasattr(app, "combo_split"):
        try:
            if app.combo_split.winfo_exists():
                app.combo_split.set(app.current_split)
        except tk.TclError:
            pass
    app.load_split_data(preferred_image=preferred_image)
    if save_session:
        app.save_session_state()


def load_project_root(app):
    directory = filedialog.askdirectory()
    if not directory:
        return
    normalized = normalize_project_root(directory)
    yolo_root = find_yolo_project_root(normalized)
    app.load_project_from_path(yolo_root or normalized)


def load_split_data(app, preferred_image=None):
    img_path = f"{app.project_root}/images/{app.current_split}"

    if app.project_root and not os.path.exists(img_path):
        fallback = next(
            (s for s in ("train", "val", "test") if os.path.exists(f"{app.project_root}/images/{s}")),
            None,
        )
        if fallback:
            app.current_split = fallback
            if hasattr(app, "combo_split"):
                try:
                    if app.combo_split.winfo_exists():
                        app.combo_split.set(app.current_split)
                except tk.TclError:
                    pass
            img_path = f"{app.project_root}/images/{app.current_split}"

    if app.project_root and os.path.exists(img_path):
        current_files = list_split_images_for_root(app.project_root, app.current_split)
        if not current_files:
            fallback_non_empty = next(
                (
                    s for s in ("train", "val", "test")
                    if os.path.exists(f"{app.project_root}/images/{s}") and list_split_images_for_root(app.project_root, s)
                ),
                None,
            )
            if fallback_non_empty and fallback_non_empty != app.current_split:
                app.current_split = fallback_non_empty
                if hasattr(app, "combo_split"):
                    try:
                        if app.combo_split.winfo_exists():
                            app.combo_split.set(app.current_split)
                    except tk.TclError:
                        pass
                img_path = f"{app.project_root}/images/{app.current_split}"

    if os.path.exists(img_path):
        app.image_files = list_split_images_for_root(app.project_root, app.current_split)
        if app.image_files:
            if preferred_image:
                name_to_idx = {
                    os.path.basename(path): i
                    for i, path in enumerate(app.image_files)
                }
                app.current_idx = name_to_idx.get(preferred_image, 0)
            else:
                app.current_idx = 0
            from ai_labeller.features import image_load
            image_load.load_image(app)
        else:
            app.current_idx = 0
            app.img_pil = None
            app.img_tk = None
            app.rects = []
            app.update_info_text()
    else:
        app.image_files = []
        app.current_idx = 0
        app.img_pil = None
        app.img_tk = None
        app.rects = []
        app.update_info_text()

    app.render()
    app.save_session_state()
