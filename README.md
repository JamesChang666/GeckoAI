# Ultimate AI Labeller

Desktop image annotation tool for object detection datasets (Tkinter + Ultralytics), with a separate web project.

## Features

- Bounding-box annotation with drag, move, and resize handles
- Multi-select boxes (`Shift/Ctrl + Click`) for batch class reassignment and delete
- Select all boxes in current image (`Ctrl+A`)
- Nested/overlapping box picking prefers inner (smaller) box for easier adjustment
- Undo/redo history (`Ctrl+Z`, `Ctrl+Y`)
- Image navigation (`F` next/save, `D` previous)
- Auto red-region proposal (`A`)
- YOLO detection from UI (`Run Detection`)
- Startup source selection:
  - Open Images Folder
  - Open YOLO Dataset
  - Open RF-DETR Dataset
- Detection model management:
  - Official model mode (`yolo26m.pt` path by default)
  - Import custom models (`.pt`, `.onnx`) via `Browse Model`
  - Select model from dropdown library
- Train from existing labels:
  - Choose training range by index
  - Save training artifacts to selected output folder
  - Non-blocking background training (continue labeling while training)
  - Built-in training monitor (command/log/progress/ETA)
- Class management:
  - Add / rename / delete class in class table
  - Deleting a class reindexes following class IDs automatically
- Auto-detect and propagate options
- Scrollable right settings panel
- Remove/restore bad frames from split
- Image dropdown jump
- Session resume (last project/split/image/model settings)
- English/Chinese UI switch and light/dark theme
- Export all annotations by format:
  - `YOLO (.txt)` export to `images/train` + `labels/train` structure
  - `JSON` full dataset export (per-image annotation json)

## Repositories

- Desktop app (this repo): `https://github.com/JamesChang666/ultimate_ai_labeller`
- Web app (separate repo): `https://github.com/JamesChang666/labeller_web`

## Dataset Structure

```text
your_project/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

- Image extensions: `.png`, `.jpg`, `.jpeg`
- Label format: YOLO txt (`class cx cy w h`, normalized)
- Full guide (ZH): `docs/dataset-structure-guide.md`

Removed frames are moved to:

```text
your_project/
  removed/
    train|val|test/
      images/
      labels/
```

## Install

From PyPI:

```bash
pip install ultimate_ai_labeller
```

From local wheel:

```bash
pip install dist/ultimate_ai_labeller-0.1.10-py3-none-any.whl
```

From source:

```bash
pip install .
```

For development:

```bash
pip install -e .
```

## Run

```bash
ai-labeller
```

Or:

```bash
python src/ai_labeller/main.py
```

## Web Version

This desktop repository includes local development files under `web_labeller/`, but the maintained web repository is:

- `https://github.com/JamesChang666/labeller_web`

Run locally:

```bash
cd web_labeller
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

## Shortcuts

- `F`: save and next image
- `D`: previous image
- `A`: auto red detection
- `Ctrl+Z`: undo
- `Ctrl+Y`: redo
- `Ctrl+A`: select all boxes in current image
- `Delete`: delete selected box

## Notes

- Default detection model mode is `Official YOLO26m.pt (Bundled)`.
- If the official model file is unavailable locally, import a custom `.pt/.onnx` model from the UI.
- To use your own Tk app icon, put `app_icon.png` in `src/ai_labeller/assets/`.
- Session file: `~/.ai_labeller_session.json`.
- Project progress YAML: `<project_root>/.ai_labeller_progress.yaml` (resume split/image and class names after reopen).
