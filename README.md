# GeckoAI

Desktop image annotation tool for object detection datasets (PySide6 + Ultralytics).

## PySide6 Migration Status

- Phase 1 complete: `geckoai-qt` launcher is available.
- Phase 2 complete: Detect setup wizard (model/source/output/conf) is Qt-native.
- Phase 3 complete: Detect setup supports golden/OCR parameters in Qt and passes them into the original detect workflow.
- Current scope:
  - Qt-native startup launcher window
  - `Label / Detect / Legacy Home` mode selection
  - Detect setup in Qt, detect workspace and runtime keep original behavior
  - Label workflow still uses the stable Tkinter app
- Install Qt launcher dependency:

```bash
pip install -e ".[qt]"
```

- Run Qt launcher:

```bash
geckoai-qt
```

## Features

- Bounding-box annotation with drag, move, and resize handles
- Rotated bounding boxes:
  - Drag rotate knob on selected box
  - 8 resize handles follow box rotation
  - Keyboard rotate (`Q/E`, `Shift+Q/E`)
- Multi-select boxes (`Shift/Ctrl + Click`) for batch class reassignment and delete
- Select all boxes in current image (`Ctrl+A`)
- Nested/overlapping box picking prefers inner (smaller) box for easier adjustment
- Undo/redo history (`Ctrl+Z`, `Ctrl+Y`)
- Image navigation (`F` next/save, `D` previous)
- YOLO detection from UI (`Run Detection`)
- Detect mode golden workflow enhancements:
  - Import golden folder and auto-load `background_cut_golden` bundle when present
  - Auto cut background and detect cut pieces for each source image
  - Piece-by-piece display and navigation in detect workspace
  - Cached detect results per source image (back/next does not re-run detection)
  - Report dedupe (same image/piece is not appended repeatedly)
  - Save rendered detect images into `detect_results_xxx/detected_images/`
- Detect mode setup wizard:
  - Step 1: Choose model
  - Step 2: Choose source (`Camera` or `Image Folder`)
  - Camera path: pick camera (when multiple cameras are found), choose auto/manual FPS mode, set confidence threshold
  - Image folder path: choose source folder, confidence threshold, run type, output folder, then start detect
  - Class color mapping:
    - List model classes and assign per-class box colors
    - Double-click class row to set color quickly
    - Classes without assigned color use auto-generated deterministic colors
- Startup source selection:
  - Dropdown chooser (default: `Open Images Folder`)
  - Open YOLO Dataset
  - Open RF-DETR Dataset
- Logo/app-name click returns to source/main page
- Detection model management:
  - Official model mode (`yolo26m.pt` path by default)
  - Import custom models (`.pt`, `.onnx`) via `Browse Model`
  - Select model from dropdown library
- Train from existing labels:
  - Choose training range by index
  - Choose weight source before training:
    - Official `yolo26m.pt`
    - Custom weight file (`.pt` / `.onnx`)
    - From scratch (`pretrained=False`)
  - Save training artifacts to selected output folder
  - Non-blocking background training (continue labeling while training)
  - Built-in training monitor (command/log/progress/ETA)
- Class management:
  - Add / rename / delete class in class table
  - Deleting a class reindexes following class IDs automatically
- Auto-detect and propagate options (3 propagate modes: no-label-only / always / selected labels only)
- OCR for golden ID/Sub ID:
  - EasyOCR first, PaddleOCR fallback
  - OCR runs on selected ID-class detection area only
  - OCR auto-tries 0/90/180/270 rotations for rotated text
- Scrollable right settings panel
- Remove/restore bad frames from split (icon buttons beside image dropdown)
- File info counters: boxes, and classes in current frame / total classes
- Image dropdown jump
- Session resume (last project/split/image/model settings)
- English UI and light/dark theme
- Export controls in top toolbar (next to undo/redo):
  - Format dropdown (`YOLO (.txt)` / `JSON`)
  - `Export`
  - `Export Golden` (export golden folder)
- Previous-label ghost workflow:
  - Optional ghost overlay of last image labels (dotted)
  - Right-click on a ghost box to paste only that clicked box
- Right-click drag to draw new box directly

## Repositories

- Desktop app (this repo): `https://github.com/JamesChang666/GeckoAI`

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
- Label format: YOLO OBB txt (`class x1 y1 x2 y2 x3 y3 x4 y4`, normalized)
- Legacy labels (`class cx cy w h`) are still readable for backward compatibility
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
pip install GeckoAI
```

From local wheel:

```bash
pip install dist/GeckoAI-*.whl
```

From source:

```bash
pip install .
```

For development:

```bash
pip install -e .
```

One-shot install for build + packaging:

```bash
pip install -e ".[build]"
```

## Run

```bash
geckoai
```

Or:

```bash
python src/ai_labeller/main.py
```

Single-mode entrypoints:

```bash
geckoai-all
geckoai-label
geckoai-detect
geckoai-qt
geckoai-report <detect_results_xxx.csv>
```

## Build EXE (Windows)

Install once:

```bash
pip install -e ".[build]"
```

Build one target:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_exe.ps1 -Target all
powershell -ExecutionPolicy Bypass -File .\scripts\build_exe.ps1 -Target label
powershell -ExecutionPolicy Bypass -File .\scripts\build_exe.ps1 -Target detect
```

Output:

```text
dist/GeckoAI-All/GeckoAI-All.exe
dist/GeckoAI-Label/GeckoAI-Label.exe
dist/GeckoAI-Detect/GeckoAI-Detect.exe
```

## Shortcuts

- `F`: save and next image
- `D`: previous image
- `Q/E`: rotate selected box (-5 deg / +5 deg)
- `Shift+Q/E`: rotate selected box faster (-15 deg / +15 deg)
- `Ctrl+Z`: undo
- `Ctrl+Y`: redo
- `Ctrl+A`: select all boxes in current image
- `Ctrl+Left Drag`: marquee multi-select boxes
- `Right Drag`: draw new box
- `Delete`: delete selected box

## Notes

- Default detection model mode is `Official YOLO26m.pt (Bundled)`.
- If the official model file is unavailable locally, import a custom `.pt/.onnx` model from the UI.
- If CUDA/GPU runtime is incompatible, detection/training automatically falls back to CPU.
- Detect mode in golden background-cut workflow:
  - Each cut piece is written as an image and detected
  - `F`/`D` navigation reuses cached results instead of re-detecting the same source image
  - CSV report rows are written once per unique image/piece key
- To use your own Tk app icon, put `app_icon.png` in `src/ai_labeller/assets/`.
- Session file: `~/.ai_labeller_session.json`.
- Project progress YAML: `<project_root>/.ai_labeller_progress.yaml` (resume split/image and class names after reopen).
- Flat image folder mode defaults classes to `0`, `1`, `2`.
