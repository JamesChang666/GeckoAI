from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any


def _log(msg: str) -> None:
    text = str(msg)
    try:
        print(text, flush=True)
        return
    except UnicodeEncodeError:
        pass
    stream = getattr(sys, "stdout", None)
    encoding = getattr(stream, "encoding", None) or "utf-8"
    safe = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(safe, flush=True)


def _convert_label_text_for_task(label_text: str, task: str) -> str:
    task_name = str(task or "detect").strip().lower()
    out_lines: list[str] = []
    for raw in str(label_text or "").splitlines():
        parts = raw.strip().split()
        if not parts:
            continue
        if task_name == "obb" and len(parts) >= 5 and len(parts) < 9:
            try:
                class_id = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:5])
            except Exception:
                continue
            x1 = cx - (bw / 2.0)
            y1 = cy - (bh / 2.0)
            x2 = cx + (bw / 2.0)
            y2 = cy - (bh / 2.0)
            x3 = cx + (bw / 2.0)
            y3 = cy + (bh / 2.0)
            x4 = cx - (bw / 2.0)
            y4 = cy + (bh / 2.0)
            vals = [x1, y1, x2, y2, x3, y3, x4, y4]
            clipped = [max(0.0, min(1.0, float(v))) for v in vals]
            out_lines.append(f"{class_id} " + " ".join(f"{v:.6f}" for v in clipped))
            continue
        if task_name != "obb" and len(parts) >= 9:
            try:
                class_id = int(float(parts[0]))
                pts = list(map(float, parts[1:9]))
            except Exception:
                continue
            px = [pts[0], pts[2], pts[4], pts[6]]
            py = [pts[1], pts[3], pts[5], pts[7]]
            x1 = min(px)
            x2 = max(px)
            y1 = min(py)
            y2 = max(py)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            out_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            continue
        out_lines.append(" ".join(parts))
    return "\n".join(out_lines) + ("\n" if out_lines else "")


def _copy_label_for_task(src_label: str, dst_label: str, task: str) -> None:
    with open(src_label, "r", encoding="utf-8") as f:
        text = f.read()
    normalized = _convert_label_text_for_task(text, task)
    with open(dst_label, "w", encoding="utf-8") as f:
        f.write(normalized)


def _prepare_dataset(job: dict[str, Any]) -> tuple[str, str]:
    class_names = [str(x) for x in (job.get("class_names") or [])]
    if not class_names:
        class_names = ["class0"]
    train_items = list(job.get("train_items") or [])
    val_items = list(job.get("val_items") or [])
    task = str(job.get("task", "detect")).strip().lower() or "detect"

    tmp_root = tempfile.mkdtemp(prefix="qt_train_")
    train_img_dir = os.path.join(tmp_root, "images", "train")
    train_lbl_dir = os.path.join(tmp_root, "labels", "train")
    val_img_dir = os.path.join(tmp_root, "images", "val")
    val_lbl_dir = os.path.join(tmp_root, "labels", "val")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    for item in train_items:
        img_path = os.path.abspath(str(item.get("image", "")).strip())
        lbl_path = os.path.abspath(str(item.get("label", "")).strip())
        if not os.path.isfile(img_path) or not os.path.isfile(lbl_path):
            continue
        name = os.path.basename(img_path)
        stem = os.path.splitext(name)[0]
        shutil.copy2(img_path, os.path.join(train_img_dir, name))
        _copy_label_for_task(lbl_path, os.path.join(train_lbl_dir, f"{stem}.txt"), task)

    for item in val_items:
        img_path = os.path.abspath(str(item.get("image", "")).strip())
        lbl_path = os.path.abspath(str(item.get("label", "")).strip())
        if not os.path.isfile(img_path) or not os.path.isfile(lbl_path):
            continue
        name = os.path.basename(img_path)
        stem = os.path.splitext(name)[0]
        shutil.copy2(img_path, os.path.join(val_img_dir, name))
        _copy_label_for_task(lbl_path, os.path.join(val_lbl_dir, f"{stem}.txt"), task)

    yaml_path = os.path.join(tmp_root, "dataset.yaml")
    yaml_lines = [
        f"path: {tmp_root.replace(os.sep, '/')}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for i, name in enumerate(class_names):
        safe = str(name).replace('"', '\\"')
        yaml_lines.append(f'  {i}: "{safe}"')
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines) + "\n")
    return tmp_root, yaml_path


def _label_file_uses_obb(label_path: str) -> bool:
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for raw in f:
                parts = raw.strip().split()
                if len(parts) >= 9:
                    return True
    except Exception:
        return False
    return False


def _detect_training_task(job: dict[str, Any]) -> str:
    for key in ("train_items", "val_items"):
        for item in list(job.get(key) or []):
            label_path = os.path.abspath(str(item.get("label", "")).strip())
            if label_path and os.path.isfile(label_path) and _label_file_uses_obb(label_path):
                return "obb"
    return "detect"


def _resolve_model_for_task(model_path: str, task: str) -> str:
    candidate = os.path.abspath(str(model_path or "").strip())
    task_name = str(task or "detect").strip().lower()
    if task_name != "obb":
        return candidate
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    bundled_obb = os.path.join(model_dir, "yolo26m-obb.pt")
    base_name = os.path.basename(candidate).lower()
    if base_name.endswith("-obb.pt") or base_name.endswith("-obb.onnx"):
        return candidate
    if os.path.isfile(bundled_obb):
        _log(f"[train] OBB dataset detected. Switching model to bundled OBB weights: {bundled_obb}")
        return os.path.abspath(bundled_obb)
    return candidate


def _is_cuda_runtime_error(text: str) -> bool:
    t = str(text or "").lower()
    keywords = [
        "cuda",
        "cudart",
        "no kernel image",
        "invalid device ordinal",
        "torch_cuda",
        "device-side assert",
    ]
    return any(k in t for k in keywords)


def _resolve_runtime_yolo_cli() -> str:
    py_dir = os.path.dirname(os.path.abspath(sys.executable))
    candidates = [
        os.path.join(py_dir, "yolo.exe"),
        os.path.join(py_dir, "yolo"),
        os.path.join(py_dir, "Scripts", "yolo.exe"),
        os.path.join(py_dir, "Scripts", "yolo"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return shutil.which("yolo") or ""


def _resolve_runtime_device(requested: str) -> str:
    req = str(requested or "").strip().lower()
    if req and req not in {"auto", "gpu"}:
        return requested
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
            _log(f"[train] CUDA arch {sm_tag} is not supported by this torch build. Fallback to CPU.")
            return "cpu"
        return "0"
    except Exception:
        return "cpu"


def _run_cli(cmd: list[str], cwd: str) -> tuple[int, str]:
    cmd_text = " ".join(f'"{part}"' if " " in part else part for part in cmd)
    _log("=" * 60)
    _log(cmd_text)
    _log("=" * 60)
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    lines: list[str] = []
    if proc.stdout is not None:
        buf = ""
        while True:
            ch = proc.stdout.read(1)
            if ch == "":
                if buf:
                    print(buf, flush=True)
                    lines.append(buf + "\n")
                    buf = ""
                break
            if ch in ("\r", "\n"):
                line = buf.rstrip("\r\n")
                if line:
                    print(line, flush=True)
                    lines.append(line + "\n")
                buf = ""
            else:
                buf += ch
    rc = proc.wait()
    return rc, "".join(lines)


def _run_python_fallback(job: dict[str, Any], yaml_path: str) -> int:
    model_path = os.path.abspath(str(job["model_path"]))
    epochs = int(job["epochs"])
    imgsz = int(job["imgsz"])
    batch = int(job["batch"])
    out_dir = os.path.abspath(str(job["out_dir"]))
    run_name = str(job["run_name"])
    device = str(job.get("device", "cpu"))
    task = str(job.get("task", "detect")).strip().lower() or "detect"
    extra_args = [str(x) for x in (job.get("extra_train_args") or [])]
    cmd_text = (
        f"yolo {task} train "
        f'model="{model_path}" '
        f'data="{yaml_path}" '
        f"epochs={epochs} imgsz={imgsz} batch={batch} "
        f'project="{out_dir}" name="{run_name}"'
    )
    _log("=" * 60)
    _log(cmd_text)
    _log("=" * 60)
    from ultralytics import YOLO

    class _Stream(io.TextIOBase):
        def __init__(self):
            self.buf = ""

        def write(self, s):  # type: ignore[override]
            text = str(s or "")
            if not text:
                return 0
            self.buf += text
            while "\n" in self.buf:
                line, self.buf = self.buf.split("\n", 1)
                line = line.rstrip("\r")
                if line.strip():
                    _log(line)
            return len(text)

        def flush(self):  # type: ignore[override]
            if self.buf.strip():
                _log(self.buf.rstrip("\r"))
            self.buf = ""

    model = YOLO(model_path)
    s = _Stream()
    try:
        with contextlib.redirect_stdout(s), contextlib.redirect_stderr(s):
            model.train(
                task=task,
                data=yaml_path,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                project=out_dir,
                name=run_name,
                device=device,
                pretrained=False if "pretrained=False" in extra_args else True,
            )
        s.flush()
        return 0
    except Exception as exc:
        s.flush()
        if str(device).strip().lower() != "cpu" and _is_cuda_runtime_error(str(exc)):
            _log("GPU failed, retrying with CPU ...")
            with contextlib.redirect_stdout(s), contextlib.redirect_stderr(s):
                model.train(
                    task=task,
                    data=yaml_path,
                    epochs=epochs,
                    imgsz=imgsz,
                    batch=batch,
                    project=out_dir,
                    name=run_name,
                    device="cpu",
                    pretrained=False if "pretrained=False" in extra_args else True,
                )
            s.flush()
            return 0
        raise


def run(job_path: str) -> int:
    with open(job_path, "r", encoding="utf-8") as f:
        job = json.load(f)
    out_dir = os.path.abspath(str(job["out_dir"]))
    run_name = str(job["run_name"])
    model_path = os.path.abspath(str(job["model_path"]))
    epochs = int(job["epochs"])
    imgsz = int(job["imgsz"])
    batch = int(job["batch"])
    device = _resolve_runtime_device(str(job.get("device", "auto")))
    task = _detect_training_task(job)
    job["task"] = task
    model_path = _resolve_model_for_task(model_path, task)
    job["model_path"] = model_path
    extra_args = [str(x) for x in (job.get("extra_train_args") or [])]
    cwd = os.path.abspath(str(job.get("cwd", os.getcwd())))
    os.makedirs(out_dir, exist_ok=True)
    _log(f"[train] task={task}")

    tmp_root, yaml_path = _prepare_dataset(job)
    try:
        cli = _resolve_runtime_yolo_cli()
        if cli:
            cmd = [
                cli,
                task,
                "train",
                f"model={model_path}",
                f"data={yaml_path}",
                f"epochs={epochs}",
                f"imgsz={imgsz}",
                f"batch={batch}",
                f"project={out_dir}",
                f"name={run_name}",
                "exist_ok=True",
                "verbose=True",
                f"device={device}",
            ]
            cmd.extend(extra_args)
            rc, out_text = _run_cli(cmd, cwd)
            if rc != 0 and str(device).strip().lower() != "cpu" and _is_cuda_runtime_error(out_text):
                _log("GPU training failed, retrying with CPU ...")
                cmd_cpu: list[str] = []
                for part in cmd:
                    if part.startswith("device="):
                        cmd_cpu.append("device=cpu")
                    else:
                        cmd_cpu.append(part)
                rc, _ = _run_cli(cmd_cpu, cwd)
        else:
            rc = _run_python_fallback(job, yaml_path)
        return int(rc)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def main() -> None:
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass
    p = argparse.ArgumentParser()
    p.add_argument("--job", required=True)
    args = p.parse_args()
    try:
        rc = run(args.job)
    except Exception as exc:
        _log(f"[ERROR] {exc}")
        rc = 1
    raise SystemExit(int(rc))


if __name__ == "__main__":
    main()
