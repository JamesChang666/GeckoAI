from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any


def _log(msg: str) -> None:
    print(str(msg), flush=True)


def _prepare_dataset(job: dict[str, Any]) -> tuple[str, str]:
    class_names = [str(x) for x in (job.get("class_names") or [])]
    if not class_names:
        class_names = ["class0"]
    train_items = list(job.get("train_items") or [])
    val_items = list(job.get("val_items") or [])

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
        shutil.copy2(lbl_path, os.path.join(train_lbl_dir, f"{stem}.txt"))

    for item in val_items:
        img_path = os.path.abspath(str(item.get("image", "")).strip())
        lbl_path = os.path.abspath(str(item.get("label", "")).strip())
        if not os.path.isfile(img_path) or not os.path.isfile(lbl_path):
            continue
        name = os.path.basename(img_path)
        stem = os.path.splitext(name)[0]
        shutil.copy2(img_path, os.path.join(val_img_dir, name))
        shutil.copy2(lbl_path, os.path.join(val_lbl_dir, f"{stem}.txt"))

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
    extra_args = [str(x) for x in (job.get("extra_train_args") or [])]
    cmd_text = (
        "yolo train "
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
    device = str(job.get("device", "cpu"))
    extra_args = [str(x) for x in (job.get("extra_train_args") or [])]
    cwd = os.path.abspath(str(job.get("cwd", os.getcwd())))
    os.makedirs(out_dir, exist_ok=True)

    tmp_root, yaml_path = _prepare_dataset(job)
    try:
        cli = shutil.which("yolo")
        if cli:
            cmd = [
                cli,
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
