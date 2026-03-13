from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

try:
    from ai_labeller.build_training_runtime import build_runtime
except Exception:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_training_runtime import build_runtime


def _run_capture(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return int(proc.returncode), str(proc.stdout or "")


def _detect_nvidia_info() -> dict[str, Any]:
    candidates = [
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        ["nvidia-smi"],
    ]
    for cmd in candidates:
        try:
            rc, out = _run_capture(cmd)
        except Exception:
            continue
        if rc != 0:
            continue
        lines = [line.strip() for line in out.splitlines() if line.strip()]
        if not lines:
            continue
        info: dict[str, Any] = {"available": True, "raw": out}
        if cmd[0:2] == ["nvidia-smi", "--query-gpu=name,driver_version"]:
            first = lines[0]
            parts = [p.strip() for p in first.split(",")]
            if parts:
                info["gpu_name"] = parts[0]
            if len(parts) >= 2:
                info["driver_version"] = parts[1]
        return info
    return {"available": False}


def _probe_runtime(runtime_dir: Path) -> dict[str, Any]:
    if os.name == "nt":
        runtime_python = runtime_dir / "Scripts" / "python.exe"
    else:
        runtime_python = runtime_dir / "bin" / "python"
    probe_script = runtime_dir / "probe_runtime.py"
    if not runtime_python.is_file() or not probe_script.is_file():
        return {"ok": False, "reason": "runtime missing probe files"}
    rc, out = _run_capture([str(runtime_python), str(probe_script)])
    if rc != 0:
        return {"ok": False, "reason": out.strip() or f"probe failed with code {rc}"}
    try:
        data = json.loads(out)
    except Exception:
        return {"ok": False, "reason": "probe output is not valid json", "raw": out}
    data["ok"] = True
    return data


def _build_attempt(
    output_dir: Path,
    runtime_name: str,
    base_python: str,
    torch_index_url: str,
    force: bool,
) -> tuple[bool, dict[str, Any]]:
    args = argparse.Namespace(
        output=str(output_dir),
        runtime_name=runtime_name,
        python=base_python,
        requirements_file="",
        torch_index_url=torch_index_url,
        extra_package=[],
        no_torch=False,
        force=force,
    )
    try:
        build_runtime(args)
    except Exception as exc:
        return False, {"reason": str(exc)}
    runtime_dir = output_dir / runtime_name
    probe = _probe_runtime(runtime_dir)
    return True, probe


def _build_cpu_attempt(
    output_dir: Path,
    runtime_name: str,
    base_python: str,
    force: bool,
) -> tuple[bool, dict[str, Any]]:
    args = argparse.Namespace(
        output=str(output_dir),
        runtime_name=runtime_name,
        python=base_python,
        requirements_file="",
        torch_index_url="",
        extra_package=[],
        no_torch=False,
        force=force,
    )
    try:
        build_runtime(args)
    except Exception as exc:
        return False, {"reason": str(exc)}
    runtime_dir = output_dir / runtime_name
    probe = _probe_runtime(runtime_dir)
    return True, probe


def auto_build(args: argparse.Namespace) -> int:
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_name = str(args.runtime_name or "training_runtime").strip() or "training_runtime"
    base_python = os.path.abspath(args.python or sys.executable)

    nvidia = _detect_nvidia_info()
    print(json.dumps({"nvidia": nvidia}, indent=2, ensure_ascii=False), flush=True)

    attempts: list[tuple[str, str]] = []
    if bool(nvidia.get("available")):
        attempts.extend(
            [
                ("cu128", "https://download.pytorch.org/whl/cu128"),
                ("cu130", "https://download.pytorch.org/whl/cu130"),
            ]
        )
    attempts.append(("cpu", ""))

    results: list[dict[str, Any]] = []
    for idx, (mode, index_url) in enumerate(attempts):
        print(f"[runtime] trying mode={mode}", flush=True)
        target_dir = output_dir / runtime_name
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        if mode == "cpu":
            ok, probe = _build_cpu_attempt(output_dir, runtime_name, base_python, True)
        else:
            ok, probe = _build_attempt(output_dir, runtime_name, base_python, index_url, True)
        result = {"mode": mode, "ok": ok, "probe": probe}
        results.append(result)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        if not ok:
            continue
        if mode == "cpu":
            print(f"[runtime] success: cpu runtime created at {target_dir}", flush=True)
            return 0
        if probe.get("cuda_available"):
            print(f"[runtime] success: {mode} runtime created at {target_dir}", flush=True)
            return 0
        print(f"[runtime] {mode} built but CUDA is not available in probe, trying next option ...", flush=True)

    print(json.dumps({"results": results}, indent=2, ensure_ascii=False), flush=True)
    return 1


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Automatically build a compatible training runtime.")
    p.add_argument("output", help="Output parent folder.")
    p.add_argument("--runtime-name", default="training_runtime", help="Folder name to create inside output.")
    p.add_argument("--python", default=sys.executable, help="Base Python used to create the runtime.")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        rc = auto_build(args)
    except Exception as exc:
        print(f"[ERROR] {exc}", flush=True)
        raise SystemExit(1)
    raise SystemExit(int(rc))


if __name__ == "__main__":
    main()
