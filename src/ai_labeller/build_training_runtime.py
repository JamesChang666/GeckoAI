from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable


def _run(cmd: list[str], *, cwd: str | None = None) -> None:
    cmd_text = " ".join(f'"{part}"' if " " in part else part for part in cmd)
    print("=" * 80, flush=True)
    print(cmd_text, flush=True)
    print("=" * 80, flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _python_in_venv(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _iter_default_packages(include_torch: bool) -> list[str]:
    packages = [
        "pip",
        "setuptools",
        "wheel",
        "numpy",
        "pillow",
        "opencv-python",
        "pyyaml",
        "matplotlib",
        "requests",
        "scipy",
        "psutil",
        "polars",
        "ultralytics-thop",
    ]
    if include_torch:
        packages.extend(["torch", "torchvision", "torchaudio"])
    return packages


def _write_runtime_manifest(
    runtime_dir: Path,
    *,
    base_python: str,
    include_torch: bool,
    torch_index_url: str,
    extra_packages: Iterable[str],
) -> None:
    manifest = {
        "runtime_dir": str(runtime_dir),
        "base_python": os.path.abspath(base_python),
        "include_torch": bool(include_torch),
        "torch_index_url": str(torch_index_url or ""),
        "extra_packages": [str(x) for x in extra_packages],
        "created_with_python": sys.version,
    }
    with open(runtime_dir / "runtime_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def _write_probe_script(runtime_dir: Path) -> None:
    probe = runtime_dir / "probe_runtime.py"
    probe.write_text(
        "import json\n"
        "info = {}\n"
        "try:\n"
        "    import torch\n"
        "    info['torch_version'] = getattr(torch, '__version__', '')\n"
        "    info['cuda_available'] = bool(torch.cuda.is_available())\n"
        "    if torch.cuda.is_available():\n"
        "        info['device_name'] = torch.cuda.get_device_name(0)\n"
        "        info['device_capability'] = list(torch.cuda.get_device_capability(0))\n"
        "        try:\n"
        "            info['arch_list'] = list(torch.cuda.get_arch_list() or [])\n"
        "        except Exception:\n"
        "            info['arch_list'] = []\n"
        "except Exception as exc:\n"
        "    info['torch_error'] = str(exc)\n"
        "try:\n"
        "    import ultralytics\n"
        "    info['ultralytics_version'] = getattr(ultralytics, '__version__', '')\n"
        "except Exception as exc:\n"
        "    info['ultralytics_error'] = str(exc)\n"
        "print(json.dumps(info, indent=2, ensure_ascii=False))\n",
        encoding="utf-8",
    )


def build_runtime(args: argparse.Namespace) -> int:
    output_root = Path(args.output).expanduser().resolve()
    runtime_dir = output_root / args.runtime_name
    if runtime_dir.exists():
        if not args.force:
            raise FileExistsError(f"Target already exists: {runtime_dir}")
        shutil.rmtree(runtime_dir)
    runtime_dir.parent.mkdir(parents=True, exist_ok=True)

    base_python = os.path.abspath(args.python or sys.executable)
    _run([base_python, "-m", "venv", str(runtime_dir)])

    runtime_python = _python_in_venv(runtime_dir)
    if not runtime_python.is_file():
        raise FileNotFoundError(f"Runtime python not found: {runtime_python}")

    _run([str(runtime_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    if args.requirements_file:
        req_path = os.path.abspath(args.requirements_file)
        _run([str(runtime_python), "-m", "pip", "install", "-r", req_path])
    else:
        packages = _iter_default_packages(not args.no_torch)
        if args.torch_index_url and not args.no_torch:
            torch_packages = ["torch", "torchvision", "torchaudio"]
            non_torch_packages = [pkg for pkg in packages if pkg not in torch_packages]
            if non_torch_packages:
                _run([str(runtime_python), "-m", "pip", "install", *non_torch_packages])
            _run(
                [
                    str(runtime_python),
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "--force-reinstall",
                    "--index-url",
                    args.torch_index_url,
                    *torch_packages,
                ]
            )
            _run([str(runtime_python), "-m", "pip", "install", "--no-deps", "ultralytics"])
        else:
            _run([str(runtime_python), "-m", "pip", "install", *packages])
            _run([str(runtime_python), "-m", "pip", "install", "--no-deps", "ultralytics"])
        if args.extra_package:
            _run([str(runtime_python), "-m", "pip", "install", *list(args.extra_package)])

    _write_runtime_manifest(
        runtime_dir,
        base_python=base_python,
        include_torch=not args.no_torch,
        torch_index_url=args.torch_index_url,
        extra_packages=args.extra_package or [],
    )
    _write_probe_script(runtime_dir)

    print(f"Runtime created at: {runtime_dir}", flush=True)
    print(f"Runtime python: {runtime_python}", flush=True)
    print("Probe command:", flush=True)
    print(f'  "{runtime_python}" "{runtime_dir / "probe_runtime.py"}"', flush=True)
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build a standalone training runtime package.")
    p.add_argument("output", help="Output parent folder.")
    p.add_argument("--runtime-name", default="training_runtime", help="Folder name to create inside output.")
    p.add_argument("--python", default=sys.executable, help="Base Python used to create the venv.")
    p.add_argument("--requirements-file", default="", help="Install from a requirements file instead of default package list.")
    p.add_argument(
        "--torch-index-url",
        default="",
        help="Optional torch wheel index URL, for example https://download.pytorch.org/whl/cu128",
    )
    p.add_argument("--extra-package", action="append", default=[], help="Additional pip package to install.")
    p.add_argument("--no-torch", action="store_true", help="Skip installing torch/torchvision/torchaudio.")
    p.add_argument("--force", action="store_true", help="Overwrite target runtime folder if it already exists.")
    return p


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        rc = build_runtime(args)
    except Exception as exc:
        print(f"[ERROR] {exc}", flush=True)
        raise SystemExit(1)
    raise SystemExit(int(rc))


if __name__ == "__main__":
    main()
