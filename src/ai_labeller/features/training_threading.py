import re
import subprocess
import time
from typing import Any

from ai_labeller.dialogs import messagebox

def stop_training(app) -> None:
    if not getattr(app, "training_running", False):
        return
    app._training_stop_requested = True
    app._append_training_log("[user] stop requested")
    proc = getattr(app, "training_process", None)
    if proc is None:
        return
    try:
        proc.terminate()
    except Exception:
        app.logger.exception("Failed to terminate training process")
    try:
        app.root.after(1200, lambda: force_kill_training_if_alive(app, proc))
    except Exception:
        pass


def force_kill_training_if_alive(app, proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.kill()
            app._append_training_log("[user] training process killed")
    except Exception:
        app.logger.exception("Failed to kill training process")


def append_training_log(app, line: str) -> None:
    log_line = line.rstrip() + "\n"
    lines = getattr(app, "_training_log_lines", None)
    if lines is None:
        app._training_log_lines = [log_line]
    else:
        lines.append(log_line)
        if len(lines) > 2000:
            app._training_log_lines = lines[-2000:]
    if getattr(app, "txt_train_log", None) is not None and app.txt_train_log.winfo_exists():
        app.txt_train_log.insert("end", log_line)
        app.txt_train_log.see("end")


def set_training_status(app, running: bool) -> None:
    if app.lbl_train_status is None or not app.lbl_train_status.winfo_exists():
        return
    status_text = app.LANG_MAP[app.lang].get("train_running", "Running") if running else app.LANG_MAP[app.lang].get("train_idle", "Idle")
    app.lbl_train_status.config(text=f"{app.LANG_MAP[app.lang].get('train_status', 'Status')}: {status_text}")


def set_training_progress(app, current_epoch: int, total_epochs: int) -> None:
    app.training_current_epoch = current_epoch
    app.training_total_epochs = total_epochs
    if app.lbl_train_progress is not None and app.lbl_train_progress.winfo_exists():
        app.lbl_train_progress.config(text=f"{app.LANG_MAP[app.lang].get('train_progress', 'Progress')}: {current_epoch}/{total_epochs}")


def format_eta_seconds(app, seconds_left: float) -> str:
    seconds = max(0, int(seconds_left))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def set_training_eta(app, eta_text: str) -> None:
    if app.lbl_train_eta is not None and app.lbl_train_eta.winfo_exists():
        app.lbl_train_eta.config(text=f"{app.LANG_MAP[app.lang].get('train_eta', 'ETA')}: {eta_text}")


def handle_training_output_line(app, line: str) -> None:
    # enqueue the raw log line
    q = getattr(app, "training_queue", None)
    if q is None:
        return
    q.put(("log", line))
    if getattr(app, "training_total_epochs", 0) <= 0:
        return
    match = re.search(r"(^|\s)(\d{1,4})/(\d{1,4})(\s|$)", line)
    if not match:
        return
    current = int(match.group(2))
    total = int(match.group(3))
    if total != app.training_total_epochs or current <= 0:
        return
    if app.training_start_time is not None:
        elapsed = max(1.0, time.time() - app.training_start_time)
        eta = (elapsed / current) * max(0, total - current)
        q.put(("progress", current, total, format_eta_seconds(app, eta)))
    else:
        q.put(("progress", current, total, "-"))


def run_training_subprocess(app, cmd: list[str], workdir: str) -> None:
    try:
        app.training_process = subprocess.Popen(
            cmd,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=getattr(app, 'WIN_NO_CONSOLE', 0),
        )
        if app.training_process.stdout is not None:
            for line in app.training_process.stdout:
                handle_training_output_line(app, line)
        rc = app.training_process.wait()
        if app._training_stop_requested:
            app.training_queue.put(("stopped",))
        elif rc == 0:
            app.training_queue.put(("done",))
        else:
            app.training_queue.put(("error", f"Process exited with code {rc}"))
    except Exception as exc:
        if app._training_stop_requested:
            app.training_queue.put(("stopped",))
        else:
            app.training_queue.put(("error", str(exc)))
    finally:
        app.training_process = None


def poll_training_queue(app) -> None:
    q = getattr(app, "training_queue", None)
    if q is None:
        return
    keep_polling = getattr(app, "training_running", False) or not q.empty()
    while not q.empty():
        event = q.get_nowait()
        kind = event[0]
        if kind == "log":
            append_training_log(app, event[1])
        elif kind == "progress":
            current, total, eta_text = event[1], event[2], event[3]
            set_training_progress(app, current, total)
            set_training_eta(app, eta_text)
        elif kind == "done":
            app.training_running = False
            set_training_status(app, False)
            set_training_eta(app, "00:00")
            if getattr(app, "training_thread", None) is not None:
                app.training_thread = None
            app._training_stop_requested = False
            output_path = getattr(app, "_last_training_output_path", "")
            try:
                messagebox.showinfo(app.LANG_MAP[app.lang]["title"], app.LANG_MAP[app.lang].get("train_done", "Training finished.\nOutput: {path}").format(path=output_path), parent=app.root)
            except Exception:
                pass
        elif kind == "error":
            app.training_running = False
            set_training_status(app, False)
            if getattr(app, "training_thread", None) is not None:
                app.training_thread = None
            app._training_stop_requested = False
            err = event[1]
            app.logger.error("Training process failed: %s", err)
            try:
                messagebox.showerror(app.LANG_MAP[app.lang]["title"], app.LANG_MAP[app.lang].get("train_failed", "Training failed: {err}").format(err=err), parent=app.root)
            except Exception:
                pass
        elif kind == "stopped":
            app.training_running = False
            set_training_status(app, False)
            set_training_eta(app, "-")
            if getattr(app, "training_thread", None) is not None:
                app.training_thread = None
            append_training_log(app, "[done] training stopped by user")
            try:
                messagebox.showinfo(app.LANG_MAP[app.lang]["title"], "Training stopped.", parent=app.root)
            except Exception:
                pass
    if keep_polling:
        try:
            app.root.after(200, lambda: poll_training_queue(app))
        except Exception:
            pass
