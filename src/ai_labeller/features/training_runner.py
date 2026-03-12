import datetime
import os
import threading
import time

from ai_labeller.constants import LANG_MAP
from ai_labeller.dialogs import filedialog, messagebox


def start_training_from_labels(app, has_yolo: bool) -> None:
    if not has_yolo:
        messagebox.showwarning("YOLO Not Available", "Please install ultralytics first.")
        return
    if app.training_running:
        messagebox.showinfo(
            LANG_MAP[app.lang]["title"],
            LANG_MAP[app.lang].get("train_already_running", "Training is already running."),
            parent=app.root,
        )
        return
    if not app.project_root:
        messagebox.showwarning(
            LANG_MAP[app.lang]["title"],
            LANG_MAP[app.lang].get("train_no_project", "No dataset loaded."),
            parent=app.root,
        )
        return
    if app.image_files and app.img_pil:
        app.save_current()

    split_roots = [s for s in ("train", "val", "test") if os.path.isdir(f"{app.project_root}/images/{s}")]
    if split_roots:
        train_split = app.current_split if app.current_split in split_roots else split_roots[0]
        train_candidates = app._list_split_labeled_images_for_root(app.project_root, train_split)
        if not train_candidates:
            messagebox.showwarning(
                LANG_MAP[app.lang]["title"],
                LANG_MAP[app.lang].get("train_no_labels", "No labeled images found for training."),
                parent=app.root,
            )
            return
        val_split = "val" if "val" in split_roots else train_split
        val_candidates = app._list_split_labeled_images_for_root(app.project_root, val_split)
    else:
        train_split = "train"
        val_split = "train"
        train_candidates = app._list_flat_labeled_images_for_root(app.project_root)
        val_candidates = []
        if not train_candidates:
            messagebox.showwarning(
                LANG_MAP[app.lang]["title"],
                LANG_MAP[app.lang].get("train_no_labels", "No labeled images found for training."),
                parent=app.root,
            )
            return

    max_idx = len(train_candidates)
    settings = app._prompt_training_runtime_settings(max_idx=max_idx)
    if settings is None:
        return
    start_idx, end_idx, epochs, imgsz, batch_size, weight_mode, custom_weight_from_settings = settings

    out_dir = filedialog.askdirectory(
        parent=app.root,
        title=LANG_MAP[app.lang].get("select_train_output", "Select Training Output Folder"),
    )
    if not out_dir:
        return
    out_dir = out_dir.replace("\\", "/")

    selected_train = train_candidates[start_idx - 1:end_idx]
    if not selected_train:
        messagebox.showwarning(
            LANG_MAP[app.lang]["title"],
            LANG_MAP[app.lang].get("train_no_labels", "No labeled images found for training."),
            parent=app.root,
        )
        return
    if val_candidates:
        selected_val = val_candidates
    else:
        val_count = max(1, int(len(selected_train) * 0.2))
        if len(selected_train) <= 1:
            selected_val = selected_train[:]
        else:
            selected_val = selected_train[-val_count:]
            selected_train = selected_train[:-val_count]
            if not selected_train:
                selected_train = selected_val[:]

    try:
        choice = weight_mode
        custom_path = custom_weight_from_settings
        extra_train_args: list[str] = []
        if choice == "official":
            model_path = app._resolve_official_model_path()
        elif choice == "custom":
            if not custom_path:
                return
            model_path = app._resolve_custom_model_path(custom_path)
        else:
            model_path = app._resolve_official_model_path()
            extra_train_args.append("pretrained=False")
        model_path = os.path.abspath(model_path)
        app.yolo_path.set(model_path)
        app._register_model_path(model_path)

        os.makedirs(out_dir, exist_ok=True)
        dataset_yaml = app._write_training_dataset_files(
            out_dir=out_dir,
            train_images=selected_train,
            val_images=selected_val,
            train_split=train_split,
            val_split=val_split,
            range_start=start_idx,
            range_end=end_idx,
        )
        run_name = f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        yolo_cli = app._resolve_yolo_cli()
        cmd = [
            yolo_cli,
            "train",
            f"model={model_path}",
            f"data={dataset_yaml}",
            f"epochs={epochs}",
            f"imgsz={imgsz}",
            f"batch={batch_size}",
            f"project={out_dir}",
            f"name={run_name}",
            "exist_ok=True",
        ]
        train_device = app._auto_runtime_device(allow_forced_cpu=False)
        cmd.append(f"device={train_device}")
        cmd.extend(extra_train_args)
        command_text = " ".join(f'"{part}"' if " " in part else part for part in cmd)
        app.train_command_var.set(command_text)
        app.open_training_monitor_popup()
        app._append_training_log("=" * 60)
        app._append_training_log(command_text)
        app._append_training_log("=" * 60)
        app._set_training_status(True)
        app._set_training_progress(0, epochs)
        app._set_training_eta("-")
        app.training_running = True
        app.training_start_time = time.time()
        app.training_total_epochs = epochs
        app.training_current_epoch = 0
        app._last_training_output_path = f"{out_dir}/{run_name}"
        app.training_thread = threading.Thread(
            target=app._run_training_subprocess,
            args=(cmd, app.project_root),
            daemon=True,
        )
        app.training_thread.start()
        app._poll_training_queue()
    except Exception as exc:
        app.logger.exception("Training from labels failed")
        messagebox.showerror(
            LANG_MAP[app.lang]["title"],
            LANG_MAP[app.lang].get("train_failed", "Training failed: {err}").format(err=exc),
            parent=app.root,
        )
