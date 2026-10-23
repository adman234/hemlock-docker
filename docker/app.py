"""Upload/download web front end for Mist v2.

The upstream mist-webui.py asks for server-side folder paths. This one takes uploads, runs attacks/mist.py on
them in a throwaway job folder, and hands back the misted images (and a zip). One job runs at a time because
each one needs the whole GPU; everyone else waits in the Gradio queue.
"""
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from pathlib import Path

import gradio as gr
import torch
from PIL import Image, ImageOps

DATA = Path(os.environ.get("DATA_DIR", "/data"))
JOBS = DATA / "jobs"
CLASS_DIR = DATA / "class"
MODEL_DIR = Path(os.environ.get("MIST_MODEL_DIR", str(DATA / "models" / "stable-diffusion-v1-5")))
MODEL_REPO = os.environ.get("MIST_HF_REPO", "stable-diffusion-v1-5/stable-diffusion-v1-5")
MAX_FILES = int(os.environ.get("MIST_MAX_FILES", "4"))
MAX_SIDE = int(os.environ.get("MIST_MAX_SIDE", "4096"))
QUEUE_MAX = int(os.environ.get("MIST_QUEUE_MAX", "20"))
NUM_CLASS_IMAGES = os.environ.get("MIST_CLASS_IMAGES", "20")
JOB_TTL = int(os.environ.get("MIST_JOB_TTL_MINUTES", "60")) * 60
PORT = int(os.environ.get("MIST_PORT", "7860"))
HAS_GPU = torch.cuda.is_available()

MODES = {
    "Mode 1 (textural, default)": "lunet",
    "Mode 2 (semantic + textural)": "fused",
    "Mode 3 (anti-DreamBooth)": "anti-db",
}

_model_state = {"ready": False, "error": "", "msg": "Starting"}


def ensure_model():
    """Download Stable Diffusion 1.5 into the data volume the first time."""
    try:
        if (MODEL_DIR / "unet" / "diffusion_pytorch_model.safetensors").exists() and (
            MODEL_DIR / "vae" / "diffusion_pytorch_model.safetensors"
        ).exists():
            _model_state.update(ready=True, msg="Ready")
            return
        _model_state["msg"] = "Downloading the Stable Diffusion 1.5 model (about 5 GB, first start only)"
        print(_model_state["msg"], flush=True)
        from huggingface_hub import snapshot_download

        snapshot_download(
            MODEL_REPO,
            local_dir=str(MODEL_DIR),
            allow_patterns=[
                "model_index.json",
                "*/*.json",
                "*/*.txt",
                "tokenizer/*",
                "unet/diffusion_pytorch_model.safetensors",
                "vae/diffusion_pytorch_model.safetensors",
                "text_encoder/model.safetensors",
            ],
        )
        _model_state.update(ready=True, msg="Ready")
        print("Model ready", flush=True)
    except Exception as e:  # shown in the UI so a bad network does not look like a hang
        _model_state.update(ready=False, error=str(e), msg="Model download failed")
        print(f"Model download failed: {e}", flush=True)


def sweep_old_jobs():
    JOBS.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - JOB_TTL
    for p in JOBS.iterdir():
        try:
            if p.stat().st_mtime < cutoff:
                shutil.rmtree(p, ignore_errors=True)
        except OSError:
            pass


def safe_stem(name):
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._") or "image"
    return stem[:80]


def run_job(files, strength, mode, epochs, keep_size, prompt, progress=gr.Progress()):
    if not _model_state["ready"]:
        raise gr.Error(_model_state["error"] or _model_state["msg"] + ". Try again in a few minutes.")
    if not files:
        raise gr.Error("Add at least one image.")
    if len(files) > MAX_FILES:
        raise gr.Error(f"At most {MAX_FILES} images per run.")

    sweep_old_jobs()
    job = JOBS / uuid.uuid4().hex
    inp, out = job / "in", job / "out"
    inp.mkdir(parents=True)
    out.mkdir()
    CLASS_DIR.mkdir(parents=True, exist_ok=True)

    # Re-encode every upload as a clean RGB PNG. Mist matches outputs to inputs by directory order, so the
    # uploads get plain numbered names here and are mapped back to the original names afterwards.
    names = {}
    used = set()
    for i, f in enumerate(files):
        src = Path(f if isinstance(f, str) else f.name)
        try:
            im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
        except Exception:
            shutil.rmtree(job, ignore_errors=True)
            raise gr.Error(f"{Path(src).name} is not an image I can read.")
        if max(im.size) > MAX_SIDE:
            im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        im.save(inp / f"{i:04d}.png")
        stem = safe_stem(getattr(f, "orig_name", None) or src.name)
        base, n = stem, 2
        while stem in used:
            stem, n = f"{base}_{n}", n + 1
        used.add(stem)
        names[f"{i:04d}"] = stem

    cmd = [
        sys.executable, "attacks/mist.py",
        "--mode", MODES[mode],
        "--pgd_eps", str(float(strength) / 255.0),
        "--instance_data_dir", str(inp),
        "--output_dir", str(out),
        "--pretrained_model_name_or_path", str(MODEL_DIR),
        "--class_data_dir", str(CLASS_DIR),
        "--instance_prompt", prompt or "a picture",
        "--class_prompt", "a picture",
        "--num_class_images", NUM_CLASS_IMAGES,
        "--max_train_steps", str(int(epochs)),
        "--mixed_precision", "bf16",
        "--report_to", "tensorboard",
        "--logging_dir", str(job / "logs"),
    ]
    if HAS_GPU:
        cmd.append("--cuda")
    if keep_size:
        cmd.append("--resize")

    progress(0, desc="Starting")
    total = int(epochs)
    done = 0
    tail = []
    proc = subprocess.Popen(
        cmd, cwd="/app", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    for line in proc.stdout:
        tail = (tail + [line.rstrip()])[-25:]
        if "Generating class images" in line:
            progress(0, desc="First run only: making comparison images")
        m = re.search(r"Epoch (\d+) ends", line)
        if m:
            done = int(m.group(1)) + 1
            progress(done / total, desc=f"Pass {done} of {total}")
    code = proc.wait()

    results = sorted(out.glob("*_noise_*.png"))
    if code != 0 or not results:
        log = "\n".join(tail)
        print(log, flush=True)
        shutil.rmtree(job, ignore_errors=True)
        raise gr.Error("Mist failed. Last output:\n" + log[-800:])

    final = job / "result"
    final.mkdir()
    outputs = []
    for r in results:
        key = r.name.split("_noise_")[1][:-4]
        dest = final / f"{names.get(key, key)}_misted.png"
        shutil.move(str(r), dest)
        outputs.append(str(dest))
    zpath = final / "misted_images.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for o in outputs:
            z.write(o, Path(o).name)
    shutil.rmtree(inp, ignore_errors=True)
    shutil.rmtree(out, ignore_errors=True)
    return outputs, [*outputs, str(zpath)]


def status_text():
    if _model_state["ready"]:
        dev = torch.cuda.get_device_name(0) if HAS_GPU else "CPU only (slow)"
        return f"Ready on {dev}."
    return _model_state["error"] or _model_state["msg"] + "..."


def build():
    with gr.Blocks(title="Mist", analytics_enabled=False) as demo:
        gr.Markdown(
            "# Mist\n"
            "Adds a faint noise to your artwork so AI models that try to copy its style or content produce "
            "scrambled results. Upload, wait a few minutes per image, download. Your images are deleted from "
            "the server after an hour.\n\n"
            "Mist raises the cost of copying an image, it is not a guarantee. Keep your originals."
        )
        status = gr.Markdown(status_text())
        with gr.Row():
            with gr.Column():
                files = gr.File(
                    label=f"Images (up to {MAX_FILES})",
                    file_count="multiple",
                    file_types=["image"],
                )
                strength = gr.Slider(
                    1, 32, step=1, value=12, label="Strength",
                    info="Higher protects more but the noise gets easier to see.",
                )
                keep_size = gr.Checkbox(True, label="Keep the original image size")
                with gr.Accordion("More options", open=False):
                    mode = gr.Radio(list(MODES), value=list(MODES)[0], label="Mode")
                    epochs = gr.Slider(
                        1, 10, step=1, value=5, label="Passes",
                        info="More passes take longer and protect somewhat better.",
                    )
                    prompt = gr.Textbox(
                        label="Short description of the images", value="a picture",
                        info="Optional, e.g. 'an anime girl' or 'a landscape painting'.",
                    )
                go = gr.Button("Protect images", variant="primary")
            with gr.Column():
                gallery = gr.Gallery(label="Result", columns=2, height=420)
                downloads = gr.File(label="Downloads", file_count="multiple")
        go.click(
            run_job,
            [files, strength, mode, epochs, keep_size, prompt],
            [gallery, downloads],
        )
        demo.load(status_text, None, status)
    return demo


if __name__ == "__main__":
    JOBS.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("MIST_SKIP_MODEL"):
        threading.Thread(target=ensure_model, daemon=True).start()
    user, pw = os.environ.get("MIST_USERNAME", ""), os.environ.get("MIST_PASSWORD", "")
    auth = (user or "mist", pw) if pw else None
    build().queue(default_concurrency_limit=1, max_size=QUEUE_MAX).launch(
        server_name="0.0.0.0",
        server_port=PORT,
        auth=auth,
        allowed_paths=[str(JOBS)],
        show_api=False,
    )
