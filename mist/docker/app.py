"""Upload/download web front end for Mist v2.

The upstream mist-webui.py asks for server-side folder paths. This one takes uploads, runs attacks/mist.py on
them in a throwaway job folder, and hands back the misted images (and a zip). One job runs at a time because
each one needs the whole GPU; everyone else waits in the Gradio queue.
"""
import html
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
PREVIEW_SIDE = int(os.environ.get("MIST_PREVIEW_SIDE", "1600"))
PORT = int(os.environ.get("MIST_PORT", "7860"))
HAS_GPU = torch.cuda.is_available()

MODES = {
    "Mode 1 (textural, default)": "lunet",
    "Mode 2 (semantic + textural)": "fused",
    "Mode 3 (anti-DreamBooth)": "anti-db",
}

STRENGTHS = {"Light": 6, "Balanced": 12, "Strong": 18}

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
        "--pgd_eps", str(STRENGTHS.get(strength, 12) / 255.0),
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
    items = []
    for r in results:
        key = r.name.split("_noise_")[1][:-4]
        name = names.get(key, key)
        dest = final / f"{name}_misted.png"
        shutil.move(str(r), dest)
        items.append(make_preview(inp / f"{key}.png", dest, final, key, name))
    zpath = final / "misted_images.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for it in items:
            z.write(it["full"], Path(it["full"]).name)
    shutil.rmtree(inp, ignore_errors=True)
    shutil.rmtree(out, ignore_errors=True)
    return compare_html(items), str(zpath), gr.update(visible=True)


def make_preview(orig_path, misted_path, final, key, name):
    """Screen-sized JPEG copies of both images for the comparison view (phones should not load 4000 px PNGs).
    When Mist did not keep the original size its output is a centre crop, so the original gets the same crop."""
    mist = Image.open(misted_path).convert("RGB")
    orig = Image.open(orig_path).convert("RGB")
    if orig.size != mist.size:
        orig = ImageOps.fit(orig, mist.size, Image.LANCZOS)
    scale = min(1.0, PREVIEW_SIDE / max(mist.size))
    if scale < 1.0:
        size = (round(mist.width * scale), round(mist.height * scale))
        mist, orig = mist.resize(size, Image.LANCZOS), orig.resize(size, Image.LANCZOS)
    mist.save(final / f"{key}_p.jpg", quality=92)
    orig.save(final / f"{key}_o.jpg", quality=92)
    return {
        "name": name,
        "full": str(misted_path),
        "mist": str(final / f"{key}_p.jpg"),
        "orig": str(final / f"{key}_o.jpg"),
        "w": mist.width,
        "h": mist.height,
    }


# Inline handlers on purpose: Gradio sets HTML component content with innerHTML, which never runs <script> tags.
# Hold handlers do nothing in slider mode: otherwise the stage would capture the pointer away from the slider
_GUARD = "if(this.closest('.cmp').dataset.mode!=='hold')return;"
_HOLD_ON = _GUARD + "this.classList.add('holding');try{this.setPointerCapture(event.pointerId)}catch(e){}"
_HOLD_OFF = "this.classList.remove('holding')"
_KEY_ON = _GUARD + "if(event.key===' '||event.key==='Enter'){event.preventDefault();this.classList.add('holding')}"
_KEY_OFF = "if(event.key===' '||event.key==='Enter'){this.classList.remove('holding')}"
_MODE = (
    "var c=this.closest('.cmp');var s=c.dataset.mode==='slide';c.dataset.mode=s?'hold':'slide';"
    "c.querySelector('.stage').classList.remove('holding');"
    "this.textContent=s?'Switch to slider':'Switch to hold';"
    "c.querySelector('.hint').textContent=s?'Press and hold the image to see the original':'Drag the slider: original on the left'"
)


def compare_html(items):
    e = html.escape
    cards = []
    for it in items:
        cards.append(f"""
<div class="cmp" data-mode="hold" style="--ar:{it['w']} / {it['h']}">
  <div class="cmp-head"><span class="cmp-name">{e(it['name'])}</span>
    <a class="cmp-dl" href="file={e(it['full'])}" download="{e(it['name'])}_misted.png">Download</a></div>
  <div class="stage" tabindex="0" role="img" aria-label="Protected image. Press and hold, or hold Space, to see the original."
       onpointerdown="{_HOLD_ON}" onpointerup="{_HOLD_OFF}" onpointercancel="{_HOLD_OFF}"
       onlostpointercapture="{_HOLD_OFF}" onkeydown="{_KEY_ON}" onkeyup="{_KEY_OFF}" onblur="{_HOLD_OFF}"
       oncontextmenu="return false">
    <img class="img-m" src="file={e(it['mist'])}" alt="" draggable="false">
    <img class="img-o" src="file={e(it['orig'])}" alt="" draggable="false">
    <span class="tag tag-m">Protected</span><span class="tag tag-o">Original</span>
    <input class="range" type="range" min="0" max="100" value="50" aria-label="Compare slider"
           oninput="this.closest('.cmp').style.setProperty('--pos',this.value+'%')">
    <div class="divider"></div>
  </div>
  <div class="cmp-foot"><span class="hint">Press and hold the image to see the original</span>
    <button type="button" class="cmp-mode" onclick="{_MODE}">Switch to slider</button></div>
</div>""")
    return '<div class="cmp-list">' + "".join(cards) + "</div>"


def status_text():
    if _model_state["ready"]:
        dev = torch.cuda.get_device_name(0) if HAS_GPU else "CPU only (slow)"
        return f"<div class='pill ok'>Ready on {html.escape(dev)}</div>"
    if _model_state["error"]:
        return f"<div class='pill bad'>{html.escape(_model_state['error'])}</div>"
    return f"<div class='pill wait'>{html.escape(_model_state['msg'])}...</div>"


CSS = """
.gradio-container{max-width:760px !important;margin:0 auto !important;padding-left:12px !important;padding-right:12px !important}
.gradio-container input,.gradio-container textarea{font-size:16px !important}  /* stops iOS zooming on focus */
button.lg,button.primary{min-height:52px;font-size:1.05rem}
#hero h1{margin:.2em 0 .1em;font-size:1.9rem}
#hero p{margin:.2em 0;opacity:.8}
.pill{display:inline-block;padding:4px 12px;border-radius:999px;font-size:.85rem;margin:6px 0}
.pill.ok{background:#1f9d5533;color:#1f9d55}.pill.wait{background:#d9a40033;color:#b58900}.pill.bad{background:#d6336c33;color:#d6336c}
.cmp-list{display:flex;flex-direction:column;gap:20px}
.cmp{border:1px solid var(--border-color-primary,#8884);border-radius:14px;padding:10px;background:var(--background-fill-secondary,#8881)}
.cmp-head,.cmp-foot{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap}
.cmp-name{font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:60%}
.cmp-dl,.cmp-mode{padding:8px 14px;min-height:40px;border-radius:10px;border:1px solid var(--border-color-primary,#8886);
  background:var(--button-secondary-background-fill,#8882);color:inherit;font-size:.9rem;text-decoration:none;cursor:pointer;
  display:inline-flex;align-items:center}
.stage{position:relative;margin:10px 0;border-radius:10px;overflow:hidden;aspect-ratio:var(--ar);width:100%;max-height:75vh;
  background:#0003;touch-action:pan-y;user-select:none;-webkit-user-select:none;-webkit-touch-callout:none;cursor:pointer;outline:none}
.stage:focus-visible{box-shadow:0 0 0 3px #7c5cff88}
.stage img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;pointer-events:none;-webkit-user-drag:none}
.img-o{opacity:0}
.cmp[data-mode=hold] .stage.holding .img-o{opacity:1}
.tag{position:absolute;left:8px;top:8px;padding:3px 10px;border-radius:999px;font-size:.78rem;color:#fff;background:#000a;pointer-events:none}
.tag-o{opacity:0}
.cmp[data-mode=hold] .stage.holding .tag-o{opacity:1}.cmp[data-mode=hold] .stage.holding .tag-m{opacity:0}
.range,.divider{display:none}
.cmp[data-mode=slide] .img-o{opacity:1;clip-path:inset(0 calc(100% - var(--pos,50%)) 0 0)}
.cmp[data-mode=slide] .tag-o{opacity:1;left:8px}.cmp[data-mode=slide] .tag-m{left:auto;right:8px}
.cmp[data-mode=slide] .stage{cursor:ew-resize}
.cmp[data-mode=slide] .range{display:block;position:absolute;inset:0;width:100%;height:100%;opacity:0;margin:0;cursor:ew-resize;z-index:3}
.cmp[data-mode=slide] .divider{display:block;position:absolute;top:0;bottom:0;left:var(--pos,50%);width:3px;margin-left:-1.5px;background:#fff;
  box-shadow:0 0 6px #0008;pointer-events:none;z-index:2}
.hint{opacity:.7;font-size:.85rem}
@media (max-width:520px){#hero h1{font-size:1.6rem}.cmp-name{max-width:50%}}
"""


def build():
    with gr.Blocks(
        title="Mist", analytics_enabled=False, css=CSS,
        theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate"),
    ) as demo:
        gr.HTML(
            "<div id='hero'><h1>Mist</h1>"
            "<p>Adds a faint noise to your artwork so AI models that try to copy its style produce scrambled results.</p>"
            "<p>Upload, wait a few minutes per image, then press and hold the result to compare it with your original. "
            "Images are deleted from the server after an hour.</p></div>"
        )
        status = gr.HTML(status_text())
        files = gr.File(
            label=f"Your images (up to {MAX_FILES})",
            file_count="multiple",
            file_types=["image"],
        )
        strength = gr.Radio(
            list(STRENGTHS), value="Balanced", label="Strength",
            info="Stronger protects more, but the noise becomes easier to see.",
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
        go = gr.Button("Protect images", variant="primary", size="lg")
        gr.Markdown("Mist makes copying harder, it is not a guarantee. Keep your originals.")
        result = gr.HTML(label="Result")
        with gr.Group(visible=False) as dl_group:
            zip_file = gr.File(label="Download everything (zip)")
        go.click(
            run_job,
            [files, strength, mode, epochs, keep_size, prompt],
            [result, zip_file, dl_group],
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
