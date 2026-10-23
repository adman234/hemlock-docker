"""Hemlock web front end: upload images, get protected images back.

An independent reimplementation of the Nightshade shading method (see research/). One job runs at a time because
each one needs the whole GPU; everyone else waits in the Gradio queue. Uploads live in a throwaway job folder that is
deleted after an hour.
"""
import html
import os
import re
import shutil
import threading
import time
import uuid
import zipfile
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image

DATA = Path(os.environ.get("DATA_DIR", "/data"))
JOBS = DATA / "jobs"
SD_DIR = Path(os.environ.get("HEMLOCK_SD_DIR", str(DATA / "models" / "stable-diffusion-v1-5")))
SD_REPO = os.environ.get("HEMLOCK_SD_REPO", "stable-diffusion-v1-5/stable-diffusion-v1-5")
BLIP_REPO = os.environ.get("HEMLOCK_BLIP_REPO", "Salesforce/blip-image-captioning-base")
CLIP_REPO = os.environ.get("HEMLOCK_CLIP_REPO", "openai/clip-vit-base-patch32")
USE_CONCEPT_MODELS = os.environ.get("HEMLOCK_CONCEPT_MODELS", "1") != "0"
LPIPS_PICKLE = os.environ.get("HEMLOCK_LPIPS_PICKLE") or None  # trusted local file, parity experiments only
MAX_FILES = int(os.environ.get("HEMLOCK_MAX_FILES", "20"))
QUEUE_MAX = int(os.environ.get("HEMLOCK_QUEUE_MAX", "20"))
JOB_TTL = int(os.environ.get("HEMLOCK_JOB_TTL_MINUTES", "60")) * 60
PREVIEW_SIDE = int(os.environ.get("HEMLOCK_PREVIEW_SIDE", "1600"))
PORT = int(os.environ.get("HEMLOCK_PORT", "7860"))
HAS_GPU = torch.cuda.is_available()

from nsre.params import compute_level_to_opt_setting, get_params, intensity_to_level  # noqa: E402

INTENSITY = {"Light": 0, "Balanced": 1, "Strong": 2}
QUALITY = {"Fast": 0, "Better": 1, "Thorough": 2, "Maximum": 3}
QUALITY_HINT = {
    "Fast": "about 20 s per image on a mid-range GPU",
    "Better": "about 30 s",
    "Thorough": "about 1 min",
    "Maximum": "about 1.5 min",
}

_state = {"ready": False, "error": "", "msg": "Starting"}
_engine = None


# --------------------------------------------------------------------------- model setup
def _snapshot(repo, local_dir, allow, ignore=None):
    from huggingface_hub import snapshot_download

    snapshot_download(repo, local_dir=str(local_dir), allow_patterns=allow, ignore_patterns=ignore)


def ensure_models():
    """Download everything into the data volume the first time, then load it onto the GPU."""
    global _engine
    try:
        from nsre.engine import Engine

        if not (SD_DIR / "unet").exists():
            _state["msg"] = "Downloading Stable Diffusion 1.5 (about 2.5 GB, first start only)"
            print(_state["msg"], flush=True)
            _snapshot(SD_REPO, SD_DIR, [
                "model_index.json", "scheduler/*.json", "tokenizer/*", "unet/config.json", "vae/config.json",
                "text_encoder/config.json", "unet/diffusion_pytorch_model.fp16.safetensors",
                "vae/diffusion_pytorch_model.fp16.safetensors", "text_encoder/model.fp16.safetensors",
            ])
        blip_dir, clip_dir = DATA / "models" / "blip", DATA / "models" / "clip"
        if USE_CONCEPT_MODELS:
            for repo, d, label in ((BLIP_REPO, blip_dir, "BLIP"), (CLIP_REPO, clip_dir, "CLIP")):
                if not (d / "config.json").exists():
                    _state["msg"] = f"Downloading {label} (tag detection)"
                    print(_state["msg"], flush=True)
                    _snapshot(repo, d, ["*.json", "*.txt", "model.safetensors"])
                    if not any(d.glob("*.safetensors")):  # older repos only ship .bin
                        _snapshot(repo, d, ["*.json", "*.txt", "pytorch_model.bin"])
            import nltk

            _state["msg"] = "Fetching the word tagger"
            for pkg in ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"):
                nltk.download(pkg, quiet=True)
        _state["msg"] = "Loading models onto the " + ("GPU" if HAS_GPU else "CPU")
        pool_file = DATA / "concepts.txt" if (DATA / "concepts.txt").exists() else Path(__file__).with_name("concepts.txt")
        pool = [ln.strip() for ln in pool_file.read_text(encoding="utf8").splitlines() if ln.strip()]
        mapping = None
        if (DATA / "mapping.json").exists():
            import json

            mapping = json.loads((DATA / "mapping.json").read_text(encoding="utf8"))
        _engine = Engine(
            str(SD_DIR), blip_dir=str(blip_dir), clip_dir=str(clip_dir), use_concept_models=USE_CONCEPT_MODELS,
            lpips_pickle=LPIPS_PICKLE, pool=pool, mapping=mapping,
        )
        _engine.load()
        _state.update(ready=True, msg="Ready")
        print("Models ready on", _engine.device, flush=True)
    except Exception as e:  # shown in the UI so a bad network does not look like a hang
        _state.update(ready=False, error=str(e), msg="Setup failed")
        print(f"Setup failed: {e!r}", flush=True)


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


# --------------------------------------------------------------------------- job
def _stage_files(files, inp):
    """Copy uploads into the job folder. PNG and JPEG are kept as they are; anything else becomes a PNG."""
    names, used, staged, skipped = {}, set(), [], []
    for i, f in enumerate(files):
        src = Path(f if isinstance(f, str) else f.name)
        orig = getattr(f, "orig_name", None) or src.name
        stem = safe_stem(orig)
        base, n = stem, 2
        while stem in used:
            stem, n = f"{base}_{n}", n + 1
        used.add(stem)
        ext = src.suffix.lower()
        if ext in (".jpg", ".jpeg"):
            dest = inp / f"{i:04d}.jpg"
            shutil.copyfile(src, dest)
        else:
            dest = inp / f"{i:04d}.png"
            try:
                Image.open(src).convert("RGB").save(dest)
            except Exception:
                skipped.append((orig, "not an image I can read"))
                continue
        names[dest] = (stem, dest.suffix)
        staged.append(dest)
    return staged, names, skipped


def _metrics(orig_img, out_path):
    out = Image.open(out_path).convert("RGB")
    if out.size != orig_img.size:
        return None
    d = np.array(out).astype(np.float64) - np.array(orig_img.convert("RGB")).astype(np.float64)
    mse = float((d ** 2).mean())
    return {"max": float(np.abs(d).max()), "psnr": float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)}


def make_preview(orig_img, out_path, final, key):
    """Screen-sized JPEG copies for the comparison view (phones should not load 5000 px PNGs)."""
    mist = Image.open(out_path).convert("RGB")
    orig = orig_img.convert("RGB")
    scale = min(1.0, PREVIEW_SIDE / max(mist.size))
    if scale < 1.0:
        size = (round(mist.width * scale), round(mist.height * scale))
        mist, orig = mist.resize(size, Image.LANCZOS), orig.resize(size, Image.LANCZOS)
    mist.save(final / f"{key}_p.jpg", quality=92)
    orig.save(final / f"{key}_o.jpg", quality=92)
    return str(final / f"{key}_p.jpg"), str(final / f"{key}_o.jpg"), mist.width, mist.height


def run_job(files, intensity, quality, tag, target, seed, progress=gr.Progress()):
    from nsre import imageio

    if not _state["ready"]:
        raise gr.Error(_state["error"] or _state["msg"] + ". Try again in a few minutes.")
    if not files:
        raise gr.Error("Add at least one image.")
    if len(files) > MAX_FILES:
        raise gr.Error(f"At most {MAX_FILES} images per run.")

    sweep_old_jobs()
    job = JOBS / uuid.uuid4().hex
    inp, final = job / "in", job / "result"
    inp.mkdir(parents=True)
    final.mkdir()
    staged, names, failures = [], {}, []
    staged, names, failures = _stage_files(files, inp)
    if not staged:
        shutil.rmtree(job, ignore_errors=True)
        raise gr.Error("None of those files could be read as images.")

    level = intensity_to_level(INTENSITY[intensity])
    params = get_params(level, compute_level_to_opt_setting(QUALITY[quality]))
    tag = (tag or "").strip() if len(staged) == 1 else ""
    target = (target or "").strip() or None
    seed = int(seed) if seed not in (None, "") else None
    n = len(staged)

    items = []
    for idx, path in enumerate(staged):
        stem, ext = names[path]
        label = f"Image {idx + 1} of {n}"
        progress(idx / n, desc=f"{label}: starting")

        def stage(s, _l=label, _i=idx):
            base = {"Reading the image": 0.0, "Making the target image": 0.04, "Crafting the protection": 0.25, "Saving": 0.98}.get(s, 0.0)
            progress((_i + base) / n, desc=f"{_l}: {s.lower()}")

        last = {"p": -1}

        def prog(pct, _l=label, _i=idx, _last=last):
            if int(pct) == _last["p"]:
                return  # one update per percent keeps the page responsive
            _last["p"] = int(pct)
            progress((_i + 0.25 + 0.73 * pct / 100.0) / n, desc=f"{_l}: crafting, {pct:.0f}%")

        try:
            res = _engine.shade(
                str(path), str(job / "out"), params, tag=tag or None, target=target, seed=seed,
                on_stage=stage, on_progress=prog, reset_exif_orientation=True, write_json=False,
            )
        except Exception as e:  # keep going so one bad file does not lose the rest
            print(f"{stem}: {e!r}", flush=True)
            failures.append((stem, str(e)))
            continue
        dest = final / f"{stem}_protected{ext}"
        shutil.move(res.output, dest)
        orig_img = imageio.load_img(str(path))
        m = _metrics(orig_img, dest)
        pm, po, w, h = make_preview(orig_img, dest, final, f"{idx:04d}")
        items.append({"name": stem, "full": str(dest), "mist": pm, "orig": po, "w": w, "h": h,
                      "tag": res.concept, "target": res.target, "seconds": res.seconds, "m": m,
                      "download": dest.name})
    if not items:
        shutil.rmtree(job, ignore_errors=True)
        raise gr.Error("Nothing could be processed. " + "; ".join(f"{a}: {b}" for a, b in failures)[:600])

    alt = "\n".join(f"{it['download']}: {it['tag']}" for it in items)
    header = ("Add the tag for each image to its ALT text (image description) when you post it online. "
              "That helps the protection take effect.")
    (final / "hemlock-tags.txt").write_text(header + "\n\n" + alt + "\n", encoding="utf8")
    zpath = final / "protected_images.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for it in items:
            z.write(it["full"], Path(it["full"]).name)
        z.write(final / "hemlock-tags.txt", "hemlock-tags.txt")
    shutil.rmtree(inp, ignore_errors=True)
    shutil.rmtree(job / "out", ignore_errors=True)
    note = ""
    if failures:
        note = "<div class='pill bad'>Skipped: " + html.escape("; ".join(f"{a} ({b})" for a, b in failures)[:300]) + "</div>"
    return note + compare_html(items), str(zpath), alt, gr.update(visible=True)


# --------------------------------------------------------------------------- comparison view
# Inline handlers on purpose: Gradio sets HTML component content with innerHTML, which never runs <script> tags.
# Hold handlers do nothing in slider mode: otherwise the stage would capture the pointer away from the slider.
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
# navigator.clipboard only exists on https or localhost, so fall back to a hidden textarea on a plain LAN address
_COPY = (
    "var t=this.dataset.text,b=this;function done(){var o=b.textContent;b.textContent='Copied';setTimeout(function(){b.textContent=o},1200)}"
    "function fb(){var a=document.createElement('textarea');a.value=t;a.style.position='fixed';a.style.opacity='0';"
    "document.body.appendChild(a);a.select();try{document.execCommand('copy');done()}catch(e){}document.body.removeChild(a)}"
    "if(navigator.clipboard&&window.isSecureContext){navigator.clipboard.writeText(t).then(done,fb)}else{fb()}"
)


def compare_html(items):
    e = html.escape
    cards = []
    for it in items:
        m = it["m"]
        stats = ""
        if m:
            psnr = "inf" if m["psnr"] == float("inf") else f"{m['psnr']:.1f}"
            stats = f"<span class='stat'>largest change {m['max']:.0f}/255</span><span class='stat'>PSNR {psnr} dB</span>"
        cards.append(f"""
<div class="cmp" data-mode="hold" style="--ar:{it['w']} / {it['h']}">
  <div class="cmp-head"><span class="cmp-name">{e(it['name'])}</span>
    <a class="cmp-dl" href="file={e(it['full'])}" download="{e(it['download'])}">Download</a></div>
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
  <div class="chips"><span class="chip">Tag: <b>{e(it['tag'])}</b>
      <button type="button" class="copy" data-text="{e(it['tag'])}" onclick="{_COPY}">Copy</button></span>
    <span class="chip">Pushes towards: <b>{e(it['target'])}</b></span>{stats}
    <span class="stat">{it['seconds']:.0f} s</span></div>
</div>""")
    return '<div class="cmp-list">' + "".join(cards) + "</div>"


def status_text():
    if _state["ready"]:
        dev = torch.cuda.get_device_name(0) if HAS_GPU else "CPU only (very slow)"
        return f"<div class='pill ok'>Ready on {html.escape(dev)}</div>"
    if _state["error"]:
        return f"<div class='pill bad'>{html.escape(_state['error'][:300])}</div>"
    return f"<div class='pill wait'>{html.escape(_state['msg'])}...</div>"


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
.cmp-dl,.cmp-mode,.copy{padding:8px 14px;min-height:40px;border-radius:10px;border:1px solid var(--border-color-primary,#8886);
  background:var(--button-secondary-background-fill,#8882);color:inherit;font-size:.9rem;text-decoration:none;cursor:pointer;
  display:inline-flex;align-items:center}
.copy{min-height:32px;padding:3px 10px;margin-left:8px;font-size:.8rem}
.stage{position:relative;margin:10px 0;border-radius:10px;overflow:hidden;aspect-ratio:var(--ar);width:100%;max-height:75vh;
  background:#0003;touch-action:pan-y;user-select:none;-webkit-user-select:none;-webkit-touch-callout:none;cursor:pointer;outline:none}
.stage:focus-visible{box-shadow:0 0 0 3px #1f9d5588}
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
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.chip b{margin-left:.3em}
.chip,.stat{display:inline-flex;align-items:center;padding:3px 10px;border-radius:999px;font-size:.8rem;background:#8882}
.stat{opacity:.8}
@media (max-width:520px){#hero h1{font-size:1.6rem}.cmp-name{max-width:50%}}
"""


def build():
    with gr.Blocks(
        title="Hemlock", analytics_enabled=False, css=CSS,
        theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="slate"),
    ) as demo:
        gr.HTML(
            "<div id='hero'><h1>Hemlock</h1>"
            "<p>Adds faint, targeted changes to your artwork so AI models that scrape it learn the wrong thing. "
            "An independent open-source take on the Nightshade method.</p>"
            "<p>Upload one or many images, wait a little per image, then press and hold a result to compare it with your "
            "original. Images are deleted from the server after an hour.</p></div>"
        )
        status = gr.HTML(status_text())
        files = gr.File(label=f"Your images (up to {MAX_FILES})", file_count="multiple", file_types=["image"])
        intensity = gr.Radio(
            list(INTENSITY), value="Balanced", label="Strength",
            info="How much each pixel may change. Light is nearly invisible, Balanced shows faint grain when zoomed in, Strong can show texture on flat areas. Strong also takes about twice as long.",
        )
        quality = gr.Radio(
            list(QUALITY), value="Fast", label="Render quality",
            info="How long it optimises. Fast about 20 s per image, Better 30 s, Thorough 1 min, Maximum 1.5 min on a mid-range GPU. Slower options use more passes and test more random crops, so the effect holds up better after cropping or resizing.",
        )
        with gr.Accordion("More options", open=False):
            tag = gr.Textbox(
                label="Poison tag (one image only)", value="",
                info="A word describing the image, e.g. 'dog'. Leave empty to detect it automatically. Add the tag to the image's ALT text when you post it.",
            )
            target = gr.Textbox(
                label="Push towards this concept (optional)", value="",
                info="Normally chosen automatically. Enter a word like 'car' to force it.",
            )
            seed = gr.Number(label="Seed (optional)", value=None, precision=0, info="Same seed and settings give the same result.")
        go = gr.Button("Protect images", variant="primary", size="lg")
        gr.Markdown(
            "Hemlock raises the cost of training on your work, it is not a guarantee, and its effect has not been "
            "independently measured. Keep your originals. Not affiliated with the Nightshade team."
        )
        result = gr.HTML(label="Result")
        with gr.Group(visible=False) as dl_group:
            zip_file = gr.File(label="Download everything (zip)")
            alt_box = gr.Textbox(label="ALT text tags (copy these when you post)", lines=4, show_copy_button=True, interactive=False)
        go.click(run_job, [files, intensity, quality, tag, target, seed], [result, zip_file, alt_box, dl_group])
        demo.load(status_text, None, status)
    return demo


if __name__ == "__main__":
    JOBS.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("HEMLOCK_SKIP_MODELS"):
        threading.Thread(target=ensure_models, daemon=True).start()
    user, pw = os.environ.get("HEMLOCK_USERNAME", ""), os.environ.get("HEMLOCK_PASSWORD", "")
    auth = (user or "hemlock", pw) if pw else None
    build().queue(default_concurrency_limit=1, max_size=QUEUE_MAX).launch(
        server_name="0.0.0.0", server_port=PORT, auth=auth, allowed_paths=[str(JOBS)], show_api=False,
    )

