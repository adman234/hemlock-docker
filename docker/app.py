"""Hemlock web app: Hemlock (a Nightshade-style poison) and Mist v2 (anti style-mimicry noise) behind one UI.

One job runs at a time because each one needs the whole GPU; everyone else waits in the queue. Models are loaded
only while they are needed and unloaded after an idle timeout (see manager.py). Uploads live in a throwaway job
folder that is deleted after a while (HEMLOCK_JOB_TTL_MINUTES).
torch is imported lazily so the idle process stays small.
"""
import collections
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
import numpy as np
from PIL import Image, ImageOps

import viewer
from manager import Cancelled, ModelManager

DATA = Path(os.environ.get("DATA_DIR", "/data"))
JOBS = DATA / "jobs"
CLASS_DIR = DATA / "class"
SD_DIR = Path(os.environ.get("HEMLOCK_SD_DIR") or os.environ.get("MIST_MODEL_DIR") or str(DATA / "models" / "stable-diffusion-v1-5"))
SD_REPO = os.environ.get("HEMLOCK_SD_REPO") or os.environ.get("MIST_HF_REPO") or "stable-diffusion-v1-5/stable-diffusion-v1-5"
BLIP_REPO = os.environ.get("HEMLOCK_BLIP_REPO", "Salesforce/blip-image-captioning-base")
CLIP_REPO = os.environ.get("HEMLOCK_CLIP_REPO", "openai/clip-vit-base-patch32")
MIST_DIR = Path(os.environ.get("MIST_DIR", "/app/mist"))
USE_CONCEPT_MODELS = os.environ.get("HEMLOCK_CONCEPT_MODELS", "1") != "0"
LPIPS_PICKLE = os.environ.get("HEMLOCK_LPIPS_PICKLE") or None  # trusted local file, parity experiments only


def _env_float(name, alias, default):
    return float(os.environ.get(name) or os.environ.get(alias or "-", "") or default)


MAX_FILES_H = int(os.environ.get("HEMLOCK_MAX_FILES", "20"))
MAX_FILES_M = int(os.environ.get("MIST_MAX_FILES", "20"))
MAX_SIDE_M = int(os.environ.get("MIST_MAX_SIDE", "4096"))
QUEUE_MAX = int(os.environ.get("HEMLOCK_QUEUE_MAX", "20"))
JOB_TTL = int(_env_float("HEMLOCK_JOB_TTL_MINUTES", "MIST_JOB_TTL_MINUTES", 60) * 60)
IDLE_SECONDS = _env_float("HEMLOCK_IDLE_UNLOAD_MINUTES", None, 10) * 60  # 0 = unload after every job, -1 = never
PREVIEW_SIDE = int(os.environ.get("HEMLOCK_PREVIEW_SIDE", "1600"))
NUM_CLASS_IMAGES = os.environ.get("MIST_CLASS_IMAGES", "20")
PORT = int(os.environ.get("HEMLOCK_PORT", "7860"))

METHOD_H = "Hemlock (poison, Nightshade-style)"
METHOD_M = "Mist v2 (anti style-copying)"
H_STRENGTH = {"Light": 0, "Balanced": 1, "Strong": 2}
H_QUALITY = {"Fast": 0, "Better": 1, "Thorough": 2, "Maximum": 3}
M_MODES = {"Mode 1 (textural, default)": "lunet", "Mode 2 (semantic + textural)": "fused", "Mode 3 (anti-DreamBooth)": "anti-db"}
M_STRENGTH = {"Light": 6, "Balanced": 12, "Strong": 18}



def _pick(name, options, fallback):
    """An env var that must be one of `options` (case-insensitive); anything else falls back with a note in the log."""
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return fallback
    for opt in options:
        if str(opt).lower() == raw.lower():
            return opt
    print(f"{name}={raw!r} is not one of {list(options)}, using {fallback!r}", flush=True)
    return fallback


DEF_METHOD = METHOD_M if (os.environ.get("HEMLOCK_DEFAULT_METHOD") or "").strip().lower() == "mist" else METHOD_H
DEF_H_STRENGTH = _pick("HEMLOCK_DEFAULT_STRENGTH", H_STRENGTH, "Balanced")
DEF_H_QUALITY = _pick("HEMLOCK_DEFAULT_QUALITY", H_QUALITY, "Better")
DEF_M_STRENGTH = _pick("MIST_DEFAULT_STRENGTH", M_STRENGTH, "Balanced")
try:
    DEF_M_PASSES = min(10, max(1, int(os.environ.get("MIST_DEFAULT_PASSES") or 5)))
except ValueError:
    DEF_M_PASSES = 5

# Time estimates. Starting guesses are RTX 4070 SUPER seconds per image times HEMLOCK_SPEED_FACTOR (4.5 is about a GTX 1070,
# use 1 for a 4070-class card). After a few runs the server uses the times it measured itself.
try:
    SPEED = float(os.environ.get("HEMLOCK_SPEED_FACTOR") or 4.5)
except ValueError:
    SPEED = 4.5
H_BASE = {"Fast": 20, "Better": 30, "Thorough": 60, "Maximum": 90}
H_STRENGTH_MULT = {"Light": 1.0, "Balanced": 1.0, "Strong": 2.0}
M_BASE = 240  # per image at 5 passes


def fmt_est(sec):
    """'45 s', '1.5 min', '12 min', '2 h'."""
    sec = max(0, float(sec))
    if sec < 60:
        return f"{max(5, int(round(sec / 5.0)) * 5)} s"
    m = sec / 60.0
    if m < 10:
        v = int(m * 2 + 0.5) / 2.0
        return f"{v:g} min"
    if m < 120:
        return f"{int(round(m))} min"
    return f"{m / 60.0:.1f} h".replace(".0 h", " h")


def h_sec(quality):
    return H_BASE[quality] * SPEED


ABOUT_HTML = f"""
<div class='about'>
<p><b>What PyTorch is.</b> PyTorch is a machine-learning library: software for building and running neural networks, the kind of model that
AI image generators and image recognisers are made of.</p>
<p><b>What the algorithm does.</b> Both methods run an optimisation. They look at your image through a neural network, then nudge the pixels by tiny
amounts, thousands of times, checking each time how the network's view of the image changes.</p>
<ul>
<li><b>Hemlock</b> (a Nightshade-style method) steers the image towards a different concept, for example "dog" towards "car". The idea is that an AI
trained on many such images, all labelled honestly, slowly learns the wrong association.</li>
<li><b>Mist</b> adds noise aimed at tools that copy an artist's style, so that attempts to imitate your work come out scrambled.</li>
</ul>
<p>These methods are deterrents, not guarantees. Your uploads are deleted after about {max(1, JOB_TTL // 60)} minutes. Please do not upload anything
you are not allowed to share, or anything illegal.</p>
</div>
"""
INFO_HTML = f"""
<div class='about'>
<p><b>Methods</b></p>
<ul>
<li><b>Hemlock</b> (poison, Nightshade-style): roughly {fmt_est(h_sec('Fast'))} to {fmt_est(h_sec('Maximum'))} per image (estimate). Pushes your image towards a different concept so an AI that <i>trains</i> on many such
images learns the wrong thing. Gives each image a tag to put in its ALT text.</li>
<li><b>Mist v2</b> (anti style-copying): slow, about {fmt_est(M_BASE * SPEED)} per image (estimate), and it may run out of memory on a small (8 GB) card. Adds noise aimed at tools that imitate an artist's style, so imitations come out scrambled.</li>
</ul>
<p><b>Models used</b></p>
<ul>
<li><b>Stable Diffusion 1.5</b> (about 5 GB, downloaded once): the image model both methods target. Hemlock also uses it to draw a "target" picture of the other concept.</li>
<li><b>BLIP</b> and <b>CLIP</b> (Hemlock only, small): BLIP describes your image, CLIP picks the main word (the tag) and a different concept to push towards.</li>
<li><b>LPIPS</b> (Hemlock only): a perceptual yardstick that keeps the changes hard to see.</li>
</ul>
<p><b>Hemlock settings</b></p>
<ul>
<li><b>Strength</b>: how much each pixel may change. Light is nearly invisible (about 6 out of 255), Balanced shows faint grain when zoomed in (about 12), Strong can show texture on flat areas (about 20) and takes about twice as long. Time per image (estimate): Fast about {fmt_est(h_sec('Fast'))}, Better {fmt_est(h_sec('Better'))}, Thorough {fmt_est(h_sec('Thorough'))}, Maximum {fmt_est(h_sec('Maximum'))}.</li>
<li><b>Render quality</b>: how long it optimises. Fast 100 passes, Better 150, Thorough 300, Maximum 500, with more random crops tested per pass as it goes up, so the effect should hold up better after cropping or resizing.</li>
<li><b>Poison tag</b> (one image only), <b>Push towards</b>, <b>ALT text template</b>, <b>metadata</b>, <b>seed</b>: in More options. Leave them empty for automatic choices. A seed makes a run repeatable.</li>
</ul>
<p><b>Mist settings</b></p>
<ul>
<li><b>Strength</b>: how far pixels may move (6, 12 or 18 out of 255).</li>
<li><b>Mode</b>: 1 targets texture (default), 2 adds meaning as well as texture, 3 is aimed at personalisation tools like DreamBooth.</li>
<li><b>Passes</b>: more passes take longer and protect somewhat better. <b>Keep the original size</b> returns the full image instead of a crop.</li>
</ul>
<p>Good starting point: Hemlock with Light and Fast on a slow card, Balanced and Better on a fast one. Times are rough and depend on the GPU and the queue. Neither method is a guarantee.</p>
</div>
"""

MANAGER = ModelManager(IDLE_SECONDS)
_setup = {"ready": False, "error": "", "msg": "Starting"}


# --------------------------------------------------------------------------- GPU info without importing torch
def gpu_name():
    try:
        import pynvml

        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))
        return n.decode() if isinstance(n, bytes) else n
    except Exception:
        return None


GPU = gpu_name()


# --------------------------------------------------------------------------- one-time downloads (disk only, nothing is loaded)
def _snapshot(repo, local_dir, allow):
    from huggingface_hub import snapshot_download

    snapshot_download(repo, local_dir=str(local_dir), allow_patterns=allow)


def ensure_downloads():
    try:
        if not (SD_DIR / "unet" / "diffusion_pytorch_model.safetensors").exists():
            _setup["msg"] = "Downloading Stable Diffusion 1.5 (about 5 GB, first start only)"
            print(_setup["msg"], flush=True)
            _snapshot(SD_REPO, SD_DIR, [
                "model_index.json", "*/*.json", "*/*.txt", "tokenizer/*",
                "unet/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.safetensors",
                "text_encoder/model.safetensors",
            ])
        for old in SD_DIR.glob("*/*.fp16.safetensors"):  # leftovers of the earlier Hemlock-only layout
            old.unlink(missing_ok=True)
        if USE_CONCEPT_MODELS:
            for repo, name, label in ((BLIP_REPO, "blip", "BLIP"), (CLIP_REPO, "clip", "CLIP")):
                d = DATA / "models" / name
                if not (d / "config.json").exists():
                    _setup["msg"] = f"Downloading {label} (tag detection)"
                    print(_setup["msg"], flush=True)
                    _snapshot(repo, d, ["*.json", "*.txt", "model.safetensors"])
                    if not any(d.glob("*.safetensors")):  # older repos only ship .bin
                        _snapshot(repo, d, ["*.json", "*.txt", "pytorch_model.bin"])
            import nltk

            _setup["msg"] = "Fetching the word tagger"
            for pkg in ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"):
                nltk.download(pkg, quiet=True)
            _setup["msg"] = "Fetching the perceptual-loss network"
            # caches torchvision's VGG16 weights under TORCH_HOME. A subprocess, so this process never imports torch
            subprocess.run([sys.executable, "-c", "import lpips; lpips.LPIPS(net='vgg')"], check=True, capture_output=True)
        _setup.update(ready=True, msg="Ready")
        print("Downloads ready", flush=True)
    except Exception as e:  # shown in the UI so a bad network does not look like a hang
        _setup.update(ready=False, error=str(e), msg="Setup failed")
        print(f"Setup failed: {e!r}", flush=True)


def make_engine():
    """The Hemlock engine runs in a worker process (see hemlock_worker.py) that is stopped when idle."""
    from worker_client import WorkerEngine

    pool_file = DATA / "concepts.txt" if (DATA / "concepts.txt").exists() else Path(__file__).with_name("concepts.txt")
    pool = [ln.strip() for ln in pool_file.read_text(encoding="utf8").splitlines() if ln.strip()]
    mapping = None
    if (DATA / "mapping.json").exists():
        import json

        mapping = json.loads((DATA / "mapping.json").read_text(encoding="utf8"))
    return WorkerEngine({
        "sd_dir": str(SD_DIR), "blip_dir": str(DATA / "models" / "blip"), "clip_dir": str(DATA / "models" / "clip"),
        "use_concept_models": USE_CONCEPT_MODELS, "lpips_pickle": LPIPS_PICKLE, "pool": pool, "mapping": mapping,
    })

# --------------------------------------------------------------------------- jobs
class JobState:
    def __init__(self):
        self.frac, self.message, self.sub = 0.0, "Starting", ""
        self.cancel = threading.Event()
        self.result, self.error = None, None

    def check(self):
        if self.cancel.is_set():
            raise Cancelled()




class Job:
    """One submitted run. Lives on the server, so a page reload or a closed tab does not lose it."""

    def __init__(self, method, count, files, o):
        self.id = uuid.uuid4().hex
        self.dir = JOBS / self.id
        self.method, self.count, self.files, self.o = method, count, files, o
        self.st = JobState()
        self.state = "queued"  # queued, running, done, error, stopped
        self.est = est_job_seconds(method, o, count)
        self.started = None


TIMINGS = {}  # "hemlock|Light|Fast" -> seconds per image measured on this server (kept in /data/timings.json)
TIMINGS_FILE = DATA / "timings.json"


def timing_key(method, o):
    if method == METHOD_H:
        return f"hemlock|{o['h_strength']}|{o['h_quality']}"
    return f"mist|{o['m_strength']}|{o['m_mode']}|{int(o['m_passes'])}"


def est_job_seconds(method, o, count):
    per = TIMINGS.get(timing_key(method, o))
    if per is None:
        if method == METHOD_H:
            per = h_sec(o["h_quality"]) * H_STRENGTH_MULT[o["h_strength"]]
        else:
            per = M_BASE * SPEED * int(o["m_passes"]) / 5.0
    return per * count


def learn_timing(job, seconds, n):
    """Remember how long this setting really takes here; used for the queue estimates."""
    if n < 1 or seconds <= 0:
        return
    key = timing_key(job.method, job.o)
    per = seconds / n
    old = TIMINGS.get(key)
    TIMINGS[key] = per if old is None else 0.6 * per + 0.4 * old
    try:
        import json

        TIMINGS_FILE.write_text(json.dumps(TIMINGS), encoding="utf8")
    except OSError:
        pass


def load_timings():
    try:
        import json

        TIMINGS.update({k: float(v) for k, v in json.loads(TIMINGS_FILE.read_text(encoding="utf8")).items()})
    except Exception:
        pass


REG = {}  # job id -> Job (the id is unguessable, it is the only thing a browser needs to remember)
QUEUE = collections.deque()
QCOND = threading.Condition()


def sweep_old_jobs():
    JOBS.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - JOB_TTL
    with QCOND:
        active = {j.id for j in REG.values() if j.state in ("queued", "running")}
    for p in JOBS.iterdir():
        try:
            if p.name not in active and p.stat().st_mtime < cutoff:
                shutil.rmtree(p, ignore_errors=True)
        except OSError:
            pass
    with QCOND:
        for jid in [k for k, j in REG.items() if j.state not in ("queued", "running") and not j.dir.exists()]:
            REG.pop(jid, None)


def _sweeper():
    while True:
        time.sleep(300)
        try:
            sweep_old_jobs()
        except Exception as e:
            print(f"sweep failed: {e!r}", flush=True)


def safe_stem(name):
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._") or "image"
    return stem[:80]


def _file_path(f):
    if isinstance(f, Path):
        return f
    return Path(f if isinstance(f, str) else f.name)


def _unique_stems(files):
    used, out = set(), []
    for f in files:
        orig = getattr(f, "orig_name", None) or _file_path(f).name
        stem = safe_stem(orig)
        base, n = stem, 2
        while stem in used:
            stem, n = f"{base}_{n}", n + 1
        used.add(stem)
        out.append((orig, stem))
    return out


def _metrics(orig_img, out_path):
    out = Image.open(out_path).convert("RGB")
    if out.size != orig_img.size:
        return None
    d = np.array(out).astype(np.float64) - np.array(orig_img.convert("RGB")).astype(np.float64)
    mse = float((d ** 2).mean())
    return {"max": float(np.abs(d).max()), "psnr": float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)}


def finalize_item(key, stem, ext, in_path, out_path, final, extra):
    """Move the result next to the full-resolution original, make screen-sized previews, return the card data."""
    from nsre import imageio

    out_path, in_path = Path(out_path), Path(in_path)
    dest = final / f"{stem}_protected{ext}"
    shutil.move(str(out_path), dest)
    orig = imageio.load_img(str(in_path))  # oriented RGB, same pixels the method saw
    prot = Image.open(dest).convert("RGB")
    if orig.size != prot.size:  # Mist without "keep size" returns a centre crop: crop the original the same way
        orig = ImageOps.fit(orig, prot.size, Image.LANCZOS)
    try:
        orientation = Image.open(in_path).getexif().get(0x0112, 1)
    except Exception:
        orientation = 1
    if orientation == 1 and Image.open(in_path).size == prot.size:
        orig_full = final / f"{stem}_original{in_path.suffix}"
        shutil.copyfile(in_path, orig_full)  # byte-exact original
    else:
        orig_full = final / f"{stem}_original.png"
        orig.save(orig_full)
    m = _metrics(orig, dest)
    scale = min(1.0, PREVIEW_SIDE / max(prot.size))
    p, o = prot, orig
    if scale < 1.0:
        size = (round(prot.width * scale), round(prot.height * scale))
        p, o = prot.resize(size, Image.LANCZOS), orig.resize(size, Image.LANCZOS)
    p.save(final / f"{key}_p.jpg", quality=92)
    o.save(final / f"{key}_o.jpg", quality=92)
    item = {"name": stem, "full": str(dest), "orig_full": str(orig_full), "mist": str(final / f"{key}_p.jpg"),
            "orig": str(final / f"{key}_o.jpg"), "w": p.width, "h": p.height, "fw": prot.width, "fh": prot.height,
            "download": dest.name, "m": m}
    item.update(extra)
    return item


def _alt(template, tag):
    try:
        return template.format(tag=tag)
    except Exception:
        return tag


def hemlock_job(st: JobState, job: Path, files, o):
    from nsre.params import compute_level_to_opt_setting, get_params, intensity_to_level

    inp, final = job / "in", job / "result"
    stems = _unique_stems(files)
    staged, skipped = [], []
    for i, (f, (orig, stem)) in enumerate(zip(files, stems)):
        src = _file_path(f)
        if src.suffix.lower() in (".jpg", ".jpeg"):
            dest = inp / f"{i:04d}.jpg"
            shutil.copyfile(src, dest)
        else:
            dest = inp / f"{i:04d}.png"
            try:
                Image.open(src).convert("RGB").save(dest)
            except Exception:
                skipped.append((orig, "not an image I can read"))
                continue
        staged.append((dest, stem))
    if not staged:
        raise RuntimeError("None of those files could be read as images.")
    params = get_params(intensity_to_level(H_STRENGTH[o["h_strength"]]), compute_level_to_opt_setting(H_QUALITY[o["h_quality"]]))
    tag = o["tag"].strip() if len(staged) == 1 else ""
    n, items, failures = len(staged), [], list(skipped)
    base = {"Reading the image": 0.0, "Making the target image": 0.04, "Crafting the protection": 0.25, "Saving": 0.98}

    def loading(s):
        st.message = s + "... (first use, or after being idle)"

    with MANAGER.use("hemlock", make_engine, on_stage=loading) as eng:
        for idx, (path, stem) in enumerate(staged):
            st.check()
            last = {"p": -1}

            def on_stage(s, _i=idx):
                st.check()
                st.message, st.frac = f"Image {_i + 1} of {n}: {s.lower()}", (_i + base.get(s, 0.0)) / n

            def on_prog(pct, _i=idx, _l=last):
                st.check()
                if int(pct) != _l["p"]:
                    _l["p"] = int(pct)
                    st.frac = (_i + 0.25 + 0.73 * pct / 100.0) / n
                    st.message = f"Image {_i + 1} of {n}: crafting, {pct:.0f}%"

            try:
                res = eng.shade(
                    str(path), str(job / "out"), params, tag=tag or None, target=o["target"].strip() or None, seed=o["seed"],
                    on_stage=on_stage, on_progress=on_prog, reset_exif_orientation=True, write_json=False,
                    description=o["template"] if o["write_meta"] else None,
                )
            except Cancelled:
                raise
            except Exception as e:  # keep going so one bad file does not lose the rest
                print(f"{stem}: {e!r}", flush=True)
                failures.append((stem, str(e)))
                continue
            items.append(finalize_item(f"{idx:04d}", stem, path.suffix if path.suffix == ".jpg" else ".png", path, res.output, final,
                                       {"tag": res.concept, "target": res.target, "seconds": res.seconds, "alt": _alt(o["template"], res.concept)}))
    return items, failures


def mist_job(st: JobState, job: Path, files, o):
    inp, out, final = job / "in", job / "out", job / "result"
    CLASS_DIR.mkdir(parents=True, exist_ok=True)
    stems = _unique_stems(files)
    names, skipped = {}, []
    k = 0
    for f, (orig, stem) in zip(files, stems):
        try:
            im = ImageOps.exif_transpose(Image.open(_file_path(f))).convert("RGB")
        except Exception:
            skipped.append((orig, "not an image I can read"))
            continue
        if max(im.size) > MAX_SIDE_M:
            im.thumbnail((MAX_SIDE_M, MAX_SIDE_M), Image.LANCZOS)
        # Mist matches outputs to inputs by directory order, so uploads get plain numbered names
        im.save(inp / f"{k:04d}.png")
        names[f"{k:04d}"] = stem
        k += 1
    if not names:
        raise RuntimeError("None of those files could be read as images.")
    total = int(o["m_passes"])
    cmd = [
        sys.executable, "attacks/mist.py", "--mode", M_MODES[o["m_mode"]], "--pgd_eps", str(M_STRENGTH[o["m_strength"]] / 255.0),
        "--instance_data_dir", str(inp), "--output_dir", str(out), "--pretrained_model_name_or_path", str(SD_DIR),
        "--class_data_dir", str(CLASS_DIR), "--instance_prompt", o["m_prompt"] or "a picture", "--class_prompt", "a picture",
        "--num_class_images", NUM_CLASS_IMAGES, "--max_train_steps", str(total), "--mixed_precision", "bf16",
        "--report_to", "tensorboard", "--logging_dir", str(job / "logs"),
    ]
    if GPU:
        cmd.append("--cuda")
    if o["m_keep"]:
        cmd.append("--resize")
    tail = []
    with MANAGER.use("mist"):  # unloads Hemlock's models first so they do not hold VRAM
        st.message = "Mist: loading the model"
        proc = subprocess.Popen(cmd, cwd=str(MIST_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                env=dict(os.environ, PYTHONUNBUFFERED="1"))
        try:
            for line in proc.stdout:
                tail = (tail + [line.rstrip()])[-25:]
                if st.cancel.is_set():
                    proc.terminate()
                    raise Cancelled()
                if "Generating class images" in line:
                    st.message, st.sub = "First run only: making comparison images (a few minutes)", ""
                m = re.search(r"Epoch (\d+) ends", line)
                if m:
                    done = int(m.group(1)) + 1
                    st.frac, st.message = done / total, f"Pass {done} of {total}"
            code = proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
    results = sorted(out.glob("*_noise_*.png"))
    if code != 0 or not results:
        log = "\n".join(tail)
        print(log, flush=True)
        raise RuntimeError("Mist failed. Last output:\n" + log[-800:])
    items = []
    for r in results:
        key = r.name.split("_noise_")[1][:-4]
        items.append(finalize_item(key, names.get(key, key), ".png", inp / f"{key}.png", r, final, {"seconds": None}))
    return items, list(skipped)


def _worker(st: JobState, method, files, o, job: Path):
    try:
        for sub in ("in", "out", "result"):
            (job / sub).mkdir(parents=True, exist_ok=True)
        sweep_old_jobs()
        items, failures = (hemlock_job if method == METHOD_H else mist_job)(st, job, files, o)
        if not items:
            raise RuntimeError("Nothing could be processed. " + "; ".join(f"{a}: {b}" for a, b in failures)[:600])
        final = job / "result"
        zpath = final / "protected_images.zip"
        alt_lines = [f"{it['download']}: {it['alt']}" for it in items if it.get("alt")]
        with zipfile.ZipFile(zpath, "w") as z:
            for it in items:
                z.write(it["full"], Path(it["full"]).name)
            if alt_lines:
                header = ("Add the tag for each image to its ALT text (image description) when you post it online. "
                          "That helps the protection take effect.")
                (final / "hemlock-tags.txt").write_text(header + "\n\n" + "\n".join(alt_lines) + "\n", encoding="utf8")
                z.write(final / "hemlock-tags.txt", "hemlock-tags.txt")
        for sub in ("in", "out", "upload"):
            shutil.rmtree(job / sub, ignore_errors=True)
        st.result = {"items": items, "failures": failures, "zip": str(zpath), "alt": "\n".join(alt_lines)}
    except Cancelled:
        st.error = "Stopped."
        shutil.rmtree(job, ignore_errors=True)
    except Exception as e:  # noqa: BLE001
        print(f"job failed: {e!r}", flush=True)
        st.error = str(e)
        shutil.rmtree(job, ignore_errors=True)


# --------------------------------------------------------------------------- UI plumbing
def status_text():
    snap = MANAGER.snapshot()
    parts = []
    if _setup["ready"]:
        parts.append(viewer.pill("ok", ("Ready on " + GPU) if GPU else "Ready, CPU only (very slow)"))
    elif _setup["error"]:
        parts.append(viewer.pill("bad", _setup["error"][:300]))
    else:
        parts.append(viewer.pill("wait", _setup["msg"] + "..."))
    if _setup["ready"]:
        if snap["busy"]:
            parts.append(viewer.pill("wait", "Working"))
        elif snap["kind"]:
            label = "Hemlock" if snap["kind"] == "hemlock" else "Mist"
            if snap["idle_left"] is not None:
                parts.append(viewer.pill("idle", f"{label} loaded, unloads if idle for {max(1, round(snap['idle_left'] / 60))} more min"))
            else:
                parts.append(viewer.pill("idle", f"{label} loaded"))
        else:
            parts.append(viewer.pill("idle", "Models unloaded, they load on first use"))
    return "".join(parts)


def _runner():
    """The single worker: takes queued jobs one at a time (each needs the whole GPU)."""
    while True:
        with QCOND:
            while not QUEUE:
                QCOND.wait()
            job = QUEUE.popleft()
            job.state = "running"
            job.started = time.time()
        try:
            _worker(job.st, job.method, job.files, job.o, job.dir)
        except Exception as e:  # noqa: BLE001
            job.st.error = str(e)
        with QCOND:
            job.state = "stopped" if job.st.error == "Stopped." else ("error" if job.st.error else "done")
            job.files = None
        if job.state == "done" and job.st.result and job.started:
            learn_timing(job, time.time() - job.started, len(job.st.result["items"]))
        try:
            os.utime(job.dir)  # the delete timer counts from when the run finished
        except OSError:
            pass


def _left(job: Job) -> float:
    """Seconds left on a running job. Starts from the estimate and leans more on its own measured pace as progress grows."""
    if job.state != "running" or not job.started:
        return 0.0
    by_est = max(0.0, job.est * (1 - job.st.frac))
    frac, elapsed = job.st.frac, time.time() - job.started
    if frac < 0.05:  # too early to read a pace (model loading counts as elapsed time)
        return by_est
    by_pace = max(0.0, elapsed / frac - elapsed)
    w = min(1.0, (frac - 0.05) / 0.15)  # fully trust the measured pace from 20% on
    return w * by_pace + (1 - w) * by_est


def job_view(job: Job):
    """What the browser shows for one run: {state, html, ahead, pct}."""
    head = f"<div class='job-head'><b>{html.escape(job.method.split(' (')[0])}</b>, {job.count} image{'s' if job.count != 1 else ''}</div>"
    clear = "<div class='job-actions'><button type='button' class='job-clear'>Clear</button></div>"
    ahead, st, eta = 0, job.st, ""
    if job.state == "queued":
        with QCOND:
            line = list(QUEUE)
            ahead = line.index(job) if job in line else 0
            wait = sum(j.est for j in line[:ahead]) + sum(_left(j) for j in REG.values() if j.state == "running")
        eta = f"about {fmt_est(wait + job.est)}"
        sub = (f"{ahead} run{'s' if ahead != 1 else ''} ahead of yours. " if ahead else "Next in line. ") + \
              f"Estimated: starts in {fmt_est(wait) if wait >= 5 else 'a moment'}, done in {fmt_est(wait + job.est)}. " \
              "It starts by itself, and you can close this page and come back to this run later."
        body = viewer.progress_bar(0, "Submitted, waiting for another run to finish", sub)
    elif job.state == "running":
        left = _left(job)
        sub = (st.sub + " " if st.sub else "") + (f"About {fmt_est(left)} left." if left >= 5 else "Almost done.")
        eta = f"about {fmt_est(left)} left" if left >= 5 else ""
        body = viewer.progress_bar(st.frac, st.message, sub)
    elif job.state == "stopped":
        body = viewer.pill("wait", "Stopped.") + clear
    elif job.state == "error":
        body = viewer.pill("bad", st.error or "Failed") + clear
    else:
        r, note = st.result, ""
        if r["failures"]:
            note = viewer.pill("bad", "Skipped: " + "; ".join(f"{a} ({b})" for a, b in r["failures"])[:300])
        body = note + viewer.cards(r["items"]) + (viewer.zip_button(r["zip"]) if len(r["items"]) > 1 else "")
    return {"state": job.state, "html": head + body, "ahead": ahead, "eta": eta, "pct": round(st.frac * 100) if job.state == "running" else 0}


def cancel_job(job: Job):
    with QCOND:
        if job.state == "queued":
            QUEUE.remove(job)
            job.state, job.st.error, job.files = "stopped", "Stopped.", None
            shutil.rmtree(job.dir, ignore_errors=True)
            return
    if job.state == "running":
        job.st.cancel.set()


def run(method, files, h_strength, h_quality, tag, target, seed, template, write_meta, m_strength, m_mode, m_passes, m_prompt, m_keep):
    """Validates, copies the uploads into the job folder, queues the run and returns its id right away."""
    if not _setup["ready"]:
        raise gr.Error(_setup["error"] or _setup["msg"] + ". Try again in a few minutes.")
    if not files:
        raise gr.Error("Add at least one image.")
    limit = MAX_FILES_H if method == METHOD_H else MAX_FILES_M
    if len(files) > limit:
        raise gr.Error(f"At most {limit} images per run for this method.")
    with QCOND:
        if len(QUEUE) >= QUEUE_MAX:
            raise gr.Error("The line is full right now. Please try again in a few minutes.")
    o = {"h_strength": h_strength, "h_quality": h_quality, "tag": tag or "", "target": target or "",
         "seed": int(seed) if seed not in (None, "") else None, "template": template or "{tag}", "write_meta": bool(write_meta),
         "m_strength": m_strength, "m_mode": m_mode, "m_passes": m_passes, "m_prompt": m_prompt or "", "m_keep": bool(m_keep)}
    job = Job(method, len(files), [], o)
    staged = []
    try:
        for i, f in enumerate(files):
            src = _file_path(f)
            d = job.dir / "upload" / f"{i:04d}"
            d.mkdir(parents=True)
            dest = d / (getattr(f, "orig_name", None) or src.name)
            shutil.copyfile(src, dest)
            staged.append(dest)
    except Exception:
        shutil.rmtree(job.dir, ignore_errors=True)
        raise gr.Error("Could not read the uploaded files. Please try again.")
    job.files = staged
    with QCOND:
        REG[job.id] = job
        QUEUE.append(job)
        QCOND.notify()
    return job.id


def build():
    with gr.Blocks(title="Hemlock", analytics_enabled=False, css=viewer.CSS, head=viewer.HEAD,
                   theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="slate")) as demo:
        gr.HTML(
            "<div id='hero'><h1>Hemlock</h1></div>"
        )
        gr.HTML(
            "<div class='note'><b>Please be patient.</b> This is a hobby project running on a spare PC in my basement, not in a data center. "
            "Your images may wait in line behind someone else's, which can take a few minutes.</div>"
        )
        gr.HTML(status_text(), elem_id="status")
        method = gr.Radio([METHOD_H, METHOD_M], value=DEF_METHOD, label="Method",
                          info=f"Hemlock poisons what AI learns from the image (Nightshade-style, roughly {fmt_est(h_sec('Fast'))} to {fmt_est(h_sec('Maximum'))} per image). "
                               f"Mist adds noise that disrupts copying an artist's style (very slow, about {fmt_est(M_BASE * SPEED)} per image).")
        with gr.Row():
            with gr.Column(scale=3, min_width=320):
                files = gr.File(label="Your images", file_count="multiple", file_types=["image"], height=380)
                gr.HTML("<div class='hint'>" + (f"Up to {MAX_FILES_H} images per run." if MAX_FILES_H == MAX_FILES_M
                        else f"Up to {MAX_FILES_H} images per run with Hemlock, {MAX_FILES_M} with Mist.") + "</div>")
            with gr.Column(scale=2, min_width=320):
                with gr.Group(visible=DEF_METHOD == METHOD_H) as h_main:
                    h_strength = gr.Radio(list(H_STRENGTH), value=DEF_H_STRENGTH, label="Strength",
                                          info="How much each pixel may change. Higher strength settings will also take longer to render.")
                    h_quality = gr.Radio(list(H_QUALITY), value=DEF_H_QUALITY, label="Render quality",
                                         info=f"How long it optimizes. Fast takes about {fmt_est(h_sec('Fast'))} per image, Better about {fmt_est(h_sec('Better'))}, Thorough about {fmt_est(h_sec('Thorough'))}, Maximum about {fmt_est(h_sec('Maximum'))}.")
                with gr.Group(visible=DEF_METHOD == METHOD_M) as m_main:
                    m_strength = gr.Radio(list(M_STRENGTH), value=DEF_M_STRENGTH, label="Strength",
                                          info="Stronger protects more, but the noise becomes easier to see.")
                    m_keep = gr.Checkbox(True, label="Keep the original image size")
        with gr.Accordion("More options", open=False):
            with gr.Group(visible=DEF_METHOD == METHOD_H) as h_more:
                tag = gr.Textbox(label="Poison tag (one image only)", value="",
                                 info="A word describing the image, e.g. 'dog'. Leave empty to detect it automatically.")
                target = gr.Textbox(label="Push towards this concept (optional)", value="",
                                    info="Normally chosen automatically. Enter a word like 'car' to force it.")
                template = gr.Textbox(label="ALT text template", value="{tag}",
                                      info="Used for the ALT text list and the Copy buttons. {tag} is replaced by the tag, e.g. 'Digital painting of a {tag}'.")
                write_meta = gr.Checkbox(True, label="Also write the ALT text into the image file's metadata",
                                         info="Most sites strip metadata, so still paste the ALT text when you post.")
                seed = gr.Number(label="Seed (optional)", value=None, precision=0, info="Same seed and settings give the same result.")
            with gr.Group(visible=DEF_METHOD == METHOD_M) as m_more:
                m_mode = gr.Radio(list(M_MODES), value=list(M_MODES)[0], label="Mode")
                m_passes = gr.Slider(1, 10, step=1, value=DEF_M_PASSES, label="Passes", info="More passes take longer and protect somewhat better.")
                m_prompt = gr.Textbox(label="Short description of the images", value="a picture",
                                      info="Optional, e.g. 'an anime girl' or 'a landscape painting'.")
        with gr.Row():
            go = gr.Button("Protect images", variant="primary", size="lg", scale=4, elem_id="go")
            stop = gr.Button("Stop", size="lg", scale=1, min_width=90)
        gr.Markdown("These methods make copying harder, they are not a guarantee, and their effect has not been independently measured here. "
                    "Keep your originals. Not affiliated with the Nightshade or Mist teams.")
        gr.HTML("<div id='hm-jobs'></div>")
        jobid = gr.Textbox(visible=False)
        with gr.Accordion("About this project: what is happening to my image?", open=False):
            gr.HTML(ABOUT_HTML)
        with gr.Accordion("Info: what the methods, models and settings mean", open=False):
            gr.HTML(INFO_HTML)

        def switch(m):
            h = m == METHOD_H
            return [gr.update(visible=h), gr.update(visible=not h), gr.update(visible=h), gr.update(visible=not h)]

        method.change(switch, method, [h_main, m_main, h_more, m_more], show_progress="hidden")
        inputs = [method, files, h_strength, h_quality, tag, target, seed, template, write_meta, m_strength, m_mode, m_passes, m_prompt, m_keep]
        go.click(run, inputs, [jobid], show_progress="hidden", trigger_mode="once").then(
            None, [jobid], None, js="(id) => { HM.added(id) }")
        stop.click(None, None, None, js="() => { HM.cancelAll() }")
    return demo


if __name__ == "__main__":
    JOBS.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("HEMLOCK_SKIP_MODELS"):
        threading.Thread(target=ensure_downloads, daemon=True).start()
    threading.Thread(target=_sweeper, daemon=True).start()
    load_timings()
    threading.Thread(target=_runner, daemon=True, name="runner").start()
    pw = os.environ.get("HEMLOCK_PASSWORD", "")
    import uvicorn
    from fastapi import FastAPI

    api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @api.get("/hm_status")
    def hm_status():
        return {"html": status_text()}

    from fastapi.responses import JSONResponse

    @api.get("/hm_job/{jid}")
    def hm_job(jid: str):
        job = REG.get(jid)
        if job is None:
            return JSONResponse({"state": "expired"}, status_code=404, headers={"Cache-Control": "no-store"})
        return JSONResponse(job_view(job), headers={"Cache-Control": "no-store"})

    @api.post("/hm_cancel/{jid}")
    def hm_cancel(jid: str):
        job = REG.get(jid)
        if job is not None:
            cancel_job(job)
        return {"ok": job is not None}

    demo = build()
    demo.queue(default_concurrency_limit=4)
    demo.show_api = False
    app = gr.mount_gradio_app(api, demo, path="/", allowed_paths=[str(JOBS)])
    if pw:
        from auth import PasswordGate

        app = PasswordGate(app, pw, max_fails=int(os.environ.get("HEMLOCK_LOGIN_MAX_FAILS", "5")),
                           lock_seconds=int(float(os.environ.get("HEMLOCK_LOGIN_LOCK_MINUTES", "15")) * 60))
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
