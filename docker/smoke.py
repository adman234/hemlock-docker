"""Build-time checks.

default:  start the app without the model download and make sure the page renders.
--vae:    download the real Stable Diffusion VAE file the app uses and make sure every weight loads.
          (This is the check that would have caught diffusers 0.14 not understanding the current VAE format.)
"""
import os
import subprocess
import sys
import time
import urllib.request


def page():
    env = dict(os.environ, HEMLOCK_SKIP_MODELS="1", DATA_DIR="/tmp/smoke", HEMLOCK_PORT="7861")
    proc = subprocess.Popen([sys.executable, "/app/app.py"], env=env)
    try:
        for _ in range(60):
            time.sleep(1)
            if proc.poll() is not None:
                sys.exit("app exited during startup")
            try:
                r = urllib.request.urlopen("http://127.0.0.1:7861/", timeout=5)
            except Exception:
                continue
            assert r.status == 200 and b"Hemlock" in r.read(), "unexpected page"
            print("page ok")
            return
        sys.exit("page never came up")
    finally:
        proc.terminate()


def vae():
    import torch
    from diffusers import AutoencoderKL
    from huggingface_hub import snapshot_download

    repo = os.environ.get("HEMLOCK_SD_REPO", "stable-diffusion-v1-5/stable-diffusion-v1-5")
    d = snapshot_download(repo, local_dir="/tmp/vae-smoke", allow_patterns=["vae/config.json", "vae/diffusion_pytorch_model.safetensors"])
    m, info = AutoencoderKL.from_pretrained(d, subfolder="vae", output_loading_info=True)
    assert not info["missing_keys"], f"VAE weights missing (would be random): {info['missing_keys'][:5]}"
    z = m.encode(torch.randn(1, 3, 64, 64)).latent_dist.mean
    assert z.shape == (1, 4, 8, 8) and torch.isfinite(z).all(), "VAE produced bad output"
    print("vae ok, no missing keys")


if __name__ == "__main__":
    if "--vae-only" in sys.argv:  # used while building the base image, where the app is not there yet
        vae()
        sys.exit(0)
    page()
    if "--vae" in sys.argv:
        vae()