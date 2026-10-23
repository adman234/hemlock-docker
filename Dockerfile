# Hemlock + Mist v2 in one image: one torch, one environment, models loaded only while needed.
# cu128 wheels cover Turing up to Blackwell (RTX 50xx). Pascal cards: --build-arg TORCH_CUDA=cu126
FROM python:3.11-slim-bookworm

ARG TORCH_VERSION=2.8.0
ARG TORCHVISION_VERSION=0.23.0
ARG TORCH_CUDA=cu128
ARG GIT_SHA=dev

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DATA_DIR=/data \
    MIST_DIR=/app/mist \
    HF_HOME=/data/cache/huggingface \
    TORCH_HOME=/data/cache/torch \
    NLTK_DATA=/data/cache/nltk \
    HF_HUB_DISABLE_TELEMETRY=1 \
    WANDB_DISABLED=true \
    GIT_SHA=${GIT_SHA} \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update \
 && apt-get install -y --no-install-recommends gosu tini ca-certificates libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Torch and the trimming happen in ONE layer: files deleted in a later layer would still be stored in this one.
# Trimmed: torch.compile's triton and the C/C++ headers shipped inside the CUDA wheels (nothing here compiles).
# Every CUDA runtime library stays, because torch links them at import; the checks further down fail the build otherwise.
RUN pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
      --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
 && pip uninstall -y triton \
 && SP="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
 && rm -rf "$SP"/nvidia/*/include "$SP"/torch/include "$SP"/torch/test \
 && find "$SP" -name '__pycache__' -type d -prune -exec rm -rf {} + ; \
    du -sh "$SP" | tail -1

COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
 && pip install setuptools
WORKDIR /app
COPY nightshade-reimplementation /tmp/nsre
RUN pip install --no-deps /tmp/nsre && rm -rf /tmp/nsre
COPY mist/attacks ./mist/attacks
COPY mist/lora_diffusion ./mist/lora_diffusion
COPY mist/data/MIST.png ./mist/data/MIST.png
COPY mist/LICENSE ./mist/LICENSE
COPY docker/app.py docker/manager.py docker/viewer.py docker/hemlock_worker.py docker/worker_client.py docker/entrypoint.sh docker/smoke.py docker/concepts.txt ./
COPY hemlock_logo.png ./
RUN chmod +x /app/entrypoint.sh

# Fail the build early if libraries stop importing together, or if pruning removed something torch needs
RUN python -c "import torch, torchvision; print('torch', torch.__version__, 'cudnn', torch.backends.cudnn.version()); torch.zeros(2).sum()" \
 && python -c "import nsre.engine, diffusers, transformers, lpips, nltk, gradio; print('hemlock imports ok')" \
 && cd /app/mist && python -c "import attacks.mist; print('mist imports ok')"

# ...that the page renders (Gradio and Starlette versions have to match), and that the real Stable Diffusion VAE
# file loads with every weight present (downloads about 330 MB at build time)
RUN python /app/smoke.py --vae && rm -rf /tmp/vae-smoke /tmp/smoke /root/.cache

EXPOSE 7860
VOLUME /data
ENTRYPOINT ["/usr/bin/tini", "--", "/app/entrypoint.sh"]
CMD ["python", "/app/app.py"]