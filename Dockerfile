# Mist v2 as a self-hosted web service: upload images, get misted images back.
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
    HF_HOME=/data/cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    WANDB_DISABLED=true \
    GIT_SHA=${GIT_SHA} \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update \
 && apt-get install -y --no-install-recommends gosu tini ca-certificates libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
      --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /app
COPY attacks ./attacks
COPY lora_diffusion ./lora_diffusion
COPY data/MIST.png ./data/MIST.png
COPY MIST_logo.png ./MIST_logo.png
COPY docker/app.py docker/entrypoint.sh ./
RUN chmod +x /app/entrypoint.sh

# Fail the build early if the old Mist code no longer imports against these library versions
RUN python -c "import attacks.mist, gradio; print('imports ok')"

EXPOSE 7860
VOLUME /data
ENTRYPOINT ["/usr/bin/tini", "--", "/app/entrypoint.sh"]
CMD ["python", "/app/app.py"]
