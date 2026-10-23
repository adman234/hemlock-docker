# App image: the Hemlock code on top of a prebuilt base (see Dockerfile.base). CI passes BASE_IMAGE. To build by hand:
#   docker build -f Dockerfile.base -t hemlock-base .
#   docker build --build-arg BASE_IMAGE=hemlock-base -t hemlock .
# (Pascal cards such as the GTX 10 series: add --build-arg TORCH_CUDA=cu126 to the first command.)
ARG BASE_IMAGE=hemlock-base
FROM ${BASE_IMAGE}

ARG GIT_SHA=dev
ENV GIT_SHA=${GIT_SHA} \
    MIST_DIR=/app/mist

WORKDIR /app
COPY nightshade-reimplementation /tmp/nsre
RUN pip install --no-deps /tmp/nsre && rm -rf /tmp/nsre
COPY mist/attacks ./mist/attacks
COPY mist/lora_diffusion ./mist/lora_diffusion
COPY mist/data/MIST.png ./mist/data/MIST.png
COPY mist/LICENSE ./mist/LICENSE
COPY docker/app.py docker/auth.py docker/manager.py docker/viewer.py docker/hemlock_worker.py docker/worker_client.py docker/entrypoint.sh docker/smoke.py docker/concepts.txt ./
COPY hemlock_logo.png ./
RUN chmod +x /app/entrypoint.sh

# Fail the build early if the app's own imports break, and make sure the page renders (Gradio and Starlette versions have to match)
RUN python -c "import nsre.engine; print('hemlock imports ok')" \
 && cd /app/mist && python -c "import attacks.mist; print('mist imports ok')" \
 && python /app/smoke.py \
 && rm -rf /tmp/smoke

EXPOSE 7860
VOLUME /data
ENTRYPOINT ["/usr/bin/tini", "--", "/app/entrypoint.sh"]
CMD ["python", "/app/app.py"]
