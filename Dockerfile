# CUDA 12.8.1 base matching torch 2.8.0+cu128, pinned by digest.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04@sha256:24c8e3581ea6330038b0d374920721983312627f8adbfcf390bdb4b399d280ed
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HOME=/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential python3.12 python3.12-venv python3.12-dev libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH VIRTUAL_ENV=/opt/venv
WORKDIR /app
COPY requirements.lock requirements.lock.sha256 /app/
RUN sha256sum -c requirements.lock.sha256 && python -m pip install --no-cache-dir -r requirements.lock
COPY . /app
EXPOSE 8080
ENTRYPOINT ["python", "-m", "open_decisions.image_jev.release.server", "--host", "0.0.0.0", "--port", "8080", "--offline", "--adapter", "adapter"]
