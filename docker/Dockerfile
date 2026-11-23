# Project image: masciarelli/masciarelli_schiavella (ALCOR naming: cognomestudente_cognomesupervisore).
#
# One Dockerfile, two CUDA builds of the same PyTorch version:
#   lab, GTX 1080 / 1080 Ti (Pascal, sm_61):  --build-arg CUDA=cu126   (default)
#   local, RTX 5070 (Blackwell, sm_120):      --build-arg CUDA=cu130   (needs a host driver for CUDA 13)
# PyTorch 2.14 is the last release with cu126 wheels, i.e. the last one that supports Pascal.
# The PyTorch wheels ship their own CUDA runtime: the host only provides the driver (via CDI).
#
# Build from the repository root:  scripts/alcor/build.sh   (or: podman build -f docker/Dockerfile .)
FROM docker.io/library/python:3.12-slim

ARG CUDA=cu126
ARG TORCH=2.14.*

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/work/project \
    MPLCONFIGDIR=/work/cache/matplotlib \
    TORCH_HOME=/work/cache/torch

RUN apt update && apt install -y --no-install-recommends git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

RUN pip3 install "torch==${TORCH}" torchvision --index-url "https://download.pytorch.org/whl/${CUDA}"

# Every other dependency comes from pyproject.toml (single source of truth).
COPY pyproject.toml /tmp/pyproject.toml
RUN python3 -c "import tomllib; print('\n'.join(tomllib.load(open('/tmp/pyproject.toml', 'rb'))['project']['dependencies']))" > /requirements.txt \
    && pip install -r /requirements.txt

RUN mkdir /.cache && chmod 777 /.cache

WORKDIR /work/project
CMD ["python3", "main.py", "env"]
