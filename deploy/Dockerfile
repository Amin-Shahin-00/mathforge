# GPU image with every engine and the Ray cluster runtime.
#
#   docker build -f deploy/Dockerfile -t mathforge .
#   docker run --gpus all -v $PWD/runs:/opt/mathforge/runs mathforge
#   docker run --gpus all -v $PWD/runs:/opt/mathforge/runs mathforge run --config configs/laptop.yaml
#
# Add Lean by uncommenting the setup line (adds ~7 GB).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-pip python3-venv git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/mathforge
COPY pyproject.toml README.md ./
COPY mathforge ./mathforge
COPY configs ./configs
COPY scripts ./scripts
RUN pip3 install --upgrade pip && pip3 install -e ".[cluster,llm,vllm,fast]" matplotlib

# RUN bash scripts/setup_lean.sh /opt/lean_workspace
# ENV MF_LEAN=/opt/lean_workspace   # then set engines.selfplay.lean.project_dir to it

VOLUME ["/opt/mathforge/runs"]
ENTRYPOINT ["mathforge"]
CMD ["run", "--config", "configs/gpu.yaml"]
