FROM pytorch/pytorch:2.9.0-cuda12.6-cudnn9-runtime


ARG BATCHBENCH_REPO="https://github.com/doublewordai/batchbench.git"
ARG BATCHBENCH_REF="main"

# git installation
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# uv installation
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && ln -s /root/.local/bin/uv /usr/local/bin/uv


RUN git clone --depth 1 --branch "${BATCHBENCH_REF}" "${BATCHBENCH_REPO}" /tmp/batchbench \
    && uv pip install --no-cache-dir /tmp/batchbench[offline,generate] --torch-backend=auto

# Default to bash for interactive usage
CMD ["/bin/bash"]
