FROM vllm/vllm-openai:v0.10.2


ARG BATCHBENCH_REPO="https://github.com/doublewordai/batchbench.git"
ARG BATCHBENCH_REF="main"

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${BATCHBENCH_REF}" "${BATCHBENCH_REPO}" /tmp/batchbench \
    && pip install --no-cache-dir /tmp/batchbench

# Default to bash for interactive usage
CMD ["/bin/bash"]
