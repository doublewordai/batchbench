#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[%s] %s\n' "$(date +'%Y-%m-%dT%H:%M:%S')" "$*" >&2
}

die() {
  log "ERROR: $*"
  exit 1
}

print_command() {
  printf '    %s' "$1" >&2
  shift
  for arg in "$@"; do
    printf ' %q' "$arg" >&2
  done
  printf '\n' >&2
}

run_offline() {
  log "Starting offline benchmark"
  local args=(python -m batchbench.offline)

  [[ -n "${OFFLINE_MODEL:-}" ]] && args+=(--model "${OFFLINE_MODEL}")
  [[ -n "${OFFLINE_NUM_REQS:-}" ]] && args+=(--num_reqs "${OFFLINE_NUM_REQS}")
  [[ -n "${OFFLINE_ICL:-}" ]] && args+=(--icl "${OFFLINE_ICL}")
  [[ -n "${OFFLINE_OCL:-}" ]] && args+=(--ocl "${OFFLINE_OCL}")
  [[ -n "${OFFLINE_THROUGHPUT_DIR:-}" ]] && args+=(--throughput_dir "${OFFLINE_THROUGHPUT_DIR}")
  [[ -n "${OFFLINE_TENSOR_PARALLEL_SIZE:-}" ]] && args+=(--tensor_parallel_size "${OFFLINE_TENSOR_PARALLEL_SIZE}")
  [[ -n "${OFFLINE_PIPELINE_PARALLEL_SIZE:-}" ]] && args+=(--pipeline_parallel_size "${OFFLINE_PIPELINE_PARALLEL_SIZE}")
  [[ -n "${OFFLINE_GPU_MEMORY_UTILIZATION:-}" ]] && args+=(--gpu_memory_utilization "${OFFLINE_GPU_MEMORY_UTILIZATION}")
  [[ -n "${OFFLINE_MAX_NUM_BATCHED_TOKENS:-}" ]] && args+=(--max_num_batched_tokens "${OFFLINE_MAX_NUM_BATCHED_TOKENS}")

  if [[ -n "${OFFLINE_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    args+=(${OFFLINE_EXTRA_ARGS})
  fi

  log "Invoking offline runner with:"; print_command "${args[@]}"
  "${args[@]}"
}

wait_for_local_http() {
  local url=$1
  local pid=$2
  local retries=${ONLINE_SERVER_WAIT_RETRIES:-60}
  local delay=${ONLINE_SERVER_WAIT_DELAY_SECS:-1}
  if ! command -v curl >/dev/null 2>&1; then
    die "curl is required to perform readiness checks"
  fi

  for ((attempt = 1; attempt <= retries; attempt++)); do
    if curl --silent --fail "$url" >/dev/null 2>&1; then
      log "Server became reachable after $attempt attempt(s)"
      return 0
    fi
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      wait "$pid" || true
      die "vLLM server process exited before readiness probe succeeded"
    fi
    sleep "$delay"
  done

  die "Timed out after $retries attempts waiting for $url"
}

cleanup_server() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    log "Stopping vLLM server (PID ${SERVER_PID})"
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" 2>/dev/null || true
    unset SERVER_PID
  fi
}

run_online() {
  log "Starting online benchmark"

  local dataset_base=${ONLINE_DATASET_PATH:-/tmp/batchbench_requests.jsonl}

  local generate_args=(python -m batchbench.generate --output "$dataset_base")
  [[ -n "${ONLINE_GENERATE_COUNT:-}" ]] && generate_args+=(--count "${ONLINE_GENERATE_COUNT}")
  [[ -n "${ONLINE_GENERATE_PREFIX_OVERLAP:-}" ]] && generate_args+=(--prefix-overlap "${ONLINE_GENERATE_PREFIX_OVERLAP}")
  [[ -n "${ONLINE_GENERATE_APPROX_INPUT_TOKENS:-}" ]] && generate_args+=(--approx-input-tokens "${ONLINE_GENERATE_APPROX_INPUT_TOKENS}")
  [[ -n "${ONLINE_GENERATE_TOKENIZER_MODEL:-}" ]] && generate_args+=(--tokenizer-model "${ONLINE_GENERATE_TOKENIZER_MODEL}")
  [[ -n "${ONLINE_GENERATE_TOKEN_TOLERANCE:-}" ]] && generate_args+=(--token-tolerance "${ONLINE_GENERATE_TOKEN_TOLERANCE}")
  [[ -n "${ONLINE_GENERATE_HUGGINGFACE_TOKEN:-}" ]] && generate_args+=(--huggingface-token "${ONLINE_GENERATE_HUGGINGFACE_TOKEN}")

  if [[ -n "${ONLINE_GENERATE_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    generate_args+=(${ONLINE_GENERATE_EXTRA_ARGS})
  fi

  log "Generating request dataset:"; print_command "${generate_args[@]}"
  local generated_output=()
  if ! mapfile -t generated_output < <("${generate_args[@]}"); then
    die "Request dataset generation failed"
  fi
  if ((${#generated_output[@]} == 0)); then
    die "Dataset generator did not return a path"
  fi
  local last_index=$(( ${#generated_output[@]} - 1 ))
  local dataset_path=${generated_output[$last_index]}
  if [[ -z "$dataset_path" ]]; then
    die "Dataset generator returned an empty path"
  fi
  log "Using dataset $dataset_path"

  local online_model=${ONLINE_MODEL:-}
  [[ -n "$online_model" ]] || die "ONLINE_MODEL must be set for online mode"

  local server_host=${ONLINE_SERVER_HOST:-0.0.0.0}
  local server_port=${ONLINE_SERVER_PORT:-8000}
  local base_url=${ONLINE_HOST:-http://127.0.0.1:$server_port}
  local readiness_url=${ONLINE_SERVER_HEALTH_URL:-$base_url/v1/models}

  local server_args=(
    python -m vllm.entrypoints.openai.api_server
    --host "$server_host"
    --port "$server_port"
    --model "$online_model"
  )
  [[ -n "${ONLINE_TENSOR_PARALLEL_SIZE:-}" ]] && server_args+=(--tensor-parallel-size "${ONLINE_TENSOR_PARALLEL_SIZE}")
  [[ -n "${ONLINE_PIPELINE_PARALLEL_SIZE:-}" ]] && server_args+=(--pipeline-parallel-size "${ONLINE_PIPELINE_PARALLEL_SIZE}")
  [[ -n "${ONLINE_MAX_NUM_BATCHED_TOKENS:-}" ]] && server_args+=(--max-num-batched-tokens "${ONLINE_MAX_NUM_BATCHED_TOKENS}")
  [[ -n "${ONLINE_GPU_MEMORY_UTILIZATION:-}" ]] && server_args+=(--gpu-memory-utilization "${ONLINE_GPU_MEMORY_UTILIZATION}")

  if [[ -n "${ONLINE_SERVER_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    server_args+=(${ONLINE_SERVER_EXTRA_ARGS})
  fi

  trap cleanup_server EXIT INT TERM

  log "Launching vLLM server:"; print_command "${server_args[@]}"
  "${server_args[@]}" &
  SERVER_PID=$!

  wait_for_local_http "$readiness_url" "$SERVER_PID"

  local online_args=(python -m batchbench.online --jsonl "$dataset_path" --model "${ONLINE_CLIENT_MODEL:-$online_model}" --host "$base_url")
  [[ -n "${ONLINE_ENDPOINT:-}" ]] && online_args+=(--endpoint "${ONLINE_ENDPOINT}")
  [[ -n "${ONLINE_USERS:-}" ]] && online_args+=(--users "${ONLINE_USERS}")
  [[ -n "${ONLINE_REQUESTS_PER_USER:-}" ]] && online_args+=(--requests-per-user "${ONLINE_REQUESTS_PER_USER}")
  [[ -n "${ONLINE_API_KEY:-}" ]] && online_args+=(--api-key "${ONLINE_API_KEY}")
  [[ -n "${ONLINE_API_KEY_ENV:-}" ]] && online_args+=(--api-key-env "${ONLINE_API_KEY_ENV}")
  [[ -n "${ONLINE_REQUEST_TIMEOUT_SECS:-}" ]] && online_args+=(--request-timeout-secs "${ONLINE_REQUEST_TIMEOUT_SECS}")
  [[ -n "${ONLINE_MAX_RETRIES:-}" ]] && online_args+=(--max-retries "${ONLINE_MAX_RETRIES}")
  [[ -n "${ONLINE_RETRY_DELAY_MS:-}" ]] && online_args+=(--retry-delay-ms "${ONLINE_RETRY_DELAY_MS}")
  [[ -n "${ONLINE_OUTPUT_TOKENS:-}" ]] && online_args+=(--output-tokens "${ONLINE_OUTPUT_TOKENS}")
  [[ -n "${ONLINE_OUTPUT_VARY:-}" ]] && online_args+=(--output-vary "${ONLINE_OUTPUT_VARY}")

  if [[ -n "${ONLINE_CLIENT_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    online_args+=(${ONLINE_CLIENT_EXTRA_ARGS})
  fi

  log "Running online benchmark:"; print_command "${online_args[@]}"
  "${online_args[@]}"
}

main() {
  local mode=${BATCHBENCH_MODE:-offline}
  mode=$(echo "$mode" | tr '[:upper:]' '[:lower:]')

  case "$mode" in
    offline)
      run_offline
      ;;
    online)
      run_online
      ;;
    *)
      die "Unknown BATCHBENCH_MODE '$mode'. Expected 'online' or 'offline'."
      ;;
  esac
}

main "$@"
