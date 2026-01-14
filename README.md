# BatchBench (Rust-only)

BatchBench is now a Rust-only CLI for generating synthetic request corpora and running online benchmarks against OpenAI-compatible endpoints.

## Build

```bash
cargo build --release --bin batchbench -p batchbench-rs
# Binary: rust-bench/target/release/batchbench
```

## Run

Generate a workload and run the benchmark (inline generation sized to `users * requests_per_user`):

```bash
./rust-bench/target/release/batchbench \
  --model gpt-4o-mini \
  --users 8 \
  --requests-per-user 2 \
  --gen-approx-input-tokens 256 \
  --output-tokens 64 \
  --output-vary 0
```

Set your API key via `--api-key` or the environment variable named by `--api-key-env` (defaults to `OPENAI_API_KEY`).

## Notes

- The CLI deterministically maps request index `m*N + n` (request m, user n) into the generated dataset.
- `build_rust.sh` simply builds the release binary; no Python packaging remains.
