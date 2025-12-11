# BatchBench Copilot Instructions

## Project Overview

BatchBench is a hybrid Python/Rust benchmarking suite for LLM inference workloads. It provides two core utilities packaged as Python CLI entrypoints:
- **`batchbench.generate`**: Creates JSONL request corpora with configurable prefix overlap and token counts
- **`batchbench.online`**: Wraps a Rust binary for parallel online requests to OpenAI-compatible endpoints

## Architecture

### Hybrid Python/Rust Design
- **Python layer** (`src/batchbench/`): CLI entrypoints and request generation
- **Rust core** (`rust-bench/`): High-performance async HTTP client for online benchmarking
- **Build bridge**: `build_rust.sh` compiles the Rust binary and copies it to `src/batchbench/bin/batchbench` for bundling in the wheel
- **Invocation**: `batchbench.online` (Python) uses `importlib.resources` to locate and subprocess-execute the bundled Rust binary

### Package Structure
- Uses `src/` layout with `setuptools.build_meta` backend
- Rust binary is packaged as data: `[tool.setuptools.package-data] "batchbench" = ["bin/batchbench"]`
- Optional dependencies: `[generate]` adds transformers/datasets

## Critical Workflows

### Rebuilding the Rust Binary
When modifying Rust code, run:
```bash
./build_rust.sh
```
This builds `rust-bench/target/release/batchbench` and copies it to `src/batchbench/bin/`. If installed in editable mode (`pip install -e .`), changes are immediate. Otherwise, reinstall the package.

### Development Installation
```bash
# Full install with all features
pip install -e ".[generate]"

# Rust binary rebuild (after Rust changes)
./build_rust.sh
```

### Running Tests
Rust tests: `cd rust-bench && cargo test`
Python tests: Not formalized yet—use example scripts in `examples/`

## JSONL Format Migration (CRITICAL)

The project recently migrated from legacy `{"text": "..."}` format to OpenAI Batch API format:
```json
{"messages": [{"role": "user", "content": "..."}], "model": "..."}
```

**Backward Compatibility**: The Rust CLI (`rust-bench/src/bin/batchbench.rs`) supports both formats:
- New format: Reads `messages[0].content` and optional `model` field
- Legacy format: Falls back to `text` field
- CLI `--model` flag is used when JSONL doesn't specify a model

**Conversion**: Use `convert_to_openai_format.py` to migrate existing files. See `OPENAI_FORMAT_MIGRATION.md` for examples.

**Generation**: `batchbench.generate` accepts optional `--model` to produce new-format files directly.

## Key Conventions

### Filename Metadata Embedding
Generated JSONL files embed run parameters in their names:
```
requests_count-32_tokens-256_prefix-0p00_tokenizer-gpt2.jsonl
```
This pattern is constructed in `generate.py` to keep runs distinct without requiring a database.

### Rust Async Patterns
- The Rust library (`rust-bench/src/lib.rs`) is fully async, requiring a Tokio runtime
- `run_benchmark()` spawns worker tasks (one per user) that independently retry failed requests
- The CLI (`batchbench.rs`) wraps the library with `#[tokio::main]` and command-line parsing via `clap`

### JSONL Reading in Rust
`batchbench.rs` reads the entire JSONL into memory, then either:
- **Per-user assignment**: Partitions records across workers (default)
- **Random pool mode** (`--random-requests`): All workers sample from the full dataset

## Docker Configuration

Docker setups use YAML config files instead of env vars. See `docker/CONFIG.md`:
- `config.online.yaml`: Online benchmark parameters (host, endpoint, users, retries)
- Mount config via `-v $(pwd)/config.yaml:/etc/batchbench/config.yaml`

Example Dockerfiles in `docker/` support different CUDA versions (cu126, cu130).

## Integration Points

### Transformers/Tokenizers
- `generate.py` uses `AutoTokenizer` to estimate token counts
- Supports both OpenAI models (via `tiktoken`) and HuggingFace models
- Tokenizer choice is embedded in output filename

### OpenAI-Compatible Endpoints
- Rust client sends JSON to `/v1/chat/completions` (configurable via `--endpoint`)
- Authentication via `Authorization: Bearer <token>` header
- Response parsing expects `usage.prompt_tokens` and `usage.completion_tokens` fields

## Common Pitfalls

1. **Forgetting to rebuild Rust**: Python changes are instant in editable mode, but Rust requires `./build_rust.sh`
2. **JSONL format assumptions**: Always handle both legacy and new formats when parsing
3. **Missing optional dependencies**: `batchbench.generate` needs `[generate]` extras

## Examples & Testing

- `examples/example_generate.sh`: Demonstrates prefix overlap and HuggingFace token usage
- `rust-bench/request.sh`: Standalone Rust CLI test without Python wrapper
