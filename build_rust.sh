#!/usr/bin/env bash
# Build the Rust binary only (no Python packaging)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUST_DIR="$SCRIPT_DIR/rust-bench"

echo "Building Rust binary in release mode..."
cd "$RUST_DIR"
cargo build --release --bin batchbench

echo "✓ Rust binary built at $RUST_DIR/target/release/batchbench"
