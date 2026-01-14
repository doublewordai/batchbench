#!/usr/bin/env bash
# Build the Rust binary only (no Python packaging)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUST_DIR="$SCRIPT_DIR/rust"
TARGET_DIR="$SCRIPT_DIR/bin"


echo "Building Rust binary in release mode..."
cd "$RUST_DIR"
cargo build --release --bin batchbench

echo "Copying binary..."
mkdir -p "$TARGET_DIR"
cp target/release/batchbench "$TARGET_DIR/batchbench"

echo "✓ Rust binary built and copied to $TARGET_DIR/batchbench"

