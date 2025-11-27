#!/usr/bin/env bash
# Build the Rust binary and copy it to the Python package for batchbench.online
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUST_DIR="$SCRIPT_DIR/rust-bench"
TARGET_DIR="$SCRIPT_DIR/src/batchbench/bin"

echo "Building Rust binary in release mode..."
cd "$RUST_DIR"
cargo build --release --bin batchbench

echo "Copying binary to Python package..."
mkdir -p "$TARGET_DIR"
cp target/release/batchbench "$TARGET_DIR/batchbench"

echo "✓ Rust binary built and copied to $TARGET_DIR/batchbench"
echo ""
echo "The binary is now bundled with the Python package."
echo "If you have batchbench installed in editable mode (pip install -e .),"
echo "the changes will be reflected immediately."
echo "Otherwise, reinstall with: pip install -e ."
