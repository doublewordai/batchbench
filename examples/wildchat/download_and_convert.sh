#!/bin/bash
# Download and convert WildChat dataset to JSONL format
#
# Usage:
#   ./download_and_convert.sh [num_samples] [output_file]
#
# Examples:
#   ./download_and_convert.sh                    # Download 50k samples (default)
#   ./download_and_convert.sh 10000              # Download 10k samples
#   ./download_and_convert.sh 10000 output.jsonl # Download 10k samples to custom file

set -e

# Default values
NUM_SAMPLES=${1:-100000}
OUTPUT_FILE=${2:-"wildchat_${NUM_SAMPLES}.jsonl"}

# Check if Python script exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/download_wildchat.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: download_wildchat.py not found in $SCRIPT_DIR"
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python3 -c "import datasets, tqdm" 2>/dev/null || {
    echo "Installing required packages..."
    pip install datasets tqdm
}

# Run the download script
echo "Downloading $NUM_SAMPLES conversations from WildChat..."
python3 "$PYTHON_SCRIPT" --num-samples "$NUM_SAMPLES" --output "$OUTPUT_FILE"

echo ""
echo "Success! Dataset saved to: $OUTPUT_FILE"
echo ""
echo "You can now use this file with batchbench:"
echo "  ./batchbench --input $OUTPUT_FILE --url http://localhost:8000/v1/chat/completions ..."
