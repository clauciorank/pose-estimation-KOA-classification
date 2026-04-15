#!/usr/bin/env bash
# Download OpenPose BODY_25 model to data/models/
# Run once from the project root: bash docker/download_models.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$(dirname "$SCRIPT_DIR")/data/models/pose/body_25"

mkdir -p "$MODELS_DIR"

MODEL_FILE="$MODELS_DIR/pose_iter_584000.caffemodel"

if [ -f "$MODEL_FILE" ]; then
    echo "Model already exists: $MODEL_FILE"
    exit 0
fi

echo "Downloading BODY_25 model (~200 MB)..."

# Primary: gaijingeek/openpose-models mirror on Hugging Face (hosted because CMU server is offline)
wget -q --show-progress \
    "https://huggingface.co/gaijingeek/openpose-models/resolve/main/models/pose/body_25/pose_iter_584000.caffemodel" \
    -O "$MODEL_FILE" && echo "Done." && exit 0

echo "Primary mirror failed, trying camenduru/openpose on Hugging Face..."

# Fallback: camenduru/openpose mirror
wget -q --show-progress \
    "https://huggingface.co/camenduru/openpose/resolve/main/pose_iter_584000.caffemodel" \
    -O "$MODEL_FILE" && echo "Done." && exit 0

echo ""
echo "ERROR: All download sources failed."
echo "Place the file manually at:"
echo "  $MODEL_FILE"
echo ""
echo "Sources to try in a browser:"
echo "  https://huggingface.co/gaijingeek/openpose-models"
echo "  https://huggingface.co/camenduru/openpose"
exit 1