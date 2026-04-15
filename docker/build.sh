#!/usr/bin/env bash
# Build the OpenPose Docker image.
# The build compiles OpenPose from source — expect 20-40 min on first run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "==> Building openpose-gait image..."
docker build \
    -f "$SCRIPT_DIR/Dockerfile.openpose" \
    -t openpose-gait:latest \
    "$PROJECT_ROOT"

echo "==> Testing GPU access..."
docker run --rm --runtime=nvidia openpose-gait:latest nvidia-smi

echo "==> Done. Run extraction with:"
echo "    docker compose -f docker/compose.yml run --rm openpose"
