#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME=${IMAGE_NAME:-colormeter:latest}
CONTAINER_NAME=${CONTAINER_NAME:-colormeter}

# Mount host data/config/results for persistence
DATA_DIR=${DATA_DIR:-$(pwd)/data}
CONFIG_DIR=${CONFIG_DIR:-$(pwd)/config}
RESULTS_DIR=${RESULTS_DIR:-$(pwd)/results}

mkdir -p "$RESULTS_DIR"

docker run --rm -it \
  --name "$CONTAINER_NAME" \
  -v "$DATA_DIR":/app/data \
  -v "$CONFIG_DIR":/app/config \
  -v "$RESULTS_DIR":/app/results \
  "$IMAGE_NAME" "$@"
