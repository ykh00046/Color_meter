#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME=${IMAGE_NAME:-colormeter:latest}
CONTAINER_NAME=${CONTAINER_NAME:-colormeter}

./scripts/build_docker.sh

docker compose down || true
IMAGE_NAME="$IMAGE_NAME" docker compose up -d --build

echo "Container running. To view logs: docker compose logs -f"
