#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME=${IMAGE_NAME:-colormeter:latest}
DOCKER_BUILDKIT=1 docker build -t "$IMAGE_NAME" .
