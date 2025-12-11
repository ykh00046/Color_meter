# Color Meter Docker Environment

## Prerequisites
- Docker 24+
- Docker Compose v2

## Build
```bash
./scripts/build_docker.sh
# or
IMAGE_NAME=colormeter:dev DOCKER_BUILDKIT=1 docker build -t colormeter:dev .
```

## Run (one-off)
```bash
./scripts/run_docker.sh --image data/raw_images/OK_001.jpg --sku SKU001
# override image/SKU
IMAGE_NAME=colormeter:dev ./scripts/run_docker.sh --image data/raw_images/OK_001.jpg --sku SKU002
```

## Compose (detached)
```bash
IMAGE_NAME=colormeter:latest docker compose up -d --build
# logs
docker compose logs -f
# stop
docker compose down
```

## Volumes
- ./data -> /app/data
- ./config -> /app/config
- ./results -> /app/results

## Environment variables
- LOG_LEVEL (default: INFO)
- SKU (default: SKU001)

## Testing inside container
```bash
docker run --rm -it -v $(pwd):/app colormeter:latest pytest
```

## Notes
- GPU/AVX 없이 CPU 모드 기준.
- OpenCV GUI 기능은 slim 이미지에서 지원되지 않으니 CLI/노트북/시각화 저장 위주로 사용.
