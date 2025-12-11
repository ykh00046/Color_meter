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
- GPU/AVX 없는 CPU 환경에서도 동작합니다.
- OpenCV GUI 기능은 slim 이미지에서 제한되지만 CLI/배치/시각화 저장은 정상 작동합니다.
- 컨테이너 내에서 pytest를 실행하여 테스트를 확인할 수 있습니다.

## Troubleshooting

### 이미지 빌드 실패
- Docker BuildKit이 활성화되어 있는지 확인: `export DOCKER_BUILDKIT=1`
- 네트워크 문제로 패키지 설치 실패 시: `--no-cache` 옵션 사용

### 컨테이너 실행 시 권한 오류
- Windows: Docker Desktop에서 파일 공유 설정 확인
- Linux: 볼륨 디렉토리 권한 확인 (`chmod -R 755 data config results`)

### 이미지/SKU 파일을 찾을 수 없음
- 볼륨 마운트 경로 확인: `docker-compose.yml`의 volumes 섹션
- 호스트 경로가 올바른지 확인: `./data`, `./config`, `./results`
