# 🚀 배포 및 환경 구성 가이드 (Docker & Config)

이 문서는 **Docker를 활용한 시스템 배포 방법**과 관련 **환경 변수 설정**, 운영 시 고려사항을 안내합니다. 개발 환경이 아니라 운영 또는 테스트 서버에서 시스템을 실행해야 할 경우 활용하세요.

## 1. Docker 환경 준비
배포를 위해서는 Docker가 설치되어 있어야 합니다. 권장 버전:
- **Docker Engine**: v24.x 이상
- **Docker Compose**: v2.x 이상 (플러그인 또는 standalone 모두 가능)

Docker가 준비됐다면, 프로젝트 루트 디렉토리에서 제공하는 설정 파일과 스크립트를 사용하여 이미지를 빌드하고 컨테이너를 실행할 수 있습니다.

## 2. Docker 이미지 빌드
프로젝트 코드 및 필요한 환경을 Docker 이미지로 패키징합니다.
```bash
# 프로젝트 루트에서 실행
./scripts/build_docker.sh
```
위 스크립트는 Dockerfile을 기반으로 이미지를 빌드하며, 기본 이미지 이름은 `colormeter:dev`입니다.
수동 빌드: 스크립트를 쓰지 않고 직접 빌드하려면 다음 명령을 사용할 수 있습니다:
```bash
docker build -t colormeter:dev .
```
(BuildKit을 사용하도록 Docker가 설정되어 있으면 빌드가 더 빠릅니다. 빌드 중 오류가 난다면 `export DOCKER_BUILDKIT=1` 설정을 확인하세요.)

## 3. 컨테이너 실행 방법
### 3.1 1회성 검사 실행 (포그라운드)
단일 이미지에 대해 바로 검사를 실행하고 종료하는 모드:
```bash
./scripts/run_docker.sh --image data/raw_images/OK_001.jpg --sku SKU001
```
스크립트 내부에서 `docker run` 명령이 호출되어, 지정한 이미지와 SKU로 검사를 수행합니다. 실행이 완료되면 컨테이너는 종료되고, 결과는 `results/` 폴더에서 확인합니다.
**이미지/SKU 변경:** 기본값으로 스크립트에 지정된 경로나 SKU를 바꾸고 싶다면 인자를 조정하거나, 환경 변수로 `IMAGE_NAME`, `SKU` 등을 넘길 수 있습니다. 예:
```bash
IMAGE_NAME=colormeter:dev ./scripts/run_docker.sh --image <경로/다른이미지.jpg> --sku SKU002
```

### 3.2 지속 실행 (백그라운드 서버 모드)
개발용 혹은 서비스 모드로 FastAPI 서버를 띄워두려면 Docker Compose를 이용합니다:
```bash
IMAGE_NAME=colormeter:latest docker compose up -d --build
```
위 명령은 `docker-compose.yml` 설정에 따라 웹 서버(Uvicorn)까지 실행된 컨테이너를 백그라운드로 띄웁니다.
`docker compose logs -f`로 실시간 로그를 볼 수 있고, `docker compose down`으로 중지시킬 수 있습니다.
기본 포트는 8000으로, 필요시 compose 파일에서 변경 가능합니다.

## 4. 볼륨 및 데이터 관리
컨테이너가 실행될 때 호스트의 몇몇 디렉토리를 마운트하여 데이터를 공유합니다 (`docker-compose.yml`에 정의):
- **호스트 `./data` -> 컨테이너 `/app/data`**
  입력 이미지 및 (선택) 배치 ZIP, 그리고 결과 CSV 등이 포함됩니다.
- **호스트 `./config` -> 컨테이너 `/app/config`**
  SKU JSON 파일 및 시스템 설정 파일을 컨테이너에서 접근할 수 있도록 합니다.
- **호스트 `./results` -> 컨테이너 `/app/results`**
  검사 결과물(오버레이 이미지, 로그 등)을 호스트에도 남깁니다.

따라서 로컬에서 보던 경로 그대로 컨테이너 안에서도 사용하면 되므로, 경로 혼동을 줄일 수 있습니다. 예를 들어 호스트의 `data/raw_images/`에 이미지를 넣고 `--image data/raw_images/XYZ.jpg`로 실행하면, 컨테이너 내에서도 동일한 `/app/data/raw_images/XYZ.jpg` 경로로 해당 파일을 찾게 됩니다.

## 5. 환경 변수 설정
Docker 컨테이너 실행 시 다음 환경 변수들을 필요에 따라 조정할 수 있습니다:
- `LOG_LEVEL` – 애플리케이션 로그 레벨. (`INFO` 기본값. 디버깅 시 `DEBUG`로 높이면 세부 로그를 확인 가능)
- `SKU` – (옵션) 컨테이너 내에서 기본으로 사용할 SKU 코드. 주로 `run_docker.sh` 1회성 실행 시 활용되며, 웹 서버 모드에서는 사용되지 않습니다.
- `DATA_PATH` – (옵션) 데이터 디렉토리 경로. 기본 `/app/data`로 마운트되므로 일반적으로 설정 필요 없지만, 커스텀 경로에 데이터를 두었다면 이 변수로 지정할 수 있습니다.

환경 변수는 Docker Compose 파일의 `environment` 섹션에서 설정하거나, `docker run -e` 옵션을 통해 주입할 수 있습니다.
예: 로그를 자세히 보고 싶다면 compose 파일에서 `LOG_LEVEL: DEBUG`로 수정한 뒤 `up -d`로 재실행하면 됩니다.

## 6. 운영 시 참고 사항
- **CPU 환경 지원:** 이미지 빌드는 GPU가 없는 CPU 환경에서도 동작하도록 설계되었습니다. OpenCV의 GUI 기능은 slim 이미지에서는 제한되지만, CLI 검사나 결과 저장 기능에는 영향이 없습니다.
- **권한 문제:** Linux 환경에서 볼륨 마운트된 디렉토리에 권한 문제가 발생하면, 호스트에서 해당 디렉토리에 대해 읽기/쓰기 권한을 부여해야 합니다. (예: `chmod -R 755 data config results`)
- **Windows 경로 이슈:** Windows에서 Docker Desktop을 사용하는 경우, 로컬 디렉토리를 마운트하려면 Docker Desktop의 File Sharing 설정에 해당 드라이브 또는 경로가 추가되어 있어야 합니다. 그렇지 않으면 컨테이너가 마운트된 폴더를 보지 못해 “파일을 찾을 수 없음” 오류가 날 수 있습니다.
- **네트워크 문제로 빌드 실패:** Docker 이미지 빌드 중 패키지 설치가 네트워크 문제로 실패할 경우, `--no-cache` 옵션을 붙여 재시도하면 캐시를 무시하고 새로 패치하므로 해결에 도움이 됩니다.

## 7. Troubleshooting (배포 관련 문제 해결)
- **이미지 빌드 실패 시:** 위의 네트워크 문제 외에도 Docker 데몬 메모리 부족 등으로 빌드가 죽을 수 있습니다. 가능하면 메모리 여유가 충분한 환경에서 빌드하고, Base image를 최신으로 유지합니다.
- **컨테이너 실행 즉시 종료:** `docker compose up -d` 후 `docker ps`에서 컨테이너가 계속 재시작하거나 내려가 있다면, `docker compose logs`로 원인을 확인하세요. 보통 SKU 파일 없음, 잘못된 환경 변수 값 등이 원인입니다. 로그에 표시된 에러 메시지에 따라 설정을 고쳐준 뒤 `down` 후 `up`으로 재실행합니다.
- **결과 파일이 안 보임:** 컨테이너 내부에서는 검사했는데 호스트 `results/` 폴더에 결과가 없으면 볼륨 마운트가 잘 되었는지 확인하세요. compose 파일의 경로와 실제 폴더 위치가 일치해야 합니다 (상대경로 주의).

이상으로 Docker 기반 배포 및 설정 가이드를 마칩니다. 이 가이드를 따르면 개발 환경과 상관없이 어디서나 일관된 환경으로 시스템을 실행할 수 있습니다. 추가 문의는 개발팀에 연락주세요.
