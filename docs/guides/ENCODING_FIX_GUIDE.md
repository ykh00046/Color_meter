# 한글 인코딩 문제 해결 가이드

## 문제 상황
Windows MSYS2/Git Bash 터미널에서 한글이 깨져서 출력되는 문제

## 원인
- 시스템 인코딩: cp949 (Windows 한국어 코드페이지)
- Python 기본 인코딩: UTF-8
- 터미널 인코딩: 설정되지 않음

## 해결 방법

### 1. BAT 파일 실행 시 (Windows CMD)

모든 `.bat` 파일에 다음 설정이 추가되었습니다:
```batch
@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
```

**수정된 파일:**
- `scripts/run_web_ui.bat` ✅
- `scripts/restart_server.bat` ✅
- `tools/install_dependencies.bat` ✅

### 2. MSYS2/Git Bash 터미널 사용 시

#### 임시 해결 (현재 세션만)
```bash
source scripts/set_encoding.sh
```

#### 영구 해결 (모든 세션)
`~/.bashrc` 파일에 다음을 추가:
```bash
export LANG=ko_KR.UTF-8
export LC_ALL=ko_KR.UTF-8
export PYTHONIOENCODING=utf-8
```

그 다음:
```bash
source ~/.bashrc
```

### 3. Web UI는 문제 없음

HTML 파일들은 이미 UTF-8로 설정되어 있습니다:
```html
<meta charset="UTF-8">
```

브라우저에서 한글은 정상적으로 표시됩니다.

## 테스트 방법

### Python 한글 출력 테스트
```bash
python -c "print('한글 테스트: 정상 출력')"
```

### Web UI 실행
```bash
./scripts/run_web_ui.bat  # Windows CMD에서
# 또는
python -m uvicorn src.web.app:app --reload  # MSYS2에서 (환경변수 설정 후)
```

## 확인 사항

✅ **완료된 작업:**
1. 모든 .bat 파일에 UTF-8 인코딩 설정 추가
2. MSYS2용 인코딩 설정 스크립트 작성 (scripts/set_encoding.sh)
3. HTML 파일 UTF-8 인코딩 확인

⚠️ **추가 권장 사항:**
1. Windows Terminal 사용 시: 설정 > 프로필 > 고급 > 텍스트 인코딩을 "UTF-8"로 설정
2. VS Code 터미널 사용 시: 자동으로 UTF-8 사용
3. Python 3.10+ 사용 (기본 UTF-8 인코딩 지원)

## 문제가 계속될 경우

1. **터미널 재시작**
2. **환경 변수 확인**:
   ```bash
   echo $PYTHONIOENCODING
   locale
   ```
3. **Python 인코딩 확인**:
   ```python
   import sys
   print(sys.getdefaultencoding())  # utf-8이어야 함
   import locale
   print(locale.getpreferredencoding())  # UTF-8 권장
   ```

## 참고
- Python 3에서는 기본적으로 UTF-8을 사용합니다
- Windows에서는 cp949를 사용하므로 명시적 설정이 필요합니다
- Web UI의 경우 브라우저가 자동으로 UTF-8을 처리합니다
