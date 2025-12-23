# Server Management Guide

**서버 시작/종료/재시작 가이드**

---

## 🚀 서버 시작

### 방법 1: 일반 실행 (추천 - 개발 중)
```bash
cd C:/X/Color_total/Color_meter
python -m uvicorn src.web.app:app --host 0.0.0.0 --port 8000 --reload
```

**특징**:
- 코드 변경 시 자동 재시작 (`--reload`)
- Ctrl+C로 쉽게 종료 가능
- 터미널 창에서 로그 실시간 확인

**단점**:
- 터미널 창을 닫으면 서버도 종료됨

---

### 방법 2: 백그라운드 실행 (장시간 운영)
```bash
cd C:/X/Color_total/Color_meter
nohup python -m uvicorn src.web.app:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
```

**특징**:
- 백그라운드에서 계속 실행
- 터미널 창을 닫아도 서버 유지
- 로그는 `server.log` 파일에 저장

**단점**:
- 코드 변경 시 수동 재시작 필요
- 프로세스를 찾아서 종료해야 함

---

## 🛑 서버 종료

### 방법 1: 일반 실행 중인 서버 종료
터미널에서 `Ctrl+C` 누르기

---

### 방법 2: 백그라운드 실행 중인 서버 종료

#### Step 1: 프로세스 찾기
```bash
ps aux | grep uvicorn | grep -v grep
```

**출력 예시**:
```
731     726     705      37856  ?         197609 14:23:25 /c/Users/.../uvicorn
```

여기서 첫 번째 숫자 `731`이 프로세스 ID (PID)입니다.

#### Step 2: 프로세스 종료
```bash
kill 731
```

또는 **강제 종료** (응답 없을 때):
```bash
kill -9 731
```

---

### 방법 3: 포트로 프로세스 찾아서 종료 (간편)
```bash
# 포트 8000을 사용하는 프로세스 찾기
netstat -ano | grep :8000

# 해당 PID 종료 (예: PID가 1234라면)
taskkill /PID 1234 /F
```

---

### 방법 4: 모든 uvicorn 프로세스 종료 (주의!)
```bash
ps aux | grep uvicorn | grep -v grep | awk '{print $2}' | xargs kill
```

**주의**: 다른 프로젝트의 uvicorn도 모두 종료됩니다!

---

## 🔄 서버 재시작

### 코드 변경 후 재시작이 필요한 경우

#### `--reload` 모드로 실행 중이라면:
→ **자동 재시작됨** (파일 저장 시)

#### 백그라운드 실행 중이라면:
```bash
# 1. 서버 종료
ps aux | grep uvicorn | grep -v grep | awk '{print $2}' | xargs kill

# 2. 잠시 대기
sleep 2

# 3. 서버 재시작
cd C:/X/Color_total/Color_meter
python -m uvicorn src.web.app:app --host 0.0.0.0 --port 8000 --reload &
```

---

## ⏸️ 쉬는 시간에 서버 관리

### 옵션 1: 서버 종료 (추천 - 리소스 절약)
```bash
# 서버 종료
ps aux | grep uvicorn | grep -v grep | awk '{print $2}' | xargs kill

# 확인
curl http://127.0.0.1:8000/health
# 출력: curl: (7) Failed to connect - OK, 서버 종료됨
```

**장점**:
- CPU/메모리 절약
- 포트 8000 해제

**단점**:
- 다시 시작할 때 명령어 입력 필요

---

### 옵션 2: 서버 유지 (테스트/데모 준비 중)
```bash
# 서버 상태 확인
curl http://127.0.0.1:8000/health

# 출력: {"status":"ok"} - 정상 동작 중
```

**장점**:
- 언제든 바로 사용 가능
- DB 데이터 계속 누적

**단점**:
- 리소스 계속 사용 (Python 프로세스 실행 중)

---

## 🩺 서버 상태 확인

### 1. Health Check
```bash
curl http://127.0.0.1:8000/health
```

**정상**: `{"status":"ok"}`
**오류**: `curl: (7) Failed to connect`

---

### 2. 프로세스 확인
```bash
ps aux | grep uvicorn | grep -v grep
```

**실행 중**: 프로세스 목록 표시
**중지**: (출력 없음)

---

### 3. 포트 확인
```bash
netstat -ano | grep :8000
```

**사용 중**: `LISTENING` 상태 표시
**미사용**: (출력 없음)

---

## 📝 추천 워크플로우

### 개발 작업 중 (코드 수정)
```bash
# 1. --reload 모드로 시작
python -m uvicorn src.web.app:app --host 0.0.0.0 --port 8000 --reload

# 2. 코드 수정 → 저장 → 자동 재시작
# 3. 브라우저에서 테스트
# 4. 작업 완료 후 Ctrl+C로 종료
```

---

### 장시간 테스트/데모
```bash
# 1. 백그라운드로 시작
nohup python -m uvicorn src.web.app:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &

# 2. 로그 확인 (필요시)
tail -f server.log

# 3. 작업 완료 후 종료
ps aux | grep uvicorn | grep -v grep | awk '{print $2}' | xargs kill
```

---

### 휴식 시간
```bash
# 옵션 A: 서버 종료 (리소스 절약)
ps aux | grep uvicorn | grep -v grep | awk '{print $2}' | xargs kill

# 옵션 B: 서버 유지 (바로 재개 가능)
# 아무것도 안 함
```

---

## 🚨 문제 해결

### 1. "Address already in use" 오류
```bash
# 원인: 포트 8000이 이미 사용 중
# 해결: 기존 프로세스 종료
ps aux | grep uvicorn | grep -v grep | awk '{print $2}' | xargs kill

# 또는 다른 포트 사용
python -m uvicorn src.web.app:app --host 0.0.0.0 --port 8001
```

---

### 2. 서버가 응답 없음
```bash
# 1. 프로세스 확인
ps aux | grep uvicorn

# 2. 강제 종료
kill -9 [PID]

# 3. 재시작
python -m uvicorn src.web.app:app --host 0.0.0.0 --port 8000 --reload
```

---

### 3. 코드 변경이 반영 안 됨
```bash
# --reload 없이 실행한 경우
# → 서버 재시작 필요

# 1. 종료
Ctrl+C (또는 kill 명령)

# 2. 재시작
python -m uvicorn src.web.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🎯 간단 명령어 모음

### 빠른 시작
```bash
cd C:/X/Color_total/Color_meter && python -m uvicorn src.web.app:app --host 0.0.0.0 --port 8000 --reload
```

### 빠른 종료
```bash
ps aux | grep uvicorn | grep -v grep | awk '{print $2}' | xargs kill
```

### 빠른 재시작
```bash
ps aux | grep uvicorn | grep -v grep | awk '{print $2}' | xargs kill && sleep 2 && cd C:/X/Color_total/Color_meter && python -m uvicorn src.web.app:app --host 0.0.0.0 --port 8000 --reload &
```

### 상태 확인
```bash
curl http://127.0.0.1:8000/health
```

---

## 💡 팁

1. **개발 중에는** `--reload` 모드 사용 추천
2. **휴식 시간에는** 서버 종료해도 됨 (DB 데이터는 유지됨)
3. **DB 데이터는** `color_meter.db` 파일에 저장되므로 서버 종료해도 안전
4. **로그 확인은** 터미널 출력 또는 `server.log` 파일
5. **포트 변경은** `--port 8001` 등으로 가능

---

**작성일**: 2025-12-19
**업데이트**: 최종
