﻿# 🌐 Web API Reference



**문서 버전**: 1.0

**최종 업데이트**: 2025-12-15

**Base URL**: `http://localhost:8000`



---



## 목차



1. [개요](#1-개요)

2. [인증 및 보안](#2-인증-및-보안)

3. [Endpoints](#3-endpoints)

   - [3.1 Health Check](#31-health-check)

   - [3.2 Web UI](#32-web-ui)

   - [3.3 Inspection Endpoints](#33-inspection-endpoints)

   - [3.4 Batch Processing](#34-batch-processing)

   - [3.5 Comparison & Analytics](#35-comparison--analytics)

   - [3.6 Results Management](#36-results-management)

4. [데이터 스키마](#4-데이터-스키마)

5. [에러 처리](#5-에러-처리)

6. [예제 코드](#6-예제-코드)



---



## 1. 개요



Color Meter Web API는 **FastAPI** 기반의 RESTful API로, 콘택트렌즈 색상 검사를 위한 프로그래매틱 인터페이스를 제공합니다.



**주요 기능**:

- ✅ 단일/배치 이미지 검사

- ✅ 파라미터 재계산 (이미지 재업로드 불필요)

- ✅ 로트 비교 분석 (Golden Sample vs Test Images)

- ✅ 결과 저장 및 조회

- ✅ 잉크 분석 (Zone-based + Image-based)



**기술 스택**:

- FastAPI 0.124+

- Uvicorn (ASGI Server)

- Python 3.8+



**Swagger UI**: `http://localhost:8000/docs`

**ReDoc**: `http://localhost:8000/redoc`



---



## 2. 인증 및 보안



### 2.1 현재 상태



**인증**: 없음 (로컬 개발 환경)



**보안 고려사항**:

- 프로덕션 배포 시 API Key 또는 OAuth2 인증 권장

- CORS 설정: 현재 모든 origin 허용 (`allow_origins=["*"]`)

- 파일 업로드 제한: 최대 50MB



### 2.2 향후 계획



**Phase 4 (예정)**:

```python

# API Key 인증

headers = {"X-API-Key": "your-api-key"}

response = requests.post(url, headers=headers, ...)

```



---



## 3. Endpoints



### 3.1 Health Check



#### `GET /health`



서버 상태 확인



**Request**:

```http

GET /health HTTP/1.1

Host: localhost:8000

```



**Response** (200 OK):

```json

{

  "status": "ok",

  "timestamp": "2025-12-15T10:30:00Z"

}

```



**사용 사례**:

- 서버 가동 여부 확인

- 헬스 체크 모니터링

- Docker 컨테이너 readiness probe



---



### 3.2 Web UI



#### `GET /`



Web UI 메인 페이지 (HTML)



**Request**:

```http

GET / HTTP/1.1

Host: localhost:8000

```



**Response** (200 OK):

- Content-Type: `text/html`

- 인터랙티브 웹 인터페이스



**기능**:

- 이미지 업로드 및 시각화

- 실시간 파라미터 조정

- 결과 시각화 (6개 탭)



---



### 3.3 Inspection Endpoints



#### `POST /inspect`



단일 이미지 검사



**Request**:

```http

POST /inspect HTTP/1.1

Host: localhost:8000

Content-Type: multipart/form-data



--boundary

Content-Disposition: form-data; name="file"; filename="lens.jpg"

Content-Type: image/jpeg



[이미지 바이너리 데이터]

--boundary

Content-Disposition: form-data; name="sku"



SKU001

--boundary

Content-Disposition: form-data; name="run_judgment"



true

--boundary--

```



**Parameters**:

| 이름 | 타입 | 필수 | 설명 |

|------|------|------|------|

| `file` | File | ✅ | 검사할 이미지 (JPG, PNG) |

| `sku` | string | ✅ | SKU 코드 (예: `SKU001`) |

| `run_judgment` | boolean | ❌ | OK/NG 판정 실행 여부 (기본: `true`) |



**Response** (200 OK):

```json

{

  "status": "success",

  "image_id": "abc123def456",

  "session_id": "session_789xyz",

  "judgment": "OK",

  "overall_delta_e": 2.5,

  "confidence": 0.92,

  "zones": [

    {

      "zone_name": "Zone_C",

      "measured": {"L": 72.1, "a": 137.2, "b": 122.7},

      "expected": {"L": 72.2, "a": 137.3, "b": 122.8},

      "delta_e": 0.15,

      "threshold": 4.0,

      "passed": true,

      "std_L": 5.2,

      "pixel_count": 35200

    }

  ],

  "decision_trace": {

    "final": "OK",

    "because": "All zones passed",

    "zone_checks": {...}

  },

  "next_actions": ["출하 승인"],

  "confidence_breakdown": {

    "pixel_count_score": 0.95,

    "transition_score": 0.90,

    "std_score": 0.92,

    "sector_uniformity": 0.88,

    "lens_detection": 0.95,

    "overall": 0.92

  },

  "ink_analysis": {

    "zone_based": {

      "ink_count": 1,

      "inks": [{

        "zone_name": "Zone_C",

        "measured": {"L": 72.1, "a": 137.2, "b": 122.7},

        "rgb": [195, 150, 135],

        "hex": "#C39687"

      }]

    },

    "image_based": {

      "ink_count": 1,

      "inks": [{

        "L": 71.8, "a": 137.0, "b": 122.5,

        "weight": 1.0,

        "rgb": [194, 149, 134],

        "hex": "#C29586"

      }],

      "meta": {

        "bic": -125430.5,

        "correction_applied": false,

        "sampling_config": {

          "chroma_threshold": 6.0,

          "sampled_pixels": 45200

        }

      }

    }

  },

  "profile_data": {...},

  "overlay_url": "/results/session_789xyz/overlay.png"

}

```



**Error Responses**:

- `400 Bad Request`: 파일 누락, SKU 없음

- `404 Not Found`: SKU 파일 없음

- `500 Internal Server Error`: 검사 파이프라인 에러



---



**추가 응답 필드**

- `metrics`: blur/histogram/dot_stats 측정 결과
  - `metrics.blur.score`: Laplacian variance 기반 선명도 점수
  - `metrics.histogram`: Lab/HSV 채널 히스토그램(정규화)
  - `metrics.dot_stats`: dot_count, dot_coverage, dot_area_mean/std 등

#### `POST /inspect_v2`



확장된 검사 엔드포인트 (추가 진단 정보)



**차이점**:

- `risk_factors` 포함

- `uniformity_metrics` 상세 정보

- `sector_statistics` 포함



**Request**: `/inspect`와 동일



**Response** (200 OK):

```json

{

  // ... /inspect 응답 내용 ...

  "risk_factors": [

    {

      "category": "sector_uniformity",

      "severity": "medium",

      "message": "Zone B 섹터 간 편차 보통",

      "details": {

        "zone": "B",

        "max_sector_std_L": 6.5,

        "worst_sector": 3

      }

    }

  ],

  "uniformity_metrics": {

    "max_std_L": 8.5,

    "warning_threshold": 10.0,

    "retake_threshold": 12.0

  },

  "sector_statistics": {

    "enabled": true,

    "num_sectors": 8,

    "max_sector_std_L": 6.5,

    "worst_zone": "B"

  }

}

```



---



#### `POST /recompute`



파라미터 재계산 (이미지 재업로드 불필요) ⭐ PHASE7 신규



**목적**:

- 같은 이미지에 대해 다른 파라미터로 재분석

- 파라미터 튜닝 워크플로우 최적화 (30× 속도 향상)



**Request**:

```http

POST /recompute HTTP/1.1

Host: localhost:8000

Content-Type: application/x-www-form-urlencoded



image_id=abc123def456&sku=SKU001&params={"smoothing_window":15,"min_gradient":3.0}

```



**Parameters**:

| 이름 | 타입 | 필수 | 설명 |

|------|------|------|------|

| `image_id` | string | ✅ | `/inspect` 응답의 `image_id` |

| `sku` | string | ✅ | SKU 코드 |

| `params` | JSON string | ✅ | 조정할 파라미터 (아래 참조) |

| `run_judgment` | boolean | ❌ | 판정 실행 여부 |



**지원 파라미터** (12개):

```json
{
  "detection_method": "gradient",      // gradient, delta_e, hybrid, variable_width
  "smoothing_window": 15,              // 1-100
  "min_gradient": 3.0,                 // 0.0-10.0
  "min_delta_e": 3.0,                  // 0.0-20.0
  "expected_zones": 3,                 // 1-20
  "uniform_split_priority": false,     // true/false
  "num_samples": 360,                  // 100-10000 (theta samples)
  "num_points": 300,                   // 50-1000 (radial points)
  "sample_percentile": 50,             // 0-100 (when set, use percentile instead of mean)
  "correction_method": "auto",         // gray_world, white_patch, auto, polynomial, gaussian, none
  "sector_count": 8,                   // 4-36
  "ring_count": 3                      // 1-10
}
```



**Response** (200 OK):

- `/inspect`와 동일한 응답 형식

- 새로운 파라미터로 재계산된 결과

- `applied_params`: 실제 적용된 파라미터 (요청에서 전달된 값만 포함)



**Error Responses**:

- `404 Not Found`: `image_id`가 캐시에 없음 (TTL 15분)

- `400 Bad Request`: 잘못된 파라미터



**사용 사례**:

```python

# 1. 이미지 업로드

resp1 = requests.post("/inspect", files={"file": img}, data={"sku": "SKU001"})

image_id = resp1.json()["image_id"]



# 2. 파라미터 튜닝 (즉시 실행)

resp2 = requests.post("/recompute", data={

    "image_id": image_id,

    "sku": "SKU001",

    "params": json.dumps({"smoothing_window": 20, "min_gradient": 2.5, "sample_percentile": 50})

})



# 3. 다른 파라미터로 다시 테스트

resp3 = requests.post("/recompute", data={

    "image_id": image_id,

    "sku": "SKU001",

    "params": json.dumps({"expected_zones": 2})

})

```



---



### 3.4 Batch Processing



#### `POST /batch`



배치 이미지 검사



**Request (ZIP 파일)**:

```http

POST /batch HTTP/1.1

Host: localhost:8000

Content-Type: multipart/form-data



--boundary

Content-Disposition: form-data; name="zip_file"; filename="batch.zip"

Content-Type: application/zip



[ZIP 바이너리 데이터]

--boundary

Content-Disposition: form-data; name="sku"



SKU001

--boundary--

```



**Request (서버 경로)**:

```http

POST /batch HTTP/1.1

Host: localhost:8000

Content-Type: application/json



{

  "path": "/data/raw_images/batch_001/",

  "sku": "SKU001",

  "run_judgment": true

}

```



**Parameters**:

| 이름 | 타입 | 필수 | 설명 |

|------|------|------|------|

| `zip_file` | File | ❌ | ZIP 파일 (또는 `path` 사용) |

| `path` | string | ❌ | 서버 상의 이미지 폴더 경로 |

| `sku` | string | ✅ | SKU 코드 |

| `run_judgment` | boolean | ❌ | 판정 실행 여부 |



**Response** (200 OK):

```json

{

  "status": "success",

  "run_id": "batch_20251215_103045",

  "total_images": 50,

  "processed": 50,

  "failed": 0,

  "summary": {

    "ok_count": 42,

    "ng_count": 5,

    "retake_count": 3,

    "ok_with_warning_count": 0,

    "avg_delta_e": 2.8,

    "max_delta_e": 5.3,

    "avg_confidence": 0.87

  },

  "results": [

    {

      "filename": "lens_001.jpg",

      "judgment": "OK",

      "overall_delta_e": 2.1,

      "confidence": 0.92

    },

    {

      "filename": "lens_002.jpg",

      "judgment": "NG",

      "overall_delta_e": 5.3,

      "confidence": 0.88,

      "ng_reasons": ["Zone_A"]

    }

  ],

  "csv_url": "/results/batch_20251215_103045/summary.csv"

}

```



**Error Responses**:

- `400 Bad Request`: ZIP 파일과 경로 모두 누락

- `404 Not Found`: 경로가 존재하지 않음



---



### 3.5 Comparison & Analytics



#### `POST /compare`



로트 비교 분석 (Golden Sample vs Test Images) ⭐ PHASE7 신규



**목적**:

- 기준 이미지(Golden Sample) 대비 테스트 이미지들의 색상 편차 분석

- 로트 전체의 일관성 평가

- 이상치(Outlier) 감지



**Request**:

```http

POST /compare HTTP/1.1

Host: localhost:8000

Content-Type: multipart/form-data



--boundary

Content-Disposition: form-data; name="reference_file"; filename="golden.jpg"

Content-Type: image/jpeg



[기준 이미지]

--boundary

Content-Disposition: form-data; name="test_files"; filename="lot_001.jpg"

Content-Type: image/jpeg



[테스트 이미지 1]

--boundary

Content-Disposition: form-data; name="test_files"; filename="lot_002.jpg"

Content-Type: image/jpeg



[테스트 이미지 2]

--boundary

Content-Disposition: form-data; name="sku"



SKU001

--boundary--

```



**Parameters**:

| 이름 | 타입 | 필수 | 설명 |

|------|------|------|------|

| `reference_file` | File | ✅ | 기준 이미지 (Golden Sample) |

| `test_files` | File[] | ✅ | 테스트 이미지 배열 (1~100개) |

| `sku` | string | ✅ | SKU 코드 |



**Response** (200 OK):

```json

{

  "status": "success",

  "reference": {

    "filename": "golden.jpg",

    "zones": [

      {

        "name": "Zone_C",

        "mean_L": 72.2,

        "mean_a": 137.3,

        "mean_b": 122.8

      }

    ]

  },

  "tests": [

    {

      "filename": "lot_001.jpg",

      "zone_deltas": [

        {

          "zone": "Zone_C",

          "delta_L": -0.5,

          "delta_a": 0.3,

          "delta_b": 0.2,

          "delta_e": 0.6

        }


      ],
      "ink_deltas": {
        "ink1": {
          "zones": ["Zone_C"],
          "mean_delta_e": 0.6,
          "max_delta_e": 0.6,
          "mean_delta_L": -0.5,
          "mean_delta_a": 0.3,
          "mean_delta_b": 0.2
        }
      },
      "ink_flags": [
        {
          "ink": "ink1",
          "metric": "max_delta_e",
          "value": 0.6,
          "threshold": 0.8
        }
      ],


      "overall_shift": "Slightly darker",

      "max_delta_e": 0.6

    },

    {

      "filename": "lot_002.jpg",

      "zone_deltas": [

        {

          "zone": "Zone_C",

          "delta_L": -2.3,

          "delta_a": 1.5,

          "delta_b": 0.8,

          "delta_e": 2.8

        }

      ],

      "overall_shift": "Darker and more yellow",

      "max_delta_e": 2.8

    }

  ],

  "ink_mapping": {
    "Zone_C": "ink1"
  },
  "ink_thresholds": {
    "ink1": {
      "max_delta_e": 0.8
    }
  },
  "batch_summary": {

    "mean_delta_e_per_zone": {

      "Zone_C": 1.7

    },

    "max_delta_e_per_zone": {

      "Zone_C": 2.8

    },

    "std_delta_e_per_zone": {

      "Zone_C": 1.1

    },

    "stability_score": 0.83,

    "outliers": ["lot_002.jpg"]

  }

}

```



**Response Fields**:

| 필드 | 설명 |

|------|------|

| `overall_shift` | 색상 변화 방향 (Darker, Lighter, More red, etc.) |

| `stability_score` | 로트 일관성 점수 (0.0~1.0, 높을수록 좋음) |

| `outliers` | 이상치로 판단된 이미지 파일명 배열 |



**Stability Score 계산**:

```python

stability_score = 1.0 - min(mean_delta_e / 10.0, 1.0)

```



**Outlier 기준**:

```python

threshold = mean + 2.0 * std

is_outlier = (delta_e > threshold)

```



**사용 사례**:

```python

# 로트 QC (Quality Control)

response = requests.post("/compare", files={

    "reference_file": open("golden_sample.jpg", "rb"),

    "test_files": [

        open("lot_001_sample1.jpg", "rb"),

        open("lot_001_sample2.jpg", "rb"),

        # ... 최대 100개

    ]

}, data={"sku": "SKU001"})



# 결과 분석

summary = response.json()["batch_summary"]

if summary["stability_score"] < 0.7:

    print("⚠️ 로트 일관성 낮음")



outliers = summary["outliers"]

if outliers:

    print(f"이상치 감지: {outliers}")

```



---



### 3.6 Results Management



#### `GET /results/{run_id}`



배치 검사 결과 조회



**Request**:

```http

GET /results/batch_20251215_103045 HTTP/1.1

Host: localhost:8000

```



**Response** (200 OK):

```json

{

  "run_id": "batch_20251215_103045",

  "timestamp": "2025-12-15T10:30:45Z",

  "sku": "SKU001",

  "total_images": 50,

  "summary": {...},

  "results": [...]

}

```



**Error Responses**:

- `404 Not Found`: `run_id`가 존재하지 않음



---



#### `GET /results/{run_id}/{filename}`



개별 결과 파일 다운로드



**Request**:

```http

GET /results/batch_20251215_103045/summary.csv HTTP/1.1

Host: localhost:8000

```



**Response** (200 OK):

- Content-Type: `text/csv` 또는 `image/png`

- 파일 다운로드



**지원 파일**:

- `summary.csv`: 배치 요약 CSV

- `lens_001_result.json`: 개별 결과 JSON

- `lens_001_overlay.png`: 오버레이 이미지



---



## 4. 데이터 스키마



### 4.1 InspectionResult



```typescript

interface InspectionResult {

  status: "success" | "error";

  image_id?: string;              // 이미지 캐시 ID (재계산용)

  session_id?: string;             // 세션 ID

  judgment: "OK" | "OK_WITH_WARNING" | "NG" | "RETAKE";

  overall_delta_e: number;

  confidence: number;              // 0.0~1.0

  zones: ZoneResult[];

  decision_trace: DecisionTrace;

  next_actions: string[];

  confidence_breakdown: ConfidenceBreakdown;

  ink_analysis: InkAnalysis;

  profile_data?: ProfileData;

  overlay_url?: string;

  risk_factors?: RiskFactor[];     // /inspect_v2에만

}

```



### 4.2 ZoneResult



```typescript

interface ZoneResult {

  zone_name: string;               // "Zone_C", "Zone_B", etc.

  measured: LabColor;

  expected: LabColor;

  delta_e: number;

  threshold: number;

  passed: boolean;

  std_L: number;                   // 균일도 (표준편차)

  std_a?: number;

  std_b?: number;

  pixel_count: number;

  rgb?: [number, number, number];

  hex?: string;

}

```



### 4.3 InkAnalysis



```typescript

interface InkAnalysis {

  zone_based: {

    ink_count: number;

    inks: ZoneBasedInk[];

  };

  image_based: {

    ink_count: number;

    inks: ImageBasedInk[];

    meta: {

      bic: number;

      correction_applied: boolean;

      sampling_config: SamplingConfig;

    };

  };

}



interface ZoneBasedInk {

  zone_name: string;

  measured: LabColor;

  rgb: [number, number, number];

  hex: string;

}



interface ImageBasedInk {

  L: number;

  a: number;

  b: number;

  weight: number;                  // 픽셀 비율 (0.0~1.0)

  rgb: [number, number, number];

  hex: string;

}



interface SamplingConfig {

  chroma_threshold: number;

  L_max: number;

  highlight_removed: boolean;

  candidate_pixels: number;

  sampled_pixels: number;

  sampling_ratio: number;          // 0.0~1.0

}

```



### 4.4 DecisionTrace



```typescript

interface DecisionTrace {

  final: "OK" | "OK_WITH_WARNING" | "NG" | "RETAKE";

  because: string;                 // 판정 이유

  overrides?: string;              // 오버라이드 정보

  zone_checks: {

    [zone: string]: {

      delta_e: number;

      threshold: number;

      passed: boolean;

    };

  };

  uniformity_check?: {

    max_std_l: number;

    warning_threshold: number;

    retake_threshold: number;

    status: "ok" | "warning" | "retake";

  };

}

```



### 4.5 ConfidenceBreakdown



```typescript

interface ConfidenceBreakdown {

  pixel_count_score: number;       // 0.0~1.0

  transition_score: number;

  std_score: number;

  sector_uniformity: number;

  lens_detection: number;

  overall: number;

}

```



### 4.6 RiskFactor



```typescript

interface RiskFactor {

  category: "delta_e_exceeded" | "sector_uniformity" | "uniformity_low" | "boundary_unclear" | "coverage_low";

  severity: "low" | "medium" | "high";

  message: string;

  details: {

    [key: string]: any;

  };

}

```



---



## 5. 에러 처리



### 5.1 에러 응답 형식



```json

{

  "status": "error",

  "error_code": "SKU_NOT_FOUND",

  "message": "SKU 'SKU999' not found in database",

  "details": {

    "sku": "SKU999",

    "available_skus": ["SKU001", "SKU002"]

  }

}

```



### 5.2 에러 코드 목록



| 코드 | HTTP Status | 설명 |

|------|-------------|------|

| `FILE_NOT_PROVIDED` | 400 | 파일이 업로드되지 않음 |

| `SKU_NOT_PROVIDED` | 400 | SKU 파라미터 누락 |

| `INVALID_FILE_FORMAT` | 400 | 지원하지 않는 파일 형식 |

| `FILE_TOO_LARGE` | 400 | 파일 크기 초과 (>50MB) |

| `SKU_NOT_FOUND` | 404 | SKU 파일 없음 |

| `IMAGE_ID_NOT_FOUND` | 404 | 캐시된 이미지 없음 (TTL 만료) |

| `RUN_ID_NOT_FOUND` | 404 | 배치 결과 없음 |

| `PIPELINE_ERROR` | 500 | 검사 파이프라인 실행 실패 |

| `LENS_NOT_DETECTED` | 500 | 렌즈 검출 실패 |

| `ZONE_DETECTION_FAILED` | 500 | Zone 분할 실패 |



### 5.3 에러 처리 예제



```python

response = requests.post("/inspect", files={...}, data={...})



if response.status_code != 200:

    error = response.json()

    error_code = error.get("error_code")



    if error_code == "SKU_NOT_FOUND":

        print(f"SKU를 찾을 수 없습니다: {error['details']['sku']}")

        print(f"사용 가능한 SKU: {error['details']['available_skus']}")

    elif error_code == "LENS_NOT_DETECTED":

        print("렌즈 검출 실패. 이미지 품질을 확인하세요.")

    else:

        print(f"에러 발생: {error['message']}")

```



---



## 6. 예제 코드



### 6.1 Python (requests)



#### 단일 이미지 검사



```python

import requests



url = "http://localhost:8000/inspect"



with open("lens_image.jpg", "rb") as f:

    response = requests.post(

        url,

        files={"file": f},

        data={"sku": "SKU001", "run_judgment": "true"}

    )



if response.status_code == 200:

    result = response.json()

    print(f"Judgment: {result['judgment']}")

    print(f"ΔE: {result['overall_delta_e']:.2f}")

    print(f"Confidence: {result['confidence']:.2f}")



    # 잉크 분석

    zone_count = result["ink_analysis"]["zone_based"]["ink_count"]

    image_count = result["ink_analysis"]["image_based"]["ink_count"]

    print(f"Zone-based: {zone_count}개, Image-based: {image_count}개")



    if zone_count != image_count:

        print("⚠️ 잉크 개수 불일치 - SKU 설정 검토 필요")

else:

    print(f"Error: {response.json()['message']}")

```



#### 파라미터 재계산



```python

# 1. 초기 검사

resp1 = requests.post("/inspect", files={"file": img}, data={"sku": "SKU001"})

image_id = resp1.json()["image_id"]



# 2. 파라미터 튜닝

params = {

    "smoothing_window": 20,

    "min_gradient": 2.5,

    "expected_zones": 2

}



resp2 = requests.post("/recompute", data={

    "image_id": image_id,

    "sku": "SKU001",

    "params": json.dumps(params)

})



result2 = resp2.json()

print(f"재계산 결과: {result2['judgment']}")

```



#### 배치 검사



```python

import zipfile

from pathlib import Path



# ZIP 파일 생성

with zipfile.ZipFile("batch.zip", "w") as zf:

    for img_path in Path("images/").glob("*.jpg"):

        zf.write(img_path, img_path.name)



# 배치 검사

with open("batch.zip", "rb") as f:

    response = requests.post(

        "http://localhost:8000/batch",

        files={"zip_file": f},

        data={"sku": "SKU001"}

    )



result = response.json()

print(f"처리: {result['processed']}/{result['total_images']}")

print(f"OK: {result['summary']['ok_count']}")

print(f"NG: {result['summary']['ng_count']}")

print(f"RETAKE: {result['summary']['retake_count']}")

```



#### 로트 비교



```python

files = {

    "reference_file": open("golden.jpg", "rb"),

    "test_files": [

        open("lot1.jpg", "rb"),

        open("lot2.jpg", "rb"),

        open("lot3.jpg", "rb")

    ]

}



response = requests.post(

    "http://localhost:8000/compare",

    files=files,

    data={"sku": "SKU001"}

)



result = response.json()

stability = result["batch_summary"]["stability_score"]



if stability < 0.7:

    print("⚠️ 로트 일관성 낮음")

    outliers = result["batch_summary"]["outliers"]

    print(f"이상치: {outliers}")

else:

    print(f"✅ 로트 일관성 양호 ({stability:.2f})")

```



---



### 6.2 JavaScript (fetch)



```javascript

// 단일 이미지 검사

async function inspectImage(file, sku) {

  const formData = new FormData();

  formData.append("file", file);

  formData.append("sku", sku);

  formData.append("run_judgment", "true");



  const response = await fetch("http://localhost:8000/inspect", {

    method: "POST",

    body: formData

  });



  const result = await response.json();

  console.log(`Judgment: ${result.judgment}`);

  console.log(`ΔE: ${result.overall_delta_e.toFixed(2)}`);



  // 잉크 분석

  const zoneBased = result.ink_analysis.zone_based.ink_count;

  const imageBased = result.ink_analysis.image_based.ink_count;

  console.log(`Zone-based: ${zoneBased}, Image-based: ${imageBased}`);



  return result;

}



// 파라미터 재계산

async function recompute(imageId, sku, params) {

  const formData = new URLSearchParams();

  formData.append("image_id", imageId);

  formData.append("sku", sku);

  formData.append("params", JSON.stringify(params));



  const response = await fetch("http://localhost:8000/recompute", {

    method: "POST",

    headers: {"Content-Type": "application/x-www-form-urlencoded"},

    body: formData

  });



  return await response.json();

}



// 사용 예시

const fileInput = document.getElementById("fileInput");

const file = fileInput.files[0];



const result1 = await inspectImage(file, "SKU001");

const imageId = result1.image_id;



// 파라미터 조정 후 재계산

const result2 = await recompute(imageId, "SKU001", {

  smoothing_window: 20,

  min_gradient: 2.5

});

```



---



### 6.3 cURL



```bash

# 단일 이미지 검사

curl -X POST "http://localhost:8000/inspect" \

  -F "file=@lens.jpg" \

  -F "sku=SKU001" \

  -F "run_judgment=true"



# 파라미터 재계산

curl -X POST "http://localhost:8000/recompute" \

  -d "image_id=abc123" \

  -d "sku=SKU001" \

  -d 'params={"smoothing_window":20,"min_gradient":2.5}'



# 배치 검사 (ZIP)

curl -X POST "http://localhost:8000/batch" \

  -F "zip_file=@batch.zip" \

  -F "sku=SKU001"



# 로트 비교

curl -X POST "http://localhost:8000/compare" \

  -F "reference_file=@golden.jpg" \

  -F "test_files=@lot1.jpg" \

  -F "test_files=@lot2.jpg" \

  -F "sku=SKU001"



# 헬스 체크

curl "http://localhost:8000/health"

```



---



## 7. 부록



### 7.1 Rate Limiting



**현재**: 없음

**향후 계획**: 분당 60 requests (개발 환경), 분당 600 requests (프로덕션)



### 7.2 CORS 설정



**현재**: 모든 origin 허용 (`allow_origins=["*"]`)

**프로덕션 권장**:

```python

app.add_middleware(

    CORSMiddleware,

    allow_origins=["https://yourdomain.com"],

    allow_methods=["POST", "GET"],

    allow_headers=["*"]

)

```



### 7.3 파일 업로드 제한



- 최대 파일 크기: **50MB**

- 지원 형식: **JPG, JPEG, PNG**

- 동시 업로드: **최대 100개** (배치)



### 7.4 캐시 정책



- 이미지 캐시 (재계산용): **TTL 15분**

- 배치 결과: **24시간 보관**



---



## 8. 변경 이력



| 날짜 | 버전 | 변경 내용 |

|------|------|----------|

| 2025-12-13 | 0.1 | 초기 API (6개 endpoints) |

| 2025-12-14 | 0.2 | `/recompute`, `/compare` 추가 (PHASE7) |

| 2025-12-14 | 0.3 | 잉크 분석 결과 포함 |

| 2025-12-15 | 1.0 | 문서 작성 완료 |



---



**문의**: 프로젝트 개발팀 또는 GitHub Issues

**마지막 검토**: 2025-12-15
