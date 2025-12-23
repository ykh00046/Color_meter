# Inspection History System Guide

**Version**: 1.0
**Date**: 2025-12-19
**Status**: Production Ready

---

## 📋 Overview

The Inspection History System provides comprehensive tracking and analysis of all lens inspection operations. It automatically stores inspection results in a database and provides both API and Web UI access to historical data.

### Key Features

- **Automatic Storage**: Every inspection result is automatically saved to the database
- **Rich Query API**: Filter and search inspection history by multiple criteria
- **Statistical Analysis**: Built-in statistics and trend analysis
- **Web UI Viewer**: Modern, responsive interface for browsing history
- **Pagination Support**: Efficient handling of large datasets
- **Export Ready**: Full analysis results stored in JSON format

---

## 🗄️ Database Schema

### InspectionHistory Table

| Field | Type | Description |
|-------|------|-------------|
| `id` | Integer | Primary key (auto-increment) |
| `session_id` | String(100) | Unique session ID from /inspect endpoint |
| `sku_code` | String(50) | SKU code |
| `image_filename` | String(500) | Original filename |
| `image_path` | String(1000) | Full path to saved image |
| `judgment` | Enum | OK, OK_WITH_WARNING, NG, RETAKE |
| `overall_delta_e` | Float | Overall color difference |
| `confidence` | Float | Confidence score (0.0 - 1.0) |
| `zones_count` | Integer | Number of zones analyzed |
| `analysis_result` | JSON | Complete analysis data |
| `ng_reasons` | JSON | List of NG reasons |
| `retake_reasons` | JSON | List of RETAKE reasons |
| `decision_trace` | JSON | Decision-making trace |
| `next_actions` | JSON | Recommended actions |
| `lens_detected` | Integer | 1 if lens detected, 0 otherwise |
| `lens_confidence` | Float | Lens detection confidence |
| `created_at` | DateTime | Timestamp |
| `operator` | String(100) | Operator name (optional) |
| `notes` | Text | Additional notes |
| `processing_time_ms` | Integer | Processing time |
| `has_warnings` | Integer | Flag for warnings |
| `has_ink_analysis` | Integer | Flag for ink analysis |
| `has_radial_profile` | Integer | Flag for radial profile |

### Indices

- `idx_inspection_sku_judgment` (sku_code, judgment)
- `idx_inspection_created` (created_at)
- `idx_inspection_delta_e` (overall_delta_e)
- `idx_inspection_confidence` (confidence)

---

## 🚀 API Reference

### Base URL

```
http://localhost:8000/api/inspection
```

### 1. List Inspection History

**GET** `/history`

Query Parameters:
- `skip` (int, default=0): Number of records to skip
- `limit` (int, default=50, max=200): Maximum records to return
- `sku_code` (string, optional): Filter by SKU
- `judgment` (string, optional): Filter by judgment (OK, OK_WITH_WARNING, NG, RETAKE)
- `min_delta_e` (float, optional): Minimum ΔE threshold
- `max_delta_e` (float, optional): Maximum ΔE threshold
- `start_date` (datetime, optional): Start date (ISO 8601)
- `end_date` (datetime, optional): End date (ISO 8601)
- `needs_action_only` (bool, default=false): Only show NG/RETAKE results

**Response:**
```json
{
  "total": 123,
  "skip": 0,
  "limit": 50,
  "results": [
    {
      "id": 1,
      "session_id": "abc123",
      "sku_code": "SKU001",
      "image_filename": "lens_001.jpg",
      "judgment": "OK",
      "overall_delta_e": 2.5,
      "confidence": 0.95,
      "zones_count": 3,
      "created_at": "2025-12-19T10:30:00",
      ...
    }
  ]
}
```

**Example:**
```bash
curl "http://localhost:8000/api/inspection/history?sku_code=SKU001&judgment=NG&limit=10"
```

---

### 2. Get Inspection by ID

**GET** `/history/{history_id}`

Query Parameters:
- `include_full_result` (bool, default=false): Include complete analysis_result JSON

**Response:**
```json
{
  "id": 1,
  "session_id": "abc123",
  "sku_code": "SKU001",
  "judgment": "OK",
  "overall_delta_e": 2.5,
  "confidence": 0.95,
  ...
}
```

**Example:**
```bash
curl "http://localhost:8000/api/inspection/history/1?include_full_result=true"
```

---

### 3. Get Inspection by Session ID

**GET** `/history/session/{session_id}`

Query Parameters:
- `include_full_result` (bool, default=false): Include complete analysis_result JSON

**Response:** Same as Get by ID

**Example:**
```bash
curl "http://localhost:8000/api/inspection/history/session/abc123"
```

---

### 4. Delete Inspection

**DELETE** `/history/{history_id}`

**Response:**
```json
{
  "message": "Inspection history 1 deleted successfully"
}
```

**Example:**
```bash
curl -X DELETE "http://localhost:8000/api/inspection/history/1"
```

---

### 5. Get Statistics Summary

**GET** `/history/stats/summary`

Query Parameters:
- `sku_code` (string, optional): Filter by SKU
- `days` (int, default=7, max=365): Number of days to include

**Response:**
```json
{
  "total_inspections": 123,
  "judgment_counts": {
    "OK": 100,
    "OK_WITH_WARNING": 10,
    "NG": 8,
    "RETAKE": 5
  },
  "pass_rate": 0.8943,
  "avg_delta_e": 3.2,
  "avg_confidence": 0.85,
  "time_range": {
    "start": "2025-12-12T00:00:00",
    "end": "2025-12-19T00:00:00",
    "days": 7
  }
}
```

**Example:**
```bash
curl "http://localhost:8000/api/inspection/history/stats/summary?days=30"
```

---

### 6. Get Statistics by SKU

**GET** `/history/stats/by-sku`

Query Parameters:
- `days` (int, default=7, max=365): Number of days to include

**Response:**
```json
{
  "time_range": {
    "start": "2025-12-12T00:00:00",
    "end": "2025-12-19T00:00:00",
    "days": 7
  },
  "stats_by_sku": {
    "SKU001": {
      "total": 50,
      "pass_count": 45,
      "pass_rate": 0.9000,
      "avg_delta_e": 2.8,
      "avg_confidence": 0.92
    },
    "SKU002": {
      "total": 30,
      "pass_count": 25,
      "pass_rate": 0.8333,
      "avg_delta_e": 3.5,
      "avg_confidence": 0.88
    }
  }
}
```

**Example:**
```bash
curl "http://localhost:8000/api/inspection/history/stats/by-sku?days=30"
```

---

## 🖥️ Web UI

### Access History Viewer

**URL:** `http://localhost:8000/history`

### Features

1. **Statistics Dashboard**
   - Total Inspections
   - Pass Rate
   - Average ΔE
   - Average Confidence

2. **Filters**
   - SKU Code
   - Judgment Type (OK, OK_WITH_WARNING, NG, RETAKE)
   - Time Range (Last 24 hours, 7 days, 30 days, 90 days)

3. **Results Table**
   - Sortable columns
   - Pagination (20 records per page)
   - Clickable rows for detailed view

4. **Detail Modal**
   - Full inspection details
   - NG/RETAKE reasons
   - Next actions
   - Complete analysis result (JSON)

### Screenshots

```
┌──────────────────────────────────────────────────────────┐
│  Inspection History                    🏠 Back to Home  │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌─────┤
│  │ Total: 123 │  │ Pass: 89%  │  │ Avg ΔE: 3.2│  │ Avg │
│  └────────────┘  └────────────┘  └────────────┘  └─────┤
│                                                          │
│  Filters: [SKU] [Judgment] [Time Range] [Apply]        │
│                                                          │
│  ┌──────────────────────────────────────────────────────┤
│  │ ID │ Date/Time │ SKU │ Image │ Judgment │ ΔE │ Conf │
│  ├────┼───────────┼─────┼───────┼──────────┼────┼──────┤
│  │  1 │ 12/19 ... │ 001 │ lens  │    OK    │2.5 │ 95%  │
│  │  2 │ 12/19 ... │ 002 │ test  │    NG    │8.2 │ 70%  │
│  └──────────────────────────────────────────────────────┤
│                                                          │
│  [◀ Previous]         Page 1          [Next ▶]         │
└──────────────────────────────────────────────────────────┘
```

---

## 🔧 Integration

### Automatic Save on Inspection

Every time you call `/inspect` endpoint, the result is automatically saved to the database.

```python
# In src/web/app.py (already implemented)

@app.post("/inspect")
async def inspect_image(...):
    # ... run inspection ...

    # Automatic save to database
    save_inspection_to_db(
        session_id=run_id,
        sku_code=sku,
        image_filename=original_name,
        image_path=str(input_path),
        result=result,
    )

    return response
```

### Programmatic Access

```python
from src.models.database import get_db
from src.models import InspectionHistory, JudgmentType

# Get database session
db = get_db()

try:
    # Query all NG results
    ng_results = db.query(InspectionHistory).filter(
        InspectionHistory.judgment == JudgmentType.NG
    ).all()

    for result in ng_results:
        print(f"{result.session_id}: {result.get_summary()}")

    # Query by SKU
    sku_results = db.query(InspectionHistory).filter(
        InspectionHistory.sku_code == "SKU001"
    ).order_by(InspectionHistory.created_at.desc()).limit(10).all()

finally:
    db.close()
```

---

## 📊 Use Cases

### 1. Quality Monitoring

Track pass rate over time:

```bash
curl "http://localhost:8000/api/inspection/history/stats/summary?days=30"
```

### 2. Root Cause Analysis

Find all failures for a specific SKU:

```bash
curl "http://localhost:8000/api/inspection/history?sku_code=SKU001&judgment=NG"
```

### 3. Operator Performance

Track inspections by operator (requires operator field to be set):

```python
# When calling /inspect, add operator parameter (future enhancement)
```

### 4. Trend Analysis

Export data for analysis:

```python
import pandas as pd
import requests

response = requests.get("http://localhost:8000/api/inspection/history?limit=1000")
data = response.json()

df = pd.DataFrame(data["results"])
df["created_at"] = pd.to_datetime(df["created_at"])

# Analyze trends
daily_pass_rate = df.groupby(df["created_at"].dt.date)["is_ok"].mean()
print(daily_pass_rate)
```

---

## 🧪 Testing

Run inspection history tests:

```bash
pytest tests/test_inspection_history.py -v
```

**Expected Output:**
```
test_save_inspection_to_history PASSED
test_inspection_history_to_dict PASSED
test_inspection_history_to_dict_full PASSED
test_inspection_history_judgment_types PASSED
test_query_by_sku PASSED
test_query_by_judgment PASSED
test_get_summary PASSED

7 passed in 4.72s
```

---

## 🔒 Data Retention

By default, inspection history is stored indefinitely. You can implement cleanup policies:

```python
from datetime import datetime, timedelta
from src.models.database import get_db
from src.models import InspectionHistory

def cleanup_old_records(days=90):
    """Delete records older than specified days"""
    db = get_db()
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        deleted = db.query(InspectionHistory).filter(
            InspectionHistory.created_at < cutoff_date
        ).delete()
        db.commit()
        print(f"Deleted {deleted} old records")
    finally:
        db.close()
```

---

## 🚀 Future Enhancements

- **Batch Export**: Export filtered results to CSV/Excel
- **Advanced Filtering**: More filter options (operator, notes, etc.)
- **Charts & Visualizations**: Interactive charts in Web UI
- **Alerts & Notifications**: Automatic alerts on quality issues
- **Operator Tracking**: Enhanced operator management
- **Comparison View**: Side-by-side comparison of results

---

## 📝 Changelog

### Version 1.0 (2025-12-19)

- Initial release
- Database schema and models
- 6 REST API endpoints
- Web UI history viewer
- Comprehensive test suite
- Automatic save integration with /inspect endpoint

---

## 🙋 Support

For questions or issues:

1. Check this guide
2. Review API examples
3. Run tests to verify system health
4. Check server logs for errors

---

**Author**: Task 4.2 Implementation Team
**License**: MIT
**Last Updated**: 2025-12-19
