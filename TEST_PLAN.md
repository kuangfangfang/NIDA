# NetGuard AI — Test Plan

## Quick Start

```bash
cd project/claude
pip install flask flask-cors numpy pandas scikit-learn xgboost
python run_tests.py
```

A `test_report.html` will be generated and auto-opened in your browser.

---

## Test Suites Overview

| # | Suite | Class | Tests | Dependencies |
|---|-------|-------|-------|-------------|
| 1 | **Unit Tests** | `TestPreprocessCSV` | 6 | None |
| 2 | **Unit Tests** | `TestRunDetection` | 3 | None |
| 3 | **Integration** | `TestDetectEndpoint` | 6 | Flask test client |
| 4 | **Input Validation** | `TestInputValidation` | 6 | Flask test client |
| 5 | **Model Performance** | `TestModelPerformance` | 2 | `test_data_labelled.csv` + models |
| 6 | **Frontend Smoke** | `TestFrontendSmoke` | 2 | Selenium + ChromeDriver |

---

## Detailed Test Cases

### 1. Unit Tests — `preprocess_csv()`

| Test | Input | Expected | Notes |
|------|-------|----------|-------|
| `test_valid_csv_returns_correct_columns` | 78-col valid CSV | DataFrame with 78 columns | — |
| `test_partial_csv_pads_missing_columns` | 20-col valid CSV | Padded to 78, missing = 0 | — |
| `test_label_column_dropped` | CSV with "Label" col | Label not in output | — |
| `test_constant_csv_raises` | All values = 1 | `ValueError` (variance check) | — |
| `test_unrelated_csv_raises` | "name,age,city" CSV | `ValueError` (column overlap) | — |
| `test_empty_csv_raises` | Header-only CSV | `ValueError` (0 rows) | — |

### 2. Unit Tests — `run_detection()`

| Test | Input | Expected |
|------|-------|----------|
| `test_returns_required_fields` | Valid DataFrame | Dict with `is_attack`, `total_flows`, etc. |
| `test_demo_mode_flag` | Valid DF, no models | `demo_mode=True` |
| `test_attack_info_structure` | Valid DF | `attack_info` has `label`, `summary`, `suggestions` |

### 3. Integration Tests — `/detect` Endpoint

| Test | Request | Expected HTTP |
|------|---------|--------------|
| `test_health` | `GET /health` | 200 |
| `test_upload_valid_csv` | POST valid CSV | 200, `is_attack` in response |
| `test_no_file_returns_400` | POST empty | 400 |
| `test_unsupported_filetype_returns_400` | POST `.txt` | 400 |
| `test_unrelated_csv_returns_400` | POST random CSV | 400 |
| `test_ddos_csv_detected` | POST DDoS-like CSV | 200 (accepted as valid format) |

### 4. Input Validation Tests

| Test | Input | Expected |
|------|-------|----------|
| `test_empty_file` | 0 bytes | 400 |
| `test_constant_csv_rejected` | All constant values | 400 |
| `test_negative_flow_duration` | Flow Duration = -100 | 400 |
| `test_port_out_of_range` | Port = 99999 | 400 |
| `test_zero_flow_bytes_rejected` | Flow Bytes/s = 0 | 400 |
| `test_few_columns_csv_rejected` | Only 3 CICIDS columns | 400 |

### 5. Model Performance (Optional)

To enable these tests, place a file named `test_data_labelled.csv` in the `claude/` directory. This CSV should:
- Contain all 78 CICIDS2017 feature columns
- Include a `Label` column (values like `BENIGN`, `DDoS`, `PortScan`, etc.)
- Contain at least 100 rows

If the file is not found, these tests are **automatically skipped**.

### 6. Frontend Smoke (Optional)

Requires `selenium` and ChromeDriver:

```bash
pip install selenium
```

If not installed, these tests are **automatically skipped**.

---

## Demo Mode vs Real Mode

When model files (`models/binary_model.pkl`, `models/multiclass_model.pkl`) are **not found**:

- The app enters **demo mode** with random predictions
- `test_demo_mode_flag` will PASS
- Detection results will include `"demo_mode": true`
- Model performance tests will be SKIPPED
- All validation tests work identically (validation happens before model inference)

---

## Reading the Report

The HTML report (`test_report.html`) shows:

- **Summary cards**: pass/fail/error/skip counts at a glance
- **Pass rate bar**: green (≥80%), orange (≥50%), red (<50%)
- **Per-test rows**: status badge, full test name, duration, and failure details

Exit code: `0` if all tests pass, `1` if any fail.

---

## Interpreting Failures

| Status | Meaning | Action |
|--------|---------|--------|
| **PASS** | Test met expectations | ✓ |
| **FAIL** | Assertion failed | Check the failure detail in the report |
| **ERROR** | Unexpected exception | Likely a code bug or missing dependency |
| **SKIP** | Precondition not met | See skip reason (e.g., no models, no Selenium) |
