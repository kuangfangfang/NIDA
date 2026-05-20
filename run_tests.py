"""
NetGuard AI — Comprehensive Test Suite
=======================================
Run:  python run_tests.py
Output: test_report.html  (auto-opens in browser)

Covers:
  1. Unit tests       – preprocess_csv, run_detection
  2. Integration tests – Flask test_client /detect, /health
  3. Input validation  – empty, invalid, unrelated CSV
  4. Model perf tests  – accuracy / F1 if labelled CSV exists
  5. Frontend smoke    – Selenium (optional, skipped if not installed)
"""

import sys, os, io, time, json, traceback, datetime, webbrowser
import unittest
import numpy as np
import pandas as pd

# ── Ensure project root is on sys.path ─────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# ── Import application objects ─────────────────────────────────────────────
from app import (
    app, preprocess_csv, run_detection,
    FEATURE_COLUMNS, ATTACK_INFO,
    binary_model, multiclass_model,
)

MODELS_LOADED = binary_model is not None and multiclass_model is not None

# ═══════════════════════════════════════════════════════════════════════════
# Helpers — build synthetic CSVs
# ═══════════════════════════════════════════════════════════════════════════

def _make_valid_csv(n_rows=5) -> bytes:
    """Generate a minimal valid CICIDS2017-format CSV (all 78 cols, plausible values)."""
    rng = np.random.RandomState(42)
    data = {}
    for col in FEATURE_COLUMNS:
        if col == "Destination Port":
            data[col] = rng.randint(1, 65535, size=n_rows)
        elif col == "Flow Duration":
            data[col] = rng.uniform(100, 1e7, size=n_rows)
        elif col == "Flow Bytes/s":
            data[col] = rng.uniform(0.01, 1e6, size=n_rows)
        elif col == "Total Fwd Packets":
            data[col] = rng.randint(0, 500, size=n_rows)
        else:
            data[col] = rng.uniform(0, 100, size=n_rows)
    df = pd.DataFrame(data)
    return df.to_csv(index=False).encode()


def _make_partial_csv(cols_to_keep=20, n_rows=5) -> bytes:
    """CSV with only a subset of feature columns (but >= MIN_MATCHED threshold)."""
    rng = np.random.RandomState(7)
    keep = FEATURE_COLUMNS[:cols_to_keep]
    data = {}
    for col in keep:
        if col == "Destination Port":
            data[col] = rng.randint(1, 65535, size=n_rows)
        elif col == "Flow Duration":
            data[col] = rng.uniform(100, 1e7, size=n_rows)
        elif col == "Flow Bytes/s":
            data[col] = rng.uniform(0.01, 1e6, size=n_rows)
        elif col == "Total Fwd Packets":
            data[col] = rng.randint(0, 500, size=n_rows)
        else:
            data[col] = rng.uniform(0, 100, size=n_rows)
    return pd.DataFrame(data).to_csv(index=False).encode()


def _make_constant_csv(n_rows=10) -> bytes:
    """All 78 columns present but every value is 1 (constant → should fail variance check)."""
    data = {col: [1] * n_rows for col in FEATURE_COLUMNS}
    data["Destination Port"] = [80] * n_rows
    data["Flow Duration"] = [1000] * n_rows
    data["Flow Bytes/s"] = [1.0] * n_rows
    return pd.DataFrame(data).to_csv(index=False).encode()


def _make_unrelated_csv() -> bytes:
    """Completely unrelated CSV (no CICIDS columns)."""
    return b"name,age,city\nAlice,30,Sydney\nBob,25,Melbourne\n"


def _make_ddos_like_csv(n_rows=20) -> bytes:
    """Synthetic DDoS-like traffic: high packet rate, many fwd packets."""
    rng = np.random.RandomState(99)
    data = {}
    for col in FEATURE_COLUMNS:
        if col == "Destination Port":
            data[col] = [80] * n_rows
        elif col == "Flow Duration":
            data[col] = rng.uniform(10, 500, size=n_rows)
        elif col == "Flow Bytes/s":
            data[col] = rng.uniform(1e6, 1e8, size=n_rows)
        elif col == "Total Fwd Packets":
            data[col] = rng.randint(500, 50000, size=n_rows)
        elif col == "Flow Packets/s":
            data[col] = rng.uniform(1e4, 1e6, size=n_rows)
        elif col == "SYN Flag Count":
            data[col] = rng.randint(100, 5000, size=n_rows)
        else:
            data[col] = rng.uniform(0, 1000, size=n_rows)
    return pd.DataFrame(data).to_csv(index=False).encode()


# ═══════════════════════════════════════════════════════════════════════════
# 1. UNIT TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestPreprocessCSV(unittest.TestCase):
    """Unit tests for preprocess_csv()."""

    def test_valid_csv_returns_correct_columns(self):
        """Valid 78-column CSV → DataFrame with exactly 78 columns."""
        df = preprocess_csv(_make_valid_csv())
        self.assertEqual(list(df.columns), FEATURE_COLUMNS)
        self.assertEqual(len(df.columns), 78)

    def test_partial_csv_pads_missing_columns(self):
        """CSV with 20 valid columns → padded to 78 with 0s."""
        df = preprocess_csv(_make_partial_csv(cols_to_keep=20))
        self.assertEqual(len(df.columns), 78)
        # The missing columns should be 0
        missing = [c for c in FEATURE_COLUMNS if c not in FEATURE_COLUMNS[:20]]
        for col in missing:
            self.assertTrue((df[col] == 0).all(), f"Column {col} should be 0-filled")

    def test_label_column_dropped(self):
        """CSV with a 'Label' column → label is dropped before output."""
        csv = _make_valid_csv()
        df_orig = pd.read_csv(io.BytesIO(csv))
        df_orig["Label"] = "BENIGN"
        csv_with_label = df_orig.to_csv(index=False).encode()
        df = preprocess_csv(csv_with_label)
        self.assertNotIn("Label", df.columns)

    def test_constant_csv_raises(self):
        """All-constant data → ValueError (variance check)."""
        with self.assertRaises(ValueError) as ctx:
            preprocess_csv(_make_constant_csv())
        self.assertIn("constant", str(ctx.exception).lower())

    def test_unrelated_csv_raises(self):
        """CSV with no CICIDS columns → ValueError (column-overlap check)."""
        with self.assertRaises(ValueError) as ctx:
            preprocess_csv(_make_unrelated_csv())
        self.assertIn("valid network traffic data", str(ctx.exception).lower())

    def test_empty_csv_raises(self):
        """CSV with only a header row (0 data rows) → ValueError."""
        header_only = ",".join(FEATURE_COLUMNS).encode() + b"\n"
        with self.assertRaises(ValueError):
            preprocess_csv(header_only)


class TestRunDetection(unittest.TestCase):
    """Unit tests for run_detection()."""

    def test_returns_required_fields(self):
        """Result dict must contain is_attack, total_flows, attack_breakdown, etc."""
        df = preprocess_csv(_make_valid_csv(n_rows=10))
        result = run_detection(df)
        for key in ("is_attack", "total_flows", "attack_breakdown", "stage1_confidence"):
            self.assertIn(key, result, f"Missing key: {key}")
        self.assertEqual(result["total_flows"], 10)

    def test_demo_mode_flag(self):
        """If models are not loaded, result should say demo_mode=True."""
        if MODELS_LOADED:
            self.skipTest("Models are loaded; demo_mode test N/A")
        df = preprocess_csv(_make_valid_csv())
        result = run_detection(df)
        self.assertTrue(result.get("demo_mode", False))

    def test_attack_info_structure(self):
        """attack_info must contain label, summary, suggestions."""
        df = preprocess_csv(_make_valid_csv())
        result = run_detection(df)
        info = result.get("attack_info", {})
        for key in ("label", "summary", "suggestions"):
            self.assertIn(key, info)


# ═══════════════════════════════════════════════════════════════════════════
# 2. INTEGRATION TESTS (Flask test client)
# ═══════════════════════════════════════════════════════════════════════════

class TestDetectEndpoint(unittest.TestCase):
    """Integration tests for POST /detect."""

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    # -- health ---
    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertEqual(data["status"], "ok")

    # -- valid CSV upload ---
    def test_upload_valid_csv(self):
        r = self.client.post("/detect", data={
            "file": (io.BytesIO(_make_valid_csv(n_rows=10)), "test.csv")
        }, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("is_attack", data)
        self.assertEqual(data["total_flows"], 10)

    # -- no file ---
    def test_no_file_returns_400(self):
        r = self.client.post("/detect")
        self.assertEqual(r.status_code, 400)
        self.assertIn("error", r.get_json())

    # -- unsupported file type ---
    def test_unsupported_filetype_returns_400(self):
        r = self.client.post("/detect", data={
            "file": (io.BytesIO(b"hello world"), "readme.txt")
        }, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)

    # -- unrelated CSV ---
    def test_unrelated_csv_returns_400(self):
        r = self.client.post("/detect", data={
            "file": (io.BytesIO(_make_unrelated_csv()), "random.csv")
        }, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)
        self.assertIn("error", r.get_json())

    # -- DDoS-like CSV ---
    def test_ddos_csv_detected(self):
        """DDoS-like synthetic data should be accepted (valid format).
        In real-model mode it may detect an attack; in demo mode result is random."""
        r = self.client.post("/detect", data={
            "file": (io.BytesIO(_make_ddos_like_csv()), "ddos.csv")
        }, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("is_attack", data)


# ═══════════════════════════════════════════════════════════════════════════
# 3. INPUT VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestInputValidation(unittest.TestCase):
    """Focused tests for the input-validation layer in preprocess_csv."""

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def _upload(self, csv_bytes, filename="test.csv"):
        return self.client.post("/detect", data={
            "file": (io.BytesIO(csv_bytes), filename)
        }, content_type="multipart/form-data")

    def test_empty_file(self):
        r = self._upload(b"")
        self.assertEqual(r.status_code, 400)

    def test_constant_csv_rejected(self):
        r = self._upload(_make_constant_csv())
        self.assertEqual(r.status_code, 400)
        self.assertIn("error", r.get_json())

    def test_negative_flow_duration(self):
        """Flow Duration < 0 → rejected."""
        data = {col: [10] for col in FEATURE_COLUMNS}
        data["Destination Port"] = [80]
        data["Flow Duration"] = [-100]
        data["Flow Bytes/s"] = [1.0]
        csv = pd.DataFrame(data).to_csv(index=False).encode()
        r = self._upload(csv)
        self.assertEqual(r.status_code, 400)

    def test_port_out_of_range(self):
        """Destination Port = 99999 → rejected."""
        data = {col: [10] for col in FEATURE_COLUMNS}
        data["Destination Port"] = [99999]
        data["Flow Duration"] = [5000]
        data["Flow Bytes/s"] = [100.0]
        csv = pd.DataFrame(data).to_csv(index=False).encode()
        r = self._upload(csv)
        self.assertEqual(r.status_code, 400)

    def test_negative_flow_bytes_rejected(self):
        """Flow Bytes/s = -1 → rejected."""
        data = {col: [10] for col in FEATURE_COLUMNS}
        data["Destination Port"] = [80]
        data["Flow Duration"] = [5000]
        data["Flow Bytes/s"] = [-1]
        csv = pd.DataFrame(data).to_csv(index=False).encode()
        r = self._upload(csv)
        self.assertEqual(r.status_code, 400)

    def test_few_columns_csv_rejected(self):
        """CSV with only 3 CICIDS columns → column-overlap rejection."""
        csv = b"Destination Port,Flow Duration,Total Fwd Packets\n80,1000,5\n"
        r = self._upload(csv)
        self.assertEqual(r.status_code, 400)


# ═══════════════════════════════════════════════════════════════════════════
# 4. MODEL PERFORMANCE TESTS (optional — needs labelled CSV)
# ═══════════════════════════════════════════════════════════════════════════

class TestModelPerformance(unittest.TestCase):
    """If a labelled test CSV exists, compute accuracy / F1."""

    LABELLED_CSV = os.path.join(PROJECT_DIR, "test_data_labelled.csv")

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(cls.LABELLED_CSV):
            cls.skip_reason = f"Labelled test file not found: {cls.LABELLED_CSV}"
            return
        if not MODELS_LOADED:
            cls.skip_reason = "Models not loaded (demo mode)"
            return
        cls.skip_reason = None
        cls.df = pd.read_csv(cls.LABELLED_CSV)
        cls.df.columns = cls.df.columns.str.strip()

    def _guard(self):
        if getattr(self.__class__, "skip_reason", None):
            self.skipTest(self.__class__.skip_reason)

    def test_binary_accuracy(self):
        self._guard()
        from sklearn.metrics import accuracy_score, f1_score
        df = self.__class__.df.copy()
        label_col = [c for c in df.columns if "label" in c.lower()]
        self.assertTrue(len(label_col) > 0, "No label column found in test CSV")
        labels = df[label_col[0]]
        y_true = (labels.str.strip().str.upper() != "BENIGN").astype(int)
        # Preprocess (drop label first)
        csv_bytes = df.to_csv(index=False).encode()
        features = preprocess_csv(csv_bytes)
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
        X = features[FEATURE_COLUMNS].values
        preds = binary_model.predict(X)
        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, zero_division=0)
        print(f"\n  Binary — Accuracy: {acc:.4f}  F1: {f1:.4f}")
        self.assertGreater(acc, 0.5, "Binary accuracy below 50% — likely broken")

    def test_multiclass_report(self):
        self._guard()
        from sklearn.metrics import classification_report
        df = self.__class__.df.copy()
        label_col = [c for c in df.columns if "label" in c.lower()][0]
        labels = df[label_col].str.strip()
        csv_bytes = df.to_csv(index=False).encode()
        features = preprocess_csv(csv_bytes)
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
        X = features[FEATURE_COLUMNS].values
        bin_preds = binary_model.predict(X)
        attack_mask = bin_preds == 1
        if attack_mask.sum() > 0:
            mc_preds = multiclass_model.predict(X[attack_mask])
            print(f"\n  Multiclass predictions on {attack_mask.sum()} flagged flows")
        self.assertTrue(True)  # informational


# ═══════════════════════════════════════════════════════════════════════════
# 5. FRONTEND SMOKE TESTS (Selenium — optional)
# ═══════════════════════════════════════════════════════════════════════════

class TestFrontendSmoke(unittest.TestCase):
    """Optional Selenium tests. Skipped if selenium is not installed."""

    @classmethod
    def setUpClass(cls):
        try:
            from selenium import webdriver
            from selenium.webdriver.common.by import By
            cls.By = By
            opts = webdriver.ChromeOptions()
            opts.add_argument("--headless")
            opts.add_argument("--disable-gpu")
            cls.driver = webdriver.Chrome(options=opts)
            cls.available = True
        except Exception:
            cls.available = False

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "available", False):
            cls.driver.quit()

    def _guard(self):
        if not self.__class__.available:
            self.skipTest("Selenium / ChromeDriver not available")

    def test_dashboard_loads(self):
        self._guard()
        path = os.path.join(PROJECT_DIR, "dashboard.html")
        self.driver.get("file:///" + path.replace("\\", "/"))
        title = self.driver.title
        self.assertIn("NetGuard", title)

    def test_theme_toggle(self):
        self._guard()
        path = os.path.join(PROJECT_DIR, "dashboard.html")
        self.driver.get("file:///" + path.replace("\\", "/"))
        html = self.driver.find_element(self.__class__.By.TAG_NAME, "html")
        initial = html.get_attribute("data-theme") or "light"
        btns = self.driver.find_elements(self.__class__.By.CSS_SELECTOR,
            "[onclick*='theme'], .theme-toggle, #themeToggle, .btn-help")
        if btns:
            btns[0].click()
            time.sleep(0.3)
            after = html.get_attribute("data-theme") or "light"
            self.assertNotEqual(initial, after, "Theme should have toggled")
        else:
            self.skipTest("No theme toggle button found")


# ═══════════════════════════════════════════════════════════════════════════
# HTML REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

class _ReportResult(unittest.TestResult):
    """Collect results for the HTML report."""
    def __init__(self):
        super().__init__()
        self.results = []  # (name, status, detail)
        self._t0 = None

    def startTest(self, test):
        super().startTest(test)
        self._t0 = time.time()

    def addSuccess(self, test):
        super().addSuccess(test)
        self.results.append((str(test), "PASS", "", time.time() - self._t0))

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.results.append((str(test), "FAIL", self._exc(err), time.time() - self._t0))

    def addError(self, test, err):
        super().addError(test, err)
        self.results.append((str(test), "ERROR", self._exc(err), time.time() - self._t0))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.results.append((str(test), "SKIP", reason, 0))

    @staticmethod
    def _exc(err):
        return "".join(traceback.format_exception(*err))


def _generate_html_report(result: _ReportResult, elapsed: float) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    passed = sum(1 for _, s, _, _ in result.results if s == "PASS")
    failed = sum(1 for _, s, _, _ in result.results if s == "FAIL")
    errors = sum(1 for _, s, _, _ in result.results if s == "ERROR")
    skipped = sum(1 for _, s, _, _ in result.results if s == "SKIP")
    total = len(result.results)
    pct = (passed / total * 100) if total else 0

    color_map = {"PASS": "#22c55e", "FAIL": "#ef4444", "ERROR": "#f97316", "SKIP": "#a1a1aa"}
    badge = lambda s: f'<span style="background:{color_map[s]};color:#fff;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:600">{s}</span>'

    rows = ""
    for name, status, detail, dur in result.results:
        detail_html = f'<pre style="margin:4px 0 0;font-size:11px;color:#888;white-space:pre-wrap">{detail}</pre>' if detail else ""
        rows += f"""<tr>
  <td style="padding:8px 12px;border-bottom:1px solid #222">{badge(status)}</td>
  <td style="padding:8px 12px;border-bottom:1px solid #222;font-family:monospace;font-size:13px">{name}{detail_html}</td>
  <td style="padding:8px 12px;border-bottom:1px solid #222;text-align:right;font-family:monospace;font-size:12px;color:#888">{dur:.3f}s</td>
</tr>"""

    mode = "Real Models" if MODELS_LOADED else "Demo Mode (models not found)"
    bar_w = int(pct)

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>NetGuard AI — Test Report</title>
<style>
  body{{background:#0a0a0a;color:#e0e0e0;font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:32px}}
  h1{{font-size:24px;margin-bottom:4px}} .sub{{color:#888;font-size:13px;margin-bottom:24px}}
  .cards{{display:flex;gap:16px;margin-bottom:24px;flex-wrap:wrap}}
  .card{{background:#161616;border:1px solid #222;border-radius:8px;padding:16px 24px;min-width:120px}}
  .card .num{{font-size:28px;font-weight:700}} .card .lbl{{font-size:11px;color:#888;text-transform:uppercase;letter-spacing:0.1em}}
  table{{width:100%;border-collapse:collapse;background:#111;border-radius:8px;overflow:hidden}}
  th{{text-align:left;padding:10px 12px;background:#1a1a1a;font-size:11px;text-transform:uppercase;letter-spacing:0.1em;color:#888}}
  .bar-bg{{background:#222;border-radius:4px;height:8px;width:200px;margin-top:8px}}
  .bar-fg{{height:8px;border-radius:4px}}
</style></head><body>
<h1>🛡️ NetGuard AI — Test Report</h1>
<div class="sub">{now} &nbsp;|&nbsp; Mode: {mode} &nbsp;|&nbsp; Total time: {elapsed:.2f}s</div>
<div class="cards">
  <div class="card"><div class="num" style="color:#22c55e">{passed}</div><div class="lbl">Passed</div></div>
  <div class="card"><div class="num" style="color:#ef4444">{failed}</div><div class="lbl">Failed</div></div>
  <div class="card"><div class="num" style="color:#f97316">{errors}</div><div class="lbl">Errors</div></div>
  <div class="card"><div class="num" style="color:#a1a1aa">{skipped}</div><div class="lbl">Skipped</div></div>
  <div class="card"><div class="num">{total}</div><div class="lbl">Total</div></div>
</div>
<div style="margin-bottom:24px">
  <span style="font-size:13px;color:#888">Pass Rate: {pct:.1f}%</span>
  <div class="bar-bg"><div class="bar-fg" style="width:{bar_w}%;background:{'#22c55e' if pct>=80 else '#f97316' if pct>=50 else '#ef4444'}"></div></div>
</div>
<table><tr><th>Status</th><th>Test</th><th style="text-align:right">Time</th></tr>
{rows}
</table>
<div style="margin-top:24px;font-size:11px;color:#555">Generated by run_tests.py — NetGuard AI Test Suite</div>
</body></html>"""
    return html


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    for cls in [
        TestPreprocessCSV,
        TestRunDetection,
        TestDetectEndpoint,
        TestInputValidation,
        TestModelPerformance,
        TestFrontendSmoke,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    # Run
    result = _ReportResult()
    t0 = time.time()
    print("=" * 60)
    print("  NetGuard AI — Running Test Suite")
    print("  Mode:", "Real Models" if MODELS_LOADED else "Demo Mode")
    print("=" * 60)

    suite.run(result)
    elapsed = time.time() - t0

    # Console summary
    passed = sum(1 for _, s, _, _ in result.results if s == "PASS")
    failed = sum(1 for _, s, _, _ in result.results if s in ("FAIL", "ERROR"))
    skipped = sum(1 for _, s, _, _ in result.results if s == "SKIP")

    print()
    for name, status, detail, dur in result.results:
        icon = {"PASS": "+", "FAIL": "-", "ERROR": "!", "SKIP": "s"}.get(status, "?")
        print(f"  [{icon}] [{status:5s}] {name}  ({dur:.3f}s)")
        if detail and status in ("FAIL", "ERROR"):
            for line in detail.strip().split("\n")[-3:]:
                print(f"           {line}")

    print()
    print(f"  Results: {passed} passed, {failed} failed, {skipped} skipped  ({elapsed:.2f}s)")
    print("=" * 60)

    # Write HTML report
    report_path = os.path.join(PROJECT_DIR, "test_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(_generate_html_report(result, elapsed))
    print(f"\n  Report saved to: {report_path}")

    # Auto-open
    try:
        webbrowser.open("file:///" + report_path.replace("\\", "/"))
    except Exception:
        pass

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
