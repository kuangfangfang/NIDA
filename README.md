# NetGuard AI — Setup Guide

Two-stage network intrusion detection system using XGBoost, trained on CICIDS2017.

---

## Project Structure

```
nids_backend/
├── app.py               ← Flask backend
├── index.html           ← Frontend (open directly in browser)
├── requirements.txt     ← Python dependencies
└── models/
    ├── binary_model.pkl     ← Your Stage 1 (benign vs attack) model
    └── multiclass_model.pkl ← Your Stage 2 (attack type) model
```

---

## Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

> Requires Python 3.9+

---

## Step 2 — Add your models

Create a `models/` folder and drop in your pkl files:

```bash
mkdir models
cp /path/to/your/binary_model.pkl    models/binary_model.pkl
cp /path/to/your/multiclass_model.pkl models/multiclass_model.pkl
```

**If your pkl files have different names**, just edit lines 22–23 in `app.py`:
```python
binary_model     = load_model("your_binary_filename.pkl")
multiclass_model = load_model("your_multiclass_filename.pkl")
```

**If your Stage 2 model uses numeric labels** (0, 1, 2…) instead of string labels,
check the `label_map` dict in `run_detection()` around line 175 and make sure
the numbers match your training labels.

**Without models**, the backend runs in **demo mode** — it returns random predictions
so you can test the UI. A yellow warning banner appears in the frontend.

---

## Step 3 — Start the backend

```bash
python app.py
```

You should see:
```
✓ Models loaded successfully
NetGuard AI backend starting on http://localhost:5000
```

---

## Step 4 — Open the frontend

Just open `index.html` in your browser — no server needed for the frontend.

```bash
# macOS
open index.html

# Windows
start index.html

# Linux
xdg-open index.html
```

---

## Uploading files

**CSV** — Should be a CICIDS2017-format CSV (output from CICFlowMeter).
The system will automatically strip any Label column and align feature columns.

**PCAP / PCAPNG** — Raw packet captures are supported. The backend uses scapy
to extract flow-level features compatible with the CICIDS2017 feature set.
Feature extraction is approximate — for best accuracy on PCAP files,
pre-process with CICFlowMeter and upload the resulting CSV instead.

---

## CORS / browser issues

If you get a CORS error in the browser console, make sure the Flask backend is running.
The frontend calls `http://localhost:5000` — flask-cors is already enabled on the backend.

For production deployment, update the `API` variable at the bottom of `index.html`:
```javascript
const API = 'https://your-deployed-backend.com';
```

---

## Feature columns

The backend expects (or extracts) the standard 78 CICIDS2017 features.
If your model was trained on a subset, edit `FEATURE_COLUMNS` in `app.py`
to match exactly the columns used during training.
