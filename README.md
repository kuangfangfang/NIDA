# NetGuard AI — Network Intrusion Detection System (NIDS)

NetGuard AI is a state-of-the-art, machine-learning-driven Network Intrusion Detection System (NIDS). It integrates a two-stage XGBoost classifier trained on the standardized **CICIDS2017** benchmark, explains predictions locally using **SHAP values**, and generates human-readable incident response playbooks powered by the **DeepSeek LLM engine**.

The system features a decoupled client-server architecture with a premium glassmorphic UI dashboard that supports direct file uploads, real-time packet sniffing, and dark/light modes with accessibility-scaled typography.

---

## 🚀 Key Features

* **Two-Stage ML Pipeline**:
  * **Stage 1 (Binary)**: Fast classification filtering normal traffic from malicious anomalies.
  * **Stage 2 (Multiclass)**: Multi-label classification identifying 7 distinct attack categories (`DDoS`, `PortScan`, `Brute Force`, `Bot`, `Infiltration`, `Web Attack`, `Heartbleed`).
* **Interactive Local Explainability (XAI)**: Calculates real-time SHAP (Shapley Additive exPlanations) values to isolate the network feature metrics (e.g., flag counts, segment sizes) driving the model's verdict.
* **DeepSeek LLM Incident Reports**: Connects to the DeepSeek Chat API to produce structured mitigation advice cards (Executive Analysis, Detailed Breakdown, Why Flagged, Action suggestions).
* **Live Sniffing Agent**: Python-based local sniffer (`live_capture.py`) intercepts active Network Interface Card (NIC) traffic, generates clean bidirectional flows via Scapy, and uploads them dynamically to the cloud API.
* **Responsive Glassmorphic UI**: High-fidelity dashboard (`dashboard.html` and `live_monitor.html`) featuring custom HSL styling, responsive CSS grids, and interactive Chart.js charts.
* **Modular Architecture**: Decouples Flask routes (`app.py`) from core ML classification functions (`detector.py`) to maximize code maintainability.
* **Empirical Feature Normalization**: Preprocessing pipeline (`preprocessing.py`) handles raw, unscaled CSV and PCAP inputs, translating them to standardized model ranges and applying split threshold corrections for robust real-time inference.
* **Dataset Portability**: Includes tools (`convert_cesnet.py`) to map NetFlow formats like CESNET directly onto the NIDS feature space.
* **Full Testing Harness**: Running `run_tests.py` launches 25 tests checking data pipelines, API integrations, and validation boundaries.

---

## 📁 Repository Structure

```
.
├── app.py                     # Flask HTTP API routing server
├── detector.py                # Core ML engine (PCAP extraction, CSV preprocessing, detection)
├── preprocessing.py           # Empirical normalization & feature alignment pipeline
├── convert_cesnet.py          # CESNET-to-CICIDS2017 feature mapping tool
├── generate_cesnet_test_data.py # Automated test data generation tool
├── index.html                 # Brand landing page
├── dashboard.html             # Historical log, CSV upload & AI reporting dashboard
├── live_monitor.html          # Live network sniffing monitor controls
├── live_capture.py            # Local interface sniffing CLI agent (Scapy)
├── requirements.txt           # Python application dependencies
├── run_tests.py               # Automated testing framework runner
├── TEST_PLAN.md               # Detailed test case specifications
├── train.py                   # XGBoost models and scaler training script
├── models/
│   ├── scaler.pkl             # Serialized MinMaxScaler
│   ├── binary_model.pkl       # Stage 1 XGBoost model
│   ├── multiclass_model.pkl   # Stage 2 XGBoost model
│   └── stage2_label_encoder.pkl # Class name encoder for multi-label predictions
└── .gitignore                 # Safe staging list (ignores .env/caches)
```

---

## 🛠️ Local Installation & Setup

### Prerequisites
* Python 3.9+ installed on your system.

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a file named `.env` in the root directory:
```env
DEEPSEEK_API_KEY=your_actual_deepseek_api_key
```

### 3. Run Training (Optional)
If model files are not found in `models/`, the server runs in **Demo Mode** with fallback simulations. To train the models on your training dataset, put your dataset CSV as `cicids2017_train.csv` and run:
```bash
python train.py
```

### 4. Start the Flask Backend
```bash
python app.py
```
The backend service will boot on `http://localhost:5000`.

### 5. Access the Frontend
Open the frontend application directly in any browser (no frontend server required):
* Landing Page: `index.html`
* Analytics Dashboard: `dashboard.html`
* Live Monitor: `live_monitor.html`

---

## 📡 Live Packet Sniffing & Ingestion

Since cloud web services block raw socket capture for security, NetGuard AI utilizes a hybrid client-server model:

1. Start your local Flask API or locate your Render deployment URL.
2. In a terminal with admin privileges (required to access physical network cards), run the sniffing agent:
   ```bash
   # Windows (Administrator command prompt)
   python live_capture.py
   
   # Linux / macOS
   sudo python live_capture.py
   ```
3. Set your target API URL and sniffing duration (e.g. 15s). The agent captures raw packets, groups them into bidrectional flows, parses them to a temporary `.pcap`, and transmits it to the Flask server for dual-stage classification.

---

## 🧪 Testing & Verification

We supply an automated verification tool validating endpoint performance, preprocessing scripts, and bad-input limits.

Run the test suite:
```bash
python run_tests.py
```
A visual `test_report.html` report will be generated and automatically opened in your default web browser detailing pass/fail summaries.

---

## ☁️ Deployment Specifications

### Render Hosting Setup
1. **GitHub Sync**: Push this repository (without the `.env` file) to a GitHub repository.
2. **Web Service Creation**: In Render, create a new **Web Service** connected to your repository.
3. **Environment Settings**:
   * **Runtime**: Python
   * **Build Command**: `pip install -r requirements.txt`
   * **Start Command**: `gunicorn app:app`
4. **Environment Variables**:
   Add `DEEPSEEK_API_KEY` under the **Environment** tab inside the Render service panel.
5. **CORS Alignment**:
   Update the backend endpoint API constant in the javascript scripts at the bottom of `dashboard.html` and `live_monitor.html` to point to your Render domain URL (e.g., `https://your-service.onrender.com`).
