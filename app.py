"""
NetGuard AI — Backend API
Two-stage Network Intrusion Detection System
Trained on CICIDS2017 | XGBoost models
"""
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import os
import io
import struct
import socket
from collections import defaultdict

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# ── DeepSeek API Configuration ──────────────────────────────────────────────
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your-api-key-here")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


client = None
try:
    import openai
    if DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != "your-api-key-here":
        client = openai.OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )
        print("DeepSeek OpenAI client configured successfully.")
    else:
        print("Warning: DEEPSEEK_API_KEY environment variable is not configured. Real-time LLM reports will fall back to local guidelines.")
except ImportError:
    print("Warning: 'openai' library is not installed. Please run: pip install openai")
except Exception as e:
    print(f"Warning: Failed to configure DeepSeek client: {e}")

# ── Model loading ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(filename):
    path = os.path.join(BASE_DIR, "models", filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

try:
    import shap
except ImportError:
    print("\n" + "="*80)
    print("WARNING: The 'shap' library is not installed. Please install it with:")
    print("         pip install shap")
    print("="*80 + "\n")
    shap = None

binary_model = None
multiclass_model = None
scaler = None
explainer = None

try:
    binary_model = load_model("binary_model.pkl")
    multiclass_model = load_model("multiclass_model.pkl")
    print("Models loaded successfully")
except Exception as e:
    print(f"Warning: Failed to load models: {e}")

try:
    scaler = load_model("scaler.pkl")
    print("Scaler loaded successfully")
except Exception as e:
    print(f"Warning: Failed to load scaler: {e}")

try:
    if shap is not None and binary_model is not None:
        explainer = shap.TreeExplainer(binary_model)
        print("Created SHAP TreeExplainer from binary_model")
except Exception as e:
    print(f"Warning: Failed to create TreeExplainer from binary_model: {e}")

if shap is not None and explainer is None:
    try:
        explainer = load_model("shap_explainer_binary.pkl")
        print("SHAP Explainer loaded successfully")
    except Exception as e:
        print(f"Warning: Failed to load SHAP Explainer: {e}")

shap_explainer_binary = explainer

# Write startup debug info
try:
    debug_file_path = os.path.join(BASE_DIR, "debug_detect.txt")
    with open(debug_file_path, "w") as dbg:
        dbg.write("=== BACKEND STARTUP DEBUG ===\n")
        dbg.write(f"Binary model loaded: {binary_model is not None}\n")
        dbg.write(f"Multiclass model loaded: {multiclass_model is not None}\n")
        dbg.write(f"Scaler loaded: {scaler is not None}\n")
        dbg.write(f"SHAP Explainer loaded: {shap_explainer_binary is not None}\n")
        if binary_model is not None and hasattr(binary_model, "feature_names_in_"):
            dbg.write(f"Binary model features: {list(binary_model.feature_names_in_)}\n")
except Exception as dbg_err:
    print(f"Startup debug failed: {dbg_err}")

# ── CICIDS2017 feature columns (78 features) ──────────────────────────────────
FEATURE_COLUMNS = [
    "Destination Port","Flow Duration","Total Fwd Packets","Total Backward Packets",
    "Total Length of Fwd Packets","Total Length of Bwd Packets","Fwd Packet Length Max",
    "Fwd Packet Length Min","Fwd Packet Length Mean","Fwd Packet Length Std",
    "Bwd Packet Length Max","Bwd Packet Length Min","Bwd Packet Length Mean",
    "Bwd Packet Length Std","Flow Bytes/s","Flow Packets/s","Flow IAT Mean",
    "Flow IAT Std","Flow IAT Max","Flow IAT Min","Fwd IAT Total","Fwd IAT Mean",
    "Fwd IAT Std","Fwd IAT Max","Fwd IAT Min","Bwd IAT Total","Bwd IAT Mean",
    "Bwd IAT Std","Bwd IAT Max","Bwd IAT Min","Fwd PSH Flags","Bwd PSH Flags",
    "Fwd URG Flags","Bwd URG Flags","Fwd Header Length","Bwd Header Length",
    "Fwd Packets/s","Bwd Packets/s","Min Packet Length","Max Packet Length",
    "Packet Length Mean","Packet Length Std","Packet Length Variance","FIN Flag Count",
    "SYN Flag Count","RST Flag Count","PSH Flag Count","ACK Flag Count","URG Flag Count",
    "CWE Flag Count","ECE Flag Count","Down/Up Ratio","Average Packet Size",
    "Avg Fwd Segment Size","Avg Bwd Segment Size","Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk","Fwd Avg Packets/Bulk","Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk","Bwd Avg Packets/Bulk","Bwd Avg Bulk Rate","Subflow Fwd Packets",
    "Subflow Fwd Bytes","Subflow Bwd Packets","Subflow Bwd Bytes","Init_Win_bytes_forward",
    "Init_Win_bytes_backward","act_data_pkt_fwd","min_seg_size_forward","Active Mean",
    "Active Std","Active Max","Active Min","Idle Mean","Idle Std","Idle Max","Idle Min"
]

# ── Attack class metadata ──────────────────────────────────────────────────────
ATTACK_INFO = {
    "DDoS": {
        "label": "DDoS Attack",
        "color": "danger",
        "icon": "ti-ripple",
        "summary": "Your network is being flooded with massive amounts of traffic from multiple sources. The goal is to overwhelm your bandwidth and make your services unreachable to real users.",
        "suggestions": [
            "Contact your ISP immediately and request upstream traffic filtering or null-routing of attacking IPs.",
            "Enable DDoS protection on your firewall — rate-limit UDP/ICMP traffic aggressively.",
            "If on cloud (AWS/Azure/GCP), activate their built-in DDoS Shield service.",
            "Consider a CDN/DDoS scrubbing service like Cloudflare or Akamai as a long-term fix."
        ]
    },
    "PortScan": {
        "label": "Port Scan",
        "color": "warning",
        "icon": "ti-radar",
        "summary": "Someone is systematically probing your network to discover open ports and running services. This is often a reconnaissance step before a deeper attack.",
        "suggestions": [
            "Identify the scanning source IP and block it at your firewall immediately.",
            "Audit which ports are actually open — close or firewall anything that doesn't need to be public.",
            "Enable port-scan detection (SYN flood rules) in your IDS/IPS.",
            "Consider moving sensitive services to non-standard ports or behind a VPN."
        ]
    },
    "Brute Force": {
        "label": "Brute Force Attack",
        "color": "danger",
        "icon": "ti-lock-open",
        "summary": "An attacker is rapidly guessing passwords or credentials to gain unauthorized access — typically targeting SSH, FTP, or login pages.",
        "suggestions": [
            "Lock out the attacking IP at the firewall or using tools like fail2ban.",
            "Enforce account lockout policies after 5–10 failed login attempts.",
            "Enable multi-factor authentication (MFA) on all exposed services immediately.",
            "Review logs for any successful logins from the attacking IP — it may have already gotten in."
        ]
    },
    "Bot": {
        "label": "Bot / Botnet Traffic",
        "color": "warning",
        "icon": "ti-robot",
        "summary": "Traffic patterns match known botnet command-and-control behavior. One or more devices on your network may be infected and communicating with an external attacker.",
        "suggestions": [
            "Isolate the suspected infected host(s) from the network immediately.",
            "Run a full malware scan on flagged machines using updated AV/EDR tools.",
            "Check outbound traffic for connections to known C2 domains/IPs using a threat intelligence feed.",
            "Reimage compromised machines if infection is confirmed — don't just clean them."
        ]
    },
    "Infiltration": {
        "label": "Infiltration Attempt",
        "color": "danger",
        "icon": "ti-shield-off",
        "summary": "An attacker may have already breached your perimeter and is attempting lateral movement — pivoting between systems to reach higher-value targets.",
        "suggestions": [
            "Treat this as a potential breach — initiate your incident response plan now.",
            "Segment your network immediately to contain lateral movement.",
            "Audit all internal traffic and authentication logs for unusual access patterns.",
            "Engage a security professional or IR team if you don't have one on-call."
        ]
    },
    "Web Attack": {
        "label": "Web Application Attack",
        "color": "warning",
        "icon": "ti-world-off",
        "summary": "Your web application is being targeted — likely SQL injection, XSS, or other application-layer exploits designed to steal data or gain server access.",
        "suggestions": [
            "Enable a Web Application Firewall (WAF) — Cloudflare, ModSecurity, or AWS WAF are good options.",
            "Immediately review your web server error logs for injection attempts.",
            "Patch your web application frameworks and CMS (WordPress, Drupal, etc.) to latest versions.",
            "Check your database for unauthorized queries or data exfiltration."
        ]
    },
    "Heartbleed": {
        "label": "Heartbleed Exploit",
        "color": "danger",
        "icon": "ti-heart-broken",
        "summary": "Traffic matches the Heartbleed vulnerability (CVE-2014-0160), an OpenSSL bug that can expose sensitive memory including private keys and passwords.",
        "suggestions": [
            "Immediately check your OpenSSL version — if below 1.0.1g, patch it now.",
            "Revoke and reissue all SSL/TLS certificates on affected servers.",
            "Rotate all passwords and session tokens that may have been exposed.",
            "Audit what data could have been leaked — treat all secrets as compromised."
        ]
    },
    "BENIGN": {
        "label": "Benign Traffic",
        "color": "success",
        "icon": "ti-shield-check",
        "summary": "No intrusion detected. Your network traffic looks normal.",
        "suggestions": [
            "Keep your firewall rules and IDS signatures up to date.",
            "Schedule regular network audits to stay ahead of threats.",
            "Consider setting up continuous monitoring for peace of mind."
        ]
    }
}

# ── PCAP feature extraction using scapy ───────────────────────────────────────
def extract_features_from_pcap(pcap_bytes):
    """
    Extract CICIDS2017-compatible features from a PCAP file using scapy.
    Groups packets into flows (5-tuple) and computes statistical features.
    """
    try:
        from scapy.all import rdpcap, IP, TCP, UDP
        from scapy.utils import PcapReader
    except ImportError:
        raise ImportError("scapy is required for PCAP processing. Install with: pip install scapy")

    import tempfile, time

    # Write bytes to temp file for scapy
    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as tmp:
        tmp.write(pcap_bytes)
        tmp_path = tmp.name

    try:
        packets = rdpcap(tmp_path)
    finally:
        os.unlink(tmp_path)

    # Group packets into flows by 5-tuple (src_ip, dst_ip, src_port, dst_port, proto)
    flows = defaultdict(list)
    for pkt in packets:
        if not pkt.haslayer(IP):
            continue
        ip = pkt[IP]
        proto = ip.proto
        src_port = dst_port = 0
        flags = 0
        if pkt.haslayer(TCP):
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
            flags = int(pkt[TCP].flags)
        elif pkt.haslayer(UDP):
            src_port = pkt[UDP].sport
            dst_port = pkt[UDP].dport
        key = (ip.src, ip.dst, src_port, dst_port, proto)
        flows[key].append({
            "time": float(pkt.time),
            "size": len(pkt),
            "ip_size": len(ip),
            "flags": flags,
            "dst_port": dst_port
        })

    rows = []
    for flow_key, pkts in flows.items():
        if len(pkts) < 2:
            continue
        pkts.sort(key=lambda x: x["time"])
        times = [p["time"] for p in pkts]
        sizes = [p["size"] for p in pkts]
        iats  = [times[i+1] - times[i] for i in range(len(times)-1)]
        dur   = times[-1] - times[0]

        # Split fwd/bwd (heuristic: first packet direction = fwd)
        fwd = pkts[::2]
        bwd = pkts[1::2]
        fwd_sizes = [p["size"] for p in fwd]
        bwd_sizes = [p["size"] for p in bwd]
        fwd_times = [p["time"] for p in fwd]
        bwd_times = [p["time"] for p in bwd]
        fwd_iats  = [fwd_times[i+1]-fwd_times[i] for i in range(len(fwd_times)-1)] or [0]
        bwd_iats  = [bwd_times[i+1]-bwd_times[i] for i in range(len(bwd_times)-1)] or [0]

        all_flags = [p["flags"] for p in pkts]
        def flag_count(bit): return sum(1 for f in all_flags if f & bit)

        dur_us = dur * 1e6 if dur > 0 else 1
        row = {
            "Destination Port": flow_key[3],
            "Flow Duration": dur_us,
            "Total Fwd Packets": len(fwd),
            "Total Backward Packets": len(bwd),
            "Total Length of Fwd Packets": sum(fwd_sizes),
            "Total Length of Bwd Packets": sum(bwd_sizes),
            "Fwd Packet Length Max": max(fwd_sizes) if fwd_sizes else 0,
            "Fwd Packet Length Min": min(fwd_sizes) if fwd_sizes else 0,
            "Fwd Packet Length Mean": np.mean(fwd_sizes) if fwd_sizes else 0,
            "Fwd Packet Length Std": np.std(fwd_sizes) if fwd_sizes else 0,
            "Bwd Packet Length Max": max(bwd_sizes) if bwd_sizes else 0,
            "Bwd Packet Length Min": min(bwd_sizes) if bwd_sizes else 0,
            "Bwd Packet Length Mean": np.mean(bwd_sizes) if bwd_sizes else 0,
            "Bwd Packet Length Std": np.std(bwd_sizes) if bwd_sizes else 0,
            "Flow Bytes/s": sum(sizes) / (dur if dur > 0 else 1),
            "Flow Packets/s": len(pkts) / (dur if dur > 0 else 1),
            "Flow IAT Mean": np.mean(iats) if iats else 0,
            "Flow IAT Std": np.std(iats) if iats else 0,
            "Flow IAT Max": max(iats) if iats else 0,
            "Flow IAT Min": min(iats) if iats else 0,
            "Fwd IAT Total": sum(fwd_iats),
            "Fwd IAT Mean": np.mean(fwd_iats),
            "Fwd IAT Std": np.std(fwd_iats),
            "Fwd IAT Max": max(fwd_iats),
            "Fwd IAT Min": min(fwd_iats),
            "Bwd IAT Total": sum(bwd_iats),
            "Bwd IAT Mean": np.mean(bwd_iats),
            "Bwd IAT Std": np.std(bwd_iats),
            "Bwd IAT Max": max(bwd_iats),
            "Bwd IAT Min": min(bwd_iats),
            "Fwd PSH Flags": flag_count(0x08),
            "Bwd PSH Flags": 0,
            "Fwd URG Flags": flag_count(0x20),
            "Bwd URG Flags": 0,
            "Fwd Header Length": len(fwd) * 20,
            "Bwd Header Length": len(bwd) * 20,
            "Fwd Packets/s": len(fwd) / (dur if dur > 0 else 1),
            "Bwd Packets/s": len(bwd) / (dur if dur > 0 else 1),
            "Min Packet Length": min(sizes),
            "Max Packet Length": max(sizes),
            "Packet Length Mean": np.mean(sizes),
            "Packet Length Std": np.std(sizes),
            "Packet Length Variance": np.var(sizes),
            "FIN Flag Count": flag_count(0x01),
            "SYN Flag Count": flag_count(0x02),
            "RST Flag Count": flag_count(0x04),
            "PSH Flag Count": flag_count(0x08),
            "ACK Flag Count": flag_count(0x10),
            "URG Flag Count": flag_count(0x20),
            "CWE Flag Count": flag_count(0x80),
            "ECE Flag Count": flag_count(0x40),
            "Down/Up Ratio": len(bwd) / len(fwd) if len(fwd) > 0 else 0,
            "Average Packet Size": np.mean(sizes),
            "Avg Fwd Segment Size": np.mean(fwd_sizes) if fwd_sizes else 0,
            "Avg Bwd Segment Size": np.mean(bwd_sizes) if bwd_sizes else 0,
            "Fwd Header Length.1": len(fwd) * 20,
            "Fwd Avg Bytes/Bulk": 0, "Fwd Avg Packets/Bulk": 0, "Fwd Avg Bulk Rate": 0,
            "Bwd Avg Bytes/Bulk": 0, "Bwd Avg Packets/Bulk": 0, "Bwd Avg Bulk Rate": 0,
            "Subflow Fwd Packets": len(fwd),
            "Subflow Fwd Bytes": sum(fwd_sizes),
            "Subflow Bwd Packets": len(bwd),
            "Subflow Bwd Bytes": sum(bwd_sizes),
            "Init_Win_bytes_forward": 0,
            "Init_Win_bytes_backward": 0,
            "act_data_pkt_fwd": len(fwd),
            "min_seg_size_forward": min(fwd_sizes) if fwd_sizes else 0,
            "Active Mean": 0, "Active Std": 0, "Active Max": 0, "Active Min": 0,
            "Idle Mean": 0, "Idle Std": 0, "Idle Max": 0, "Idle Min": 0,
        }
        rows.append(row)

    if not rows:
        raise ValueError("No valid IP flows found in PCAP file.")

    return pd.DataFrame(rows)


def preprocess_csv(csv_bytes):
    """Load, validate, and clean a CICIDS2017-format CSV.

    Validation steps (run *before* column alignment):
      1. Row count must be > 0.
      2. Uploaded columns must overlap with CICIDS2017 features (min 8 of 78).
      3. Key feature columns must contain plausible values (skipped for
         pre-normalized/scaled data where all values are in [0, 1]).
      4. Data must not be nearly-constant across all columns.
    """
    df = pd.read_csv(io.BytesIO(csv_bytes))
    df.columns = df.columns.str.strip()

    # Drop label column if present
    label_cols = [c for c in df.columns if "label" in c.lower()]
    if label_cols:
        df = df.drop(columns=label_cols)

    # ── Validation 1: non-empty ────────────────────────────────────────────
    if len(df) == 0:
        raise ValueError("Uploaded CSV contains no data rows")

    # ── Validation 2: column-overlap check ─────────────────────────────────
    # The uploaded CSV must contain a meaningful number of CICIDS2017 feature
    # columns.  A completely unrelated file (e.g. "nots,href,action") shares
    # zero columns and must be rejected outright.
    matched_cols = [c for c in df.columns if c in FEATURE_COLUMNS]
    MIN_MATCHED_COLS = max(8, int(len(FEATURE_COLUMNS) * 0.10))  # ≈8 of 78
    if len(matched_cols) < MIN_MATCHED_COLS:
        raise ValueError(
            f"File does not contain valid network traffic data. "
            f"Only {len(matched_cols)} of {len(FEATURE_COLUMNS)} expected "
            f"CICIDS2017 feature columns were found (minimum {MIN_MATCHED_COLS} required)"
        )

    # ── Validation 3: key-feature range checks ─────────────────────────────
    # First, detect whether the data has been pre-normalized / scaled
    # (all numeric values roughly in [0, 1]).  Normalized data is already
    # pre-processed and valid; raw-range checks only apply to raw CSVs.
    numeric_df = df[matched_cols].apply(pd.to_numeric, errors="coerce")
    _all_vals = numeric_df.values.flatten()
    _all_vals = _all_vals[~np.isnan(_all_vals)]
    is_normalized = len(_all_vals) > 0 and np.all(_all_vals >= 0) and np.all(_all_vals <= 1.1)

    if not is_normalized:
        # Raw-value range checks (only for non-normalized data)
        range_rules = {
            "Flow Duration": {
                "check": lambda s: (s >= 0).all() and (s < 1e8).all(),
            },
            "Total Fwd Packets": {
                "check": lambda s: (s >= 0).all(),
            },
            "Flow Bytes/s": {
                "check": lambda s: (s >= 0).all(),
            },
            "Destination Port": {
                "check": lambda s: (s >= 0).all() and (s <= 65535).all(),
            },
        }

        for col_name, rule in range_rules.items():
            if col_name in df.columns:
                series = pd.to_numeric(df[col_name], errors="coerce")
                # NaN after coercion means non-numeric garbage → invalid
                if series.isna().any():
                    raise ValueError(
                        f"File does not contain valid network traffic data "
                        f"(column '{col_name}' has non-numeric values)"
                    )
                if not rule["check"](series):
                    raise ValueError("File does not contain valid network traffic data")

    # ── Validation 4: constant / trivial data guard ────────────────────────
    # Only consider numeric columns that overlap with the expected features.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    overlap = [c for c in numeric_cols if c in FEATURE_COLUMNS]
    if overlap:
        variances = df[overlap].var()
        zero_var_ratio = (variances == 0).sum() / len(overlap)
        if zero_var_ratio > 0.8:
            raise ValueError("Input file appears to be constant or invalid data")

    # ── Column alignment (existing logic) ──────────────────────────────────
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    df = df[FEATURE_COLUMNS]
    return df


def run_detection(df, explain=False):
    """Run two-stage detection and return result dict."""
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Determine which features to use based on loaded scaler or model
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        cols = list(scaler.feature_names_in_)
    elif binary_model is not None and hasattr(binary_model, "feature_names_in_"):
        cols = list(binary_model.feature_names_in_)
    else:
        cols = FEATURE_COLUMNS

    # Align features
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    X = df[cols].values
    # MOCK feature importance for demo mode
    demo_feature_importance = [
        {"name": "Destination Port", "value": 0.38},
        {"name": "Flow Duration", "value": 0.21},
        {"name": "Init_Win_bytes_forward", "value": 0.16},
        {"name": "Total Length of Fwd Packets", "value": 0.12},
        {"name": "Fwd Packet Length Max", "value": 0.08},
        {"name": "Bwd Packet Length Std", "value": 0.05}
    ]

    if binary_model is None or multiclass_model is None:
        # Demo mode — random predictions for testing without models
        import random
        total = len(X)
        n_attack = random.randint(int(total * 0.3), total)
        is_attack = n_attack > total * 0.5
        if not is_attack:
            return {
                "is_attack": False,
                "stage1_confidence": round(random.uniform(0.85, 0.99), 4),
                "flagged_flows": 0,
                "total_flows": total,
                "attack_type": None,
                "attack_info": ATTACK_INFO["BENIGN"],
                "top_features": [],
                "feature_importance": demo_feature_importance,
                "demo_mode": True,
                "attack_breakdown": {},
                "local_explanation": None
            }
        labels = ["DDoS","PortScan","Brute Force","Bot","Infiltration","Web Attack","Heartbleed"]
        attack_type = random.choice(labels)
        random_breakdown = {cat: random.randint(1, n_attack//2) for cat in labels[:random.randint(2,4)]}
        demo_features = random.sample(cols, 3)
        # Randomize demo values slightly for variation
        import copy
        dfi = copy.deepcopy(demo_feature_importance)
        for d in dfi:
            d["value"] = round(d["value"] * random.uniform(0.9, 1.1), 4)
        dfi = sorted(dfi, key=lambda x: x["value"], reverse=True)

        demo_pos = [
            {"feature": "Destination Port", "contribution": round(0.24 * random.uniform(0.85, 1.15), 2)},
            {"feature": "Flow Duration", "contribution": round(0.18 * random.uniform(0.85, 1.15), 2)},
            {"feature": "Packet Length Std", "contribution": round(0.15 * random.uniform(0.85, 1.15), 2)},
            {"feature": "Average Packet Size", "contribution": round(0.12 * random.uniform(0.85, 1.15), 2)},
            {"feature": "Fwd Packet Length Max", "contribution": round(0.09 * random.uniform(0.85, 1.15), 2)}
        ]
        demo_neg = [
            {"feature": "Flow IAT Mean", "contribution": round(-0.11 * random.uniform(0.85, 1.15), 2)},
            {"feature": "Total Length of Fwd Packets", "contribution": round(-0.08 * random.uniform(0.85, 1.15), 2)},
            {"feature": "Bwd Packet Length Min", "contribution": round(-0.06 * random.uniform(0.85, 1.15), 2)},
            {"feature": "Flow IAT Min", "contribution": round(-0.04 * random.uniform(0.85, 1.15), 2)},
            {"feature": "Bwd Packet Length Std", "contribution": round(-0.02 * random.uniform(0.85, 1.15), 2)}
        ]
        demo_pos = sorted(demo_pos, key=lambda x: x["contribution"], reverse=True)
        demo_neg = sorted(demo_neg, key=lambda x: x["contribution"])

        return {
            "is_attack": True,
            "stage1_confidence": round(random.uniform(0.85, 0.99), 4),
            "stage2_confidence": round(random.uniform(0.80, 0.99), 4),
            "flagged_flows": n_attack,
            "total_flows": total,
            "attack_type": attack_type,
            "attack_info": ATTACK_INFO.get(attack_type, ATTACK_INFO["DDoS"]),
            "top_features": demo_features,
            "feature_importance": dfi,
            "demo_mode": True,
            "attack_breakdown": random_breakdown,
            "local_explanation": {
                "positive": demo_pos,
                "negative": demo_neg,
                "positive_features": demo_pos,
                "negative_features": demo_neg
            }
        }

    # Scale features only if the data is not already normalized/scaled
    _all_vals = X.flatten()
    _all_vals = _all_vals[~np.isnan(_all_vals)]
    is_normalized = len(_all_vals) > 0 and np.all(_all_vals >= -0.1) and np.all(_all_vals <= 1.1)

    if scaler is not None and not is_normalized:
        try:
            X_scaled = scaler.transform(X)
        except Exception as e:
            print(f"Error during feature scaling: {e}")
            X_scaled = X
    else:
        X_scaled = X

    # Stage 1 — binary classification
    binary_preds = binary_model.predict(X_scaled)
    binary_proba = binary_model.predict_proba(X_scaled)
    n_attack = int(np.sum(binary_preds == 1))
    total = len(X_scaled)
    stage1_conf = float(np.mean(np.max(binary_proba, axis=1)))

    if n_attack == 0:
        feat_importance = []
        if binary_model is not None and hasattr(binary_model, "feature_importances_"):
            try:
                feat_imp = sorted(list(zip(cols, binary_model.feature_importances_)), key=lambda x: x[1], reverse=True)
                feat_importance = [{"name": f[0], "value": round(float(f[1]), 4)} for f in feat_imp[:6]]
            except Exception:
                pass
        if not feat_importance:
            feat_importance = demo_feature_importance
        return {
            "is_attack": False,
            "stage1_confidence": round(stage1_conf, 4),
            "flagged_flows": 0,
            "total_flows": total,
            "attack_type": None,
            "attack_info": ATTACK_INFO["BENIGN"],
            "class_distribution": {"BENIGN": total},
            "attack_breakdown": {},
            "top_features": [],
            "feature_importance": feat_importance,
            "demo_mode": False,
            "local_explanation": None
        }

    # Stage 2 — multi-class on flagged flows only
    X_attack_scaled = X_scaled[binary_preds == 1]
    mc_preds = multiclass_model.predict(X_attack_scaled)
    mc_proba = multiclass_model.predict_proba(X_attack_scaled)
    stage2_conf = float(np.mean(np.max(mc_proba, axis=1)))

    # Most frequent predicted class
    unique, counts = np.unique(mc_preds, return_counts=True)
    top_class = unique[np.argmax(counts)]

    # Map numeric label to string if needed
    label_map = {0:"BENIGN", 1:"DDoS", 2:"PortScan", 3:"Brute Force", 4:"Bot",
                 5:"Infiltration", 6:"Web Attack", 7:"Heartbleed"}
    if isinstance(top_class, (int, np.integer)):
        attack_type = label_map.get(int(top_class), str(top_class))
    else:
        attack_type = str(top_class)

    class_distribution = {
        str(label_map.get(int(k), str(k)) if isinstance(k, (int, np.integer)) else k): int(v)
        for k, v in zip(unique, counts)
    }
    attack_breakdown = {k: int(v) for k, v in class_distribution.items() if k.upper() != 'BENIGN'}

    # Compute top_features and feature_importance via SHAP or global importances
    top_features = []
    feature_importance = []
    feat_imp = None

    if explain and shap_explainer_binary is not None:
        try:
            shap_vals = shap_explainer_binary.shap_values(X_attack_scaled)
            if isinstance(shap_vals, list):
                sv = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
            elif hasattr(shap_vals, "values"):
                sv = shap_vals.values
            else:
                sv = shap_vals
            mean_abs_shap = np.mean(np.abs(sv), axis=0)
            total_shap = np.sum(mean_abs_shap)
            if total_shap > 0:
                feat_imp = sorted(list(zip(cols, mean_abs_shap / total_shap)), key=lambda x: x[1], reverse=True)
            else:
                feat_imp = sorted(list(zip(cols, mean_abs_shap)), key=lambda x: x[1], reverse=True)
        except Exception as shap_err:
            print(f"Error computing SHAP: {shap_err}")

    if feat_imp is None:
        if hasattr(binary_model, "feature_importances_"):
            try:
                feat_imp = sorted(list(zip(cols, binary_model.feature_importances_)), key=lambda x: x[1], reverse=True)
            except Exception:
                pass

    if feat_imp:
        top_features = [f[0] for f in feat_imp[:3]]
        feature_importance = [{"name": f[0], "value": round(float(f[1]), 4)} for f in feat_imp[:6]]
    else:
        top_features = ["Destination Port", "Flow Duration", "Total Fwd Packets"]
        feature_importance = demo_feature_importance

    local_explanation = None
    if explainer is not None:
        try:
            attack_indices = np.where(binary_preds == 1)[0]
            if len(attack_indices) > 0:
                print("Computing SHAP explanations...")
                
                # Sample up to 200 attack flows to keep computation fast
                if len(attack_indices) > 200:
                    sampled_indices = attack_indices[:200]
                else:
                    sampled_indices = attack_indices

                X_scaled_attack = X_scaled[sampled_indices]
                shap_values = explainer.shap_values(X_scaled_attack)

                if isinstance(shap_values, list):
                    shap_matrix = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                elif hasattr(shap_values, "values"):
                    shap_matrix = shap_values.values
                    if len(shap_matrix.shape) == 3:
                        shap_matrix = shap_matrix[:, :, 1]
                elif isinstance(shap_values, np.ndarray):
                    if len(shap_values.shape) == 3:
                        shap_matrix = shap_values[:, :, 1]
                    else:
                        shap_matrix = shap_values
                else:
                    shap_matrix = np.array(shap_values)

                # Average contributions over all sampled attack flows
                mean_shap = np.mean(shap_matrix, axis=0)

                pos_features = []
                neg_features = []
                for col_name, val in zip(cols, mean_shap):
                    val_f = float(val)
                    if val_f > 0:
                        pos_features.append({"feature": col_name, "contribution": round(val_f, 4)})
                    elif val_f < 0:
                        neg_features.append({"feature": col_name, "contribution": round(val_f, 4)})

                pos_features = sorted(pos_features, key=lambda x: abs(x["contribution"]), reverse=True)[:5]
                neg_features = sorted(neg_features, key=lambda x: abs(x["contribution"]), reverse=True)[:5]

                local_explanation = {
                    "positive": pos_features,
                    "negative": neg_features,
                    "positive_features": pos_features,
                    "negative_features": neg_features
                }
        except Exception as shap_err:
            print(f"Error computing local SHAP explanation: {shap_err}")

    return {
        "is_attack": True,
        "stage1_confidence": round(stage1_conf, 4),
        "stage2_confidence": round(stage2_conf, 4),
        "flagged_flows": n_attack,
        "total_flows": total,
        "attack_type": attack_type,
        "attack_info": ATTACK_INFO.get(attack_type, ATTACK_INFO["DDoS"]),
        "class_distribution": class_distribution,
        "attack_breakdown": attack_breakdown,
        "top_features": top_features,
        "feature_importance": feature_importance,
        "demo_mode": False,
        "local_explanation": local_explanation
    }


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return app.send_static_file('dashboard.html')
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": binary_model is not None and multiclass_model is not None,
        "demo_mode": binary_model is None
    })


@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    filename = file.filename.lower()
    file_bytes = file.read()

    try:
        if filename.endswith(".csv"):
            df = preprocess_csv(file_bytes)
        elif filename.endswith(".pcap") or filename.endswith(".pcapng"):
            df = extract_features_from_pcap(file_bytes)
        else:
            return jsonify({"error": "Unsupported file type. Upload a .csv or .pcap file."}), 400

        if df.empty or len(df) == 0:
            return jsonify({"error": "No valid flows found in the uploaded file."}), 400

        explain = request.args.get("explain", "false").lower() == "true"
        result = run_detection(df, explain=explain)

        # DEBUGGING CODE
        try:
            debug_path = os.path.join(BASE_DIR, "debug_detect.txt")
            with open(debug_path, "w") as f:
                f.write(f"Filename: {filename}\n")
                f.write(f"DF shape: {df.shape}\n")
                f.write(f"Binary model type: {type(binary_model)}\n")
                f.write(f"Multiclass model type: {type(multiclass_model)}\n")
                if hasattr(binary_model, "feature_names_in_"):
                    f.write(f"Binary model feature_names_in_: {list(binary_model.feature_names_in_)}\n")
                if hasattr(multiclass_model, "feature_names_in_"):
                    f.write(f"Multiclass model feature_names_in_: {list(multiclass_model.feature_names_in_)}\n")
                
                # Re-run prediction locally to capture detailed info
                df_clean = df.replace([np.inf, -np.inf], np.nan).fillna(0)
                if scaler is not None and hasattr(scaler, "feature_names_in_"):
                    cols_dbg = list(scaler.feature_names_in_)
                elif binary_model is not None and hasattr(binary_model, "feature_names_in_"):
                    cols_dbg = list(binary_model.feature_names_in_)
                else:
                    cols_dbg = FEATURE_COLUMNS

                for c in cols_dbg:
                    if c not in df_clean.columns:
                        df_clean[c] = 0.0
                X = df_clean[cols_dbg].values
                if scaler is not None:
                    X_scaled = scaler.transform(X)
                else:
                    X_scaled = X
                
                f.write(f"X shape: {X_scaled.shape}\n")
                if binary_model is not None:
                    bin_preds = binary_model.predict(X_scaled)
                    bin_proba = binary_model.predict_proba(X_scaled)
                    f.write(f"Binary predictions: {list(bin_preds)}\n")
                    f.write(f"Binary probabilities (class 1): {list(bin_proba[:, 1])}\n")
                if multiclass_model is not None:
                    mc_classes = getattr(multiclass_model, "classes_", None)
                    f.write(f"Multiclass classes: {list(mc_classes) if mc_classes is not None else None}\n")
        except Exception as debug_err:
            print(f"Debug logging failed: {debug_err}")

        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except ImportError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


@app.route("/live/capture", methods=["POST"])
def live_capture():
    data = request.get_json() or {}
    duration = data.get("duration", 30)

    try:
        duration = int(duration)
        if duration < 5 or duration > 120:
            return jsonify({"error": "Duration must be between 5 and 120 seconds."}), 400
    except ValueError:
        return jsonify({"error": "Invalid duration parameter."}), 400

    try:
        from scapy.all import sniff, wrpcap
    except ImportError:
        return jsonify({"error": "scapy is not installed. Please install scapy using: pip install scapy"}), 500

    import tempfile

    tmp_path = None
    try:
        # Check permissions / start sniffing
        try:
            packets = sniff(timeout=duration)
            if not packets or len(packets) == 0:
                return jsonify({
                    "error": "No network packets were captured. Ensure there is active traffic on the monitored interface."
                }), 400

            with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as tmp:
                tmp_path = tmp.name

            wrpcap(tmp_path, packets)

            with open(tmp_path, "rb") as f:
                pcap_bytes = f.read()

            df = extract_features_from_pcap(pcap_bytes)
            simulated = False
        except Exception as e:
            # Fallback to simulation if raw capture is not permitted or fails (e.g. on Render)
            import random
            print(f"Packet sniffing failed ({e}). Falling back to simulated live traffic.")
            
            sample_files = ["sample_attack.csv", "sample_benign.csv", "netguard_sample_flows.csv"]
            loaded = False
            for sf in sample_files:
                sf_path = os.path.join(BASE_DIR, sf)
                if os.path.exists(sf_path):
                    try:
                        df = pd.read_csv(sf_path)
                        # Slice a dynamic random sample of flows
                        sample_size = min(len(df), random.randint(15, 45))
                        df = df.sample(n=sample_size).reset_index(drop=True)
                        loaded = True
                        break
                    except Exception as csv_err:
                        print(f"Failed to read sample csv {sf}: {csv_err}")
            
            if not loaded:
                total_rows = random.randint(10, 30)
                data_dict = {col: [random.uniform(0, 100) for _ in range(total_rows)] for col in FEATURE_COLUMNS}
                df = pd.DataFrame(data_dict)
            simulated = True

        explain = request.args.get("explain", "false").lower() == "true"
        result = run_detection(df, explain=explain)
        result["success"] = True
        result["simulated"] = simulated
        return jsonify(result)

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal processing error: {str(e)}"}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass


@app.route("/explain", methods=["POST"])
def generate_explanation():
    data = request.get_json() or {}
    
    # Extract necessary fields
    is_attack = data.get("is_attack", False)
    attack_type = data.get("attack_type", "Unknown")
    confidence_s1 = data.get("stage1_confidence", 0.0)
    confidence_s2 = data.get("stage2_confidence", 0.0)
    flagged_flows = data.get("flagged_flows", 0)
    total_flows = data.get("total_flows", 0)
    attack_breakdown = data.get("attack_breakdown", {})
    local_explanation = data.get("local_explanation")  # SHAP features
    
    # Extract top 3 features from local_explanation
    top_features = ""
    if local_explanation:
        pos_list = local_explanation.get("positive") or local_explanation.get("positive_features")
        if pos_list:
            top_features = "\n".join([
                f"  • {f['feature']}: 推高攻击概率 (+{f['contribution']:.4f})"
                for f in pos_list[:3]
            ])
    
    # If client is not configured, fallback to offline report generation
    if client is None:
        if is_attack:
            breakdown_str = ", ".join([f"{k}: {v} flows" for k, v in attack_breakdown.items()])
            explanation = f"""### 📋 AI Integrated Report

**[Analysis]**
The system has detected active **{attack_type}** network traffic patterns with high confidence (S1: {confidence_s1:.1%}, S2: {confidence_s2:.1%}). This suggests potential host compromise or targeted network intrusion attempts.

**[Attack Breakdown]**
- {breakdown_str if breakdown_str else "No breakdown available"}

**[Why it was flagged as an attack]**
The intrusion detection model flagged these flows primarily due to anomalies in the following features:
{top_features if top_features else "  • Anomalous traffic volume and packet length statistics."}

**[Action Recommendations]**
1. **[Immediate]** Isolate or block the source IP addresses identified in the traffic logs.
   *Command Example:* `netsh advfirewall firewall add rule name="Block Attack" dir=in action=block remoteip=<source_ip>`
2. **[Immediate]** Rate-limit incoming requests on the targeted destination ports.
3. **[Long-term]** Update intrusion detection signatures and configure firewalls to block port-scanning behaviors.
4. **[Long-term]** Conduct a vulnerability scan on target assets to ensure no services are exposed.

---
*Note: This is a local template-based report because the `DEEPSEEK_API_KEY` environment variable is not set. Configure it to enable live AI reports.*"""
        else:
            explanation = f"""### 📋 AI Integrated Report

The Network Intrusion Detection System analyzed {total_flows} network flows. **No intrusion was detected, and all traffic looks benign.**

* **Status:** Clean
* **System Confidence:** {confidence_s1:.2%}

All analyzed metrics (flow duration, packet size variance, flag distributions) remain well within normal thresholds. Keep maintaining standard security policies and routine diagnostics to ensure persistent security.

---
*Note: This is a local template-based report because the `DEEPSEEK_API_KEY` environment variable is not set. Configure it to enable live AI reports.*"""

        return jsonify({
            "success": True,
            "explanation": explanation
        })

    # Construct DeepSeek prompt
    if is_attack:
        breakdown_str = ", ".join([f"{k}: {v}" for k, v in attack_breakdown.items()])
        prompt = f"""
You are a cybersecurity AI. Based on the detection results, generate a response in English containing:

**[Analysis]**
Briefly explain the detected attack type and its severity.

**[Attack Breakdown]**
{breakdown_str}

**[Why it was flagged as attack]**
The model relied on these features:
{top_features if top_features else "Pattern-based detection."}

**[Action Recommendations]**
Provide 3-4 specific, actionable steps to mitigate the {attack_type} attack.
Order them from immediate to long-term. Include command examples if applicable (e.g., `block ip`).

Keep the total length between 200-300 words.
"""
    else:
        prompt = f"""
你是一个网络安全分析 AI。检测结果显示：**未检测到入侵，流量正常**。

置信度：{confidence_s1:.2%}

请用英文生成一段简洁的确认信息（约 50-80 字），让用户知道系统已检查了 {total_flows} 条网络流，未发现异常。
语气要积极，可以建议用户继续保持良好的安全习惯。
"""
    
    # Call DeepSeek API
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个网络安全分析专家，专门解读入侵检测系统结果并给出应对建议。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800,
            timeout=30
        )
        
        explanation = response.choices[0].message.content
        
        return jsonify({
            "success": True,
            "explanation": explanation
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    import webbrowser
    
    port = int(os.environ.get("PORT", 5000))
    # On Render, we must listen on 0.0.0.0
    is_render = os.environ.get("RENDER") is not None
    host = "0.0.0.0" if is_render else "127.0.0.1"
    
    # Only open browser in parent process, avoiding opening twice due to Werkzeug reloader
    if not os.environ.get("WERKZEUG_RUN_MAIN") and not is_render:
        webbrowser.open(f"http://localhost:{port}")
    
    print(f"NetGuard AI backend starting on http://{host}:{port}")
    app.run(debug=not is_render, host=host, port=port)

