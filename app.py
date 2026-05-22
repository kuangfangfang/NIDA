"""
NetGuard AI — Backend API
Two-stage Network Intrusion Detection System
Trained on CICIDS2017 | XGBoost models

This file handles HTTP routing only.
All ML/detection logic lives in detector.py.
"""
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import os

# ── Import detection engine ───────────────────────────────────────────────────
from detector import (
    FEATURE_COLUMNS,
    ATTACK_INFO,
    binary_model,
    multiclass_model,
    scaler,
    preprocess_csv,
    run_detection,
    extract_features_from_pcap,
)

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
            sample_files = ["sample_attack.csv", "sample_benign.csv", "netguard_sample_flows.csv"]
            existing_files = [sf for sf in sample_files if os.path.exists(os.path.join(BASE_DIR, sf))]
            loaded = False

            if existing_files:
                random.shuffle(existing_files)
                for sf in existing_files:
                    try:
                        df = pd.read_csv(os.path.join(BASE_DIR, sf))
                        # Slice a dynamic random sample of flows
                        sample_size = min(len(df), random.randint(15, 45))
                        df = df.sample(n=sample_size).reset_index(drop=True)
                        loaded = True
                        print(f"Simulation successfully loaded sample from: {sf}")
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
