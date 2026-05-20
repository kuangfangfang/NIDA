import requests
import json
import sys

# Ensure terminal outputs UTF-8 characters cleanly on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def test_explain_endpoint():
    print("=== Testing /explain with Attack Telemetry ===")
    attack_payload = {
        "is_attack": True,
        "attack_type": "PortScan",
        "stage1_confidence": 0.9989,
        "stage2_confidence": 0.9876,
        "flagged_flows": 28,
        "total_flows": 200,
        "attack_breakdown": {"PortScan": 22, "DDoS": 6},
        "local_explanation": {
            "positive": [
                {"feature": "Flow Duration", "contribution": 0.234},
                {"feature": "Packet Length Std", "contribution": 0.187}
            ]
        }
    }
    
    try:
        resp = requests.post("http://127.0.0.1:5000/explain", json=attack_payload)
        print(f"Status Code: {resp.status_code}")
        print("Response JSON:")
        print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error connecting to server: {e}")
        sys.exit(1)

    print("\n=== Testing /explain with Benign Telemetry ===")
    benign_payload = {
        "is_attack": False,
        "stage1_confidence": 0.9995,
        "total_flows": 1500,
        "local_explanation": None
    }
    
    try:
        resp = requests.post("http://127.0.0.1:5000/explain", json=benign_payload)
        print(f"Status Code: {resp.status_code}")
        print("Response JSON:")
        print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error connecting to server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_explain_endpoint()
