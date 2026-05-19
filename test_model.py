import os
import joblib
import pandas as pd
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SAMPLE_PATH = os.path.join(os.path.dirname(BASE_DIR), "sample_attack.csv")

def main():
    print("=== NIDS Model Test Script ===")
    
    # 1. Load models & scaler
    print("Loading models and scaler...")
    try:
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        binary_model = joblib.load(os.path.join(MODELS_DIR, "binary_model.pkl"))
        multiclass_model = joblib.load(os.path.join(MODELS_DIR, "multiclass_model.pkl"))
        print("✓ Scaler and models loaded successfully.")
    except Exception as e:
        print(f"Error loading models from {MODELS_DIR}: {e}")
        return

    # Load SHAP explainer
    shap_explainer = None
    try:
        shap_explainer = joblib.load(os.path.join(MODELS_DIR, "shap_explainer_binary.pkl"))
        print("✓ SHAP explainer loaded successfully.")
    except Exception as e:
        print(f"Warning: SHAP explainer could not be loaded: {e}")

    # 2. Load test CSV
    global SAMPLE_PATH
    if not os.path.exists(SAMPLE_PATH):
        # Fall back to root folder or current folder
        if os.path.exists("sample_attack.csv"):
            SAMPLE_PATH = "sample_attack.csv"
        else:
            print(f"Error: test CSV file 'sample_attack.csv' not found at {SAMPLE_PATH}.")
            return
            
    print(f"Reading test data from {SAMPLE_PATH}...")
    df = pd.read_csv(SAMPLE_PATH)
    print(f"Data shape: {df.shape}")

    # 3. Preprocess and align features
    print("Aligning features...")
    # Import FEATURE_COLUMNS from app.py
    import sys
    sys.path.append(BASE_DIR)
    from app import FEATURE_COLUMNS
    
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.strip()
    
    # Remove label column if present
    for col in ["Label", "label", "Class", "class"]:
        if col in df_clean.columns:
            df_clean.drop(columns=[col], inplace=True)
            
    for col in FEATURE_COLUMNS:
        if col not in df_clean.columns:
            df_clean[col] = 0.0
            
    X = df_clean[FEATURE_COLUMNS].values

    # 4. Standardize/scale features
    print("Scaling features using scaler.transform()...")
    X_scaled = scaler.transform(X)

    # 5. Run predictions
    print("\n--- Running Inference ---")
    binary_preds = binary_model.predict(X_scaled)
    binary_proba = binary_model.predict_proba(X_scaled)
    
    n_attack = np.sum(binary_preds == 1)
    total = len(X_scaled)
    
    print(f"Total flows tested: {total}")
    print(f"Flagged as Attack (1): {n_attack} / {total} ({(n_attack/total)*100:.2f}%)")
    print(f"Flagged as Benign (0): {total - n_attack} / {total} ({((total - n_attack)/total)*100:.2f}%)")

    if n_attack > 0:
        # Predict attack types
        X_attack_scaled = X_scaled[binary_preds == 1]
        mc_preds = multiclass_model.predict(X_attack_scaled)
        
        label_map = {0:"BENIGN", 1:"DDoS", 2:"PortScan", 3:"Brute Force", 4:"Bot",
                     5:"Infiltration", 6:"Web Attack", 7:"Heartbleed"}
        
        unique, counts = np.unique(mc_preds, return_counts=True)
        print("\nPredicted Attack Breakdown:")
        for val, count in zip(unique, counts):
            lbl = label_map.get(int(val), str(val))
            print(f" - {lbl}: {count} flows")

        # Explain top features for the attacks using SHAP
        if shap_explainer is not None:
            try:
                print("\nCalculating SHAP feature importances for attack flows...")
                shap_vals = shap_explainer.shap_values(X_attack_scaled)
                if isinstance(shap_vals, list):
                    sv = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
                elif hasattr(shap_vals, "values"):
                    sv = shap_vals.values
                else:
                    sv = shap_vals
                mean_abs_shap = np.mean(np.abs(sv), axis=0)
                feat_imp = sorted(list(zip(FEATURE_COLUMNS, mean_abs_shap)), key=lambda x: x[1], reverse=True)
                print("Top 5 contributing features for flagged attacks:")
                for i, (feat, val) in enumerate(feat_imp[:5]):
                    print(f" {i+1}. {feat} (mean |SHAP| = {val:.6f})")
            except Exception as e:
                print(f"Could not compute SHAP explanations: {e}")
    else:
        print("No attack flows flagged.")

    print("\n=== Test script executed successfully ===")

if __name__ == "__main__":
    main()
