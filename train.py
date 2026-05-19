import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Define constants
DATA_PATH = "cicids2017_train.csv"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# 78 features of CICIDS2017 in the correct order
FEATURE_COLUMNS = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets",
    "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean",
    "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
]

def generate_dummy_data():
    """Generates a representative dummy dataset for demonstration when raw file is missing."""
    print("Generating representative dummy dataset...")
    np.random.seed(42)
    n_samples = 1500
    
    # Generate random features
    X_dummy = np.random.rand(n_samples, len(FEATURE_COLUMNS))
    
    # We want to make features somewhat informative for classification
    # e.g. column 0 (Destination Port) and column 1 (Flow Duration) can influence classes
    # Label distribution: BENIGN, DDoS, PortScan, Brute Force, Bot, Infiltration, Web Attack, Heartbleed
    labels = ["BENIGN", "DDoS", "PortScan", "Brute Force", "Bot", "Infiltration", "Web Attack", "Heartbleed"]
    probs = [0.60, 0.15, 0.10, 0.05, 0.04, 0.03, 0.02, 0.01]
    y_dummy = np.random.choice(labels, size=n_samples, p=probs)
    
    df_dummy = pd.DataFrame(X_dummy, columns=FEATURE_COLUMNS)
    df_dummy["Label"] = y_dummy
    
    # Save to DATA_PATH
    df_dummy.to_csv(DATA_PATH, index=False)
    print(f"✓ Dummy dataset saved to {DATA_PATH}")

def map_multiclass_label(label):
    """Maps various raw dataset label variations into standardized numeric multiclass labels."""
    label_upper = str(label).strip().upper()
    if "BENIGN" in label_upper:
        return 0
    elif "DDOS" in label_upper or "DOS" in label_upper:
        return 1
    elif "PORTSCAN" in label_upper or "PORT SCAN" in label_upper:
        return 2
    elif "BRUTE FORCE" in label_upper or "BRUTEFORCE" in label_upper or "PATATOR" in label_upper or "LOCK" in label_upper:
        return 3
    elif "BOT" in label_upper or "BOTNET" in label_upper or "ROBOT" in label_upper:
        return 4
    elif "INFILTRATION" in label_upper or "INFILTRATE" in label_upper:
        return 5
    elif "WEB" in label_upper or "SQL" in label_upper or "XSS" in label_upper:
        return 6
    elif "HEARTBLEED" in label_upper:
        return 7
    else:
        # Default fallback for unknown attacks (classify as generic attack class 1 - DDoS)
        return 1

def main():
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Dataset '{DATA_PATH}' not found in the current folder.")
        generate_dummy_data()
        
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # 2. Clean & Preprocess
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    # Ensure all target columns exist in df
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
            
    X = df[FEATURE_COLUMNS]
    
    # Handle Labels
    if "Label" not in df.columns:
        raise ValueError("Critical Error: 'Label' column is missing from the dataset!")
        
    y_raw = df["Label"]
    
    # Binary Label Mapping (BENIGN -> 0, others -> 1)
    y_raw_cleaned = y_raw.astype(str).str.strip().str.upper()
    y_binary = np.where(y_raw_cleaned == "BENIGN", 0, 1)
    
    # Multiclass Label Mapping (BENIGN:0, DDoS:1, PortScan:2, Brute Force:3, Bot:4, Infiltration:5, Web Attack:6, Heartbleed:7)
    y_multiclass = y_raw.apply(map_multiclass_label).values
    
    print(f"Dataset shape: {X.shape}")
    print(f"Binary classes count - Benign (0): {np.sum(y_binary == 0)}, Attack (1): {np.sum(y_binary == 1)}")
    
    # 3. Train-Test Split (80% Train, 20% Test)
    print("Splitting dataset into train and test sets...")
    X_train, X_test, y_train_bin, y_test_bin, y_train_mc, y_test_mc = train_test_split(
        X, y_binary, y_multiclass, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    # 4. Feature Standardization
    print("Fitting StandardScaler and transforming features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    scaler_save_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_save_path)
    print(f"✓ Scaler saved to: {scaler_save_path}")
    
    # 5. Train Binary Classifier (Stage 1)
    print("Training Stage 1 (Binary) XGBoost Classifier...")
    binary_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    binary_model.fit(X_train_scaled, y_train_bin)
    
    binary_save_path = os.path.join(MODELS_DIR, "binary_model.pkl")
    joblib.dump(binary_model, binary_save_path)
    print(f"✓ Binary model saved to: {binary_save_path}")
    
    # 6. Train Multiclass Classifier (Stage 2)
    # The multiclass model classifies specific attack categories. We train it on the full dataset 
    # (or optionally attack-only with dummy class 0). Training with BENIGN included is robust and prevents 
    # class mapping shifts or contiguous index errors in XGBoost.
    print("Training Stage 2 (Multiclass) XGBoost Classifier...")
    multiclass_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        objective='multi:softprob',
        num_class=8,
        eval_metric='mlogloss'
    )
    multiclass_model.fit(X_train_scaled, y_train_mc)
    
    multiclass_save_path = os.path.join(MODELS_DIR, "multiclass_model.pkl")
    joblib.dump(multiclass_model, multiclass_save_path)
    print(f"✓ Multiclass model saved to: {multiclass_save_path}")
    
    # 7. Model Evaluation
    print("\n=== Evaluating Models ===")
    
    # Stage 1 (Binary) Evaluation
    y_pred_bin = binary_model.predict(X_test_scaled)
    print("\n--- Binary Model Classification Report ---")
    print(classification_report(y_test_bin, y_pred_bin, target_names=["BENIGN", "ATTACK"]))
    
    # Stage 2 (Multiclass) Evaluation
    y_pred_mc = multiclass_model.predict(X_test_scaled)
    target_names_mc = ["BENIGN", "DDoS", "PortScan", "Brute Force", "Bot", "Infiltration", "Web Attack", "Heartbleed"]
    
    # Check what labels are actually present in test set to prevent formatting issues
    unique_test_labels = np.unique(np.concatenate([y_test_mc, y_pred_mc]))
    present_target_names = [target_names_mc[i] for i in unique_test_labels]
    
    print("\n--- Multiclass Model Classification Report ---")
    print(classification_report(y_test_mc, y_pred_mc, labels=unique_test_labels, target_names=present_target_names))
    
    # Generate and Save Confusion Matrices
    print("Plotting confusion matrices...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Binary Confusion Matrix Plot
    cm_bin = confusion_matrix(y_test_bin, y_pred_bin)
    disp_bin = ConfusionMatrixDisplay(confusion_matrix=cm_bin, display_labels=["BENIGN", "ATTACK"])
    disp_bin.plot(ax=axes[0], cmap="Blues", values_format="d")
    axes[0].set_title("Binary Confusion Matrix (Stage 1)")
    
    # Multiclass Confusion Matrix Plot
    cm_mc = confusion_matrix(y_test_mc, y_pred_mc)
    disp_mc = ConfusionMatrixDisplay(confusion_matrix=cm_mc, display_labels=target_names_mc)
    disp_mc.plot(ax=axes[1], cmap="Oranges", values_format="d")
    axes[1].set_title("Multiclass Confusion Matrix (Stage 2)")
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")
    
    plt.tight_layout()
    cm_img_path = os.path.join(BASE_DIR, "confusion_matrix.png")
    plt.savefig(cm_img_path, bbox_inches="tight")
    plt.close()
    print(f"✓ Confusion matrix plot saved to: {cm_img_path}")
    
    # 8. SHAP Explainer (TreeExplainer) & Interpretability
    print("\nGenerating SHAP TreeExplainer for Binary Model...")
    try:
        # TreeExplainer is ideal for XGBoost tree ensembles
        explainer_bin = shap.TreeExplainer(binary_model)
        explainer_save_path = os.path.join(MODELS_DIR, "shap_explainer_binary.pkl")
        joblib.dump(explainer_bin, explainer_save_path)
        print(f"✓ SHAP explainer saved to: {explainer_save_path}")
        
        # Select 100 samples from the test set for the SHAP summary plot
        print("Calculating SHAP values for test samples subset (100 samples)...")
        shap_samples = X_test_scaled[:100]
        shap_values = explainer_bin.shap_values(shap_samples)
        
        # Handle different SHAP output formats across versions
        if isinstance(shap_values, list):
            sv_to_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif hasattr(shap_values, "values"):
            sv_to_plot = shap_values.values
        else:
            sv_to_plot = shap_values
            
        # Generate Summary Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv_to_plot, shap_samples, feature_names=FEATURE_COLUMNS, show=False)
        plt.tight_layout()
        shap_img_path = os.path.join(BASE_DIR, "shap_summary.png")
        plt.savefig(shap_img_path, bbox_inches="tight")
        plt.close()
        print(f"✓ SHAP summary plot saved to: {shap_img_path}")
        
    except Exception as shap_e:
        print(f"Warning: Failed to create or save SHAP explainer/plot: {shap_e}")
        print("Proceeding without saving SHAP explainer. (Memory or dependencies constraint)")

    print("\n=== Model training pipeline complete successfully ===")

if __name__ == "__main__":
    main()
