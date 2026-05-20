import pickle
import os
import pandas as pd
import numpy as np
import shap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(filename):
    path = os.path.join(BASE_DIR, "models", filename)
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

# Load scaler & model
scaler = load_model("scaler.pkl")
binary_model = load_model("binary_model.pkl")

print("Binary Model:", type(binary_model))
print("Scaler:", type(scaler))

# Load sample attack
df = pd.read_csv("sample_attack.csv")
print("DF shape:", df.shape)

# Preprocess columns
FEATURE_COLUMNS = list(binary_model.feature_names_in_)
X = df[FEATURE_COLUMNS].values
X_scaled = scaler.transform(X)

# Let's mock all rows as attacks to force local explanation computation
attack_indices = np.arange(len(X_scaled))

explainer = shap.TreeExplainer(binary_model)
print("TreeExplainer created:", explainer)

try:
    X_scaled_attack = X_scaled[attack_indices]
    print("X_scaled_attack shape:", X_scaled_attack.shape)
    shap_values = explainer.shap_values(X_scaled_attack)
    print("shap_values type:", type(shap_values))
    if isinstance(shap_values, list):
        print("shap_values list length:", len(shap_values))
        for idx, s in enumerate(shap_values):
            print(f"  [{idx}] shape:", s.shape)
    elif hasattr(shap_values, "shape"):
        print("shap_values shape:", shap_values.shape)
    else:
        print("shap_values has no shape/list properties")
except Exception as e:
    import traceback
    print("Error computing SHAP values:")
    traceback.print_exc()
