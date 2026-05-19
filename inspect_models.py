import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(filename):
    path = os.path.join(BASE_DIR, "models", filename)
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

bm = load_model("binary_model.pkl")
mm = load_model("multiclass_model.pkl")

print("Binary model type:", type(bm))
print("Multiclass model type:", type(mm))

if hasattr(bm, "steps"):
    print("Binary model steps:", bm.steps)
if hasattr(mm, "steps"):
    print("Multiclass model steps:", mm.steps)
