from pathlib import Path
import joblib

# 🔥 AUTO ROOT FINDER (works on Cloud + local)
ROOT = Path(__file__).resolve()

while not (ROOT / "data").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent

PROJECT_ROOT = ROOT
MODEL_PATH = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data" / "processed"


# =========================
# SAFE MODEL LOADER
# =========================
def safe_load_model(filename):
    try:
        path = MODEL_PATH / filename
        if path.exists():
            return joblib.load(path)
    except:
        return None
    return None


# =========================
# SAFE DATA LOADER
# =========================
def get_data_path(filename):
    return DATA_PATH / filename