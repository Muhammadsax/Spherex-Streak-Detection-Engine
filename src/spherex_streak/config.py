"""
config.py — نقطة التحميل المركزية لمتغيرات البيئة
كل ملفات الكود تستورد من هنا فقط، ولا أحد يقرأ .env مباشرة
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# تحميل .env من جذر المشروع
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")


# ---------------------------
# مسارات البيانات
# ---------------------------
DATA_DIR       = Path(os.getenv("DATA_DIR", "data/raw"))
PROCESSED_DIR  = Path(os.getenv("PROCESSED_DIR", "data/processed"))
MASKS_DIR      = Path(os.getenv("MASKS_DIR", "data/masks"))

# ---------------------------
# مجموعة بيانات SPARKLES
# ---------------------------
ZENODO_RECORD_ID     = os.getenv("ZENODO_RECORD_ID", "10935280")
ZENODO_ACCESS_TOKEN  = os.getenv("ZENODO_ACCESS_TOKEN")   # ← يُقرأ من .env فقط

# ---------------------------
# إعدادات النموذج
# ---------------------------
MODEL_CHECKPOINT = Path(os.getenv("MODEL_CHECKPOINT", "checkpoints/deepstreaks_best.pt"))
DEVICE           = os.getenv("DEVICE", "cpu")

# ---------------------------
# إعدادات التدريب
# ---------------------------
BATCH_SIZE    = int(os.getenv("BATCH_SIZE", 16))
NUM_EPOCHS    = int(os.getenv("NUM_EPOCHS", 50))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 1e-4))
FOCAL_GAMMA   = float(os.getenv("FOCAL_GAMMA", 2.0))
FOCAL_ALPHA   = float(os.getenv("FOCAL_ALPHA", 0.25))

# ---------------------------
# إعدادات PQM
# ---------------------------
PQM_THRESHOLD = float(os.getenv("PQM_THRESHOLD", 0.5))

# ---------------------------
# Streamlit UI
# ---------------------------
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", 8501))
