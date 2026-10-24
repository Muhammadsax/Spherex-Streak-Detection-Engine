"""
app/main.py — Main Entry Point for Phase 5 Streamlit Application
================================================================
Redirects to the core interactive dashboard in app/streamlit_app.py.
Can be executed with:
    streamlit run app/main.py
or
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Ensure src/ and repo root are in sys.path for Streamlit Cloud deployment
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.streamlit_app import main

if __name__ == "__main__":
    main()
