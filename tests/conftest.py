"""
tests/conftest.py — Pytest shared configuration and fixtures
============================================================
Sets up test environment:
- Forces matplotlib into headless 'Agg' backend to avoid Tcl/Tk dependency
  in CI and cloud test environments.
"""

import matplotlib
matplotlib.use("Agg")  # Must be called before any other matplotlib import
