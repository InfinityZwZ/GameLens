"""Stable project paths, independent of working directory and module depth."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
