"""Path utilities for consistent, repo-relative file access."""

from __future__ import annotations
from pathlib import Path

def project_root() -> Path:
    # Assumes this file lives in src/yelp_risk/
    return Path(__file__).resolve().parents[2]

def data_raw_dir() -> Path:
    return project_root() / "data" / "raw"

def data_processed_dir() -> Path:
    return project_root() / "data" / "processed"

def results_dir() -> Path:
    return project_root() / "results"

def figures_dir() -> Path:
    return project_root() / "figures"
