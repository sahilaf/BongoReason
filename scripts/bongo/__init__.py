"""Shared library for the BongoReason data pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATASET = ROOT / "dataset"

__all__ = ["ROOT", "DATASET"]
