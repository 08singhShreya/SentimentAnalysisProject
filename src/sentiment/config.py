"""Project settings in one place.

Change numbers here, not inside the training code.
"""
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class Config:
    """Read-only settings (frozen, so nothing can edit them mid-run)."""

    dataset_name: str = "amazon_polarity"
    sample_size: int = 10_000   #laptop constraint
    test_size: float = 0.2      
    random_state: int = 42      # same split every run, results are repeatable
    max_features: int = 20_000  # caps vocabulary size, keeps the model small
    model_dir: Path = PROJECT_ROOT / "models"
    model_path: Path = PROJECT_ROOT / "models" / "model.joblib"