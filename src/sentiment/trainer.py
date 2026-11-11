"""Builds, trains and saves the sentiment model."""

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import Config
from .exceptions import ModelTrainingError
from .logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """Turns training reviews into a saved model file."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()

    def build_pipeline(self) -> Pipeline:
        """Empty model: TF-IDF turns text into numbers, then the classifier.

        ngram_range=(1, 2) keeps single words and two-word phrases, so
        "not good" is seen as its own feature and not just "not" and "good".
        """
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=self.config.max_features,
            )),
            ("clf", LogisticRegression(max_iter=1000)),
        ])

    def train(self, X_train: list[str], y_train: list[int]) -> Pipeline:
        """Fit a fresh pipeline on the training data and return it."""
        if not X_train or len(X_train) != len(y_train):
            raise ModelTrainingError("Training texts and labels are empty or mismatched.")

        logger.info("Training on %d reviews", len(X_train))
        model = self.build_pipeline()
        try:
            model.fit(X_train, y_train)
        except Exception as exc:
            raise ModelTrainingError(f"Training failed: {exc}") from exc

        logger.info("Training finished")
        return model

    def save(self, model: Pipeline, path: Path | None = None) -> Path:
        """Write the model to disk. Creates the folder if it's missing."""
        path = path or self.config.model_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, path)
        except OSError as exc:
            raise ModelTrainingError(f"Could not save model to {path}: {exc}") from exc

        logger.info("Model saved to %s", path)
        return path