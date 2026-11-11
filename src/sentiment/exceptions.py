"""Errors specific to this project.

Everything inherits from SentimentError, so the API can catch one
type and handle all of our failures in one place.
"""

class SentimentError(Exception):
    """Parent of every error we raise ourselves."""

class DataLoadError(SentimentError):
    """Dataset download or parsing failed (usually network trouble)."""
    
class ModelTrainingError(SentimentError):
    """Something went wrong during fit() or while saving the model."""


class ModelNotFoundError(SentimentError):
    """No model file on disk. Run scripts/train.py first."""


class PredictionError(SentimentError):
    """Model is loaded but could not score the given text."""    