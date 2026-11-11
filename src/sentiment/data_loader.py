"""Downloads reviews from Hugging Face and splits them for training."""

from dataclasses import dataclass
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from .config import Config
from .exceptions import DataLoadError
from .logger import get_logger

logger = get_logger(__name__)

@dataclass
class DataSplit:
    """The four lists the model needs: texts and labels, train and test."""
    X_train : list[str]
    X_test: list[str]
    y_train : list[int]
    y_test : list[int]

class DataLoader:
    """Fetches the review sample and splits it.

    Takes a Config so sample size and split ratio are never hardcoded here.
    """
    def __init__(self, config: Config | None = None)-> None:
        self.config = config or Config()

    def load_reviews(self)-> tuple[list[str], list[int]]:
        """Stream sample_size reviews. Returns (texts, labels).

        Title and content are joined, because short titles like
        "Stopped working" carry a lot of the sentiment.
        """
        cfg = self.config
        logger.info("Loading %d reviews from %s", cfg.sample_size, cfg.dataset_name)
        texts: list[str] = []
        labels: list[int] = []

        try:
            stream = load_dataset(cfg.dataset_name, split="train", streaming=True)
            stream = stream.shuffle(
                seed=cfg.random_state, buffer_size=cfg.sample_size
            ).take(cfg.sample_size)

            for row in stream:
                texts.append(f"{row['title']}. {row['content']}")
                labels.append(row["label"])
        except Exception as exc:
            raise DataLoadError(f"Could not load {cfg.dataset_name}: {exc}") from exc

        if not texts:
            raise DataLoadError("Dataset returned zero reviews.")

        logger.info("Loaded %d reviews", len(texts))
        return texts, labels

    def split(self)-> DataSplit:
        """Load the reviews and split them into train and test sets."""
        texts, labels = self.load_reviews()
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                texts,
                labels,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=labels, # keep the 50/50 balance in both parts
            ) 
        except ValueError as exc:
            raise DataLoadError(f"Could not split the data: {exc}") from exc

        logger.info("Split into %d train / %d test", len(X_train), len(X_test))
        return DataSplit(X_train, X_test, y_train, y_test)          
