"""Deep learning utilities for recurrent sequence classification.

The package is intentionally small:
- ``models`` defines the recurrent classifiers
- ``data`` loads and reshapes repository datasets into Hugging Face datasets
- ``trainer`` contains the Trainer subclass, metrics, and logging helpers
- ``train`` provides the CLI entrypoint and training orchestration
"""

from .models import GRUClassifier, LSTMClassifier

__all__ = ["LSTMClassifier", "GRUClassifier"]
