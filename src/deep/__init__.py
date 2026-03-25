"""Deep learning utilities for recurrent sequence classification.

The package is intentionally small:
- ``models`` defines the recurrent classifiers
- ``data`` loads and reshapes repository datasets into Hugging Face datasets
- ``train`` provides the Trainer-based CLI entrypoint with grouped validation
"""

from .models import GRUClassifier, LSTMClassifier

__all__ = ["LSTMClassifier", "GRUClassifier"]
