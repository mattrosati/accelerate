"""Deep learning utilities for recurrent sequence classification.

The package is intentionally small:
- ``models`` defines the recurrent classifiers
- ``data`` loads and reshapes repository datasets into sequence tensors
- ``train`` provides the CLI entrypoint with grouped validation and W&B logging
"""

from .models import GRUClassifier, LSTMClassifier

__all__ = ["LSTMClassifier", "GRUClassifier"]
