"""Recurrent neural network predictors used by the deep training CLI."""

import torch
from torch import nn


class RecurrentPredictor(nn.Module):
    """Shared recurrent backbone with a scalar prediction head.

    The model consumes tensors shaped ``[batch, timesteps, channels]`` and
    predicts one score per window. The task decides whether that score is a
    classification logit or a regression output.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        dropout,
        bidirectional,
        rnn_type,
        task="classification",
        pos_weight=None,
    ):
        super().__init__()
        self.task = task
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = rnn_type(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=bidirectional,
        )
        output_dim = hidden_dim * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(output_dim)
        self.head = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, 1),
        )
        if pos_weight is None:
            self.register_buffer("pos_weight", None)
        else:
            self.register_buffer(
                "pos_weight",
                torch.tensor([float(pos_weight)], dtype=torch.float32),
            )

    def _forward_logits(self, x):
        """Return one scalar prediction for each sequence window."""
        _, hidden = self.rnn(x)
        if isinstance(hidden, tuple):
            hidden = hidden[0]

        # Use the final hidden state from the last recurrent layer. For
        # bidirectional models we concatenate the last forward/backward states.
        if self.rnn.bidirectional:
            final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            final_hidden = hidden[-1]

        final_hidden = self.norm(final_hidden)
        logits = self.head(final_hidden).squeeze(-1)
        return logits

    def forward(self, features=None, labels=None, x=None):
        """Support both Trainer-style dict inputs and direct tensor calls."""
        if features is None:
            features = x
        if features is None:
            raise ValueError("Expected `features` or `x` input for recurrent model.")

        logits = self._forward_logits(features)
        if labels is None:
            return logits

        labels = labels.to(logits.dtype)
        if self.task == "classification":
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        else:
            loss_fn = nn.MSELoss()

        return {
            "loss": loss_fn(logits, labels),
            "logits": logits,
        }


class LSTMClassifier(RecurrentPredictor):
    """LSTM-backed recurrent predictor."""

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        bidirectional=False,
        task="classification",
        pos_weight=None,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            rnn_type=nn.LSTM,
            task=task,
            pos_weight=pos_weight,
        )


class GRUClassifier(RecurrentPredictor):
    """GRU-backed recurrent predictor."""

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        bidirectional=False,
        task="classification",
        pos_weight=None,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            rnn_type=nn.GRU,
            task=task,
            pos_weight=pos_weight,
        )
