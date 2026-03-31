"""Neural network predictors used by the deep training CLI."""

import torch
import torch.nn.functional as F
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
        self.input_dropout = nn.Dropout(dropout)
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
        x = self.input_dropout(x)
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


# ---------------------------------------------------------------------------
# MOMENT foundation model wrapper
# ---------------------------------------------------------------------------

_MOMENT_D_MODEL = {"small": 512, "base": 768, "large": 1024}
_MOMENT_SEQ_LEN = 512


class MomentPredictor(nn.Module):
    """MOMENT foundation model wrapper matching the RecurrentPredictor interface.

    Classification: uses MOMENT's built-in 2-class head, converts to scalar
    logit for BCEWithLogitsLoss compatibility.
    Regression: uses MOMENT in embedding mode with a custom linear head.
    """

    def __init__(
        self,
        n_channels,
        seq_len,
        task="classification",
        pos_weight=None,
        moment_size="large",
        freeze_backbone=True,
    ):
        super().__init__()
        from momentfm import MOMENTPipeline

        if seq_len > _MOMENT_SEQ_LEN:
            raise ValueError(
                f"MOMENT supports max {_MOMENT_SEQ_LEN} timesteps but got {seq_len}. "
                "Use a shorter window size (≤512s at 1Hz)."
            )

        self.task = task
        self.seq_len = seq_len
        d_model = _MOMENT_D_MODEL[moment_size]

        task_name = "classification" if task == "classification" else "embedding"
        model_kwargs = {
            "task_name": task_name,
            "n_channels": n_channels,
            "freeze_encoder": freeze_backbone,  # Freeze the patch embedding layer
            "freeze_embedder": freeze_backbone,  # Freeze the transformer encoder
            "freeze_head": False,  # The linear forecasting head must be trained
            ## NOTE: Disable gradient checkpointing to supress the warning when linear probing the model as MOMENT encoder is frozen
            "enable_gradient_checkpointing": not freeze_backbone,
            # Choose how embedding is obtained from the model: One of ['mean', 'concat']
            # Multi-channel embeddings are obtained by either averaging or concatenating patch embeddings
            # along the channel dimension. 'concat' results in embeddings of size (n_channels * d_model),
            # while 'mean' results in embeddings of size (d_model)
            "reduction": "mean",
        }

        if task == "classification":
            model_kwargs["num_class"] = 2

        self.moment = MOMENTPipeline.from_pretrained(
            f"AutonLab/MOMENT-1-{moment_size}",
            model_kwargs=model_kwargs,
        )
        self.moment.init()

        self.regression_head = None
        if task != "classification":
            self.regression_head = nn.Linear(d_model, 1)

        if pos_weight is None:
            self.register_buffer("pos_weight", None)
        else:
            self.register_buffer(
                "pos_weight",
                torch.tensor([float(pos_weight)], dtype=torch.float32),
            )

    def _pad_to_moment(self, x):
        """Reshape and pad input for MOMENT.

        Args:
            x: ``[batch, timesteps, channels]``

        Returns:
            padded: ``[batch, channels, 512]``
            mask: ``[batch, 512]`` with 1 for real positions, 0 for padding.
        """
        batch, t, c = x.shape
        # MOMENT expects [batch, n_channels, seq_len]
        x = x.transpose(1, 2)  # [batch, channels, timesteps]
        pad_len = _MOMENT_SEQ_LEN - t
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))  # pad last dim (time)
        mask = torch.zeros(batch, _MOMENT_SEQ_LEN, device=x.device, dtype=x.dtype)
        mask[:, :t] = 1.0
        return x, mask

    def forward(self, features=None, labels=None, x=None):
        """Support both Trainer-style dict inputs and direct tensor calls."""
        if features is None:
            features = x
        if features is None:
            raise ValueError("Expected `features` or `x` input for MOMENT model.")

        padded, mask = self._pad_to_moment(features)
        output = self.moment(x_enc=padded, input_mask=mask)

        if self.task == "classification":
            # MOMENT returns [batch, 2] logits; convert to scalar for BCE
            logits = output.logits[:, 1] - output.logits[:, 0]
        else:
            # Embedding mode: [batch, d_model] → scalar
            logits = self.regression_head(output.embeddings).squeeze(-1)

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
