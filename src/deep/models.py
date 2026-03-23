import torch
from torch import nn


class RecurrentClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        dropout,
        bidirectional,
        rnn_type,
    ):
        super().__init__()
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

    def forward(self, x):
        _, hidden = self.rnn(x)
        if isinstance(hidden, tuple):
            hidden = hidden[0]

        if self.rnn.bidirectional:
            final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            final_hidden = hidden[-1]

        final_hidden = self.norm(final_hidden)
        logits = self.head(final_hidden).squeeze(-1)
        return logits


class LSTMClassifier(RecurrentClassifier):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        bidirectional=False,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            rnn_type=nn.LSTM,
        )


class GRUClassifier(RecurrentClassifier):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        bidirectional=False,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            rnn_type=nn.GRU,
        )
