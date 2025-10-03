import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self, dim: int, hidden_dim: int | None, multiple_of: int, dropout: float
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * (
                (hidden_dim + multiple_of - 1) // multiple_of
            )

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = nn.functional.silu(self.w1(x))
        x3 = self.w3(x)
        x = self.w2(x1 * x3)
        x = self.dropout(x)
        return x
