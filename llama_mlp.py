import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self, dim: int, hidden_dim: int, multiple_of: int, dropout: float
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * (
                (hidden_dim + multiple_of - 1) // multiple_of
            )
