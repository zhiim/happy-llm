import torch.nn as nn

from config import ModelConfig
from llama_attention import Attention
from llama_mlp import MLP
from rms_norm import RMSNorm
from rope import precompute_freqs_cis


class DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, args: ModelConfig):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.ffn = MLP(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )

        self.layer_id = layer_id

        self.atten_norm = RMSNorm(dim=args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(dim=args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cis):
        # pre-norm
        x = x + self.attention(self.atten_norm(x), freqs_cis)
        x = x + self.ffn(self.ffn_norm(x))
        return x


if __name__ == "__main__":
    import torch

    args = ModelConfig()

    decoder_layer = DecoderLayer(0, args)

    dim = args.dim
    seq_len = 50

    x = torch.randn(2, seq_len, dim)

    freqs_cis = precompute_freqs_cis(dim // args.n_heads, seq_len)

    out = decoder_layer(x, freqs_cis)

    print(out.shape)
