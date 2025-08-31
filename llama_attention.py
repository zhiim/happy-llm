import torch
import torch.nn as nn

from config import ModelConfig


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """用于将键值对重复n_rep次，扩展到和查询一致的维度"""
    bs, slen, n_kv_heads, head_dim = x.shape

    if n_rep == 1:
        return x

    # 在多头的维度上扩展
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshpae(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()

        # GQA每组的头数
        self.n_kv_heads = (
            args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        )
        assert args.n_heads % self.n_kv_heads == 0

        # 并行处理
        model_parallel_size = 1
        # 本地头数
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        # q, k, v的线性变换
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(
            args.dim, args.n_kv_heads * self.head_dim, bias=False
        )  # k, v 被共享
        self.wv = nn.Linear(
            args.dim, args.n_kv_heads * self.head_dim, bias=False
        )

        # 输出的线性变换
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        self.res_dropout = nn.Dropout(args.dropout)

        self.flash = hasattr(nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: Flash Attention not available, using manual attention."
            )
            mask = torch.full(
                (1, 1, args.max_seq_len, args.max_seq_len), float("-inf")
            )
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

    def forward(self, x, freqs_cis):
        pass
