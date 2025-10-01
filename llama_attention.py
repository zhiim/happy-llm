import math

import torch
import torch.nn as nn

from config import ModelConfig
from rope import apply_rotary_emb, precompute_freqs_cis


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """用于将键值对重复n_rep次，扩展到和查询一致的维度，以实现kv cache"""
    bs, slen, n_kv_heads, head_dim = x.shape

    if n_rep == 1:
        return x

    # 在多头的维度上扩展
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
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
        # 多个head使用统一族kv
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        # q, k, v的线性变换
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(
            args.dim, args.n_kv_heads * self.head_dim, bias=False
        )  # k, v 被共享，直接变换到小维度
        self.wv = nn.Linear(
            args.dim, args.n_kv_heads * self.head_dim, bias=False
        )

        # 输出的线性变换
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        self.res_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        self.flash = hasattr(nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: Flash Attention not available, using manual attention"
            )
            mask = torch.full(
                (1, 1, args.max_seq_len, args.max_seq_len), float("-inf")
            )
            mask = torch.triu(mask, diagonal=1)  # casual mask
            self.register_buffer("mask", mask)

    def forward(self, x, freqs_cis):
        bs, slen, _ = x.shape

        # 计算q，k，v
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # 正常的多头
        xq = xq.view(bs, slen, self.n_local_heads, self.head_dim)
        # 只是用少数的头
        xk = xk.view(bs, slen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bs, slen, self.n_local_kv_heads, self.head_dim)

        # 添加rope
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # 扩展k, v到和q一样的头数
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if self.flash:
            output = nn.functional.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(
                self.head_dim
            )
            assert hasattr(self, "mask")
            scores = scores + self.mask[:, :, :slen, :slen]
            scores = nn.functional.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)

        output = output.transpose(1, 2).contiguous().view(bs, slen, -1)

        output = self.wo(output)
        output = self.res_dropout(output)
        return output


if __name__ == "__main__":
    from config import ModelConfig

    args = ModelConfig()

    attn = Attention(args)

    batch_size = 2
    seq_len = 50
    dim = args.dim

    x = torch.randn(batch_size, seq_len, dim)

    freqs_cis = precompute_freqs_cis(dim // args.n_heads, seq_len)

    output = attn(x, freqs_cis)

    print(output.shape)
