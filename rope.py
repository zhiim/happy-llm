import torch


def precompute_freqs_cis(dim: int, slen: int, theta=10000.0):
    freqs = 1.0 / (theta ** torch.arange(0, dim, 2)[: dim // 2].float() / dim)
    seq_idx = torch.arange(slen)  # 序列索引
    freqs = torch.outer(seq_idx, freqs).float()  # m \theta
    return torch.polar(torch.ones_like(freqs), freqs)  # 返回复数形式


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
):
    xq_reshape = xq.view(*xq.shape[:-1], -1, 2)
    xk_reshape = xk.view(*xk.shape[:-1], -1, 2)

    xq_complex = torch.view_as_complex(xq_reshape)
    xk_complex = torch.view_as_complex(xk_reshape)

    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(-2)

    return xq_out, xk_out
