import torch


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """用于将键值对重复n_rep次，扩展到和查询一致的维度"""
    bs, slen, n_kv_heads, head_dim = x.shape

    if n_rep == 1:
        return x

    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshpae(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (torch.arange(0, dim, 2)[: dim // 2].float() / dim)
    t = torch.arange(end, devcie=freqs.device)
    freqs = torch.out(t, freqs).float()
