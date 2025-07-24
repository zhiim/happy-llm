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
    """计算旋转矩阵

    Args:
        dim: 特征的维度
        end: 序列的长度
        theta: 旋转矩阵的缩放因子，默认值为10000.0
    """
    # e^{-2(i - 1)/d}
    freqs = 1.0 / (theta ** torch.arange(0, dim, 2)[: dim // 2].float() / dim)
    t = torch.arange(end, devcie=freqs.device)  # 序列索引
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin


def reshape_for_broadcast(freq_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim

    assert 1 < ndim
    assert freq_cis.shape == (x.shape[1], x.shape[-1])

    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freq_cis.view(shape)  # 为旋转矩阵扩展维度，使其可以broadcast


def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
    # xq.shape[:-1] + (-1, 2): (bs, slen, -1 , 2)
    # 将最后一个维度拆分成两个维度
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(dim=-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(dim=-1)

    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # 计算旋转编码
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_sin - xk_i * freqs_sin
    xk_out_i = 
