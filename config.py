from transformers import PretrainedConfig


class ModelConfig(PretrainedConfig):
    """Model config.

    Attributes:
        dim: 模型维度
        n_layers: Transformer的层数
        n_heads: 注意力机制的头数
        n_kv_heads: 键值头的数量
        vocab_size: 词汇表大小
        hidden_dim: 隐藏层维度
        multiple_of:
        norm_eps: 归一化层的eps
        max_seq_len: 最大序列长度
        dropout: dropout概率
        flash_attn: 是否使用Flash Attention
    """

    model_type = "Tiny-K"

    def __init__(
        self,
        dim: int = 768,
        n_layers: int = 12,
        n_heads: int = 16,
        n_kv_heads: int = 8,
        vocab_size: int = 6144,
        hidden_dim: int = None,
        multiple_of: int = 64,
        norm_eps: float = 1e-5,
        max_seq_len: int = 512,
        dropout: float = 0.0,
        flash_attn: bool = True,
        **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        super().__init__(**kwargs)
