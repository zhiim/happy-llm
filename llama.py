import math

import torch.nn as nn
from transformers import CausalLMOutputWithPast, PreTrainedModel

from config import ModelConfig
from llama_decoder import DecoderLayer
from rms_norm import RMSNorm
from rope import precompute_freqs_cis


class LLaMa(PreTrainedModel):
    config_class = ModelConfig

    def __init__(self, args: ModelConfig):
        super().__init__()

        self.args = args

        self.vocab_size = args.vocab_size

        self.n_layers = args.n_layers

        self.token_embedding = nn.Embedding(args.vocab_size, args.dim)

        self.dropout = nn.Dropout(args.dropout)

        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(DecoderLayer(layer_id=layer_id, args=args))

        self.norm = RMSNorm(dim=args.dim)

        self.output_layer = nn.Linear(args.dim, args.vocab_size, bias=False)

        # embedding 和 output layer 共享权重
        self.token_embedding.weight = self.output_layer.weight

        # 预计算 rotary embedding 的频率
        freq_cis = precompute_freqs_cis(
            dim=args.dim // args.n_heads, slen=args.max_seq_len
        )
        self.register_buffer("freqs_cis", freq_cis, persistent=False)

        self.apply(self._init_weights)  # 递归的作用于所有的子模块
        # 单独处理投影层的参数初始化
        for pn, p in self.named_parameters():
            if pn.endswith("w3.weight") or pn.endswith("wo.weight"):
                nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * args.n_layers)
                )

        self.last_loss = None
        self.OUT = CausalLMOutputWithPast()
        self._no_split_modules = [name for name, _ in self.named_modules()]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
