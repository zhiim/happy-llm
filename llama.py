import math
from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from config import ModelConfig
from llama_decoder import DecoderLayer
from rms_norm import RMSNorm
from rope import precompute_freqs_cis


class LLaMa(PreTrainedModel):
    config_class = ModelConfig

    def __init__(self, args: ModelConfig):
        super().__init__(args)

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

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        - tokens: Optional[torch.Tensor], 输入 token 张量。
        - targets: Optional[torch.Tensor], 目标 token 张量。
        - kv_cache: bool, 是否使用键值缓存。
        - kwargs: 其他关键字参数。

        - self.OUT: CausalLMOutputWithPast, 包含 logits 和损失。
        """
        if "input_ids" in kwargs:
            tokens = kwargs["input_ids"]

        _bsz, seqlen = tokens.size()
        # 将每个词元 embedding
        h = self.token_embedding(tokens)  # [b, s, d]
        h = self.dropout(h)
        # 获取与输入序列长度匹配的 rotary embedding 频率
        freqs_cis = self.freqs_cis[:seqlen]

        # 依次通过每一层解码器
        for layer in self.layers:
            h = layer(h, freqs_cis=freqs_cis)
        h = self.norm(h)

        if targets is not None:
            logits = self.output_layer(h)
            self.last_loss = nn.functional.cross_entropy(
                # 将所有句子的所有词元展开，便于计算
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,
                reduction="none",
            )
        else:
            # 只对最后一个词元进行预测
            logits = self.output_layer(h[:, [-1], :])
            self.last_loss = None

        self.OUT.__setitem__("logits", logits)
        self.OUT.__setitem__("last_loss", self.last_loss)

        return self.OUT

    @torch.inference_mode()
    def generate(
        self, idx, stop_id=None, max_new_tokens=256, temperature=1.0, top_k=None
    ):
        index = idx.shape[1]
        for _ in range(max_new_tokens):
            # 仅使用最后 max_seq_len 个 token 进行预测
            idx_cond = (
                idx
                if idx.size(1) <= self.args.max_seq_len
                else idx[:, -self.args.max_seq_len :]
            )

            logits = self(idx_cond).logits
            logits = logits[:, -1, :]  # 只保留最脏一个词元的预测结果

            if temperature == 0.0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # 对 logits 进行温度缩放
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    # 只使用 topk 计算 softmax
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = nn.functional.softmax(logits, dim=-1)
                # 根据概率分布采样下一个词元的idx
                idx_next = torch.multinomial(probs, num_samples=1)

            # 下一个词元预测为 end
            if idx_next == stop_id:
                break

            # 将新预测的词元拼接到输入序列后面，参与下次预测
            idx = torch.cat((idx, idx_next), dim=1)

        return idx[:, index:]  # 只返回 model 的 response


if __name__ == "__main__":
    x = torch.randint(0, 6144, (2, 50))

    args = ModelConfig()
    model = LLaMa(args=args)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params / 1e6:.2f}M")

    out = model(x)
    print(out.logits.shape)
