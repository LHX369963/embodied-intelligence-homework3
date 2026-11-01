"""从随机参数构建的小型 GPT；仅使用 PyTorch 基础模块。"""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab_size: int = 4096
    context: int = 128
    width: int = 256
    heads: int = 4
    layers: int = 4
    dropout: float = 0.1


class CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.proj = nn.Linear(config.width, config.width)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [a.view(b, t, self.heads, c // self.heads).transpose(1, 2)
                   for a in (q, k, v)]
        # is_causal=True 屏蔽未来位置；评估和生成时关闭注意力 Dropout。
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).contiguous().view(b, t, c))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.width)
        self.norm2 = nn.LayerNorm(config.width)
        self.attn = CausalAttention(config)
        self.ffn = nn.Sequential(
            nn.Linear(config.width, 4 * config.width), nn.GELU(),
            nn.Linear(4 * config.width, config.width))
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        return x + self.drop(self.ffn(self.norm2(x)))


class GPT(nn.Module):
    def __init__(self, config=Config()):
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.width)
        self.position = nn.Embedding(config.context, config.width)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.width)
        self.lm_head = nn.Linear(config.width, config.vocab_size, bias=False)
        self.apply(self._init)
        self.lm_head.weight = self.token.weight
        # 对残差分支的输出投影缩小初始化尺度。
        for name, parameter in self.named_parameters():
            if name.endswith('proj.weight') or name.endswith('ffn.2.weight'):
                nn.init.normal_(parameter, std=0.02 / math.sqrt(2 * config.layers))

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def encode(self, ids):
        if ids.shape[1] > self.config.context:
            raise ValueError('输入超过模型上下文长度')
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.drop(self.token(ids) + self.position(positions))
        return self.norm(self.blocks(x))

    def forward(self, ids, targets=None):
        logits = self.lm_head(self.encode(ids))
        loss = None if targets is None else F.cross_entropy(
            logits.flatten(0, 1), targets.flatten())
        return logits, loss

    @torch.no_grad()
    def generate(self, ids, new_tokens=80, temperature=0.8, top_k=40, eos_id=1):
        self.eval()
        for _ in range(new_tokens):
            logits, _ = self(ids[:, -self.config.context:])
            scores = logits[:, -1] / temperature
            cutoff = scores.topk(min(top_k, scores.size(-1))).values[:, -1:]
            scores = scores.masked_fill(scores < cutoff, float('-inf'))
            nxt = torch.multinomial(scores.softmax(-1), num_samples=1)
            ids = torch.cat((ids, nxt), dim=1)
            if int(nxt.item()) == eos_id:
                break
        return ids


class SentimentModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.config.width, 2)

    def forward(self, ids, lengths):
        hidden = self.backbone.encode(ids)
        # 右侧 padding；因果注意力保证有效位置不读取右侧 padding。
        pooled = hidden[torch.arange(len(ids), device=ids.device), lengths - 1]
        return self.head(pooled)


def load_gpt(path, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    model = GPT(Config(**saved['config']))
    model.load_state_dict(saved['model'])
    return model.to(device).eval(), saved


def save_gpt(model, path, **metadata):
    torch.save(dict(config=asdict(model.config), model=model.state_dict(), **metadata), path)
