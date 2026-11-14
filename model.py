"""从随机初始化构建的 GPT 语言模型与情感分类模型；仅使用 PyTorch 基础算子。"""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab_size: int = 32768
    context: int = 512
    width: int = 1280
    heads: int = 20
    layers: int = 24
    dropout: float = 0.1


class CausalAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        if config.width % config.heads != 0:
            raise ValueError('隐藏维度必须整除注意力头数')
        self.heads = config.heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.proj = nn.Linear(config.width, config.width)

    def forward(self, x, kv_cache=None, use_cache=False):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        head_dim = c // self.heads
        q = q.view(b, t, self.heads, head_dim).transpose(1, 2)
        k = k.view(b, t, self.heads, head_dim).transpose(1, 2)
        v = v.view(b, t, self.heads, head_dim).transpose(1, 2)
        if kv_cache is not None:
            k = torch.cat((kv_cache[0], k), dim=2)
            v = torch.cat((kv_cache[1], v), dim=2)
        causal = kv_cache is None and t > 1
        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = self.proj(y.transpose(1, 2).contiguous().view(b, t, c))
        return (out, (k, v)) if use_cache else out


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.width)
        self.norm2 = nn.LayerNorm(config.width)
        self.attn = CausalAttention(config)
        self.ffn = nn.Sequential(
            nn.Linear(config.width, 4 * config.width),
            nn.GELU(),
            nn.Linear(4 * config.width, config.width),
        )
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x, kv_cache=None, use_cache=False):
        if use_cache:
            attn_out, next_cache = self.attn(self.norm1(x), kv_cache=kv_cache, use_cache=True)
            x = x + self.drop(attn_out)
            x = x + self.drop(self.ffn(self.norm2(x)))
            return x, next_cache
        x = x + self.drop(self.attn(self.norm1(x)))
        return x + self.drop(self.ffn(self.norm2(x)))


class GPT(nn.Module):
    def __init__(self, config: Config = Config()):
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.width)
        self.position = nn.Embedding(config.context, config.width)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.width)
        self.lm_head = nn.Linear(config.width, config.vocab_size, bias=False)
        self.apply(self._init)
        self.lm_head.weight = self.token.weight
        scale = 0.02 / math.sqrt(2 * config.layers)
        for name, parameter in self.named_parameters():
            if name.endswith('proj.weight') or name.endswith('ffn.2.weight'):
                nn.init.normal_(parameter, std=scale)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def encode(self, ids, kv_caches=None, use_cache=False):
        past_len = 0 if kv_caches is None else kv_caches[0][0].size(2)
        if past_len + ids.shape[1] > self.config.context:
            raise ValueError('输入序列超过模型最大上下文长度')
        positions = torch.arange(past_len, past_len + ids.shape[1], device=ids.device)
        x = self.drop(self.token(ids) + self.position(positions))
        if not use_cache:
            for block in self.blocks:
                x = block(x)
            return self.norm(x)
        next_caches = []
        for i, block in enumerate(self.blocks):
            layer_cache = None if kv_caches is None else kv_caches[i]
            x, updated = block(x, kv_cache=layer_cache, use_cache=True)
            next_caches.append(updated)
        return self.norm(x), next_caches

    def forward(self, ids, targets=None):
        logits = self.lm_head(self.encode(ids))
        loss = None if targets is None else F.cross_entropy(
            logits.flatten(0, 1), targets.flatten()
        )
        return logits, loss

    @torch.no_grad()
    def generate(self, ids, new_tokens=80, temperature=0.8, top_k=40, eos_id=1):
        self.eval()
        kv_caches = None
        for _ in range(new_tokens):
            if kv_caches is not None and kv_caches[0][0].size(2) >= self.config.context:
                kv_caches = None
            if kv_caches is None:
                window = ids[:, -self.config.context:]
                hidden, kv_caches = self.encode(window, use_cache=True)
            else:
                hidden, kv_caches = self.encode(ids[:, -1:], kv_caches=kv_caches, use_cache=True)
            logits = self.lm_head(hidden[:, -1:])
            if temperature <= 0:
                nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
            else:
                scores = logits[:, -1] / temperature
                if top_k is not None and top_k > 0:
                    cutoff = scores.topk(min(top_k, scores.size(-1))).values[:, -1:]
                    scores = scores.masked_fill(scores < cutoff, float('-inf'))
                nxt = torch.multinomial(scores.softmax(-1), num_samples=1)
            ids = torch.cat((ids, nxt), dim=1)
            if eos_id is not None and int(nxt.item()) == eos_id:
                break
        return ids


class SentimentModel(nn.Module):
    def __init__(self, backbone: GPT):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.config.width, 2)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, ids, lengths):
        hidden = self.backbone.encode(ids)
        pooled = hidden[torch.arange(len(ids), device=ids.device), lengths - 1]
        return self.head(pooled)


def _clean_state_dict(module: nn.Module, save_dtype=torch.bfloat16):
    raw = module._orig_mod if hasattr(module, '_orig_mod') else module
    state = {}
    memo = {}
    for k, v in raw.state_dict().items():
        sig = (v.data_ptr(), v.shape, v.stride())
        if sig in memo:
            state[k] = memo[sig]
            continue
        tensor = v.detach().cpu()
        if tensor.is_floating_point() and save_dtype is not None:
            tensor = tensor.to(save_dtype)
        memo[sig] = tensor
        state[k] = tensor
    return state


def default_infer_dtype(device):
    dev = torch.device(device)
    if dev.type == 'cuda':
        return torch.bfloat16
    return torch.float32


def save_gpt(model: GPT, path, save_dtype=torch.bfloat16, **metadata):
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    payload = dict(
        config=asdict(raw.config),
        storage_dtype=str(save_dtype).replace('torch.', ''),
        model=_clean_state_dict(raw, save_dtype=save_dtype),
        **metadata,
    )
    torch.save(payload, path)


def load_gpt(path, device='cpu', dtype=None):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    target_dtype = default_infer_dtype(device) if dtype is None else dtype
    model = GPT(Config(**saved['config'])).to(dtype=target_dtype)
    state = {
        k.removeprefix('_orig_mod.'): (v.to(target_dtype) if v.is_floating_point() else v)
        for k, v in saved['model'].items()
    }
    model.load_state_dict(state)
    return model.to(device).eval(), saved


def save_classifier(model: SentimentModel, path, save_dtype=torch.bfloat16, **metadata):
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    backbone = raw.backbone._orig_mod if hasattr(raw.backbone, '_orig_mod') else raw.backbone
    payload = dict(
        config=asdict(backbone.config),
        storage_dtype=str(save_dtype).replace('torch.', ''),
        model=_clean_state_dict(raw, save_dtype=save_dtype),
        **metadata,
    )
    torch.save(payload, path)


def load_classifier(path, device='cpu', dtype=None):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    target_dtype = default_infer_dtype(device) if dtype is None else dtype
    backbone = GPT(Config(**saved['config']))
    model = SentimentModel(backbone).to(dtype=target_dtype)
    state = {
        k.removeprefix('_orig_mod.'): (v.to(target_dtype) if v.is_floating_point() else v)
        for k, v in saved['model'].items()
    }
    model.load_state_dict(state)
    return model.to(device).eval(), saved
