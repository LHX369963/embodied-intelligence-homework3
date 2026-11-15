"""从随机初始化构建的 Qwen3 / GPT 语言模型与情感分类模型；仅使用 PyTorch 基础算子。"""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    arch: str = 'qwen3'
    vocab_size: int = 32768
    context: int = 512
    width: int = 1024
    heads: int = 16
    kv_heads: int = 8
    head_dim: int = 128
    ffn_width: int = 2880
    layers: int = 28
    dropout: float = 0.0
    rope_theta: float = 10000.0
    rms_eps: float = 1e-6
    qk_norm: bool = True
    attn_gate: bool = True


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (out * self.weight.float()).to(x.dtype)


def build_rope_cache(seq_len: int, head_dim: int, theta: float = 10000.0, device=None):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_out = (q.float() * cos + rotate_half(q.float()) * sin).to(q.dtype)
    k_out = (k.float() * cos + rotate_half(k.float()) * sin).to(k.dtype)
    return q_out, k_out


class QwenAttention(nn.Module):
    """集成 2:1 GQA、逐头 QK-Norm、RoPE 与 Sigmoid 输出门控的因果自注意力。"""
    def __init__(self, config: Config):
        super().__init__()
        if config.heads % config.kv_heads != 0:
            raise ValueError('查询头数必须整除键值头数')
        self.heads = config.heads
        self.kv_heads = config.kv_heads
        self.head_dim = config.head_dim
        self.q_dim = config.heads * config.head_dim
        self.kv_dim = config.kv_heads * config.head_dim
        self.dropout = config.dropout
        self.q_proj = nn.Linear(config.width, self.q_dim, bias=False)
        self.k_proj = nn.Linear(config.width, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(config.width, self.kv_dim, bias=False)
        self.gate_proj = nn.Linear(config.width, self.q_dim, bias=False) if config.attn_gate else None
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_eps) if config.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_eps) if config.qk_norm else nn.Identity()
        self.o_proj = nn.Linear(self.q_dim, config.width, bias=False)

    def forward(self, x, cos, sin, kv_cache=None, use_cache=False):
        b, t, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(b, t, self.heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(b, t, self.kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        if kv_cache is not None:
            k = torch.cat((kv_cache[0], k), dim=2)
            v = torch.cat((kv_cache[1], v), dim=2)
        causal = kv_cache is None and t > 1
        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=causal,
            dropout_p=self.dropout if self.training else 0.0,
            enable_gqa=(self.kv_heads != self.heads),
        )
        y = y.transpose(1, 2).contiguous().view(b, t, self.q_dim)
        if self.gate_proj is not None:
            y = y * torch.sigmoid(self.gate_proj(x))
        out = self.o_proj(y)
        return (out, (k, v)) if use_cache else out


class SwiGLU(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.width, config.ffn_width, bias=False)
        self.up_proj = nn.Linear(config.width, config.ffn_width, bias=False)
        self.down_proj = nn.Linear(config.ffn_width, config.width, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.norm1 = RMSNorm(config.width, eps=config.rms_eps)
        self.norm2 = RMSNorm(config.width, eps=config.rms_eps)
        self.attn = QwenAttention(config)
        self.ffn = SwiGLU(config)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x, cos, sin, kv_cache=None, use_cache=False):
        if use_cache:
            attn_out, next_cache = self.attn(self.norm1(x), cos, sin, kv_cache=kv_cache, use_cache=True)
            x = x + self.drop(attn_out)
            x = x + self.drop(self.ffn(self.norm2(x)))
            return x, next_cache
        x = x + self.drop(self.attn(self.norm1(x), cos, sin))
        return x + self.drop(self.ffn(self.norm2(x)))


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
        self.drop = nn.Dropout(config.dropout)
        if config.arch == 'qwen3':
            cos, sin = build_rope_cache(config.context, config.head_dim, config.rope_theta)
            self.register_buffer('rope_cos', cos, persistent=False)
            self.register_buffer('rope_sin', sin, persistent=False)
            self.blocks = nn.ModuleList([QwenBlock(config) for _ in range(config.layers)])
            self.norm = RMSNorm(config.width, eps=config.rms_eps)
        else:
            self.position = nn.Embedding(config.context, config.width)
            self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
            self.norm = nn.LayerNorm(config.width)
        self.lm_head = nn.Linear(config.width, config.vocab_size, bias=False)
        self.apply(self._init)
        self.lm_head.weight = self.token.weight
        scale = 0.02 / math.sqrt(2 * config.layers)
        for name, parameter in self.named_parameters():
            if name.endswith(('attn.proj.weight', 'ffn.2.weight', 'o_proj.weight', 'down_proj.weight')):
                nn.init.normal_(parameter, std=scale)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def encode(self, ids, kv_caches=None, use_cache=False):
        past_len = 0 if kv_caches is None else kv_caches[0][0].size(2)
        seq_len = ids.shape[1]
        if past_len + seq_len > self.config.context:
            raise ValueError('输入序列超过模型最大上下文长度')
        if self.config.arch == 'qwen3':
            x = self.drop(self.token(ids))
            cos = self.rope_cos[past_len:past_len + seq_len]
            sin = self.rope_sin[past_len:past_len + seq_len]
            if not use_cache:
                for block in self.blocks:
                    x = block(x, cos, sin)
                return self.norm(x)
            next_caches = []
            for i, block in enumerate(self.blocks):
                layer_cache = None if kv_caches is None else kv_caches[i]
                x, updated = block(x, cos, sin, kv_cache=layer_cache, use_cache=True)
                next_caches.append(updated)
            return self.norm(x), next_caches

        positions = torch.arange(past_len, past_len + seq_len, device=ids.device)
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
        if self.backbone.config.arch == 'qwen3':
            pos = (lengths - 2).clamp(min=0)
        else:
            pos = lengths - 1
        pooled = hidden[torch.arange(len(ids), device=ids.device), pos]
        return self.head(pooled)


def _parse_config(raw_cfg: dict, dropout=None) -> Config:
    cfg = dict(raw_cfg)
    if 'arch' not in cfg:
        cfg['arch'] = 'gpt2'
    if dropout is not None:
        cfg['dropout'] = dropout
    return Config(**cfg)


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


def load_gpt(path, device='cpu', dtype=None, dropout=None):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    target_dtype = default_infer_dtype(device) if dtype is None else dtype
    model = GPT(_parse_config(saved['config'], dropout=dropout)).to(dtype=target_dtype)
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
    backbone = GPT(_parse_config(saved['config']))
    model = SentimentModel(backbone).to(dtype=target_dtype)
    state = {
        k.removeprefix('_orig_mod.'): (v.to(target_dtype) if v.is_floating_point() else v)
        for k, v in saved['model'].items()
    }
    model.load_state_dict(state)
    return model.to(device).eval(), saved
