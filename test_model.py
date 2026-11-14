"""模型结构、因果掩码、KV 缓存一致性、权重共享与保存加载单元测试。"""
from pathlib import Path
import json
import tempfile
import torch
from torch.nn import functional as F
from model import (
    Config, GPT, SentimentModel,
    load_classifier, load_gpt, save_classifier, save_gpt,
)


def run_tests():
    torch.manual_seed(2026)
    cfg = Config(vocab_size=128, context=32, width=64, heads=4, layers=3, dropout=0.0)
    model = GPT(cfg).eval()

    x = torch.randint(3, 128, (2, 14))
    y = x.clone()
    y[:, 8:] = torch.randint(3, 128, (2, 6))
    a, _ = model(x)
    b, _ = model(y)
    causal_diff = (a[:, :8] - b[:, :8]).abs().max().item()
    assert causal_diff < 1e-6, '未来位置输入改变了前序位置的预测结果'

    assert model.token.weight is model.lm_head.weight

    full_logits, _ = model(x)
    h_prefill, caches = model.encode(x[:, :10], use_cache=True)
    step_logits = [model.lm_head(h_prefill)]
    for pos in range(10, x.shape[1]):
        h_step, caches = model.encode(x[:, pos:pos + 1], kv_caches=caches, use_cache=True)
        step_logits.append(model.lm_head(h_step))
    cached_logits = torch.cat(step_logits, dim=1)
    kv_diff = (full_logits - cached_logits).abs().max().item()
    assert kv_diff < 1e-5, f'KV 缓存增量解码与完整前向不一致: {kv_diff}'

    targets = torch.randint(0, 128, x.shape)
    logits, loss = model(x, targets)
    assert torch.allclose(loss, F.cross_entropy(logits.flatten(0, 1), targets.flatten()))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

    with tempfile.TemporaryDirectory() as tmp:
        gpt_path = Path(tmp) / 'gpt.pt'
        save_gpt(model, gpt_path, save_dtype=torch.float32, step=10)
        restored, meta = load_gpt(gpt_path, device='cpu', dtype=torch.float32)
        assert meta['step'] == 10
        assert restored.token.weight is restored.lm_head.weight
        assert torch.allclose(model(x)[0], restored(x)[0], atol=1e-6)

        classifier = SentimentModel(model).eval()
        short = torch.tensor([[3, 4, 5, 1]])
        padded = torch.tensor([[3, 4, 5, 1, 0, 0, 0]])
        assert torch.allclose(
            classifier(short, torch.tensor([4])),
            classifier(padded, torch.tensor([4])),
            atol=1e-6,
        )

        cls_path = Path(tmp) / 'cls.pt'
        save_classifier(classifier, cls_path, save_dtype=torch.bfloat16, epoch=2)
        restored_cls, cls_meta = load_classifier(cls_path, device='cpu', dtype=torch.float32)
        assert cls_meta['epoch'] == 2
        assert restored_cls(short, torch.tensor([4])).shape == (1, 2)

    from experiment import EarlyStopping
    stopper = EarlyStopping(patience=2)
    for value in (1.0, 0.9, 0.91):
        stopper.update(value)
        assert not stopper.should_stop
    stopper.update(0.92)
    assert stopper.should_stop
    stopper.update(0.8)
    assert stopper.bad_checks == 0 and not stopper.should_stop

    default_gpt_params = (
        Config().vocab_size * Config().width
        + Config().context * Config().width
        + Config().layers * (
            4 * Config().width
            + 3 * Config().width * Config().width + 3 * Config().width
            + Config().width * Config().width + Config().width
            + 4 * Config().width * Config().width + 4 * Config().width
            + 4 * Config().width * Config().width + Config().width
        )
        + 2 * Config().width
    )

    result = dict(
        default_config_parameters=default_gpt_params,
        causal_max_difference=causal_diff,
        kv_cache_max_difference=kv_diff,
        tied_embeddings=True,
        loss_matches_cross_entropy=True,
        finite_gradients=True,
        save_load_consistent=True,
        right_padding_invariant=True,
        early_stopping_counter_and_reset=True,
    )
    Path(__file__).with_name('model_tests.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    run_tests()
