"""核心正确性检查，不涉及训练或测试集选参。"""
import io
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from model import Config, GPT, SentimentModel


def run_tests():
    torch.manual_seed(2026)
    model = GPT(Config(vocab_size=64, context=16, width=32, heads=4, layers=2)).eval()
    x = torch.randint(3, 64, (2, 12))
    y = x.clone()
    y[:, 7:] = torch.randint(3, 64, (2, 5))
    a, _ = model(x)
    b, _ = model(y)
    error = (a[:, :7] - b[:, :7]).abs().max().item()
    assert error < 1e-6, '未来词元改变了前面位置的预测'
    assert model.token.weight is model.lm_head.weight
    targets = torch.randint(0, 64, x.shape)
    logits, loss = model(x, targets)
    assert torch.allclose(loss, F.cross_entropy(logits.flatten(0, 1), targets.flatten()))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = GPT(model.config).eval()
    restored.load_state_dict(torch.load(stream, weights_only=True))
    assert torch.allclose(model(x)[0], restored(x)[0])
    classifier = SentimentModel(model).eval()
    short = torch.tensor([[3, 4, 1]])
    padded = torch.tensor([[3, 4, 1, 0, 0]])
    assert torch.allclose(classifier(short, torch.tensor([3])),
                          classifier(padded, torch.tensor([3])), atol=1e-6)
    from experiment import EarlyStopping
    stopper = EarlyStopping(patience=2)
    for value in (1.0, 0.9, 0.91):
        stopper.update(value)
        assert not stopper.should_stop
    stopper.update(0.92)
    assert stopper.should_stop
    stopper.update(0.8)
    assert stopper.bad_checks == 0 and not stopper.should_stop
    result = dict(causal_max_difference=error, tied_embeddings=True,
                  loss_matches_cross_entropy=True, finite_gradients=True,
                  save_load_consistent=True, right_padding_invariant=True,
                  early_stopping_counter_and_reset=True)
    Path(__file__).with_name('model_tests.json').write_text(json.dumps(result, indent=2))
    print(result, flush=True)


if __name__ == '__main__':
    run_tests()
