"""按作业顺序运行预训练、监督微调和最终评估。"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from tokenizers import Tokenizer
from model import Config, GPT, SentimentModel, load_gpt, save_gpt

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = ROOT / 'checkpoints'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 2026


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)


class EarlyStopping:
    """验证损失连续 patience 次评估保持在历史最低值以上时停止。"""
    def __init__(self, patience=2, best=float('inf')):
        self.patience = patience
        self.best = best
        self.bad_checks = 0

    def update(self, loss):
        improved = loss < self.best
        if improved:
            self.best = loss
            self.bad_checks = 0
        else:
            self.bad_checks += 1
        return improved

    @property
    def should_stop(self):
        return self.bad_checks >= self.patience


def write_json(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def read_tokens(split):
    return torch.from_numpy(np.load(ROOT / 'data' / f'wiki_{split}.npy').astype(np.int64)).to(DEVICE)


@torch.no_grad()
def language_evaluate(model, tokens, batch_size=64):
    """遍历连续、不重叠的完整上下文块，按有效词元数累计损失。"""
    model.eval()
    context = model.config.context
    blocks = (len(tokens) - 1) // context
    total = count = 0
    for start in range(0, blocks, batch_size):
        offsets = torch.arange(start, min(start + batch_size, blocks), device=DEVICE) * context
        indices = offsets[:, None] + torch.arange(context, device=DEVICE)
        _, loss = model(tokens[indices], tokens[indices + 1])
        n = indices.numel()
        total += loss.item() * n
        count += n
    value = total / count
    return dict(loss=value, perplexity=math.exp(value), evaluated_tokens=count,
                omitted_tail_tokens=len(tokens) - 1 - count)


def pretrain(steps=20000):
    seed_all()
    CHECKPOINTS.mkdir(exist_ok=True)
    tokenizer = Tokenizer.from_file(str(ROOT / 'tokenizer.json'))
    config = Config(vocab_size=tokenizer.get_vocab_size())
    model = GPT(config).to(DEVICE)
    train, val = read_tokens('train'), read_tokens('validation')
    settings = dict(seed=SEED, max_steps=steps, batch_size=64, context=config.context,
                    peak_lr=6e-4, minimum_lr=6e-5, warmup_steps=100,
                    lr_decay_steps=2500, validation_interval=250, patience=2,
                    weight_decay=0.1, grad_clip=1.0, optimizer='AdamW', precision='float32')
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['peak_lr'],
                                  betas=(0.9, 0.95), weight_decay=0.1)
    initial = language_evaluate(model, val)
    record = dict(config=asdict(config), settings=settings,
                  parameters=sum(p.numel() for p in model.parameters()),
                  initial_validation=initial, history=[],
                  initial_test=None)
    # 初始模型测试留待最终统一评价；此处只观察验证集。
    prompts = ['The history of', 'In the early years', 'The film was']
    def samples():
        torch.manual_seed(SEED)
        values = []
        for prompt in prompts:
            ids = torch.tensor([tokenizer.encode(prompt).ids], device=DEVICE)
            output = model.generate(ids, new_tokens=70)
            values.append(dict(prompt=prompt, text=tokenizer.decode(output[0].tolist())))
        return values
    record['initial_generation'] = samples()
    torch.manual_seed(SEED + 1)
    stopper = EarlyStopping(patience=settings['patience'], best=initial['loss'])
    save_gpt(model, CHECKPOINTS / 'pretrained.pt', step=0, validation_loss=stopper.best)
    rolling = []
    record['stop_reason'] = 'max_steps'
    print('PRETRAIN', record['parameters'], 'parameters', flush=True)
    for step in range(1, steps + 1):
        model.train()
        if step <= 100:
            rate = settings['peak_lr'] * step / 100
        else:
            # 保持原学习率计划：第2500步降至下限，之后维持下限。
            ratio = min(1.0, (step - 100) / (settings['lr_decay_steps'] - 100))
            rate = settings['minimum_lr'] + 0.5 * (settings['peak_lr'] - settings['minimum_lr']) * (1 + math.cos(math.pi * ratio))
        for group in optimizer.param_groups:
            group['lr'] = rate
        starts = torch.randint(0, len(train) - config.context, (64,), device=DEVICE)
        indices = starts[:, None] + torch.arange(config.context, device=DEVICE)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(train[indices], train[indices + 1])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        rolling.append(loss.item())
        if step % 50 == 0:
            print(f'PRETRAIN step={step}/{steps} loss={sum(rolling[-50:])/len(rolling[-50:]):.4f}', flush=True)
        if step % 250 == 0 or step == steps:
            metrics = language_evaluate(model, val)
            row = dict(step=step, train_loss=sum(rolling[-250:]) / len(rolling[-250:]),
                       validation_loss=metrics['loss'], validation_perplexity=metrics['perplexity'],
                       learning_rate=rate)
            if stopper.update(metrics['loss']):
                save_gpt(model, CHECKPOINTS / 'pretrained.pt', step=step, validation_loss=stopper.best)
            row['bad_checks'] = stopper.bad_checks
            record['history'].append(row)
            write_json('pretraining.json', record)
            print('VALIDATION', json.dumps(row), flush=True)
            if stopper.should_stop:
                record['stop_reason'] = 'validation_early_stopping'
                print('EARLY STOP PRETRAIN', step, flush=True)
                break
    record['stopped_step'] = step
    record['sampled_training_tokens'] = step * 64 * config.context
    model, saved = load_gpt(CHECKPOINTS / 'pretrained.pt', DEVICE)
    record['selected_step'] = saved['step']
    record['validation'] = language_evaluate(model, val)
    record['generation'] = samples()
    test = read_tokens('test')
    record['test'] = language_evaluate(model, test)
    seed_all()
    initial_model = GPT(config).to(DEVICE)
    record['initial_test'] = language_evaluate(initial_model, test)
    write_json('pretraining.json', record)
    print('PRETRAIN COMPLETE', record['test'], flush=True)


def encode_reviews(tokenizer, context):
    datasets = {}
    for split in ('train', 'validation', 'test'):
        rows = pq.read_table(ROOT / 'data' / f'sentiment_{split}.parquet').to_pylist()
        ids = torch.zeros((len(rows), context), dtype=torch.long)
        lengths = torch.empty(len(rows), dtype=torch.long)
        labels = torch.tensor([row['label'] for row in rows])
        truncated = 0
        for i, row in enumerate(rows):
            content = tokenizer.encode(row['text']).ids
            truncated += len(content) > context - 1
            content = content[:context - 1] + [tokenizer.token_to_id('<eos>')]
            ids[i, :len(content)] = torch.tensor(content)
            lengths[i] = len(content)
        datasets[split] = dict(ids=ids.to(DEVICE), lengths=lengths.to(DEVICE),
                               labels=labels.to(DEVICE), rows=rows, truncated=truncated)
    return datasets


def metrics_from_predictions(labels, predictions):
    matrix = [[sum(y == i and p == j for y, p in zip(labels, predictions))
               for j in range(2)] for i in range(2)]
    scores = []
    for i in range(2):
        tp = matrix[i][i]
        fp = matrix[1-i][i]
        fn = matrix[i][1-i]
        scores.append(2 * tp / max(1, 2 * tp + fp + fn))
    return dict(accuracy=sum(y == p for y, p in zip(labels, predictions)) / len(labels),
                macro_f1=sum(scores) / 2, confusion_matrix=matrix,
                correct=sum(y == p for y, p in zip(labels, predictions)), examples=len(labels))


@torch.no_grad()
def classify_evaluate(model, dataset):
    model.eval()
    losses = 0.0
    predictions, probabilities = [], []
    size = len(dataset['labels'])
    for start in range(0, size, 64):
        stop = min(start + 64, size)
        logits = model(dataset['ids'][start:stop], dataset['lengths'][start:stop])
        losses += nn.functional.cross_entropy(logits, dataset['labels'][start:stop], reduction='sum').item()
        predictions += logits.argmax(-1).tolist()
        probabilities += logits.softmax(-1)[:, 1].tolist()
    metrics = metrics_from_predictions(dataset['labels'].tolist(), predictions)
    metrics['loss'] = losses / size
    return metrics, predictions, probabilities


def finetune(epochs=50):
    tokenizer = Tokenizer.from_file(str(ROOT / 'tokenizer.json'))
    config = Config(vocab_size=tokenizer.get_vocab_size())
    datasets = encode_reviews(tokenizer, config.context)
    record = dict(seed=SEED, max_epochs=epochs, patience=2, batch_size=64, backbone_lr=2e-4,
                  head_lr=1e-3, weight_decay=0.01, grad_clip=1.0,
                  classification_pooling='last real token (EOS)',
                  truncated={k:v['truncated'] for k,v in datasets.items()}, variants={})
    train, val, test = [datasets[k] for k in ('train','validation','test')]
    for name in ('frozen', 'scratch', 'full'):
        seed_all()
        if name == 'scratch':
            backbone = GPT(config).to(DEVICE)
        else:
            backbone, _ = load_gpt(CHECKPOINTS / 'pretrained.pt', DEVICE)
        # 独立固定分类头的初值，使三组在相同起点比较。
        torch.manual_seed(SEED + 10)
        model = SentimentModel(backbone).to(DEVICE)
        if name == 'frozen':
            for parameter in model.backbone.parameters():
                parameter.requires_grad = False
        groups = [dict(params=model.head.parameters(), lr=1e-3)]
        if name != 'frozen':
            groups.append(dict(params=model.backbone.parameters(), lr=2e-4))
        optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
        variant = dict(history=[], initial_validation=classify_evaluate(model, val)[0],
                       trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        stopper = EarlyStopping(patience=record['patience'])
        variant['stop_reason'] = 'max_epochs'
        for epoch in range(1, epochs + 1):
            model.train()
            if name == 'frozen':
                model.backbone.eval()
            generator = torch.Generator().manual_seed(SEED + epoch)
            permutation = torch.randperm(len(train['labels']), generator=generator).to(DEVICE)
            losses = count = 0
            for start in range(0, len(permutation), 64):
                indices = permutation[start:start + 64]
                optimizer.zero_grad(set_to_none=True)
                logits = model(train['ids'][indices], train['lengths'][indices])
                loss = nn.functional.cross_entropy(logits, train['labels'][indices])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses += loss.item() * len(indices)
                count += len(indices)
            metrics = classify_evaluate(model, val)[0]
            if stopper.update(metrics['loss']):
                torch.save(dict(config=asdict(config), model=model.state_dict(), epoch=epoch,
                                validation_loss=stopper.best, variant=name), CHECKPOINTS / f'{name}.pt')
            variant['history'].append(dict(epoch=epoch, train_loss=losses/count,
                                           bad_checks=stopper.bad_checks, **metrics))
            print('FINETUNE', name, epoch, json.dumps(metrics), flush=True)
            if stopper.should_stop:
                variant['stop_reason'] = 'validation_early_stopping'
                print('EARLY STOP FINETUNE', name, epoch, flush=True)
                break
        variant['stopped_epoch'] = epoch
        saved = torch.load(CHECKPOINTS / f'{name}.pt', map_location=DEVICE, weights_only=False)
        model.load_state_dict(saved['model'])
        variant['selected_epoch'] = saved['epoch']
        variant['train'] = classify_evaluate(model, train)[0]
        variant['validation'] = classify_evaluate(model, val)[0]
        metrics, predictions, probabilities = classify_evaluate(model, test)
        variant['test'] = metrics
        variant['test_predictions'] = [dict(index=i, text=row['text'], label=row['label'],
                                             prediction=predictions[i], positive_probability=probabilities[i])
                                        for i,row in enumerate(test['rows'])]
        record['variants'][name] = variant
        write_json('finetuning.json', record)
        print('TEST', name, json.dumps(metrics), flush=True)
    write_json('finetuning.json', record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['pretrain','finetune','all'])
    parser.add_argument('--steps', type=int, default=20000, help='预训练步数上限')
    parser.add_argument('--epochs', type=int, default=50, help='监督训练轮数上限')
    args = parser.parse_args()
    if args.stage in ('pretrain', 'all'):
        pretrain(args.steps)
    if args.stage in ('finetune', 'all'):
        finetune(args.epochs)
