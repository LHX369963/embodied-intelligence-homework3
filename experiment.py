"""按顺序执行 515M GPT 预训练、三组监督微调对照与最终测试集评估。"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import gc
import json
import math
from pathlib import Path
import platform
import random
import time
import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from tokenizers import Tokenizer
from model import (
    Config, GPT, SentimentModel,
    load_classifier, load_gpt, save_classifier, save_gpt,
)

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = ROOT / 'checkpoints'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 2026

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(8)


def autocast_ctx():
    if DEVICE.type == 'cuda':
        return torch.amp.autocast('cuda', dtype=torch.bfloat16)
    return nullcontext()


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class EarlyStopping:
    """验证损失连续 patience 次评估未刷新历史最低值时触发早停。"""
    def __init__(self, patience=3, best=float('inf')):
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


def save_environment():
    info = dict(
        python=platform.python_version(),
        platform=platform.platform(),
        torch=torch.__version__,
        device=str(DEVICE),
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        gpu_total_memory_gb=round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        if torch.cuda.is_available() else None,
    )
    write_json('environment.json', info)
    return info


def build_adamw(named_params, lr, weight_decay, betas=(0.9, 0.95)):
    decay, nodecay = [], []
    for _, param in named_params:
        if not param.requires_grad:
            continue
        (decay if param.ndim >= 2 else nodecay).append(param)
    groups = [
        dict(params=decay, lr=lr, weight_decay=weight_decay),
        dict(params=nodecay, lr=lr, weight_decay=0.0),
    ]
    kwargs = dict(betas=betas)
    if DEVICE.type == 'cuda':
        kwargs['fused'] = True
    return torch.optim.AdamW(groups, **kwargs)


def cosine_lr(step, total_steps, warmup_steps, peak_lr, min_lr):
    if step <= warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def read_tokens_cpu(split) -> np.ndarray:
    return np.load(ROOT / 'data' / f'wiki_{split}.npy', mmap_mode='r')


@torch.no_grad()
def language_evaluate(model, tokens_np: np.ndarray, batch_size=16, utf8_bytes=None):
    """遍历连续互不重叠的完整上下文窗口，按词元总数累计交叉熵与每字节比特数。"""
    model.eval()
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    context = raw.config.context
    blocks = (len(tokens_np) - 1) // context
    total = 0.0
    count = 0
    for start in range(0, blocks, batch_size):
        end = min(start + batch_size, blocks)
        span = np.asarray(tokens_np[start * context : end * context + 1], dtype=np.int64)
        offsets = np.arange(end - start)[:, None] * context + np.arange(context + 1)[None, :]
        chunk = torch.from_numpy(span[offsets]).to(DEVICE, non_blocking=True)
        with autocast_ctx():
            _, loss = model(chunk[:, :-1], chunk[:, 1:])
        n = (end - start) * context
        total += loss.item() * n
        count += n
    value = total / count
    result = dict(
        loss=value,
        perplexity=math.exp(min(value, 20.0)),
        evaluated_tokens=count,
        omitted_tail_tokens=int(len(tokens_np) - 1 - count),
    )
    if utf8_bytes:
        effective_bytes = utf8_bytes * (count / len(tokens_np))
        result['bits_per_byte'] = (total / math.log(2.0)) / effective_bytes
    return result


def pretrain(steps=5000, micro_batch=8, grad_accum=8, use_compile=True):
    seed_all()
    save_environment()
    CHECKPOINTS.mkdir(exist_ok=True)
    tokenizer = Tokenizer.from_file(str(ROOT / 'tokenizer.json'))
    data_info = json.loads((ROOT / 'data_info.json').read_text())
    val_bytes = data_info['wiki']['validation'].get('utf8_bytes')
    test_bytes = data_info['wiki']['test'].get('utf8_bytes')

    config = Config(vocab_size=tokenizer.get_vocab_size())
    model = GPT(config).to(DEVICE)
    forward_model = torch.compile(model) if (use_compile and DEVICE.type == 'cuda') else model

    train_np = read_tokens_cpu('train')
    val_np = read_tokens_cpu('validation')
    batch_size = micro_batch * grad_accum
    settings = dict(
        seed=SEED,
        max_steps=steps,
        micro_batch_size=micro_batch,
        grad_accum_steps=grad_accum,
        effective_batch_size=batch_size,
        tokens_per_step=batch_size * config.context,
        context=config.context,
        peak_lr=3e-4,
        minimum_lr=3e-5,
        warmup_steps=200,
        lr_decay_steps=steps,
        validation_interval=250,
        patience=4,
        weight_decay=0.1,
        grad_clip=1.0,
        optimizer='AdamW',
        precision='bfloat16' if DEVICE.type == 'cuda' else 'float32',
        checkpoint_dtype='bfloat16',
        compiled=bool(use_compile and DEVICE.type == 'cuda'),
    )
    optimizer = build_adamw(model.named_parameters(), lr=settings['peak_lr'], weight_decay=0.1)
    initial_val = language_evaluate(forward_model, val_np, utf8_bytes=val_bytes)
    prompts = [
        'The history of',
        'In the early twentieth century,',
        'The film delivers a',
        'Artificial intelligence and',
    ]

    def generate_samples(target_model):
        torch.manual_seed(SEED)
        values = []
        eos_id = tokenizer.token_to_id('<eos>')
        for prompt in prompts:
            ids = torch.tensor([tokenizer.encode(prompt).ids], device=DEVICE)
            with autocast_ctx():
                output = target_model.generate(ids, new_tokens=80, temperature=0.8, top_k=40, eos_id=eos_id)
            values.append(dict(prompt=prompt, text=tokenizer.decode(output[0].tolist())))
        return values

    record = dict(
        config=asdict(config),
        settings=settings,
        parameters=sum(p.numel() for p in model.parameters()),
        corpus_tokens=int(len(train_np)),
        initial_validation=initial_val,
        initial_generation=generate_samples(model),
        history=[],
        initial_test=None,
    )
    rng = np.random.default_rng(SEED + 1)
    stopper = EarlyStopping(patience=settings['patience'], best=initial_val['loss'])
    save_gpt(model, CHECKPOINTS / 'pretrained.pt', step=0, validation_loss=stopper.best)
    rolling = []
    record['stop_reason'] = 'max_steps'
    print(f"PRETRAIN {record['parameters']:,} parameters | corpus={len(train_np):,} tokens", flush=True)

    t_start = time.perf_counter()
    train_elapsed = 0.0
    max_start = len(train_np) - config.context - 1
    offsets_grid = np.arange(config.context + 1, dtype=np.int64)[None, :]

    for step in range(1, steps + 1):
        step_t0 = time.perf_counter()
        forward_model.train()
        rate = cosine_lr(step, steps, settings['warmup_steps'], settings['peak_lr'], settings['minimum_lr'])
        for group in optimizer.param_groups:
            group['lr'] = rate
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(grad_accum):
            starts = rng.integers(0, max_start, size=(micro_batch, 1), dtype=np.int64)
            batch_np = np.asarray(train_np[starts + offsets_grid], dtype=np.int64)
            batch = torch.from_numpy(batch_np).to(DEVICE, non_blocking=True)
            with autocast_ctx():
                _, loss = forward_model(batch[:, :-1], batch[:, 1:])
                scaled = loss / grad_accum
            scaled.backward()
            step_loss += loss.item() / grad_accum
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        train_elapsed += time.perf_counter() - step_t0
        rolling.append(step_loss)

        if step % 50 == 0 or step == 1:
            avg_loss = sum(rolling[-50:]) / len(rolling[-50:])
            tok_per_s = (step * settings['tokens_per_step']) / max(train_elapsed, 1e-6)
            print(
                f'PRETRAIN step={step}/{steps} loss={avg_loss:.4f} lr={rate:.6f} '
                f'grad={grad_norm:.3f} tok/s={tok_per_s:.0f}',
                flush=True,
            )

        if step % settings['validation_interval'] == 0 or step == steps:
            metrics = language_evaluate(forward_model, val_np, utf8_bytes=val_bytes)
            row = dict(
                step=step,
                sampled_tokens=step * settings['tokens_per_step'],
                train_loss=sum(rolling[-settings['validation_interval']:]) / len(rolling[-settings['validation_interval']:]),
                validation_loss=metrics['loss'],
                validation_perplexity=metrics['perplexity'],
                validation_bpb=metrics.get('bits_per_byte'),
                learning_rate=rate,
                grad_norm=grad_norm,
                elapsed_seconds=round(train_elapsed, 2),
            )
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
    record['sampled_training_tokens'] = step * settings['tokens_per_step']
    record['training_seconds'] = round(train_elapsed, 2)
    record['wall_seconds'] = round(time.perf_counter() - t_start, 2)
    if torch.cuda.is_available():
        record['peak_gpu_memory_gb'] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    del forward_model, model, optimizer
    release_memory()

    best_model, saved = load_gpt(CHECKPOINTS / 'pretrained.pt', DEVICE, dtype=torch.bfloat16 if DEVICE.type == 'cuda' else torch.float32)
    record['selected_step'] = saved['step']
    record['checkpoint_size_mb'] = round((CHECKPOINTS / 'pretrained.pt').stat().st_size / (1024 * 1024), 2)
    record['validation'] = language_evaluate(best_model, val_np, utf8_bytes=val_bytes)
    record['generation'] = generate_samples(best_model)
    test_np = read_tokens_cpu('test')
    record['test'] = language_evaluate(best_model, test_np, utf8_bytes=test_bytes)
    del best_model
    release_memory()

    seed_all()
    initial_model = GPT(config).to(DEVICE, dtype=torch.bfloat16 if DEVICE.type == 'cuda' else torch.float32)
    record['initial_test'] = language_evaluate(initial_model, test_np, utf8_bytes=test_bytes)
    del initial_model
    release_memory()

    write_json('pretraining.json', record)
    print('PRETRAIN COMPLETE', json.dumps(record['test']), flush=True)


def encode_reviews(tokenizer, context):
    raw_splits = {}
    max_len = 0
    eos_id = tokenizer.token_to_id('<eos>')
    for split in ('train', 'validation', 'test'):
        rows = pq.read_table(ROOT / 'data' / f'sentiment_{split}.parquet').to_pylist()
        encoded_list = []
        truncated = 0
        for row in rows:
            content = tokenizer.encode(row['text']).ids
            truncated += len(content) > context - 1
            seq = content[:context - 1] + [eos_id]
            max_len = max(max_len, len(seq))
            encoded_list.append(seq)
        raw_splits[split] = (rows, encoded_list, truncated)

    pad_len = min(context, ((max_len + 7) // 8) * 8)
    datasets = {}
    for split, (rows, encoded_list, truncated) in raw_splits.items():
        ids = torch.zeros((len(rows), pad_len), dtype=torch.long)
        lengths = torch.empty(len(rows), dtype=torch.long)
        labels = torch.tensor([row['label'] for row in rows], dtype=torch.long)
        for i, seq in enumerate(encoded_list):
            ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            lengths[i] = len(seq)
        datasets[split] = dict(
            ids=ids.to(DEVICE),
            lengths=lengths.to(DEVICE),
            labels=labels.to(DEVICE),
            rows=rows,
            truncated=truncated,
            padded_length=pad_len,
            max_sequence_length=max_len,
        )
    return datasets


def metrics_from_predictions(labels, predictions):
    matrix = [[sum(y == i and p == j for y, p in zip(labels, predictions))
               for j in range(2)] for i in range(2)]
    scores = []
    for i in range(2):
        tp = matrix[i][i]
        fp = matrix[1 - i][i]
        fn = matrix[i][1 - i]
        scores.append(2 * tp / max(1, 2 * tp + fp + fn))
    return dict(
        accuracy=sum(y == p for y, p in zip(labels, predictions)) / len(labels),
        macro_f1=sum(scores) / 2,
        confusion_matrix=matrix,
        correct=sum(y == p for y, p in zip(labels, predictions)),
        examples=len(labels),
    )


@torch.no_grad()
def classify_evaluate(model, dataset, batch_size=64):
    model.eval()
    losses = 0.0
    predictions, probabilities = [], []
    size = len(dataset['labels'])
    for start in range(0, size, batch_size):
        stop = min(start + batch_size, size)
        with autocast_ctx():
            logits = model(dataset['ids'][start:stop], dataset['lengths'][start:stop])
            loss = nn.functional.cross_entropy(logits.float(), dataset['labels'][start:stop], reduction='sum')
        losses += loss.item()
        probs = logits.float().softmax(-1)
        predictions += probs.argmax(-1).tolist()
        probabilities += probs[:, 1].tolist()
    metrics = metrics_from_predictions(dataset['labels'].tolist(), predictions)
    metrics['loss'] = losses / size
    return metrics, predictions, probabilities


def finetune(epochs=10, micro_batch=32, grad_accum=2):
    CHECKPOINTS.mkdir(exist_ok=True)
    tokenizer = Tokenizer.from_file(str(ROOT / 'tokenizer.json'))
    saved_meta = torch.load(CHECKPOINTS / 'pretrained.pt', map_location='cpu', weights_only=False)
    config = Config(**saved_meta['config'])
    del saved_meta
    datasets = encode_reviews(tokenizer, config.context)
    batch_size = micro_batch * grad_accum
    variant_specs = {
        'frozen': dict(backbone_lr=0.0, head_lr=1e-3, weight_decay=0.01),
        'scratch': dict(backbone_lr=1e-4, head_lr=2e-4, weight_decay=0.01),
        'full': dict(backbone_lr=2e-5, head_lr=2e-4, weight_decay=0.01),
    }
    record = dict(
        seed=SEED,
        max_epochs=epochs,
        patience=3,
        micro_batch_size=micro_batch,
        grad_accum_steps=grad_accum,
        effective_batch_size=batch_size,
        grad_clip=1.0,
        warmup_ratio=0.1,
        classification_pooling='last real token (EOS)',
        padded_sequence_length=datasets['train']['padded_length'],
        max_review_tokens=datasets['train']['max_sequence_length'],
        truncated={k: v['truncated'] for k, v in datasets.items()},
        variant_hyperparameters=variant_specs,
        variants={},
    )
    train, val, test = [datasets[k] for k in ('train', 'validation', 'test')]
    steps_per_epoch = math.ceil(len(train['labels']) / batch_size)
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, int(0.1 * total_steps))

    for name in ('frozen', 'scratch', 'full'):
        release_memory()
        spec = variant_specs[name]
        seed_all()
        if name == 'scratch':
            backbone = GPT(config).to(DEVICE, dtype=torch.float32)
        else:
            backbone, _ = load_gpt(CHECKPOINTS / 'pretrained.pt', DEVICE, dtype=torch.float32)
        torch.manual_seed(SEED + 10)
        model = SentimentModel(backbone).to(DEVICE, dtype=torch.float32)

        if name == 'frozen':
            for parameter in model.backbone.parameters():
                parameter.requires_grad = False

        param_groups = [
            dict(params=[model.head.weight], lr=spec['head_lr'], base_lr=spec['head_lr'], weight_decay=spec['weight_decay']),
            dict(params=[model.head.bias], lr=spec['head_lr'], base_lr=spec['head_lr'], weight_decay=0.0),
        ]
        if name != 'frozen':
            b_decay, b_nodecay = [], []
            for _, p in model.backbone.named_parameters():
                (b_decay if p.ndim >= 2 else b_nodecay).append(p)
            param_groups.extend([
                dict(params=b_decay, lr=spec['backbone_lr'], base_lr=spec['backbone_lr'], weight_decay=spec['weight_decay']),
                dict(params=b_nodecay, lr=spec['backbone_lr'], base_lr=spec['backbone_lr'], weight_decay=0.0),
            ])

        opt_kwargs = dict(betas=(0.9, 0.999))
        if DEVICE.type == 'cuda':
            opt_kwargs['fused'] = True
        optimizer = torch.optim.AdamW(param_groups, **opt_kwargs)

        variant = dict(
            hyperparameters=spec,
            history=[],
            initial_validation=classify_evaluate(model, val)[0],
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            total_parameters=sum(p.numel() for p in model.parameters()),
        )
        stopper = EarlyStopping(patience=record['patience'])
        variant['stop_reason'] = 'max_epochs'
        global_step = 0
        t0 = time.perf_counter()

        for epoch in range(1, epochs + 1):
            model.train()
            if name == 'frozen':
                model.backbone.eval()
            generator = torch.Generator().manual_seed(SEED + epoch)
            permutation = torch.randperm(len(train['labels']), generator=generator).to(DEVICE)
            losses = 0.0
            count = 0
            for batch_start in range(0, len(permutation), batch_size):
                global_step += 1
                lr_scale = cosine_lr(global_step, total_steps, warmup_steps, 1.0, 0.1)
                for group in optimizer.param_groups:
                    group['lr'] = group['base_lr'] * lr_scale
                optimizer.zero_grad(set_to_none=True)
                batch_indices = permutation[batch_start:batch_start + batch_size]
                sub_chunks = batch_indices.split(micro_batch)
                total_in_step = len(batch_indices)
                for sub_idx in sub_chunks:
                    weight = len(sub_idx) / total_in_step
                    with autocast_ctx():
                        logits = model(train['ids'][sub_idx], train['lengths'][sub_idx])
                        loss = nn.functional.cross_entropy(logits.float(), train['labels'][sub_idx])
                    (loss * weight).backward()
                    losses += loss.item() * len(sub_idx)
                    count += len(sub_idx)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            metrics = classify_evaluate(model, val)[0]
            if stopper.update(metrics['loss']):
                save_classifier(
                    model,
                    CHECKPOINTS / f'{name}.pt',
                    epoch=epoch,
                    validation_loss=stopper.best,
                    variant=name,
                )
            row = dict(
                epoch=epoch,
                train_loss=losses / count,
                bad_checks=stopper.bad_checks,
                elapsed_seconds=round(time.perf_counter() - t0, 2),
                **metrics,
            )
            variant['history'].append(row)
            print('FINETUNE', name, epoch, json.dumps(row), flush=True)
            if stopper.should_stop:
                variant['stop_reason'] = 'validation_early_stopping'
                print('EARLY STOP FINETUNE', name, epoch, flush=True)
                break

        variant['stopped_epoch'] = epoch
        variant['training_seconds'] = round(time.perf_counter() - t0, 2)
        del model, backbone, optimizer
        release_memory()

        eval_dtype = torch.bfloat16 if DEVICE.type == 'cuda' else torch.float32
        best_model, saved_ckpt = load_classifier(CHECKPOINTS / f'{name}.pt', DEVICE, dtype=eval_dtype)
        variant['selected_epoch'] = saved_ckpt['epoch']
        variant['checkpoint_size_mb'] = round((CHECKPOINTS / f'{name}.pt').stat().st_size / (1024 * 1024), 2)
        variant['train'] = classify_evaluate(best_model, train)[0]
        variant['validation'] = classify_evaluate(best_model, val)[0]
        metrics, predictions, probabilities = classify_evaluate(best_model, test)
        variant['test'] = metrics
        variant['test_predictions'] = [
            dict(
                index=i,
                text=row['text'],
                label=row['label'],
                prediction=predictions[i],
                positive_probability=probabilities[i],
            )
            for i, row in enumerate(test['rows'])
        ]
        del best_model
        release_memory()

        record['variants'][name] = variant
        write_json('finetuning.json', record)
        print('TEST', name, json.dumps(metrics), flush=True)

    write_json('finetuning.json', record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['pretrain', 'finetune', 'all'])
    parser.add_argument('--steps', type=int, default=5000, help='预训练参数更新步数上限')
    parser.add_argument('--epochs', type=int, default=10, help='监督微调轮数上限')
    parser.add_argument('--micro-batch', type=int, default=8, help='预训练单次前向微批大小')
    parser.add_argument('--grad-accum', type=int, default=8, help='预训练梯度累积步数')
    parser.add_argument('--no-compile', action='store_true', help='关闭 torch.compile')
    args = parser.parse_args()
    if args.stage in ('pretrain', 'all'):
        pretrain(
            steps=args.steps,
            micro_batch=args.micro_batch,
            grad_accum=args.grad_accum,
            use_compile=not args.no_compile,
        )
    if args.stage in ('finetune', 'all'):
        finetune(epochs=args.epochs)
