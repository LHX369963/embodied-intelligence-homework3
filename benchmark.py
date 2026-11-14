"""4.24M 与 515M GPT 跨规模公平横向评测：BPB/词级困惑度、BLiMP 语法最小对、零样本/少样本提示、2x3 监督矩阵与 KV 缓存加速比。"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time
import urllib.request
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from model import Config, GPT, SentimentModel, load_classifier, load_gpt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
ENDPOINTS = [
    os.environ.get('HF_ENDPOINT', '').rstrip('/'),
    'https://hf-mirror.com',
    'https://huggingface.co',
]

BLIMP_TASKS = [
    ('regular_plural_subject_verb_agreement_1', '主谓单复数一致'),
    ('determiner_noun_agreement_1', '限定词名词一致'),
    ('anaphor_number_agreement', '反身代词指代'),
    ('irregular_past_participle_verbs', '不规则过去分词'),
    ('wh_questions_object_gap', '疑问句宾语空位'),
]

HARD_MARKERS = {
    'not', 'never', 'neither', 'nor', 'hardly', 'barely', 'without',
    'despite', 'although', 'though', 'yet', 'but', 'however',
}


def ensure_blimp_parquet(task_name: str) -> Path:
    DATA.mkdir(exist_ok=True)
    target = DATA / f'blimp_{task_name}.parquet'
    if target.exists() and target.stat().st_size > 0:
        return target
    hosts = [h for i, h in enumerate(ENDPOINTS) if h and h not in ENDPOINTS[:i]]
    last_err = None
    for host in hosts:
        url = f'{host}/api/datasets/nyu-mll/blimp/parquet/{task_name}/train/0.parquet'
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as resp, open(target, 'wb') as out:
                out.write(resp.read())
            return target
        except Exception as exc:
            last_err = exc
            if target.exists():
                target.unlink()
    raise RuntimeError(f'无法下载 BLiMP 子集 {task_name}: {last_err}')


def roc_auc_binary(labels, scores) -> float:
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum_pos = 0.0
    i = 0
    n = len(pairs)
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        pos_in_tie = sum(pairs[k][1] for k in range(i, j))
        rank_sum_pos += pos_in_tie * avg_rank
        i = j
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def is_negation_or_contrast(text: str) -> bool:
    lowered = text.lower()
    if "n't" in lowered:
        return True
    words = {w.strip(".,!?;:\"'()[]{}") for w in lowered.split()}
    return bool(words & HARD_MARKERS)


@torch.no_grad()
def eval_wikitext_fair(model: GPT, tokenizer: Tokenizer, window: int, device: str, batch_size: int = 16):
    rows = pq.read_table(DATA / 'wiki_test.parquet').to_pylist()
    non_empty = [r['text'] for r in rows if r['text']]
    full_text = ''.join(non_empty)
    total_utf8_bytes = len(full_text.encode('utf-8'))
    total_words = len(full_text.split())

    encoded = tokenizer.encode_batch(non_empty)
    token_ids = [tid for item in encoded for tid in item.ids]
    total_tokens = len(token_ids)
    blocks = (total_tokens - 1) // window
    total_nats = 0.0
    evaluated_tokens = 0

    amp_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' else torch.no_grad()
    for start in range(0, blocks, batch_size):
        end = min(start + batch_size, blocks)
        batch_seqs = [
            token_ids[b * window : (b + 1) * window + 1]
            for b in range(start, end)
        ]
        chunk = torch.tensor(batch_seqs, dtype=torch.long, device=device)
        with amp_ctx:
            logits, _ = model(chunk[:, :-1])
            loss = F.cross_entropy(
                logits.float().flatten(0, 1),
                chunk[:, 1:].flatten(),
                reduction='sum',
            )
        total_nats += loss.item()
        evaluated_tokens += (end - start) * window

    coverage = evaluated_tokens / (total_tokens - 1)
    effective_bytes = total_utf8_bytes * coverage
    effective_words = total_words * coverage
    token_ce = total_nats / evaluated_tokens
    bpb = (total_nats / math.log(2.0)) / effective_bytes
    word_nll = total_nats / effective_words
    return dict(
        window=window,
        total_utf8_bytes=total_utf8_bytes,
        total_words=total_words,
        total_tokens=total_tokens,
        evaluated_tokens=evaluated_tokens,
        coverage=round(coverage, 6),
        token_loss=round(token_ce, 4),
        token_perplexity=round(math.exp(min(token_ce, 20.0)), 2),
        bits_per_byte=round(bpb, 4),
        word_perplexity=round(math.exp(min(word_nll, 20.0)), 2),
    )


@torch.no_grad()
def sequence_logprobs(model: GPT, token_lists, device: str, batch_size: int = 64):
    """计算给定词元序列从第二个位置起的总对数概率 log P(t_1..t_m | t_0)。"""
    results = []
    max_ctx = model.config.context
    amp_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' else torch.no_grad()
    for start in range(0, len(token_lists), batch_size):
        batch = [seq[:max_ctx + 1] for seq in token_lists[start:start + batch_size]]
        max_len = max(len(seq) for seq in batch)
        inp = torch.zeros((len(batch), max_len - 1), dtype=torch.long, device=device)
        tgt = torch.full((len(batch), max_len - 1), -100, dtype=torch.long, device=device)
        for i, seq in enumerate(batch):
            inp[i, :len(seq) - 1] = torch.tensor(seq[:-1], dtype=torch.long)
            tgt[i, :len(seq) - 1] = torch.tensor(seq[1:], dtype=torch.long)
        with amp_ctx:
            logits, _ = model(inp)
            token_nll = F.cross_entropy(
                logits.float().transpose(1, 2),
                tgt,
                ignore_index=-100,
                reduction='none',
            )
        seq_logp = (-token_nll.sum(dim=1)).tolist()
        results.extend(seq_logp)
    return results


@torch.no_grad()
def eval_blimp(model: GPT, tokenizer: Tokenizer, device: str):
    per_task = {}
    total_correct = 0
    total_pairs = 0
    for task_id, cn_name in BLIMP_TASKS:
        path = ensure_blimp_parquet(task_id)
        rows = pq.read_table(path).to_pylist()
        good_seqs = [tokenizer.encode('\n' + r['sentence_good']).ids for r in rows]
        bad_seqs = [tokenizer.encode('\n' + r['sentence_bad']).ids for r in rows]
        good_lp = sequence_logprobs(model, good_seqs, device=device)
        bad_lp = sequence_logprobs(model, bad_seqs, device=device)
        correct = sum(g > b for g, b in zip(good_lp, bad_lp))
        acc = correct / len(rows)
        per_task[task_id] = dict(name=cn_name, accuracy=round(acc, 4), correct=correct, total=len(rows))
        total_correct += correct
        total_pairs += len(rows)
    return dict(
        macro_accuracy=round(total_correct / total_pairs, 4),
        total_correct=total_correct,
        total_pairs=total_pairs,
        tasks=per_task,
    )


def select_fewshot_demos():
    train_rows = pq.read_table(DATA / 'sentiment_train.parquet').to_pylist()
    neg_demos = [r['text'].strip() for r in train_rows if r['label'] == 0 and 6 <= len(r['text'].split()) <= 10][:2]
    pos_demos = [r['text'].strip() for r in train_rows if r['label'] == 1 and 6 <= len(r['text'].split()) <= 10][:2]
    demos = [
        (pos_demos[0], 'good'),
        (neg_demos[0], 'bad'),
        (pos_demos[1], 'good'),
        (neg_demos[1], 'bad'),
    ]
    return '\n'.join(f'Review: {txt}\nOverall: {lbl}.' for txt, lbl in demos) + '\n'


@torch.no_grad()
def eval_icl_sentiment(model: GPT, tokenizer: Tokenizer, device: str):
    test_rows = pq.read_table(DATA / 'sentiment_test.parquet').to_pylist()
    labels = [r['label'] for r in test_rows]
    fewshot_prefix = select_fewshot_demos()
    results = {}

    for mode, prefix in [('zero_shot', ''), ('four_shot', fewshot_prefix)]:
        null_pos = tokenizer.encode(prefix + 'Review:\nOverall: good.').ids
        null_neg = tokenizer.encode(prefix + 'Review:\nOverall: bad.').ids
        null_lp = sequence_logprobs(model, [null_pos, null_neg], device=device)
        prior_bias = null_lp[0] - null_lp[1]

        pos_seqs = []
        neg_seqs = []
        max_ctx = model.config.context
        pos_suffix = tokenizer.encode('\nOverall: good.').ids
        neg_suffix = tokenizer.encode('\nOverall: bad.').ids
        head_ids = tokenizer.encode(prefix + 'Review: ').ids

        for r in test_rows:
            rev_ids = tokenizer.encode(r['text'].strip()).ids
            budget = max_ctx + 1 - len(head_ids) - max(len(pos_suffix), len(neg_suffix))
            rev_ids = rev_ids[:max(8, budget)]
            pos_seqs.append(head_ids + rev_ids + pos_suffix)
            neg_seqs.append(head_ids + rev_ids + neg_suffix)

        pos_lp = sequence_logprobs(model, pos_seqs, device=device)
        neg_lp = sequence_logprobs(model, neg_seqs, device=device)
        diffs = [(p - n) - prior_bias for p, n in zip(pos_lp, neg_lp)]
        preds = [1 if d > 0 else 0 for d in diffs]
        correct = sum(y == p for y, p in zip(labels, preds))
        results[mode] = dict(
            accuracy=round(correct / len(labels), 4),
            roc_auc=round(roc_auc_binary(labels, diffs), 4),
            correct=correct,
            total=len(labels),
        )
    return results


@torch.no_grad()
def eval_supervised_suite(ckpt_dir: Path, tokenizer: Tokenizer, device: str):
    test_rows = pq.read_table(DATA / 'sentiment_test.parquet').to_pylist()
    labels = [r['label'] for r in test_rows]
    hard_mask = [is_negation_or_contrast(r['text']) for r in test_rows]
    long_mask = [len(r['text'].split()) >= 22 for r in test_rows]
    eos_id = tokenizer.token_to_id('<eos>')

    out = {}
    for name in ('frozen', 'scratch', 'full'):
        model, meta = load_classifier(ckpt_dir / f'{name}.pt', device=device)
        ctx = model.backbone.config.context
        encoded = [tokenizer.encode(r['text']).ids[:ctx - 1] + [eos_id] for r in test_rows]
        pad_len = max(len(s) for s in encoded)
        ids = torch.zeros((len(encoded), pad_len), dtype=torch.long, device=device)
        lengths = torch.zeros(len(encoded), dtype=torch.long, device=device)
        for i, seq in enumerate(encoded):
            ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            lengths[i] = len(seq)

        probs_pos = []
        preds = []
        loss_sum = 0.0
        amp_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' else torch.no_grad()
        y_tensor = torch.tensor(labels, dtype=torch.long, device=device)
        for start in range(0, len(encoded), 64):
            stop = min(start + 64, len(encoded))
            with amp_ctx:
                logits = model(ids[start:stop], lengths[start:stop]).float()
                loss_sum += F.cross_entropy(logits, y_tensor[start:stop], reduction='sum').item()
            p = logits.softmax(-1)
            probs_pos.extend(p[:, 1].tolist())
            preds.extend(p.argmax(-1).tolist())

        tp0 = sum(y == 0 and p == 0 for y, p in zip(labels, preds))
        fp0 = sum(y == 1 and p == 0 for y, p in zip(labels, preds))
        fn0 = sum(y == 0 and p == 1 for y, p in zip(labels, preds))
        tp1 = sum(y == 1 and p == 1 for y, p in zip(labels, preds))
        fp1 = sum(y == 0 and p == 1 for y, p in zip(labels, preds))
        fn1 = sum(y == 1 and p == 0 for y, p in zip(labels, preds))
        f1_0 = 2 * tp0 / max(1, 2 * tp0 + fp0 + fn0)
        f1_1 = 2 * tp1 / max(1, 2 * tp1 + fp1 + fn1)

        hard_total = sum(hard_mask)
        hard_correct = sum(y == p for y, p, m in zip(labels, preds, hard_mask) if m)
        long_total = sum(long_mask)
        long_correct = sum(y == p for y, p, m in zip(labels, preds, long_mask) if m)
        short_total = len(labels) - long_total
        short_correct = sum(y == p for y, p, m in zip(labels, preds, long_mask) if not m)

        out[name] = dict(
            selected_epoch=meta.get('epoch'),
            accuracy=round(sum(y == p for y, p in zip(labels, preds)) / len(labels), 4),
            macro_f1=round(0.5 * (f1_0 + f1_1), 4),
            roc_auc=round(roc_auc_binary(labels, probs_pos), 4),
            loss=round(loss_sum / len(labels), 4),
            confusion_matrix=[[tp0, fn0], [fp0, tp1]],
            hard_negation_contrast=dict(
                accuracy=round(hard_correct / max(1, hard_total), 4),
                correct=hard_correct,
                total=hard_total,
            ),
            long_reviews=dict(
                accuracy=round(long_correct / max(1, long_total), 4),
                correct=long_correct,
                total=long_total,
            ),
            short_reviews=dict(
                accuracy=round(short_correct / max(1, short_total), 4),
                correct=short_correct,
                total=short_total,
            ),
        )
        del model
    return out


@torch.no_grad()
def eval_kv_cache_speedup(model_cpu: GPT, tokenizer: Tokenizer, gen_tokens: int = 96):
    torch.set_num_threads(8)
    prompt_ids = tokenizer.encode('The history of artificial intelligence began in the mid twentieth century').ids
    x0 = torch.tensor([prompt_ids], dtype=torch.long)

    # 预热一次消除首次分配开销
    _ = model_cpu.generate(x0.clone(), new_tokens=4, temperature=0.0, eos_id=None)

    # 1. 开启 KV 缓存的贪心生成
    t0 = time.perf_counter()
    cached_ids = model_cpu.generate(x0.clone(), new_tokens=gen_tokens, temperature=0.0, eos_id=None)
    cached_sec = time.perf_counter() - t0

    # 2. 关闭 KV 缓存的全序列重算贪心生成
    ids = x0.clone()
    t1 = time.perf_counter()
    for _ in range(gen_tokens):
        window = ids[:, -model_cpu.config.context:]
        logits, _ = model_cpu(window)
        nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
        ids = torch.cat((ids, nxt), dim=1)
    uncached_sec = time.perf_counter() - t1

    exact_match = bool(torch.equal(cached_ids, ids))
    return dict(
        generated_tokens=gen_tokens,
        exact_token_match=exact_match,
        kv_cached_seconds=round(cached_sec, 4),
        kv_cached_tok_per_s=round(gen_tokens / max(cached_sec, 1e-6), 2),
        uncached_seconds=round(uncached_sec, 4),
        uncached_tok_per_s=round(gen_tokens / max(uncached_sec, 1e-6), 2),
        speedup_ratio=round(uncached_sec / max(cached_sec, 1e-6), 2),
    )


def load_4m_tokenizer() -> Tokenizer:
    raw_json = subprocess.check_output(
        ['git', '-C', str(ROOT), 'show', 'HEAD:tokenizer.json']
    ).decode('utf-8')
    return Tokenizer.from_str(raw_json)


def run_scale(scale: str, ckpt_dir: Path, device: str, run_cpu_speed: bool):
    if scale == '4m':
        tokenizer = load_4m_tokenizer()
    else:
        tokenizer = Tokenizer.from_file(str(ROOT / 'tokenizer.json'))

    gpt_model, gpt_meta = load_gpt(ckpt_dir / 'pretrained.pt', device=device)
    params = sum(p.numel() for p in gpt_model.parameters())
    print(f'[{scale}] 参数量={params:,} | 词表={tokenizer.get_vocab_size()} | 设备={device}', flush=True)

    lm_results = {}
    lm_results['ctx_128'] = eval_wikitext_fair(gpt_model, tokenizer, window=128, device=device)
    print(f'[{scale}] WikiText ctx=128:', lm_results['ctx_128'], flush=True)
    if gpt_model.config.context >= 512:
        lm_results['ctx_512'] = eval_wikitext_fair(gpt_model, tokenizer, window=512, device=device)
        print(f'[{scale}] WikiText ctx=512:', lm_results['ctx_512'], flush=True)

    blimp = eval_blimp(gpt_model, tokenizer, device=device)
    print(f'[{scale}] BLiMP 语法最小对准确率:', blimp['macro_accuracy'], flush=True)

    icl = eval_icl_sentiment(gpt_model, tokenizer, device=device)
    print(f'[{scale}] 零样本/少样本提示:', icl, flush=True)

    del gpt_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    supervised = eval_supervised_suite(ckpt_dir, tokenizer, device=device)
    print(f'[{scale}] 监督微调三组测试结果:', {k: v['accuracy'] for k, v in supervised.items()}, flush=True)

    speed = None
    if run_cpu_speed:
        cpu_model, _ = load_gpt(ckpt_dir / 'pretrained.pt', device='cpu', dtype=torch.float32)
        speed = eval_kv_cache_speedup(cpu_model, tokenizer, gen_tokens=96)
        print(f'[{scale}] CPU KV-Cache 加速实测:', speed, flush=True)
        del cpu_model

    return dict(
        parameters=params,
        vocab_size=tokenizer.get_vocab_size(),
        config=gpt_meta['config'],
        selected_pretrain_step=gpt_meta.get('step'),
        wikitext_test=lm_results,
        blimp_zero_shot=blimp,
        sentiment_icl=icl,
        sentiment_supervised=supervised,
        cpu_kv_cache_benchmark=speed,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', choices=['4m', '515m'], required=True)
    parser.add_argument('--checkpoints', type=Path, default=ROOT / 'checkpoints')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--skip-cpu-speed', action='store_true')
    args = parser.parse_args()

    out_path = ROOT / 'benchmark.json'
    data = json.loads(out_path.read_text()) if out_path.exists() else {'models': {}}
    data['models'][args.scale] = run_scale(
        args.scale,
        ckpt_dir=args.checkpoints,
        device=args.device,
        run_cpu_speed=not args.skip_cpu_speed,
    )
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f'已更新 {out_path}', flush=True)


if __name__ == '__main__':
    main()
