"""命令行推理与一键演示入口；缺失本地权重时自动从 GitHub Release 下载。"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import urllib.request
import torch
from tokenizers import Tokenizer
from model import load_classifier, load_gpt

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = ROOT / 'checkpoints'
RELEASE_BASE = 'https://github.com/LHX369963/embodied-intelligence-homework3/releases/download/submission'
VALID_NAMES = ('pretrained', 'full', 'frozen', 'scratch')


def candidate_urls(filename: str):
    raw_url = f'{RELEASE_BASE}/{filename}'
    custom_proxy = os.environ.get('GITHUB_PROXY', '').strip().rstrip('/')
    urls = []
    if custom_proxy:
        urls.append(f'{custom_proxy}/{raw_url}')
    urls.append(raw_url)
    urls.append(f'https://ghfast.top/{raw_url}')
    urls.append(f'https://gh-proxy.com/{raw_url}')
    return urls


def ensure_checkpoint(name: str) -> Path:
    if name not in VALID_NAMES:
        raise ValueError(f'未知权重名称: {name}')
    CHECKPOINTS.mkdir(exist_ok=True)
    target = CHECKPOINTS / f'{name}.pt'
    if target.exists() and target.stat().st_size > 1024 * 1024:
        return target

    part = CHECKPOINTS / f'{name}.pt.part'
    last_error = None
    for url in candidate_urls(f'{name}.pt'):
        try:
            print(f'正在获取权重 {name}.pt <- {url}', file=sys.stderr, flush=True)
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as resp:
                expected = int(resp.headers.get('Content-Length') or 0)
                downloaded = 0
                t0 = time.perf_counter()
                last_print = 0.0
                with open(part, 'wb') as out:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        downloaded += len(chunk)
                        now = time.perf_counter()
                        if now - last_print >= 0.5 or (expected and downloaded == expected):
                            speed = (downloaded / (1024 * 1024)) / max(now - t0, 1e-3)
                            if expected > 0:
                                pct = downloaded * 100.0 / expected
                                msg = (f'\r  已下载 {downloaded / (1024 * 1024):.1f} / '
                                       f'{expected / (1024 * 1024):.1f} MiB ({pct:.1f}%) '
                                       f'{speed:.1f} MiB/s')
                            else:
                                msg = f'\r  已下载 {downloaded / (1024 * 1024):.1f} MiB ({speed:.1f} MiB/s)'
                            print(msg, end='', file=sys.stderr, flush=True)
                            last_print = now
                print(file=sys.stderr, flush=True)
                if expected > 0 and downloaded != expected:
                    raise IOError(f'文件字节数不完整: {downloaded} != {expected}')
            part.replace(target)
            return target
        except Exception as exc:
            last_error = exc
            if part.exists():
                part.unlink()
            print(f'下载源切换 ({exc})', file=sys.stderr, flush=True)
    raise RuntimeError(
        f'自动下载 {name}.pt 失败 ({last_error})。请检查网络连接，或手动将 {name}.pt 放入 {CHECKPOINTS}'
    )


def select_device(device_arg: str) -> str:
    if device_arg != 'auto':
        return device_arg
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def run_generate(tokenizer: Tokenizer, text: str, device: str, new_tokens: int,
                 temperature: float, top_k: int):
    ckpt = ensure_checkpoint('pretrained')
    model, _ = load_gpt(ckpt, device=device)
    ids = tokenizer.encode(text).ids or [tokenizer.token_to_id('<eos>')]
    prompt_tensor = torch.tensor([ids], device=device)
    t0 = time.perf_counter()
    generated = model.generate(
        prompt_tensor,
        new_tokens=new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=tokenizer.token_to_id('<eos>'),
    )
    elapsed = time.perf_counter() - t0
    out_ids = generated[0].tolist()
    if out_ids and out_ids[-1] == tokenizer.token_to_id('<eos>'):
        out_ids = out_ids[:-1]
    return tokenizer.decode(out_ids), len(generated[0]) - len(ids), elapsed, model


def run_classify(tokenizer: Tokenizer, text: str, device: str, variant: str):
    ckpt = ensure_checkpoint(variant)
    model, _ = load_classifier(ckpt, device=device)
    cfg = model.backbone.config
    ids = tokenizer.encode(text).ids[:cfg.context - 1] + [tokenizer.token_to_id('<eos>')]
    with torch.no_grad():
        logits = model(
            torch.tensor([ids], device=device),
            torch.tensor([len(ids)], device=device),
        )
        probabilities = logits.float().softmax(-1)[0]
    pred = int(probabilities.argmax().item())
    return dict(
        variant=variant,
        label='正面' if pred == 1 else '负面',
        positive_probability=round(float(probabilities[1].item()), 4),
        negative_probability=round(float(probabilities[0].item()), 4),
    )


def run_demo(tokenizer: Tokenizer, device: str):
    print(f'=== 作业三 515M GPT 演示 (运行设备: {device}) ===')
    prompts = [
        'The history of artificial intelligence began in',
        'The film delivers a remarkably compelling performance',
    ]
    model_ref = None
    for p in prompts:
        torch.manual_seed(2026)
        text, n_tok, dt, model_ref = run_generate(
            tokenizer, p, device=device, new_tokens=60, temperature=0.8, top_k=40
        )
        if p == prompts[0]:
            params = sum(param.numel() for param in model_ref.parameters())
            print(f'语言模型参数量: {params:,} | 配置: {model_ref.config}\n')
        print(f'[续写提示] {p}')
        print(f'[生成文本] {text}')
        print(f'[推理速度] 生成 {n_tok} 词元, 耗时 {dt:.2f}s ({n_tok / max(dt, 1e-6):.1f} tok/s)\n')

    reviews = [
        'A deeply moving, visually stunning masterpiece with unforgettable performances.',
        'consistently clever and suspenseful .',
        'A tedious, unimaginative script that wastes a talented cast on flat dialogue.',
        'this slender plot feels especially thin stretched over the nearly 80-minute running time .',
    ]
    print('=== 影评情感分类演示 (全参数微调模型 full.pt) ===')
    for rev in reviews:
        res = run_classify(tokenizer, rev, device=device, variant='full')
        print(f"[{res['label']} | 正面概率 {res['positive_probability']:.4f}] {rev}")


def main():
    parser = argparse.ArgumentParser(description='515M GPT 文本续写与影评情感分类推理工具')
    parser.add_argument('task', choices=['generate', 'classify', 'demo', 'download'],
                        help='执行任务类型：generate(续写) / classify(分类) / demo(一键演示) / download(预下载权重)')
    parser.add_argument('text', nargs='?', default='', help='输入英文提示词或待分类影评文本')
    parser.add_argument('--device', default='cpu', help='运行设备: cpu / cuda / mps / auto (默认 cpu)')
    parser.add_argument('--new-tokens', type=int, default=60, help='续写最大新增词元数')
    parser.add_argument('--temperature', type=float, default=0.8, help='采样温度 (0 表示贪心解码)')
    parser.add_argument('--top-k', type=int, default=40, help='Top-k 采样截断数')
    parser.add_argument('--variant', choices=['full', 'frozen', 'scratch'], default='full',
                        help='分类模型权重版本 (默认 full)')
    parser.add_argument('--all', action='store_true', help='download 模式下下载全部四份实验权重')
    args = parser.parse_args()

    torch.set_num_threads(min(8, os.cpu_count() or 4))
    torch.manual_seed(2026)
    device = select_device(args.device)

    if args.task == 'download':
        targets = VALID_NAMES if args.all else ('pretrained', 'full')
        for name in targets:
            p = ensure_checkpoint(name)
            print(f'已就绪: {p} ({p.stat().st_size / (1024 * 1024):.1f} MiB)')
        return

    tokenizer = Tokenizer.from_file(str(ROOT / 'tokenizer.json'))
    if args.task == 'demo':
        run_demo(tokenizer, device=device)
    elif args.task == 'generate':
        if not args.text:
            parser.error('generate 模式需要提供起始英文文本')
        out_text, _, _, _ = run_generate(
            tokenizer, args.text, device=device,
            new_tokens=args.new_tokens, temperature=args.temperature, top_k=args.top_k,
        )
        print(out_text)
    elif args.task == 'classify':
        if not args.text:
            parser.error('classify 模式需要提供待分类影评文本')
        result = run_classify(tokenizer, args.text, device=device, variant=args.variant)
        print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
