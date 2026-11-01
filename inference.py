"""可直接使用的命令行推理入口，默认在 CPU 上运行。"""
import argparse
from pathlib import Path
import json
import torch
from tokenizers import Tokenizer
from model import Config, GPT, SentimentModel, load_gpt

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['generate','classify'])
    parser.add_argument('text')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--new-tokens', type=int, default=60)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    tokenizer = Tokenizer.from_file(str(ROOT / 'tokenizer.json'))
    if args.task == 'generate':
        model, _ = load_gpt(ROOT / 'checkpoints/pretrained.pt', args.device)
        ids = tokenizer.encode(args.text).ids or [1]
        generated = model.generate(torch.tensor([ids], device=args.device), args.new_tokens)
        print(tokenizer.decode(generated[0].tolist()))
    else:
        saved = torch.load(ROOT / 'checkpoints/full.pt', map_location='cpu', weights_only=False)
        config = Config(**saved['config'])
        model = SentimentModel(GPT(config)).to(args.device).eval()
        model.load_state_dict(saved['model'])
        ids = tokenizer.encode(args.text).ids[:config.context - 1] + [tokenizer.token_to_id('<eos>')]
        with torch.no_grad():
            probabilities = model(torch.tensor([ids], device=args.device),
                                  torch.tensor([len(ids)], device=args.device)).softmax(-1)[0]
        print(json.dumps(dict(label='正面' if int(probabilities.argmax()) else '负面',
                              positive_probability=float(probabilities[1])), ensure_ascii=False))


if __name__ == '__main__':
    main()
