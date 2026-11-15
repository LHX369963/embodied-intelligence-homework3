"""下载公开语料划分，训练字节级 BPE 分词器（词表 32768），生成词元序列与数据核查记录。"""
import json
import os
import urllib.request
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
VOCAB_SIZE = 32768
ENDPOINTS = [
    os.environ.get('HF_ENDPOINT', '').rstrip('/'),
    'https://hf-mirror.com',
    'https://huggingface.co',
]


def download_parquet(repo_path: str, target: Path):
    if target.exists() and target.stat().st_size > 0:
        return
    hosts = [h for i, h in enumerate(ENDPOINTS) if h and h not in ENDPOINTS[:i]]
    last_error = None
    for host in hosts:
        url = f'{host}/api/datasets/{repo_path}'
        try:
            print(f'下载 {target.name} <- {host}', flush=True)
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=60) as resp, open(target, 'wb') as out:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            return
        except Exception as exc:
            last_error = exc
            if target.exists():
                target.unlink()
    raise RuntimeError(f'无法下载 {repo_path}: {last_error}')


def normalize_text(text: str) -> str:
    return ' '.join(text.lower().split())


def clean_imdb(text: str) -> str:
    return text.replace('<br /><br />', '\n').replace('<br />', '\n').replace('<br/>', '\n').strip()


def encode_lines(tok: Tokenizer, lines, append_eos_id=None, chunk_size=8192) -> np.ndarray:
    chunks = []
    for start in range(0, len(lines), chunk_size):
        batch = lines[start:start + chunk_size]
        encoded = tok.encode_batch(batch)
        if append_eos_id is None:
            flat = [tid for item in encoded for tid in item.ids]
        else:
            flat = [tid for item in encoded for tid in (item.ids + [append_eos_id])]
        if flat:
            chunks.append(np.asarray(flat, dtype=np.uint16))
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.uint16)


def is_top_article_header(line: str) -> bool:
    return line.startswith(' = ') and not line.startswith(' = = ')


def prepare():
    DATA.mkdir(exist_ok=True)
    parquet_specs = {
        'wiki_train_0': 'Salesforce/wikitext/parquet/wikitext-103-raw-v1/train/0.parquet',
        'wiki_train_1': 'Salesforce/wikitext/parquet/wikitext-103-raw-v1/train/1.parquet',
        'wiki_validation': 'Salesforce/wikitext/parquet/wikitext-103-raw-v1/validation/0.parquet',
        'wiki_test': 'Salesforce/wikitext/parquet/wikitext-103-raw-v1/test/0.parquet',
        'imdb_unsupervised': 'stanfordnlp/imdb/parquet/plain_text/unsupervised/0.parquet',
        'sentiment_train': 'cornell-movie-review-data/rotten_tomatoes/parquet/default/train/0.parquet',
        'sentiment_validation': 'cornell-movie-review-data/rotten_tomatoes/parquet/default/validation/0.parquet',
        'sentiment_test': 'cornell-movie-review-data/rotten_tomatoes/parquet/default/test/0.parquet',
    }
    for name, repo_path in parquet_specs.items():
        download_parquet(repo_path, DATA / f'{name}.parquet')

    sentiment_rows = {}
    sentiment_norm = {}
    sentiment_meta = {}
    for split in ('train', 'validation', 'test'):
        rows = pq.read_table(DATA / f'sentiment_{split}.parquet').to_pylist()
        sentiment_rows[split] = rows
        sentiment_norm[split] = {normalize_text(r['text']) for r in rows}
        sentiment_meta[split] = dict(
            examples=len(rows),
            negative=sum(r['label'] == 0 for r in rows),
            positive=sum(r['label'] == 1 for r in rows),
        )

    holdout_norm = sentiment_norm['validation'] | sentiment_norm['test']
    raw_imdb = pq.read_table(DATA / 'imdb_unsupervised.parquet').to_pylist()
    imdb_texts = []
    removed_imdb_overlap = 0
    for row in raw_imdb:
        cleaned = clean_imdb(row['text'])
        if not cleaned:
            continue
        if normalize_text(cleaned) in holdout_norm:
            removed_imdb_overlap += 1
            continue
        imdb_texts.append(cleaned)

    wiki_lines = {'train': [], 'validation': [], 'test': []}
    wiki_meta = {}
    for split, files in [
        ('train', ['wiki_train_0.parquet', 'wiki_train_1.parquet']),
        ('validation', ['wiki_validation.parquet']),
        ('test', ['wiki_test.parquet']),
    ]:
        total_rows = 0
        non_empty = []
        for fname in files:
            rows = pq.read_table(DATA / fname).to_pylist()
            total_rows += len(rows)
            non_empty.extend(r['text'] for r in rows if r['text'])
        wiki_lines[split] = non_empty
        chars = sum(len(s) for s in non_empty)
        utf8_bytes = sum(len(s.encode('utf-8')) for s in non_empty)
        wiki_meta[split] = dict(rows=total_rows, non_empty_lines=len(non_empty),
                                characters=chars, utf8_bytes=utf8_bytes)

    print('训练字节级 BPE 分词器...', flush=True)
    tok = Tokenizer(models.BPE(unk_token='<unk>'))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=['<pad>', '<eos>', '<unk>'],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    def train_corpus_iterator():
        for line in wiki_lines['train']:
            yield line
        for doc in imdb_texts:
            yield doc + '\n'

    tok.train_from_iterator(train_corpus_iterator(), trainer=trainer)
    tok.save(str(ROOT / 'tokenizer.json'))
    eos_id = tok.token_to_id('<eos>')

    print('编码 WikiText-103 与 IMDb 无标签语料...', flush=True)
    # 在百科词条起始处插入 <eos>，使模型学习完整的文档结束信号。
    wiki_train_articles = []
    current = []
    for line in wiki_lines['train']:
        if is_top_article_header(line) and current:
            wiki_train_articles.append(''.join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        wiki_train_articles.append(''.join(current))

    wiki_train_ids = encode_lines(tok, wiki_train_articles, append_eos_id=eos_id, chunk_size=2048)
    imdb_train_ids = encode_lines(tok, imdb_texts, append_eos_id=eos_id, chunk_size=4096)
    pretrain_train_ids = np.concatenate([wiki_train_ids, imdb_train_ids])
    np.save(DATA / 'wiki_train.npy', pretrain_train_ids)

    for split in ('validation', 'test'):
        split_ids = encode_lines(tok, wiki_lines[split], append_eos_id=None, chunk_size=8192)
        np.save(DATA / f'wiki_{split}.npy', split_ids)
        wiki_meta[split]['tokens'] = int(len(split_ids))

    wiki_meta['train']['wiki_tokens'] = int(len(wiki_train_ids))
    wiki_meta['train']['imdb_unsupervised_tokens'] = int(len(imdb_train_ids))
    wiki_meta['train']['tokens'] = int(len(pretrain_train_ids))
    wiki_meta['train']['articles'] = len(wiki_train_articles)

    sample_text = 'Language models learn general representations from large-scale text corpora.'
    metadata = dict(
        sources={
            'wikitext_103': 'https://huggingface.co/datasets/Salesforce/wikitext (wikitext-103-raw-v1)',
            'imdb_unsupervised': 'https://huggingface.co/datasets/stanfordnlp/imdb (unsupervised split)',
            'rotten_tomatoes': 'https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes',
        },
        vocab_size=tok.get_vocab_size(),
        wiki=wiki_meta,
        imdb_unsupervised=dict(
            documents=len(imdb_texts),
            characters=sum(len(s) for s in imdb_texts),
            tokens=int(len(imdb_train_ids)),
        ),
        pretrain_total_tokens=int(len(pretrain_train_ids)),
        sentiment=sentiment_meta,
        tokenization_example=dict(
            text=sample_text,
            tokens=tok.encode(sample_text).tokens,
            ids=tok.encode(sample_text).ids,
        ),
    )
    (ROOT / 'data_info.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    prepare()
