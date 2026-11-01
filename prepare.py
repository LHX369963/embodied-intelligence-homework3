"""下载公开划分，训练仅见训练语料的字节级 BPE 分词器。"""
import json
import urllib.request
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'


def prepare():
    DATA.mkdir(exist_ok=True)
    sources = {
        'wiki': ('Salesforce/wikitext', 'wikitext-2-raw-v1'),
        'sentiment': ('cornell-movie-review-data/rotten_tomatoes', 'default')}
    metadata = {'sources': {}, 'wiki': {}, 'sentiment': {}}
    wiki = {}
    for name, (repository, subset) in sources.items():
        metadata['sources'][name] = f'https://huggingface.co/datasets/{repository}'
        for split in ('train', 'validation', 'test'):
            target = DATA / f'{name}_{split}.parquet'
            url = (f'https://huggingface.co/datasets/{repository}/resolve/'
                   f'refs%2Fconvert%2Fparquet/{subset}/{split}/0000.parquet')
            if not target.exists():
                print('下载', name, split, flush=True)
                urllib.request.urlretrieve(url, target)
            rows = pq.read_table(target).to_pylist()
            if name == 'wiki':
                # 保留原行次序和换行，避免在文档级官方划分之间拼接。
                wiki[split] = ''.join(row['text'] for row in rows)
                metadata[name][split] = dict(rows=len(rows), characters=len(wiki[split]))
            else:
                metadata[name][split] = dict(
                    examples=len(rows), negative=sum(r['label'] == 0 for r in rows),
                    positive=sum(r['label'] == 1 for r in rows))
    tok = Tokenizer(models.BPE(unk_token='<unk>'))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=4096, min_frequency=2,
        special_tokens=['<pad>', '<eos>', '<unk>'],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(wiki['train'].splitlines(keepends=True), trainer=trainer)
    tok.save(str(ROOT / 'tokenizer.json'))
    for split, text in wiki.items():
        ids = np.asarray(tok.encode(text).ids, dtype=np.int32)
        np.save(DATA / f'wiki_{split}.npy', ids)
        metadata['wiki'][split]['tokens'] = len(ids)
    metadata['vocab_size'] = tok.get_vocab_size()
    metadata['tokenization_example'] = {
        'text': 'Language models learn from text.',
        'tokens': tok.encode('Language models learn from text.').tokens,
        'ids': tok.encode('Language models learn from text.').ids}
    (ROOT / 'data_info.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    prepare()
