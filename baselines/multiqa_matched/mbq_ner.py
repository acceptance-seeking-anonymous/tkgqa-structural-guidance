#!/usr/bin/env python3
"""mbq_ner.py -- MultiQA NER preprocessing (port of czy1999/MultiTQ@94ad61c MultiQA/ner_task.py).

Identical per-question logic to the official script:
  flair/ner-english-large tagger, predict in batches of 64, difflib.get_close_matches(text, KG names, n=1)
  for every NER span text not already an exact KG name, extract_time() regex + dateutil, fields
  'time', 'entities', 'entity_positions' added to each question.
Adaptations (recorded in the registration): flair version (0.11.3 is not installable on Python 3.12;
the current flair release is used with the same HF checkpoint); questions are processed per file
in the order dev_sel2000, test, train(k10 members), train(rest) so that the 1/5/10% levels do not
wait for the full train pass; worker pool capped at 16 processes (shared host).
Inputs only use the question text (test is stripped to {quid, question}). Output: compact JSONL.GZ.
"""
import difflib, gzip, json, os, re, sys, time
import multiprocessing as mp
import flair, torch
from flair.data import Sentence
from flair.models import SequenceTagger
from dateutil.parser import parse

W = os.path.abspath(os.getcwd())
flair.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
_KEYS = None
_KEYS_SET = None


def _worker_init(keys_list, keys_set):
    global _KEYS, _KEYS_SET
    _KEYS = keys_list
    _KEYS_SET = keys_set


def _fuzzy_match(text):
    if text in _KEYS_SET:
        return text, (text,)
    m = difflib.get_close_matches(text, _KEYS, n=1)
    return text, tuple(m)


def extract_time(sentence: str):
    date_pattern = r'\b(?:\d{1,2}\s)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s?\d{1,2}?(?:\s|,)?\s?\d{2,4}\b|\b\d{4}\b'
    date_match = re.search(date_pattern, sentence)
    if date_match:
        date_str = date_match.group()
        date_obj = parse(date_str)
        date_format = ''
        if len(date_str.split(' ')) == 3:
            date_format = f"{date_obj.year}-{date_obj.month:02d}-{date_obj.day:02d}"
        elif len(date_str.split(' ')) == 2:
            date_format = f"{date_obj.year}-{date_obj.month:02d}"
        elif len(date_str.split(' ')) == 1:
            date_format = f"{date_obj.year}"
        return [date_format]
    else:
        return []


def log(msg):
    with open(os.path.join(W, 'ner_log.txt'), 'a') as f:
        f.write(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + ' ' + msg + '\n')
    print(msg, flush=True)


def process(name, dataset, tagger, keys, keys_set, nworkers):
    t0 = time.time()
    BATCH_SIZE = 64
    questions = [Sentence(x['question']) for x in dataset]
    for start in range(0, len(questions), BATCH_SIZE):
        tagger.predict(questions[start:start + BATCH_SIZE], mini_batch_size=BATCH_SIZE)
        if (start // BATCH_SIZE) % 200 == 0:
            log(f'[{name}] NER {start}/{len(questions)} {time.time()-t0:.0f}s')
    t1 = time.time()
    per_q_spans, unique_texts = [], set()
    for s in questions:
        spans = []
        for ent in s.get_spans('ner'):
            spans.append((ent.text, ent.start_position, ent.end_position))
            unique_texts.add(ent.text)
        per_q_spans.append(spans)
    unique_texts = sorted(unique_texts)
    text_to_match = {}
    if unique_texts:
        ctx = mp.get_context('fork')
        with ctx.Pool(processes=nworkers, initializer=_worker_init, initargs=(keys, keys_set)) as pool:
            chunksize = max(1, len(unique_texts) // (nworkers * 8))
            for text, matched in pool.imap_unordered(_fuzzy_match, unique_texts, chunksize=chunksize):
                text_to_match[text] = matched
    t2 = time.time()
    out = os.path.join(W, f'ner_{name}.jsonl.gz')
    tmp = out + '.tmp'
    with gzip.open(tmp, 'wt', encoding='utf-8') as f:
        for ix, spans in enumerate(per_q_spans):
            e = []
            for text, start, end in spans:
                matched = list(text_to_match.get(text, ()))
                e.append({'entity': matched, 'position': [start, end]})
            clean = [y for y in e if len(y['entity']) > 0]
            rec = {'quid': dataset[ix]['quid'], 'time': extract_time(dataset[ix]['question']),
                   'entities': [x['entity'][0] for x in clean], 'entity_positions': clean}
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    os.replace(tmp, out)
    log(f'[{name}] DONE n={len(dataset)} unique_texts={len(unique_texts)} ner_s={t1-t0:.1f} match_s={t2-t1:.1f}')
    return {'n': len(dataset), 'ner_s': t1 - t0, 'match_s': t2 - t1, 'unique_texts': len(unique_texts)}


def main():
    parts = sys.argv[1].split(',') if len(sys.argv) > 1 else ['dev_sel2000', 'test', 'train_k10', 'train_rest']
    nworkers = int(os.environ.get('NER_WORKERS', '16'))
    with open(os.path.join(W, 'data/MultiTQ/kg/tkbc_processed_data/ent_id'), 'r', encoding='utf-8') as f:
        keys = [l.strip().split('\t')[0] for l in f.readlines()]
    keys_set = set(keys)
    log(f'flair {flair.__version__} torch {torch.__version__} device {flair.device} '
        f'gpu {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None} keys {len(keys)}')
    tagger = SequenceTagger.load('flair/ner-english-large')
    log('tagger loaded')
    train = json.load(open(os.path.join(W, 'qin/train_all.json')))
    k10 = set(json.load(open(os.path.join(W, 'lists/k10_s0.json'))))
    srcs = {
        'dev_sel2000': lambda: json.load(open(os.path.join(W, 'qin/dev_sel2000.json'))),
        'test': lambda: json.load(open(os.path.join(W, 'qin/test_stripped.json'))),
        'train_k10': lambda: [q for q in train if q['quid'] in k10],
        'train_rest': lambda: [q for q in train if q['quid'] not in k10],
    }
    summary = {'flair': flair.__version__, 'torch': torch.__version__, 'parts': {}}
    for p in parts:
        summary['parts'][p] = process(p, srcs[p](), tagger, keys, keys_set, nworkers)
        json.dump(summary, open(os.path.join(W, 'ner_summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
