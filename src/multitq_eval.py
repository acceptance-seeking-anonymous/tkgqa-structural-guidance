#!/usr/bin/env python3
"""multitq_eval.py -- MultiTQ scorer reproducing the official MultiQA semantics.

Reference implementation: czy1999/MultiTQ @ 94ad61c, MultiQA/utils.py::eval_multi
(lines 365-407) and MultiQA/ner_task.py::extract_time.

Official semantics (reproduced exactly):
  * predicted = ranked[:k]   (the system's ranked answer strings, no de-duplication)
  * if answer_type == 'time':  L = len(answers[0]);
        L == 4  -> predicted = [x[:4] for x in predicted]   (year)
        L == 7  -> predicted = [x[:7] for x in predicted]   (month)
        (L == 10, day, untouched).  Truncation is applied to EVERY prediction string,
        including entity strings (harmless: they cannot equal a date).
  * hit@k = 1  iff  set(answers) & set(predicted) is non-empty  (any gold in top k).
  * Hits@k = mean over ALL questions of the split (missing prediction -> miss).
  * Strata: qtype; 'Single' if 'Single' in qlabel else 'Multiple'; answer_type;
    official granularity table ("time_level_count_all"): only questions with
    qtype in {equal, equal_multi}, or qtype == before_after AND the question carries an
    explicit time (extract_time(question) non-empty); "<qtype>-<time_level>" cells
    over the same subset.  We additionally report time_level over ALL questions
    (labelled `time_level_all`, NOT an official stratum).

Only deviation option (default ON, documented): entity strings are compared after
mapping '_' -> ' '.  The official predictions come from tkbc_processed_data/ent_id
(space form) while kg/full.txt and entity2id.json use underscores; the mapping is a
bijection over the 10,488 KG entity names and no gold answer contains '_' (verified
in multitq_data_profile.json), so it is a format conversion, not a relaxation.
Pass normalize_underscore=False / --raw to disable.

Prediction file formats accepted:
  * JSONL: one {"quid": int, "ranked": [str, ...]} per line
  * JSON : {"<quid>": [str, ...], ...}  or  [{"quid":..., "ranked": [...]}, ...]

CLI:
  python multitq_eval.py --gold test.json --pred preds.jsonl [--k 1 10]
         [--per-question out.jsonl] [--json out.json] [--strict] [--raw]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import OrderedDict, defaultdict

__version__ = "1.0.0"

QTYPES = ["equal", "before_after", "first_last", "equal_multi", "after_first", "before_last"]
GRAN_QTYPES = ["equal", "before_after", "equal_multi"]
TIME_LEVELS = ["day", "month", "year"]

# ---------------------------------------------------------------------------
# extract_time: verbatim port of MultiQA/ner_task.py (regex + dateutil.parse)
# ---------------------------------------------------------------------------
_DATE_PATTERN = (r'\b(?:\d{1,2}\s)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|'
                 r'Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|'
                 r'Dec(?:ember)?)\s?\d{1,2}?(?:\s|,)?\s?\d{2,4}\b|\b\d{4}\b')


def extract_time(sentence: str):
    """Exact port of MultiQA ner_task.extract_time (used only for the official
    granularity-table subset: before_after questions with an explicit time)."""
    from dateutil.parser import parse  # same dependency as the official script
    m = re.search(_DATE_PATTERN, sentence)
    if not m:
        return []
    date_str = m.group()
    date_obj = parse(date_str)
    n = len(date_str.split(' '))
    if n == 3:
        return [f"{date_obj.year}-{date_obj.month:02d}-{date_obj.day:02d}"]
    if n == 2:
        return [f"{date_obj.year}-{date_obj.month:02d}"]
    if n == 1:
        return [f"{date_obj.year}"]
    return [""]  # official code returns [''] (non-empty list) for >3 tokens


# ---------------------------------------------------------------------------
# core scoring
# ---------------------------------------------------------------------------
def _norm(s: str, normalize_underscore: bool) -> str:
    return s.replace('_', ' ') if normalize_underscore else s


def hit_at_k(question: dict, ranked, k: int, normalize_underscore: bool = True) -> int:
    """Official hit for one question (1/0)."""
    if ranked is None:
        return 0
    predicted = [_norm(str(x), normalize_underscore) for x in list(ranked)[:k]]
    answers = question['answers']
    if question['answer_type'] == 'time':
        L = len(answers[0])
        if L == 4:
            predicted = [x[:4] for x in predicted]
        elif L == 7:
            predicted = [x[:7] for x in predicted]
    return 1 if len(set(answers).intersection(set(predicted))) > 0 else 0


def qlabel_bucket(q: dict) -> str:
    return 'Single' if 'Single' in q['qlabel'] else 'Multiple'


def in_official_granularity_subset(q: dict, time_annot=None) -> bool:
    t = extract_time(q['question']) if time_annot is None else time_annot
    return (q['qtype'] == 'before_after' and len(t) > 0) or q['qtype'] in ('equal', 'equal_multi')


def load_predictions(path: str) -> dict:
    """Return {quid(int): ranked list}."""
    with open(path, 'r', encoding='utf-8') as f:
        txt = f.read()
    preds = {}
    s = txt.strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        obj = None  # -> JSONL
    if isinstance(obj, dict):
        if 'quid' in obj and 'ranked' in obj:  # single-line JSONL
            preds[int(obj['quid'])] = list(obj['ranked'])
        else:
            for k, v in obj.items():
                preds[int(k)] = list(v)
        return preds
    if isinstance(obj, list):
        for r in obj:
            q = int(r['quid'])
            if q in preds:
                raise ValueError(f'duplicate quid {q} in prediction file')
            preds[q] = list(r['ranked'])
        return preds
    for line in s.splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        q = int(r['quid'])
        if q in preds:
            raise ValueError(f'duplicate quid {q} in prediction file')
        preds[q] = list(r['ranked'])
    return preds


def score(questions, preds: dict, ks=(1, 10), normalize_underscore: bool = True,
          strict: bool = False, time_annots: dict | None = None):
    """Score a split. Returns (summary dict, per-question list).

    questions: list of gold dicts (the official split file, all questions).
    preds: {quid: ranked list}.  Missing quids score 0 (official behaviour: the
           official loop evaluates every question in the split); strict=True raises.
    time_annots: optional {quid: extract_time(question)} cache.
    """
    gold_quids = [int(q['quid']) for q in questions]
    gq = set(gold_quids)
    missing = [q for q in gold_quids if q not in preds]
    extra = [q for q in preds if q not in gq]
    if strict and (missing or extra):
        raise ValueError(f'coverage error: {len(missing)} missing, {len(extra)} extra quids')
    n_dup = 0
    short = defaultdict(int)
    for q in gold_quids:
        r = preds.get(q)
        if r is None:
            continue
        rr = [_norm(str(x), normalize_underscore) for x in r[:max(ks)]]
        if len(set(rr)) < len(rr):
            n_dup += 1
        for k in ks:
            if len(r) < k:
                short[k] += 1

    per_q = []
    strata = OrderedDict()
    for k in ks:
        strata[k] = {'overall': [], 'qtype': defaultdict(list), 'qlabel': defaultdict(list),
                     'answer_type': defaultdict(list), 'time_level_official': defaultdict(list),
                     'qtype_time_level_official': defaultdict(list), 'time_level_all': defaultdict(list),
                     'qtype_answer_type': defaultdict(list)}
    for q in questions:
        quid = int(q['quid'])
        r = preds.get(quid)
        t = time_annots.get(quid) if time_annots is not None else extract_time(q['question'])
        in_gran = in_official_granularity_subset(q, t)
        row = {'quid': quid}
        for k in ks:
            h = hit_at_k(q, r, k, normalize_underscore)
            row[f'hit@{k}'] = h
            S = strata[k]
            S['overall'].append(h)
            S['qtype'][q['qtype']].append(h)
            S['qlabel'][qlabel_bucket(q)].append(h)
            S['answer_type'][q['answer_type']].append(h)
            S['time_level_all'][q['time_level']].append(h)
            S['qtype_answer_type'][q['qtype'] + '-' + q['answer_type']].append(h)
            if in_gran:
                S['time_level_official'][q['time_level']].append(h)
                S['qtype_time_level_official'][q['qtype'] + '-' + q['time_level']].append(h)
        per_q.append(row)

    def agg(v):
        return {'hits': (sum(v) / len(v)) if v else float('nan'), 'n': len(v), 'correct': int(sum(v))}

    summary = OrderedDict()
    summary['scorer'] = f'multitq_eval.py v{__version__} (official MultiQA eval_multi semantics)'
    summary['normalize_underscore'] = normalize_underscore
    summary['n_questions'] = len(questions)
    summary['n_missing_predictions'] = len(missing)
    summary['n_extra_predictions'] = len(extra)
    summary['n_questions_with_duplicate_strings_in_topk'] = n_dup
    summary['n_short_lists'] = {f'<{k}': short[k] for k in ks}
    for k in ks:
        S = strata[k]
        block = OrderedDict()
        block['overall'] = agg(S['overall'])
        for name in ['qlabel', 'answer_type', 'qtype', 'time_level_official',
                     'qtype_time_level_official', 'time_level_all', 'qtype_answer_type']:
            block[name] = OrderedDict((key, agg(S[name][key])) for key in sorted(S[name], key=str.lower))
        summary[f'hits@{k}'] = block
    return summary, per_q


def headline_row(summary: dict, k: int = 1) -> OrderedDict:
    """PI layout: Overall | Single, Multiple | Entity, Time."""
    b = summary[f'hits@{k}']
    g = lambda d, key: d.get(key, {}).get('hits', float('nan'))
    return OrderedDict([('Overall', b['overall']['hits']),
                        ('Single', g(b['qlabel'], 'Single')), ('Multiple', g(b['qlabel'], 'Multiple')),
                        ('Entity', g(b['answer_type'], 'entity')), ('Time', g(b['answer_type'], 'time'))])


# ---------------------------------------------------------------------------
# paired Poisson(1) bootstrap over questions (house protocol, B = 2000)
# ---------------------------------------------------------------------------
def paired_poisson_bootstrap(hits_a, hits_b, B: int = 2000, seed: int = 0, alpha: float = 0.05):
    """delta = mean(a) - mean(b); Poisson(1) weights per question shared by both arms."""
    import numpy as np
    a = np.asarray(hits_a, dtype=float)
    b = np.asarray(hits_b, dtype=float)
    assert a.shape == b.shape and a.ndim == 1
    rng = np.random.default_rng(seed)
    d = a - b
    deltas = np.empty(B)
    for i in range(B):
        w = rng.poisson(1.0, size=d.shape[0])
        sw = w.sum()
        deltas[i] = (w * d).sum() / sw if sw > 0 else 0.0
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return {'delta': float(d.mean()), 'ci_low': float(lo), 'ci_high': float(hi),
            'p_one_sided_le0': float((deltas <= 0).mean()), 'B': B, 'seed': seed}


def _fmt(x):
    return '  -  ' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:.3f}'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--gold', required=True)
    ap.add_argument('--pred', required=True)
    ap.add_argument('--k', type=int, nargs='+', default=[1, 10])
    ap.add_argument('--json', default=None, help='write summary JSON here')
    ap.add_argument('--per-question', default=None, help='write per-question hits JSONL here')
    ap.add_argument('--strict', action='store_true', help='fail on missing/extra quids')
    ap.add_argument('--raw', action='store_true', help="disable '_'->' ' entity normalisation")
    a = ap.parse_args(argv)
    with open(a.gold, encoding='utf-8') as f:
        questions = json.load(f)
    preds = load_predictions(a.pred)
    summary, per_q = score(questions, preds, ks=tuple(a.k), normalize_underscore=not a.raw, strict=a.strict)
    for k in a.k:
        row = headline_row(summary, k)
        print(f'Hits@{k}: ' + ' | '.join(f'{n} {_fmt(v)}' for n, v in row.items()))
    print(f"missing={summary['n_missing_predictions']} extra={summary['n_extra_predictions']} "
          f"dup_topk={summary['n_questions_with_duplicate_strings_in_topk']}")
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(summary, f, indent=1)
    if a.per_question:
        with open(a.per_question, 'w') as f:
            for r in per_q:
                f.write(json.dumps(r) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
