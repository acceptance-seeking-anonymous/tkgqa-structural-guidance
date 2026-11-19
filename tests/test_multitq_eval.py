#!/usr/bin/env python3
"""Unit tests for multitq_eval.py.

Run:  python test_multitq_eval.py   (or pytest test_multitq_eval.py)
Env:  MULTITQ_DIR = directory holding questions/{dev,test}.json and kg/ (the unzipped
      data/MultiTQ of czy1999/MultiTQ @ 94ad61c).  Tests needing it are skipped if absent.

T1  toy cases with hand-computed expectations (granularity truncation, any-gold rule,
    rank cut-off, missing predictions, Single/Multiple bucketing, underscore mapping).
T2  CLI round-trip on a toy gold/pred file pair (JSONL, dict-JSON and list-JSON inputs).
T3  extract_time port: never raises on any question of train/dev/test (the official
    script would crash if it did) and returns the documented formats.
T4  equivalence with a VERBATIM port of MultiQA/utils.py::eval_multi (lines 365-407)
    on the full dev and test splits with 3 synthetic prediction regimes; every stratum
    (counts and hit sums) must match exactly at k = 1 and k = 10.
T5  paired Poisson bootstrap sanity (identical arms -> delta 0, CI [0, 0]; dominance).
"""
import json
import os
import random
import subprocess
import sys
import tempfile
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'src')
sys.path.insert(0, SRC)
import multitq_eval as M  # noqa: E402

DATA = os.environ.get('MULTITQ_DIR', os.path.join(HERE, 'mtq', 'MultiTQ-94ad61c', 'data', 'MultiTQ'))


def Q(quid, answers, at, tl, qtype='equal', qlabel='Single', question='Who visited Japan in 2010?'):
    return {'quid': quid, 'question': question, 'answers': answers, 'answer_type': at,
            'time_level': tl, 'qtype': qtype, 'qlabel': qlabel}


TOY = [
    Q(1, ['China'], 'entity', 'year'),                                   # hit@1
    Q(2, ['China', 'Japan'], 'entity', 'day', qlabel='Multiple', qtype='after_first',
      question='Who first visited Japan after China did?'),               # 2nd gold at rank 5 -> hit@10 only
    Q(3, ['2014-11'], 'time', 'month', qtype='first_last', question='When did X last praise Y?'),  # day pred truncated -> hit@1
    Q(4, ['2014'], 'time', 'year', qtype='first_last', question='In which year did X praise Y?'),  # day pred truncated -> hit@1
    Q(5, ['2014-11-05'], 'time', 'day', qtype='first_last', question='When did X last praise Y?'),  # month pred: no truncation -> miss
    Q(6, ['Head of Government (Togo)'], 'entity', 'year', qtype='before_after',
      question='Before Togo, who visited Mali?'),                         # underscore form, rank 1
    Q(7, ['Iran'], 'entity', 'month', qtype='before_after', question='Before Dec, 2008, who visited Mali?'),  # rank 11 -> miss@10
    Q(8, ['Iran'], 'entity', 'day', qtype='equal_multi', qlabel='Multiple',
      question='In 2012, who last did Barack Obama appeal for?'),         # no prediction -> miss
]
TOY_PRED = {
    1: ['China', 'Japan'],
    2: ['Iran', 'Iraq', 'Peru', 'Chile', 'Japan', 'Mali'],
    3: ['2014-11-30', '2014-10-01'],
    4: ['2014-03-03'],
    5: ['2014-11', '2014-11-06'],
    6: ['Head_of_Government_(Togo)'],
    7: ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'Iran'],
}
# expected (hit@1, hit@10) per quid
TOY_EXPECT = {1: (1, 1), 2: (0, 1), 3: (1, 1), 4: (1, 1), 5: (0, 0), 6: (1, 1), 7: (0, 0), 8: (0, 0)}


def test_T1_toy():
    s, pq = M.score(TOY, TOY_PRED, ks=(1, 10))
    got = {r['quid']: (r['hit@1'], r['hit@10']) for r in pq}
    assert got == TOY_EXPECT, got
    assert s['n_missing_predictions'] == 1
    assert abs(s['hits@1']['overall']['hits'] - 4 / 8) < 1e-12
    assert abs(s['hits@10']['overall']['hits'] - 5 / 8) < 1e-12
    assert s['hits@1']['qlabel']['Multiple']['n'] == 2 and s['hits@10']['qlabel']['Multiple']['correct'] == 1
    assert s['hits@1']['answer_type']['time'] == {'hits': 2 / 3, 'n': 3, 'correct': 2}
    # official granularity subset: equal(1) + equal_multi(8) + before_after with explicit time (7 only;
    # 6 has none) -> 3 questions
    tl = s['hits@1']['time_level_official']
    assert sum(v['n'] for v in tl.values()) == 3, tl
    # raw mode: underscore entity no longer matches
    s_raw, pq_raw = M.score(TOY, TOY_PRED, ks=(1,), normalize_underscore=False)
    assert {r['quid']: r['hit@1'] for r in pq_raw}[6] == 0
    # strict mode raises on missing quid
    try:
        M.score(TOY, TOY_PRED, strict=True)
        raise AssertionError('strict did not raise')
    except ValueError:
        pass
    # headline layout
    row = M.headline_row(s, 1)
    assert list(row) == ['Overall', 'Single', 'Multiple', 'Entity', 'Time']


def test_T2_cli_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        g = os.path.join(d, 'gold.json')
        json.dump(TOY, open(g, 'w'))
        variants = {
            'p.jsonl': '\n'.join(json.dumps({'quid': k, 'ranked': v}) for k, v in TOY_PRED.items()),
            'p_dict.json': json.dumps({str(k): v for k, v in TOY_PRED.items()}, indent=2),
            'p_list.json': json.dumps([{'quid': k, 'ranked': v} for k, v in TOY_PRED.items()], indent=1),
        }
        for name, txt in variants.items():
            p = os.path.join(d, name)
            open(p, 'w').write(txt)
            out = os.path.join(d, name + '.summary.json')
            pqf = os.path.join(d, name + '.pq.jsonl')
            r = subprocess.run([sys.executable, os.path.join(SRC, 'multitq_eval.py'), '--gold', g, '--pred', p,
                                '--json', out, '--per-question', pqf], capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
            s = json.load(open(out))
            assert abs(s['hits@1']['overall']['hits'] - 0.5) < 1e-12, (name, s['hits@1']['overall'])
            assert abs(s['hits@10']['overall']['hits'] - 0.625) < 1e-12
            rows = [json.loads(x) for x in open(pqf)]
            assert {x['quid']: (x['hit@1'], x['hit@10']) for x in rows} == TOY_EXPECT


def _load(split):
    p = os.path.join(DATA, 'questions', f'{split}.json')
    return json.load(open(p)) if os.path.exists(p) else None


def test_T3_extract_time_all_splits():
    n_seen = 0
    for split in ['train', 'dev', 'test']:
        qs = _load(split)
        if qs is None:
            print('  [skip T3: data not found]')
            return
        for q in qs:
            t = M.extract_time(q['question'])  # must not raise
            assert isinstance(t, list) and len(t) <= 1
            if t and t[0]:
                assert len(t[0]) in (4, 7, 10), (q['question'], t)
            n_seen += 1
    assert n_seen == 386787 + 57979 + 54584
    assert M.extract_time('Who visited Japan in 2013?') == ['2013']
    assert M.extract_time('Who visited Japan?') == []


# ---------------------------------------------------------------------------
# verbatim port of the official aggregation loop (MultiQA/utils.py lines 365-407)
# ---------------------------------------------------------------------------
def official_eval_multi_reference(data, topk_answers, k_list=(1, 10)):
    from collections import defaultdict
    out = {}
    for k in k_list:
        hits_at_k = 0
        total = 0
        question_types_count = defaultdict(list)
        simple_complex_count = defaultdict(list)
        entity_time_count = defaultdict(list)
        time_level_count = defaultdict(list)
        time_level_count_all = defaultdict(list)
        for i, question in enumerate(data):
            actual_answers = question['answers']
            question_type = question['qtype']
            time_level = question['time_level']
            if 'Single' in question['qlabel']:
                simple_complex_type = 'Single'
            else:
                simple_complex_type = 'Multiple'
            entity_time_type = question['answer_type']
            predicted = topk_answers[i][:k]

            # multiple time answers - hard way
            if question['answer_type'] == 'time':
                if len(question['answers'][0]) == 4:
                    predicted = [x[:4] for x in predicted]
                elif len(question['answers'][0]) == 7:
                    predicted = [x[:7] for x in predicted]

            if len(set(actual_answers).intersection(set(predicted))) > 0:
                val_to_append = 1
                hits_at_k += 1
            else:
                val_to_append = 0

            question_types_count[question_type].append(val_to_append)
            if (question['qtype'] == 'before_after' and len(question['time']) > 0) or (
                    question['qtype'] == 'equal') or (question['qtype'] == 'equal_multi'):
                time_level_count_all[time_level].append(val_to_append)
            if (question['qtype'] == 'before_after' and len(question['time']) > 0) or (
                    question['qtype'] == 'equal') or (question['qtype'] == 'equal_multi'):
                time_level_count[question['qtype'] + '-' + time_level].append(val_to_append)
            simple_complex_count[simple_complex_type].append(val_to_append)
            entity_time_count[entity_time_type].append(val_to_append)
            total += 1
        out[k] = {'overall': (hits_at_k, total), 'qtype': question_types_count, 'qlabel': simple_complex_count,
                  'time_level_official': time_level_count_all, 'qtype_time_level_official': time_level_count,
                  'answer_type': entity_time_count}
    return out


def _ent_names():
    p = os.path.join(DATA, 'kg', 'tkbc_processed_data', 'ent_id')
    return [line.rstrip('\n').split('\t')[0] for line in open(p, encoding='utf-8')]


def _ts_names():
    p = os.path.join(DATA, 'kg', 'tkbc_processed_data', 'ts_id')
    return [line.rstrip('\n').split('\t')[0] for line in open(p, encoding='utf-8')]


def _synthetic_topk(qs, regime, rng, ents, tss):
    """Official-style top-10 lists: entity names in ent_id (space) form, times as days."""
    out = []
    for q in qs:
        lst = []
        if q['answer_type'] == 'entity':
            pool = ents
        else:
            g = q['answers'][0]
            near = [t for t in tss if t.startswith(g[:4])]  # same-year days -> truncation collisions
            pool = near + tss[:50]
        while len(lst) < 10:
            x = rng.choice(pool if rng.random() < 0.8 else ents + tss)
            if x not in lst:
                lst.append(x)
        p_gold = {'A': 0.35, 'B': 0.7, 'C': 0.15}[regime]
        if rng.random() < p_gold:
            g = rng.choice(q['answers'])
            if q['answer_type'] == 'time' and len(g) < 10:
                cands = [t for t in tss if t.startswith(g)]
                g = rng.choice(cands) if cands else g
            pos = rng.randrange(10 if regime != 'C' else 30)
            if pos < 10:
                lst[pos] = g
        out.append(lst)
    return out


def test_T4_equivalence_with_official_loop():
    for split in ['dev', 'test']:
        qs = _load(split)
        if qs is None:
            print('  [skip T4: data not found]')
            return
        ents, tss = _ent_names(), _ts_names()
        tann = {q['quid']: M.extract_time(q['question']) for q in qs}
        data = [dict(q, time=tann[q['quid']]) for q in qs]
        for regime in ['A', 'B', 'C']:
            rng = random.Random(f'{split}-{regime}')
            topk = _synthetic_topk(qs, regime, rng, ents, tss)
            ref = official_eval_multi_reference(data, topk)
            preds = {q['quid']: topk[i] for i, q in enumerate(qs)}
            mine, _ = M.score(qs, preds, ks=(1, 10), time_annots=tann)
            for k in (1, 10):
                rh, rt = ref[k]['overall']
                assert mine[f'hits@{k}']['overall']['correct'] == rh and mine[f'hits@{k}']['overall']['n'] == rt
                for name in ['qtype', 'qlabel', 'time_level_official', 'qtype_time_level_official', 'answer_type']:
                    R = {key: (sum(v), len(v)) for key, v in ref[k][name].items()}
                    Mi = {key: (v['correct'], v['n']) for key, v in mine[f'hits@{k}'][name].items()}
                    assert R == Mi, (split, regime, k, name, R, Mi)
            print(f'  T4 {split}/{regime}: H@1={mine["hits@1"]["overall"]["hits"]:.4f} '
                  f'H@10={mine["hits@10"]["overall"]["hits"]:.4f} identical to official loop')
            # underscore-form predictions give identical results with normalisation ON
            preds_us = {k_: [x.replace(' ', '_') if not x[:4].isdigit() else x for x in v] for k_, v in preds.items()}
            mine_us, _ = M.score(qs, preds_us, ks=(1, 10), time_annots=tann)
            for k in (1, 10):
                assert mine_us[f'hits@{k}']['overall'] == mine[f'hits@{k}']['overall']


def test_T5_bootstrap():
    import numpy as np
    rng = np.random.default_rng(1)
    a = rng.integers(0, 2, 5000)
    r = M.paired_poisson_bootstrap(a, a, B=200)
    assert r['delta'] == 0 and r['ci_low'] == 0 and r['ci_high'] == 0
    b = a.copy()
    b[:500] = 0
    a2 = a.copy()
    a2[:500] = 1
    r2 = M.paired_poisson_bootstrap(a2, b, B=500)
    assert r2['ci_low'] > 0 and r2['delta'] > 0


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS {name}')
