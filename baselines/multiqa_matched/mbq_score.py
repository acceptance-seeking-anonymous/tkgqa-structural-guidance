#!/usr/bin/env python3
"""mbq_score.py -- single test pass for the matched MultiQA baseline (run ONCE, after the freeze).

usage: python mbq_score.py <freeze.json> <test.json> <src/multitq_eval.py> <k% per-question hits csv.gz> <selection.json> <out_prefix>
Inputs: the frozen prediction files listed in the freeze JSON (sha256 re-checked before scoring),
the official MultiTQ test.json, the project scorer multitq_eval.py (official MultiQA eval_multi
semantics), the per-question hits of the frozen arms, the selection record.
Outputs: per-question hits CSV (keyed by quid), results JSON.
Statistics: Hits@1/@10 with Poisson(1) bootstrap 95% CI (B=2000, seed 0, two-sided percentile);
paired contrasts with the same paired bootstrap (descriptive, no pass/fail margin).
"""
import gzip, hashlib, importlib.util, json, os, sys
import numpy as np
import pandas as pd


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''):
            h.update(ch)
    return h.hexdigest()


def boot_mean(x, B=2000, seed=0):
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    bs = np.empty(B)
    for i in range(B):
        w = rng.poisson(1.0, size=x.shape[0]); s = w.sum()
        bs[i] = (w * x).sum() / s
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return float(x.mean()), [float(lo), float(hi)]


def paired(a, b, B=2000, seed=0):
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    rng = np.random.default_rng(seed)
    bs = np.empty(B)
    for i in range(B):
        w = rng.poisson(1.0, size=d.shape[0]); s = w.sum()
        bs[i] = (w * d).sum() / s
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return {'delta': float(d.mean()), 'ci95': [float(lo), float(hi)], 'B': B, 'seed': seed,
            'n_a_only': int(((np.asarray(a) > np.asarray(b))).sum()), 'n_b_only': int(((np.asarray(b) > np.asarray(a))).sum())}


def main(freeze_path, test_json, evaluator_py, hits_csv, selection_json, out_prefix):
    spec = importlib.util.spec_from_file_location('multitq_eval', evaluator_py)
    ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
    freeze = json.load(open(freeze_path))
    sel = json.load(open(selection_json))
    gold = json.load(open(test_json))
    quids = [q['quid'] for q in gold]
    n = len(gold)
    cols = {}
    hash_check = []
    for f in freeze['files']:
        p = f['local_path']
        h = sha256(p)
        ok = (h == f['sha256'])
        hash_check.append({'file': f['name'], 'ok': ok})
        if not ok:
            raise SystemExit(f'hash mismatch {p}')
        opener = gzip.open if p.endswith('.gz') else open
        with opener(p, 'rt', encoding='utf-8') as fh:
            preds = {}
            for line in fh:
                r = json.loads(line); preds[int(r['quid'])] = r['ranked']
        assert len(preds) == n and set(preds) == set(quids), f['name']
        for k in (1, 10):
            cols[f"{f['arm']}@{k}"] = np.array([ev.hit_at_k(q, preds[q['quid']], k) for q in gold], dtype=np.float64)
    df = pd.DataFrame({'quid': quids})
    for c in sorted(cols):
        df[c] = cols[c]
    # seed means per (level, emb, ckpt)
    groups = {}
    for f in freeze['files']:
        groups.setdefault(f['group'], []).append(f['arm'])
    for g, arms in groups.items():
        for k in (1, 10):
            df[f'{g}@{k}'] = np.mean([df[f'{a}@{k}'] for a in arms], axis=0)
    # primary arm per level = dev-selected embedding, primary checkpoint
    for lvl, s in sel['selected'].items():
        for k in (1, 10):
            df[f'MBQ_{lvl}@{k}'] = df[f"MBQ_{s['emb'].upper()}_{lvl}@{k}"]
    ref = pd.read_csv(hits_csv)
    ref = ref.set_index('quid').loc[quids].reset_index()
    meta = pd.DataFrame(gold)[['quid', 'qtype', 'qlabel', 'answer_type', 'time_level']]
    meta['qlabel2'] = np.where(meta['qlabel'].str.contains('Single'), 'Single', 'Multiple')
    res = {'n': n, 'hash_check_all_ok': all(x['ok'] for x in hash_check), 'arms': {}, 'contrasts': {}, 'strata': {}}
    arm_cols = sorted({c.rsplit('@', 1)[0] for c in df.columns if c != 'quid'})
    for a in arm_cols:
        r = {}
        for k in (1, 10):
            m, ci = boot_mean(df[f'{a}@{k}'])
            r[f'hits@{k}'] = m; r[f'hits@{k}_ci95'] = ci
        res['arms'][a] = r
    comps = {'RULE': 'RULE_{k}@1', 'S_seedmean': 'S_{k}@1', 'G': 'G_{k}@1', 'P_seedmean': 'P_{k}@1'}
    for lvl in sel['selected']:
        kk = lvl
        a = df[f'MBQ_{lvl}@1'].values
        for name, tmpl in comps.items():
            col = tmpl.format(k=kk)
            if col not in ref.columns:
                continue
            b = ref[col].values
            cm, cci = boot_mean(b)
            res['contrasts'][f'MBQ_{lvl} - {name}_{kk}'] = dict(paired(a, b), comparator_col=col, comparator_hits1=cm,
                                                               comparator_hits1_ci95=cci, mbq_hits1=float(a.mean()))
        # strata
        st = {}
        for sname, scol in [('qlabel', 'qlabel2'), ('answer_type', 'answer_type'), ('qtype', 'qtype')]:
            for v in sorted(meta[scol].unique()):
                m = (meta[scol] == v).values
                row = {'n': int(m.sum()), 'MBQ': float(a[m].mean())}
                for name, tmpl in comps.items():
                    col = tmpl.format(k=kk)
                    if col in ref.columns:
                        row[name] = float(ref[col].values[m].mean())
                st[f'{sname}={v}'] = row
        res['strata'][lvl] = st
    df.to_csv(out_prefix + '_per_question_hits.csv.gz', index=False, float_format='%.6g')
    json.dump(res, open(out_prefix + '_test_scores.json', 'w'), indent=1)
    return res


if __name__ == '__main__':
    main(*sys.argv[1:7])
