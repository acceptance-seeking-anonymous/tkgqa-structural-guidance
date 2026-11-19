#!/usr/bin/env python3
"""type_breakdown.py -- per-question-type breakdown of test Hits@1 (overall, Single / Multiple, entity / time answers)
for the 5% arms (Ranker S_5, Reader G_5, Parser P_5) and Reader G_1 / G_100, with 95% bootstrap CIs (1,000 ordinary
resamples per group x arm, one numpy default_rng(0) stream in the fixed group / arm order) and the difference from the
published AT2QA per-type scores. Ranker / Parser hits = per-question mean over the three Ranker seeds.

  python type_breakdown.py --test-json $DATA_DIR/questions/test.json \
         --hits multitq_kpct_test_per_question_hits.csv.gz --out multitq_pertype_ci.csv
(--hits is the per-question CSV written by kpct/test_pass/multitq_kpct_test_pass.py analyze.)
"""
import argparse
import json

import numpy as np
import pandas as pd

AT2QA = {'overall': 0.887, 'multiple': 0.751, 'single': 0.942, 'entity': 0.864, 'time': 0.945}   # published


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-json', required=True)
    ap.add_argument('--hits', required=True)
    ap.add_argument('--out', default='multitq_pertype_ci.csv')
    a = ap.parse_args()

    test = json.load(open(a.test_json))
    ph = pd.read_csv(a.hits)
    meta = pd.DataFrame([{k: q[k] for k in ('quid', 'answer_type', 'time_level', 'qtype', 'qlabel')} for q in test])
    M = ph.merge(meta, on='quid', how='left')
    assert M['qlabel'].notna().all() and len(M) == 54584

    def smean(k):
        return M[[f'S_{k}_seed{i}@1' for i in range(3)]].mean(axis=1)

    def pmean(k):
        return M[[f'P_{k}_seed{i}@1' for i in range(3)]].mean(axis=1)

    arms = {
        'S_1': smean(1), 'G_1': M['G_1@1'], 'P_1': pmean(1),
        'S_5': smean(5), 'G_5': M['G_5@1'], 'P_5': pmean(5),
        'S_10': smean(10), 'G_10': M['G_10@1'], 'P_10': pmean(10),
        'S_100': smean(100), 'G_100': M['G_100@1'], 'P_100': pmean(100)}
    groups = {
        'overall': np.ones(len(M), bool),
        'multiple': (M.qlabel == 'Multiple').values,
        'single': (M.qlabel == 'Single').values,
        'entity': (M.answer_type == 'entity').values,
        'time': (M.answer_type == 'time').values}

    rng = np.random.default_rng(0)
    rows = []
    for g, mask in groups.items():
        idx = np.where(mask)[0]
        for arm in ('S_5', 'G_5', 'P_5', 'G_1', 'G_100'):
            v = arms[arm].values[idx]
            bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(1000)]
            lo, hi = np.quantile(bs, [.025, .975])
            rows.append(dict(group=g, n=len(idx), arm=arm, hits1=round(v.mean(), 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                             AT2QA_published=AT2QA[g], diff_pts=round((v.mean() - AT2QA[g]) * 100, 2)))
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
