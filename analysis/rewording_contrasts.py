#!/usr/bin/env python3
"""rewording_contrasts.py -- post-hoc, descriptive parser-vs-reader contrasts on the R1 rewrites (P3, LLM-judge
accepted): drop(G_k) - drop(P_k), drop(S_k) - drop(P_k), and rewritten-wording differences P_k - G_k, P_k - GU_k,
G_k - GU_k at k = 1, 5, 100 (paired Poisson(1) bootstrap, B = 2000, seed 0, of rescore_after_review.boot).
Not a registered analysis.

  python rewording_contrasts.py --gold paraphrase_review_gold.json --pred-dir <extracted R1 probe predictions> \
         [--scorer src/multitq_eval.py] --out robustness_posthoc_parser_vs_reader.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _r1_common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gold', required=True)
    ap.add_argument('--pred-dir', required=True)
    ap.add_argument('--scorer', default=None)
    ap.add_argument('--out', default='robustness_posthoc_parser_vs_reader.csv')
    a = ap.parse_args()
    ME = C.scorer(a.scorer)
    RS = C.rescorer()
    gold = C.gold(a.gold)
    files = RS.arm_files()
    boot = RS.boot
    pred = {arm: [RS.load_pred(os.path.join(a.pred_dir, f)) for f in fs] for arm, fs in files.items()}
    quids = sorted(gold)

    def hits(arm, vi):
        return np.array([float(np.mean([ME.hit_at_k(gold[q], P_.get(10 * q + vi), 1) for P_ in pred[arm]])) for q in quids])

    H = {(arm, v): hits(arm, vi) for arm in pred for v, vi in (('ORIG', 0), ('P3', 3))}
    out = []

    def bc(name, a_, b_):
        r = boot(a_, b_)
        out.append((name, round(r['delta'] * 100, 2), [round(x * 100, 2) for x in r['ci']]))

    for K in ('1', '5', '100'):
        d = {x: H[(f'{x}_{K}', 'ORIG')] - H[(f'{x}_{K}', 'P3')] for x in ('S', 'G', 'P')}
        bc(f'drop(G_{K})-drop(P_{K})', d['G'], d['P'])
        bc(f'drop(S_{K})-drop(P_{K})', d['S'], d['P'])
        bc(f'P3: P_{K}-G_{K}', H[(f'P_{K}', 'P3')], H[(f'G_{K}', 'P3')])
        if f'GU_{K}' in pred:
            bc(f'P3: P_{K}-GU_{K}', H[(f'P_{K}', 'P3')], H[(f'GU_{K}', 'P3')])
            bc(f'P3: G_{K}-GU_{K}', H[(f'G_{K}', 'P3')], H[(f'GU_{K}', 'P3')])

    R = pd.DataFrame(out, columns=['contrast', 'delta_pts', 'ci95_pts'])
    R['n'] = len(quids)
    R['label'] = 'POST_HOC_DESCRIPTIVE (not registered; LLM-judge-accepted rewrites)'
    R.to_csv(a.out, index=False)
    print(R.to_string(index=False))


if __name__ == '__main__':
    main()
