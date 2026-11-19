#!/usr/bin/env python3
"""tipping_point.py -- pre-review tipping point of the R1 reader-vs-parser drop contrast c = drop(G_k) - drop(P_k):
the smallest number m of rewrites, removed in order of decreasing c (ties by quid), after which the one-sided paired
Poisson(1) bootstrap p (B = 2000, seed 0) of drop(G_k) - drop(P_k) reaches alpha; reported for alpha = 0.05 and the
Bonferroni-3 level 0.05/3, at k = 1, 5, 100, with the counts of questions favouring P and G.

  python tipping_point.py --gold paraphrase_review_gold.json --pred-dir <extracted R1 probe predictions> \
         [--scorer src/multitq_eval.py] --out tipping_point_R1_pre_review.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _r1_common as C  # noqa: E402

VIDX = {'ORIG': 0, 'P3': 3}


def p_one(a_, b_, B=2000, seed=0):
    d = np.asarray(a_, float) - np.asarray(b_, float)
    rng = np.random.default_rng(seed)
    ds = np.empty(B)
    for i in range(B):
        w = rng.poisson(1.0, size=d.shape[0])
        sw = w.sum()
        ds[i] = (w * d).sum() / sw if sw > 0 else 0.0
    return float((ds <= 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gold', required=True)
    ap.add_argument('--pred-dir', required=True)
    ap.add_argument('--scorer', default=None)
    ap.add_argument('--out', default='tipping_point_R1_pre_review.csv')
    a = ap.parse_args()
    ME = C.scorer(a.scorer)
    RS = C.rescorer()
    gold = C.gold(a.gold)
    files = RS.arm_files()
    pred = {arm: [RS.load_pred(os.path.join(a.pred_dir, f)) for f in fs] for arm, fs in files.items()}
    quids = sorted(gold)

    def hits(arm, vi):
        return np.array([float(np.mean([ME.hit_at_k(gold[q], P_.get(10 * q + vi), 1) for P_ in pred[arm]])) for q in quids])

    H = {(arm, v): hits(arm, vi) for arm in pred for v, vi in VIDX.items()}

    def tip(K, alpha):
        dG = H[(f'G_{K}', 'ORIG')] - H[(f'G_{K}', 'P3')]
        dP = H[(f'P_{K}', 'ORIG')] - H[(f'P_{K}', 'P3')]
        c = dG - dP
        order = np.lexsort((np.array(quids), -c))

        def fails(m):
            return p_one(dG[np.sort(order[m:])], dP[np.sort(order[m:])]) >= alpha
        if fails(0):   # precondition: the unflipped R1 contrast must clear the decision rule
            raise SystemExit(f'K={K}: the observed G-vs-P drop contrast does not reach p < {alpha} before any flip; '
                             'the tipping point is undefined for these predictions')
        lo, hi = 0, len(c)
        while lo < hi:
            mid = (lo + hi) // 2
            if fails(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo, int((c > 0).sum()), int((c < 0).sum())

    TP = []
    for K in ('1', '5', '100'):
        m05, npos, nneg = tip(K, 0.05)
        m017, _, _ = tip(K, 0.05 / 3)
        TP.append(dict(k=K, n=len(quids), q_favour_P=npos, q_favour_G=nneg, m_holm=m05, m_over_n_holm=round(m05 / len(quids), 4),
                       m_bonf3=m017, m_over_n_bonf3=round(m017 / len(quids), 4)))
    TP = pd.DataFrame(TP)
    print(TP.to_string(index=False))
    TP.to_csv(a.out, index=False)


if __name__ == '__main__':
    main()
