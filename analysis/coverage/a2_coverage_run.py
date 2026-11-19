#!/usr/bin/env python3
"""a2_coverage_run.py -- A2 of protocol/multitq_output_analyses_plan.json: coverage-conditioned Hits@1 on the official
MultiTQ test split (descriptive; frozen, already scored outputs only).

  python a2_coverage_run.py --test-json $DATA_DIR/questions/test.json --hits <k% per-question hits csv.gz> \
      --bundle 1=<bundles/k01_s0/test.jsonl> --bundle 5=<bundles/k05_s0/test.jsonl> --bundle 100=<bundles/k100/test.jsonl> \
      --g-pred 1=<G_1 test predictions> --g-pred 5=<G_5 test predictions> --g-pred 100=<G_100 test predictions> --out out

Per question and level k: F_k = gold string among the displayed provenance facts of the Reader bundle, C_k = gold among
the displayed top-20 candidates (a2_coverage.bundle_coverage; renderer kpct/reader/gk_common.py), Gans_in_cands_k =
the Reader's answer is one of the displayed candidates; outcomes G_k, P_k (mean and per seed), S_k (seed mean), RULE_k,
GU_1, GU_100 from the per-question hits. Tables (a2_tables.cell_rows; paired Poisson(1) bootstrap, B = 2000, seed 0):
F x C cells and margins per k, Parser seeds per cell, and the subset covered at every k.
Writes A2_coverage_per_question.csv and A2_coverage_tables.csv.
"""
import argparse, importlib.util, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a2_coverage as AC
import a2_tables as AT

KS = ['1', '5', '100']


def load_pred(path):
    P = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            P[int(r['quid'])] = r.get('ranked', r.get('answers', []))
    return P


def kv(items):
    return dict(x.split('=', 1) for x in items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-json', required=True); ap.add_argument('--hits', required=True)
    ap.add_argument('--bundle', action='append', required=True, help='K=path (K in 1, 5, 100)')
    ap.add_argument('--g-pred', action='append', required=True, help='K=path of the Reader G_K test predictions')
    ap.add_argument('--scorer', default=os.path.join(HERE, '..', '..', 'src', 'multitq_eval.py'))
    ap.add_argument('--gk-common', default=os.path.join(HERE, '..', '..', 'kpct', 'reader', 'gk_common.py'))
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    bundles, gpaths = kv(a.bundle), kv(a.g_pred)
    assert set(bundles) == set(gpaths) == set(KS), (bundles, gpaths)
    os.makedirs(a.out, exist_ok=True)
    GC = AC.load_module('gk_common_frozen', a.gk_common)
    ME = AC.load_module('multitq_eval', a.scorer)
    gold_t = {int(q['quid']): q for q in json.load(open(a.test_json))}
    H = pd.read_csv(a.hits)
    H.columns = [c.strip() for c in H.columns]
    cov = {K: AC.bundle_coverage(bundles[K], gold_t, ME, GC) for K in KS}
    gpreds = {K: load_pred(gpaths[K]) for K in KS}

    D = H[['quid']].copy()
    D['qtype'] = [gold_t[q]['qtype'] for q in D.quid]
    D['answer_type'] = [gold_t[q]['answer_type'] for q in D.quid]
    for K in KS:
        cv = pd.DataFrame.from_dict(cov[K], orient='index')
        D[f'F_{K}'] = D.quid.map(cv.F).astype(int)
        D[f'C_{K}'] = D.quid.map(cv.C).astype(int)
        D[f'nfacts_{K}'] = D.quid.map(cv.n_facts)
        D[f'ncands_{K}'] = D.quid.map(cv.n_cands)
        D[f'G_{K}'] = H[f'G_{K}@1'].values
        for s in range(3):
            D[f'P_{K}_seed{s}'] = H[f'P_{K}_seed{s}@1'].values
        D[f'P_{K}'] = D[[f'P_{K}_seed{s}' for s in range(3)]].mean(1)
        D[f'S_{K}'] = H[[f'S_{K}_seed{s}@1' for s in range(3)]].mean(1).values
        D[f'RULE_{K}'] = H[f'RULE_{K}@1'].values
    for K in ('1', '100'):
        D[f'GU_{K}'] = H[f'GU_{K}@1'].values
    for K in KS:
        cands = {}
        with open(bundles[K]) as f:
            for line in f:
                b = GC.normalize_bundle(json.loads(line))
                cands[b['quid']] = set(c['candidate'] for c in b['candidates'])
        D[f'Gans_in_cands_{K}'] = [int(gpreds[K][q][0] in cands[q]) for q in D.quid]
    pqcols = ['quid', 'qtype', 'answer_type'] + \
        [f'{x}_{K}' for K in KS for x in ('F', 'C', 'nfacts', 'ncands', 'Gans_in_cands')] + \
        [f'{x}_{K}' for K in KS for x in ('G', 'P', 'S', 'RULE')] + \
        [f'P_{K}_seed{s}' for K in KS for s in range(3)] + ['GU_1', 'GU_100']
    D[pqcols].to_csv(os.path.join(a.out, 'A2_coverage_per_question.csv'), index=False, float_format='%.6f')

    cellspec = [('F1_C1', 1, 1), ('F1_C0', 1, 0), ('F0_C1', 0, 1), ('F0_C0', 0, 0)]
    T2 = []
    for K in KS:
        cols = [f'G_{K}', f'P_{K}', f'S_{K}', f'RULE_{K}'] + ([f'GU_{K}'] if K in ('1', '100') else [])
        cons = [(f'G_{K}', f'P_{K}'), (f'G_{K}', f'S_{K}'), (f'P_{K}', f'S_{K}')] + (
            [(f'G_{K}', f'GU_{K}'), (f'P_{K}', f'GU_{K}')] if K in ('1', '100') else [])
        for lab, fv, cv_ in cellspec:
            m = ((D[f'F_{K}'] == fv) & (D[f'C_{K}'] == cv_)).values
            T2 += AT.cell_rows(D, m, cols, lab, extra=dict(k=K, table='bundle_FxC'), contrasts=cons)
        T2 += AT.cell_rows(D, np.ones(len(D), bool), cols, 'ALL', extra=dict(k=K, table='bundle_FxC'), contrasts=cons)
        for lab, m in [('F1', D[f'F_{K}'].values == 1), ('F0', D[f'F_{K}'].values == 0),
                       ('C1', D[f'C_{K}'].values == 1), ('C0', D[f'C_{K}'].values == 0)]:
            T2 += AT.cell_rows(D, m, cols, lab, extra=dict(k=K, table='bundle_marginal'), contrasts=cons)
    T2s = []
    for K in KS:
        cols = [f'P_{K}_seed{s}' for s in range(3)]
        for lab, fv, cv_ in cellspec:
            m = ((D[f'F_{K}'] == fv) & (D[f'C_{K}'] == cv_)).values
            T2s += AT.cell_rows(D, m, cols, lab, extra=dict(k=K, table='bundle_FxC_parser_seeds'),
                                contrasts=[(f'G_{K}', f'P_{K}_seed{s}') for s in range(3)])
    allk = np.ones(len(D), bool)
    for K in KS:
        allk &= ((D[f'F_{K}'] == 1) & (D[f'C_{K}'] == 1)).values
    colsall = ['G_1', 'G_5', 'G_100', 'P_1', 'P_5', 'P_100', 'S_1', 'S_5', 'S_100', 'RULE_1', 'RULE_5', 'RULE_100', 'GU_1', 'GU_100']
    consall = [('G_1', 'P_1'), ('G_5', 'P_5'), ('G_100', 'P_100'), ('G_1', 'GU_1'), ('G_100', 'GU_100'), ('P_1', 'GU_1'), ('P_100', 'GU_100')]
    T3 = AT.cell_rows(D, allk, colsall, 'bundle_FC_all_k', extra=dict(k='all', table='common_covered_bundle'), contrasts=consall)
    pd.concat([pd.DataFrame(T2), pd.DataFrame(T2s), pd.DataFrame(T3)], ignore_index=True).to_csv(
        os.path.join(a.out, 'A2_coverage_tables.csv'), index=False, float_format='%.6f')
    print(json.dumps({'n': int(len(D)), **{f'F_{K}': float(D[f'F_{K}'].mean()) for K in KS}, **{f'C_{K}': float(D[f'C_{K}'].mean()) for K in KS},
                      'covered_all_k': int(allk.sum())}))


if __name__ == '__main__':
    main()
