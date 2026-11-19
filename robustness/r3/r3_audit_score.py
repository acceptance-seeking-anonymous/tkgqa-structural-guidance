"""Score the single-annotator R3 audit (targeted + random 100) and run the registered BOUND analysis.

Inputs are the R3 perturbed set, the returned audit sheet, the analyst-only selection key and the per-question R3 hits of
the three robustness methods (Parser-first, question normalizer, Parser-grounded Reader). The analysis uses
r3_common.method_tests / bound_analysis / wilson unchanged:

  accepted row   same_answer_set == Y and same_constraint_time == Y (N, unsure or blank = rejected)
  random100      acceptance and rejection rates with Wilson 95% CIs
  targeted       acceptance with Wilson 95% CI
  pre_audit      every method test on all judge-accepted R3 rewrites (reproduces the method tracks' registered pass)
  bound          every method test on the judge-accepted rewrites minus the human-rejected rows; Holm within method;
                 BOUND_HOLDS iff the Holm-adjusted one-sided p < 0.05. Asserts that no unreviewed row has a positive
                 contrast, so unreviewed rows cannot create positive evidence.

Usage:
  python r3_audit_score.py --perturbed-set multitq_robustness_R3_perturbed_set.json \
      --sheet paraphrase_audit_sheet_R3_396_<initials>.xlsx --selection-key audit_R3_selection_key.csv \
      --parser-first-rewrite-hits multitq_parser_first_rewrite_hits.csv.gz \
      --normalizer-contrasts multitq_normalizer_r3_contrasts.csv \
      --grounded-reader-hits grounded_reader_per_question_hits.csv.gz \
      --out audit_R3_result.json --table audit_R3_bound_table.csv
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROB = os.path.dirname(_HERE)
for _p in (os.path.join(os.path.dirname(_ROB), 'src'), os.path.join(os.path.dirname(_ROB), 'src', 'struct'), _ROB,
           os.path.join(_ROB, 'r2', 'local'), os.path.join(_ROB, 'r1', 'local'), _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)   # release layout: robustness/r3 next to robustness/r1/local and robustness/r2/local
import r3_common as R3C  # noqa: E402

PF_METHOD, PF_BASE, PF_LEVEL = 'V3(-0.003)', 'V0', 5   # dev-selected Parser-first variant; only k = 5 differs from P_k


def yes(v):
    return str(v).strip().upper() in ('Y', 'YES')


def read_sheet(path):
    if path.endswith('.xlsx'):
        return pd.read_excel(path, sheet_name=0, dtype=str).fillna('')
    return pd.read_csv(path, dtype=str).fillna('')


def primary_quids(perturbed_set):
    recs = perturbed_set['records']
    return sorted(int(q) for q, rec in recs.items() if any(a.get('accepted') for a in rec['attempts']))


def build_levels(prim, pf_rw, nz, gr):
    out = {}
    r3 = pf_rw[pf_rw.probe == 'R3']

    def pf(cand):
        d = r3[(r3.level == PF_LEVEL) & (r3.cand == cand)].set_index('quid').loc[prim]
        return d.orig_h1.values.astype(float), d.rw_h1.values.astype(float)
    om, rm = pf(PF_METHOD)
    ob, rb = pf(PF_BASE)
    out['Parser-first'] = {str(PF_LEVEL): dict(o_m=om, r_m=rm, o_b=ob, r_b=rb)}
    out['Normalizer'] = {}
    for arm, g in nz.groupby('arm'):
        g = g.set_index('quid').loc[prim]
        out['Normalizer'][arm] = dict(o_m=g.method_orig.values.astype(float), r_m=g.method_rw.values.astype(float),
                                      o_b=g.base_orig.values.astype(float), r_b=g.base_rw.values.astype(float))
    out['Grounded Reader'] = {}
    for k in ('5', '100'):
        def get(s, arm):
            return gr[(gr['set'] == s) & (gr.arm == arm)].set_index('quid').loc[prim, 'hit1'].values.astype(float)
        out['Grounded Reader'][k] = dict(o_m=get('r3_orig', f'GR_{k}'), r_m=get('r3_rw', f'GR_{k}'),
                                         o_b=get('r3_orig', f'G_{k}'), r_b=get('r3_rw', f'G_{k}'))
    return out


def to_json(o):
    if isinstance(o, dict):
        return {str(k): to_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_json(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_json(o.tolist())
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def main(argv=None):
    ap = argparse.ArgumentParser()
    for a in ('--perturbed-set', '--sheet', '--selection-key', '--parser-first-rewrite-hits', '--normalizer-contrasts',
              '--grounded-reader-hits', '--out'):
        ap.add_argument(a, required=True)
    ap.add_argument('--table')
    a = ap.parse_args(argv)

    prim = primary_quids(json.load(open(a.perturbed_set)))
    sheet = read_sheet(a.sheet).set_index('audit_id')
    key = pd.read_csv(a.selection_key)
    assert set(key.audit_id) == set(sheet.index), 'sheet must cover exactly the audit ids of the selection key'
    assert key.quid.isin(prim).all()
    key['acc'] = [yes(sheet.loc[i, 'same_answer_set']) and yes(sheet.loc[i, 'same_constraint_time']) for i in key.audit_id]
    rejected = set(key.loc[~key.acc, 'quid'].astype(int))
    reviewed = set(key.quid.astype(int))
    rnd, tgt = key[key.in_random100], key[key.in_targeted]
    audit = dict(n_primary=len(prim), n_reviewed=len(reviewed), n_rejected=len(rejected),
                 random100=dict(accepted=int(rnd.acc.sum()), n=len(rnd),
                                acceptance_wilson95=R3C.wilson(int(rnd.acc.sum()), len(rnd)),
                                reject_rate_wilson95=R3C.wilson(int((~rnd.acc).sum()), len(rnd))),
                 targeted=dict(accepted=int(tgt.acc.sum()), n=len(tgt),
                               acceptance_wilson95=R3C.wilson(int(tgt.acc.sum()), len(tgt))))

    levels = build_levels(prim, pd.read_csv(a.parser_first_rewrite_hits), pd.read_csv(a.normalizer_contrasts),
                          pd.read_csv(a.grounded_reader_hits))
    pre = {m: R3C.method_tests(lv) for m, lv in levels.items()}
    bound = {m: R3C.bound_analysis({m: lv}, prim, rejected, reviewed)[m] for m, lv in levels.items()}

    rows = []
    for m in levels:
        for lev in levels[m]:
            p, b = pre[m][lev], bound[m][lev]
            for cn in ('c_drop', 'c_rw'):
                rows.append(dict(method=m, arm_level=lev, contrast=cn, n_pre=p['n'], pre_delta=p[cn]['delta'],
                                 pre_holm_p=p[cn]['p_holm_one_sided'], pre_label=p[cn]['label'], n_post=b['n'],
                                 post_delta=b[cn]['delta'], post_ci95_lo=b[cn]['ci95'][0], post_ci95_hi=b[cn]['ci95'][1],
                                 post_holm_p=b[cn]['p_holm_one_sided'], bound_label=b[cn]['bound_label'],
                                 post_method_orig=b['method']['orig'], post_base_orig=b['baseline']['orig'],
                                 post_method_rw=b['method']['rw'], post_base_rw=b['baseline']['rw']))
    json.dump(to_json(dict(audit=audit, rejected_quids=sorted(rejected), pre_audit=pre, bound=bound)),
              open(a.out, 'w'), indent=1)
    if a.table:
        pd.DataFrame(rows).to_csv(a.table, index=False)
    print(json.dumps(to_json(audit)))
    return rows


if __name__ == '__main__':
    main()
