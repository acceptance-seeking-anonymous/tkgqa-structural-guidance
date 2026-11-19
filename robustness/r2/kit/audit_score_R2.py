"""audit_score_R2.py -- score the R2 targeted + random human audit (rule R2 + amendment A2; logic of audit_score_R1.py,
with the R2 registered bootstrap B = 10,000, seed 20261029). Hits come from score_R2.Data (same files, freeze check).

Accepted row: same_answer_set == Y and same_constraint_time == Y (N / unsure / blank = rejected).
random100     acceptance of the random 100 with Wilson 95% CI; rejection rate with Wilson 95% CI.
targeted      acceptance of the rows with c_k > 0 at any k, Wilson 95% CI.
BOUND         H1-H3 (drop(G_k) - drop(P_k) > 0) on all judge-accepted rewrites minus the human-rejected rows; one-sided,
              Holm over k; BOUND_HOLDS iff Holm-adjusted p < 0.05, else BOUND_FAILS. Every unreviewed row has c_k <= 0 at
              every k (asserted), so leaving unreviewed rows in cannot create a positive contrast.
descriptive   contrasts on the audited-and-accepted rows only; per-arm drops after removal.

Usage:
  python audit_score_R2.py --sheet paraphrase_audit_sheet_R2_<N>_<initials>.xlsx \\
      --selection key_do_not_open_before_review/audit_R2_selection_key.csv \\
      --gold key_do_not_open_before_review/multitq_robustness_R2_gold.json --pred-dir <extracted R2 predictions> \\
      --scorer multitq_eval.py --freeze multitq_robustness_R2_freeze.json --out audit_R2_result.json
"""
import argparse, csv, json, os, sys
import importlib.util
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_R2 as SR


def main(argv=None):
    ap = argparse.ArgumentParser()
    for x in ('--sheet', '--selection', '--gold', '--pred-dir', '--scorer'):
        ap.add_argument(x, required=True)
    ap.add_argument('--freeze'); ap.add_argument('--no-freeze-ok', action='store_true', help='tests only')
    ap.add_argument('--B', type=int, default=SR.B_REG); ap.add_argument('--seed', type=int, default=SR.SEED_REG)
    ap.add_argument('--out')
    a = ap.parse_args(argv)
    assert a.freeze or a.no_freeze_ok, 'score only against the sha256 freeze (--freeze)'
    spec = importlib.util.spec_from_file_location('multitq_eval', a.scorer)
    ME = importlib.util.module_from_spec(spec); spec.loader.exec_module(ME)
    gold = {int(q): v for q, v in json.load(open(a.gold))['questions'].items()}
    sel = {r['audit_id']: r for r in csv.DictReader(open(a.selection, encoding='utf-8'))}
    rows = SR.read_sheet(a.sheet)
    assert {r['audit_id'] for r in rows} == set(sel), f'sheet must cover exactly the {len(sel)} audit ids'
    acc = {int(sel[r['audit_id']]['quid']): SR.yes(r.get('same_answer_set', '')) and SR.yes(r.get('same_constraint_time', '')) for r in rows}
    assert set(acc) <= set(gold)
    quids = sorted(gold)
    D = SR.Data(a.pred_dir, gold, ME, a.freeze, with_gu=True)
    reviewed = np.array([q in acc for q in quids]); rejected = np.array([q in acc and not acc[q] for q in quids])
    rnd = {int(r['quid']) for r in sel.values() if r['in_random100'] == 'True'}
    tgt = {int(r['quid']) for r in sel.values() if r['in_targeted'] == 'True'}
    xr, xt = sum(acc[q] for q in rnd), sum(acc[q] for q in tgt)
    out = dict(protocol='multitq_robustness_rule_R2 + amendment A2 (targeted + random audit)', B=a.B, seed=a.seed,
               n_rewrites=len(quids), n_reviewed=int(reviewed.sum()), n_rejected=int(rejected.sum()),
               rejected_quids=sorted(q for q in acc if not acc[q]),
               random100=dict(accepted=xr, n=len(rnd), acceptance_wilson95=SR.wilson(xr, len(rnd)), reject_rate_wilson95=SR.wilson(len(rnd) - xr, len(rnd))),
               targeted=dict(accepted=xt, n=len(tgt), acceptance_wilson95=SR.wilson(xt, len(tgt))),
               bound={}, audited_accepted_only={}, drops_after_removal={})
    ps = []
    for _, K in SR.KS:
        c = D.drop(f'G_{K}', quids) - D.drop(f'P_{K}', quids)
        flagged = np.array([q in tgt for q in quids])
        assert (c[~reviewed] <= 1e-12).all(), f'unreviewed row with positive contrast at k={K}'
        assert ((c > 1e-12) <= flagged).all(), f'a positive-contrast row is not in the targeted set at k={K}'
        keep = ~rejected
        r = SR.summarise(c[keep], a.B, a.seed); r['n'] = int(keep.sum()); out['bound'][K] = r; ps.append(r['p_one_sided'])
        m = reviewed & ~rejected
        out['audited_accepted_only'][K] = dict(SR.summarise(c[m], a.B, a.seed), n=int(m.sum()), label='descriptive')
    for (_, K), pa in zip(SR.KS, SR.holm(ps)):
        out['bound'][K]['p_holm_one_sided'] = pa
        out['bound'][K]['label'] = 'BOUND_HOLDS' if pa < SR.ALPHA else 'BOUND_FAILS'
    keep = ~rejected
    kq = [q for q, k in zip(quids, keep) if k]
    for arm in SR.DROP_ARMS:
        if arm in D.pred:
            s = SR.summarise(D.drop(arm, kq), 2000, 0)
            out['drops_after_removal'][arm] = dict(drop=s['delta'], ci95=s['ci95'], n=len(kq))
    s = json.dumps(out, indent=1)
    if a.out:
        open(a.out, 'w').write(s)
    print(json.dumps(dict(random100=out['random100'], targeted=out['targeted'],
                          bound={K: (round(100 * v['delta'], 2), round(v['p_holm_one_sided'], 4), v['label']) for K, v in out['bound'].items()})))
    return out


if __name__ == '__main__':
    main()
