"""rescore_after_review.py -- recompute the six registered robustness contrasts and the per-arm P3 drops
on the subset of accepted paraphrases that human reviewers confirm.  Output label: POST_HOC_HUMAN_FILTER.

The registered analysis (multitq_robustness_rule.json v1.0, scored once by rob_score.py) is NOT changed by this
script; it prints a descriptive post-hoc sensitivity.  Statistics follow rob_score.py exactly:
  hits      official multitq_eval.hit_at_k (Hits@1); S and P = mean over the 3 seed files, G and GU = seed 0
  drop      Hits@1 on the original wording of the kept P3 questions minus Hits@1 on their paraphrase
  contrast  Delta_<arm> = (arm - S_k) on P3 minus (arm - S_k) on the original wording = drop(S_k) - drop(arm)
  bootstrap paired Poisson(1), B = 2000, seed 0, two-sided 95% percentile CI, p_two = 2 min(P(d<=0), P(d>=0))
  Holm      over the six G/P contrasts; REPAIRS iff delta >= +0.01 and Holm p < 0.05 (AMPLIFIES symmetric)
  drop label BRITTLE iff drop >= 0.02 and CI low > 0; ROBUST iff CI high < 0.02; else INCONCLUSIVE

Keep rule: a row is kept iff EVERY supplied reviewer sheet marks it Y in both same_answer_set and
same_constraint_time (one sheet = single reviewer; two sheets = both reviewers).  'unsure', N or blank = not kept.

Usage:
  python rescore_after_review.py --sheet reviewer_A.csv [--sheet reviewer_B.csv] --key paraphrase_review_key.csv \
      --gold paraphrase_review_gold.json --pred-dir <extracted multitq_robustness_predictions.tgz> \
      --scorer multitq_eval.py [--freeze multitq_robustness_freeze.json] [--registered multitq_robustness_results.json] \
      [--out rescore_result.json]
Sheets may be .csv or .xlsx (xlsx needs openpyxl).
"""
import argparse, csv, gzip, hashlib, importlib.util, json, os, sys
import numpy as np

KS = [('k01_s0', '1'), ('k05_s0', '5'), ('k100', '100')]
VIDX = {'ORIG': 0, 'P3': 3}
FAMILY = ['G_1', 'P_1', 'G_5', 'P_5', 'G_100', 'P_100']          # registered primary family (rob_score.py order)
DROP_ARMS = ['S_1', 'G_1', 'P_1', 'GU_1', 'S_5', 'G_5', 'P_5', 'S_100', 'G_100', 'P_100', 'GU_100']
LABEL = 'POST_HOC_HUMAN_FILTER'


def arm_files():
    a = {}
    for k, K in KS:
        a[f'S_{K}'] = [f'preds/{k}/variants_S_seed{s}.jsonl' for s in range(3)]
        a[f'G_{K}'] = [f'G/{k}/pred_G_variants_G_{k}.jsonl']
        a[f'P_{K}'] = [f'P/{k}/pred_P_{k}_variants_P_{k}_tauoff_seed{s}.jsonl' for s in range(3)]
    for k, K in (KS[0], KS[2]):
        a[f'GU_{K}'] = [f'GU/{k}/pred_GU_variants_GU_{k}.jsonl']
    return a


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_pred(p):
    out = {}
    op = gzip.open if p.endswith('.gz') else open
    with op(p, 'rt') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rk = r['ranked']
                out[int(r['quid'])] = json.loads(rk.replace("'", '"')) if isinstance(rk, str) else rk
    return out


def boot(a, b, B=2000, seed=0):
    """identical to rob_score.boot (multitq_eval.paired_poisson_bootstrap stream)"""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    deltas = np.empty(B)
    for i in range(B):
        w = rng.poisson(1.0, size=d.shape[0])
        sw = w.sum()
        deltas[i] = (w * d).sum() / sw if sw > 0 else 0.0
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    p_le, p_ge = float((deltas <= 0).mean()), float((deltas >= 0).mean())
    return dict(delta=float(d.mean()), ci=[float(lo), float(hi)], p_two=float(min(1.0, 2 * min(p_le, p_ge))), B=B, n=int(d.shape[0]))


def holm(ps):
    m = len(ps); order = np.argsort(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj.tolist()


def read_sheet(path):
    if path.endswith('.xlsx'):
        import openpyxl
        ws = openpyxl.load_workbook(path, read_only=True).active
        rows = list(ws.iter_rows(values_only=True))
        hdr = [str(h) for h in rows[0]]
        return [dict(zip(hdr, ['' if v is None else str(v) for v in r])) for r in rows[1:] if any(v is not None for v in r)]
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def yes(v):
    return str(v).strip().upper() in ('Y', 'YES')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--sheet', action='append', required=True)
    ap.add_argument('--key', required=True)
    ap.add_argument('--gold', required=True)
    ap.add_argument('--pred-dir', required=True)
    ap.add_argument('--scorer', required=True)
    ap.add_argument('--freeze')
    ap.add_argument('--registered')
    ap.add_argument('--out')
    a = ap.parse_args(argv)

    spec = importlib.util.spec_from_file_location('multitq_eval', a.scorer)
    ME = importlib.util.module_from_spec(spec); spec.loader.exec_module(ME)
    key = {r['review_id']: int(r['quid']) for r in csv.DictReader(open(a.key, encoding='utf-8'))}
    gold = {int(q): v for q, v in json.load(open(a.gold))['questions'].items()}
    assert set(key.values()) == set(gold), 'key and gold cover different questions'

    sheets = [read_sheet(p) for p in a.sheet]
    keep = set(key.values())
    for s in sheets:
        ids = {r['review_id'] for r in s}
        assert ids == set(key), f'sheet does not cover exactly the {len(key)} review ids'
        keep &= {key[r['review_id']] for r in s if yes(r.get('same_answer_set', '')) and yes(r.get('same_constraint_time', ''))}
    quids = sorted(keep)
    dropped = sorted(set(key.values()) - keep)
    assert quids, 'no row kept'

    files = arm_files()
    if a.freeze:
        fz = json.load(open(a.freeze))['files']
        bad = [f for fs in files.values() for f in fs if sha256_file(os.path.join(a.pred_dir, f)) != fz[f]]
        assert not bad, f'prediction files differ from the freeze: {bad}'
    pred = {arm: [load_pred(os.path.join(a.pred_dir, f)) for f in fs] for arm, fs in files.items()}

    def hits(arm, var):
        vi = VIDX[var]
        return np.array([float(np.mean([ME.hit_at_k(gold[q], P_.get(10 * q + vi), 1) for P_ in pred[arm]])) for q in quids])

    H = {(arm, v): hits(arm, v) for arm in pred for v in VIDX}
    out = dict(label=LABEL, note='descriptive post-hoc sensitivity on human-confirmed rewrites; the registered labels are unchanged',
               n_reviewers=len(sheets), n_kept=len(quids), n_dropped=len(dropped), dropped_review_ids=sorted(r for r, q in key.items() if q in set(dropped)),
               drops={}, contrasts={})
    for arm in DROP_ARMS:
        d = boot(H[(arm, 'ORIG')], H[(arm, 'P3')])
        lab = 'BRITTLE' if (d['delta'] >= 0.02 and d['ci'][0] > 0) else ('ROBUST' if d['ci'][1] < 0.02 else 'INCONCLUSIVE')
        out['drops'][arm] = dict(d, orig_hits1=float(H[(arm, 'ORIG')].mean()), p3_hits1=float(H[(arm, 'P3')].mean()), label=lab)
    raw = []
    for arm in FAMILY:
        S = 'S_' + arm.split('_')[1]
        c = boot(H[(arm, 'P3')] - H[(S, 'P3')], H[(arm, 'ORIG')] - H[(S, 'ORIG')])
        out['contrasts'][f'Delta_{arm}'] = dict(c, definition=f'drop({S}) - drop({arm})'); raw.append(c['p_two'])
    for (name, c), pa in zip(out['contrasts'].items(), holm(raw)):
        c['p_holm'] = pa
        c['label'] = 'REPAIRS' if (c['delta'] >= 0.01 and pa < 0.05) else ('AMPLIFIES' if (c['delta'] <= -0.01 and pa < 0.05) else 'NO_DETECTABLE_DIFFERENCE')
    if a.registered:
        R = json.load(open(a.registered))
        out['registered_comparison'] = dict(
            contrasts={k: dict(registered=dict(delta=R['contrasts']['P3'][k]['delta'], ci=R['contrasts']['P3'][k]['ci'],
                                               p_holm=R['contrasts']['P3'][k]['p_holm'], label=R['contrasts']['P3'][k]['label']),
                               identical=(len(quids) == R['n']['P3'] and all(R['contrasts']['P3'][k][f] == v[f] for f in ('delta', 'ci', 'p_two', 'p_holm', 'label'))))
                       for k, v in out['contrasts'].items()},
            drops={k: dict(registered=dict(delta=R['drops'][k]['P3']['drop_hits1']['delta'], label=R['drops'][k]['P3']['label']),
                           identical=(len(quids) == R['n']['P3'] and R['drops'][k]['P3']['drop_hits1']['delta'] == v['delta']
                                      and R['drops'][k]['P3']['drop_hits1']['ci'] == v['ci'] and R['drops'][k]['P3']['label'] == v['label']))
                   for k, v in out['drops'].items()})
    print(f'{LABEL}: n_kept = {len(quids)} (dropped {len(dropped)}), reviewers = {len(sheets)}')
    print('%-14s%24s%20s%8s%8s  label' % ('contrast', 'drop(S)-drop(arm), pts', '95% CI', 'p_two', 'Holm p'))
    for k, c in out['contrasts'].items():
        cis = '[%+.2f, %+.2f]' % (100 * c['ci'][0], 100 * c['ci'][1])
        print('%-14s%+24.2f%20s%8.3f%8.3f  %s' % (k, 100 * c['delta'], cis, c['p_two'], c['p_holm'], c['label']))
    print('%-8s%8s%8s%11s%18s  label' % ('arm', 'orig', 'P3', 'drop, pts', '95% CI'))
    for k, d in out['drops'].items():
        cis = '[%.2f, %.2f]' % (100 * d['ci'][0], 100 * d['ci'][1])
        print('%-8s%8.2f%8.2f%11.2f%18s  %s' % (k, 100 * d['orig_hits1'], 100 * d['p3_hits1'], 100 * d['delta'], cis, d['label']))
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)
    return out


if __name__ == '__main__':
    main()
