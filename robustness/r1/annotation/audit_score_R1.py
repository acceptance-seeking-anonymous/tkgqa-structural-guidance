"""Score the R1 targeted + random human audit (protocol v2, multitq_robustness_R1_audit_protocol.json).

Usage:
  python audit_score_R1.py --sheet paraphrase_audit_sheet_R1_269_<initials>.xlsx --selection audit_R1_selection_key.csv \
      --review-key paraphrase_review_key.csv --gold paraphrase_review_gold.json --pred-dir <robustness predictions dir> \
      --scorer multitq_eval.py --out audit_R1_result.json

Accepted row: same_answer_set == Y and same_constraint_time == Y (unsure / blank / N = rejected).
Bound analysis: drop(G_k) - drop(P_k) on all 569 rewrites minus the human-rejected rows; every unreviewed row has
per-question contrast <= 0 at every k (asserted), so leaving unreviewed rows in cannot create a positive contrast.
"""
import argparse, csv, gzip, importlib.util, json, math, os
import numpy as np

KS = [('k01_s0', '1'), ('k05_s0', '5'), ('k100', '100')]
VIDX = {'ORIG': 0, 'P3': 3}


def arm_files():
    a = {}
    for k, K in KS:
        a[f'S_{K}'] = [f'preds/{k}/variants_S_seed{s}.jsonl' for s in range(3)]
        a[f'G_{K}'] = [f'G/{k}/pred_G_variants_G_{k}.jsonl']
        a[f'P_{K}'] = [f'P/{k}/pred_P_{k}_variants_P_{k}_tauoff_seed{s}.jsonl' for s in range(3)]
    for k, K in (KS[0], KS[2]):
        a[f'GU_{K}'] = [f'GU/{k}/pred_GU_variants_GU_{k}.jsonl']
    return a


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


def boot(d, B=2000, seed=0):
    """paired Poisson bootstrap, same stream as rob_score.boot; returns delta, 95% CI, one-sided p(delta <= 0)"""
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    ds = np.empty(B)
    for i in range(B):
        w = rng.poisson(1.0, size=d.shape[0])
        sw = w.sum()
        ds[i] = (w * d).sum() / sw if sw > 0 else 0.0
    lo, hi = np.quantile(ds, [0.025, 0.975])
    return dict(delta=float(d.mean()), ci=[float(lo), float(hi)], p_one=float((ds <= 0).mean()), n=int(d.shape[0]))


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj.tolist()


def wilson(x, n, z=1.959964):
    if n == 0:
        return [float('nan'), float('nan')]
    p = x / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [c - h, c + h]


def read_sheet(path):
    if path.endswith('.xlsx'):
        import openpyxl
        ws = openpyxl.load_workbook(path, read_only=True).active
        rows = list(ws.iter_rows(values_only=True))
        hdr = [str(h) for h in rows[0]]
        return [dict(zip(hdr, ['' if v is None else str(v) for v in r])) for r in rows[1:] if any(v is not None for v in r)]
    return list(csv.DictReader(open(path, encoding='utf-8-sig')))


def yes(v):
    return str(v).strip().upper() in ('Y', 'YES')


def main(argv=None):
    ap = argparse.ArgumentParser()
    for a in ('--sheet', '--selection', '--review-key', '--gold', '--pred-dir', '--scorer'):
        ap.add_argument(a, required=True)
    ap.add_argument('--out')
    a = ap.parse_args(argv)
    spec = importlib.util.spec_from_file_location('multitq_eval', a.scorer)
    ME = importlib.util.module_from_spec(spec); spec.loader.exec_module(ME)
    gold = {int(q): v for q, v in json.load(open(a.gold))['questions'].items()}
    rkey = {r['review_id']: int(r['quid']) for r in csv.DictReader(open(a.review_key, encoding='utf-8'))}
    sel = {r['audit_id']: r for r in csv.DictReader(open(a.selection, encoding='utf-8'))}
    rows = read_sheet(a.sheet)
    assert {r['audit_id'] for r in rows} == set(sel), 'sheet must cover exactly the 269 audit ids'
    acc = {int(sel[r['audit_id']]['quid']): yes(r.get('same_answer_set', '')) and yes(r.get('same_constraint_time', '')) for r in rows}
    assert set(acc) <= set(gold) and set(rkey.values()) == set(gold)
    quids = sorted(gold)
    files = arm_files()
    pred = {arm: [load_pred(os.path.join(a.pred_dir, f)) for f in fs] for arm, fs in files.items()}

    def hits(arm, var):
        vi = VIDX[var]
        return np.array([float(np.mean([ME.hit_at_k(gold[q], P_.get(10 * q + vi), 1) for P_ in pred[arm]])) for q in quids])

    H = {(arm, v): hits(arm, v) for arm in pred for v in VIDX}
    reviewed = np.array([q in acc for q in quids])
    rejected = np.array([q in acc and not acc[q] for q in quids])
    rnd = {int(r['quid']) for r in sel.values() if r['in_random100'] == 'True'}
    tgt = {int(r['quid']) for r in sel.values() if r['in_targeted'] == 'True'}
    out = dict(protocol='multitq_robustness_R1_audit_protocol v2 (targeted + random)', n_rewrites=len(quids), n_reviewed=int(reviewed.sum()),
               n_rejected=int(rejected.sum()),
               random100=dict(accepted=sum(acc[q] for q in rnd), n=len(rnd), reject_rate_wilson95=wilson(sum(not acc[q] for q in rnd), len(rnd))),
               targeted=dict(accepted=sum(acc[q] for q in tgt), n=len(tgt)), bound={}, audited_accepted_only={}, drops_after_removal={})
    ps, keys = [], []
    for _, K in KS:
        dG = H[(f'G_{K}', 'ORIG')] - H[(f'G_{K}', 'P3')]
        dP = H[(f'P_{K}', 'ORIG')] - H[(f'P_{K}', 'P3')]
        c = dG - dP
        assert (c[~reviewed] <= 1e-12).all(), f'unreviewed row with positive contrast at k={K}'
        keep = ~rejected
        r = boot(c[keep]); out['bound'][K] = r; ps.append(r['p_one']); keys.append(K)
        m = reviewed & ~rejected
        out['audited_accepted_only'][K] = dict(boot(c[m]), label='descriptive')
    for K, pa in zip(keys, holm(ps)):
        out['bound'][K]['p_holm'] = pa
        out['bound'][K]['label'] = 'BOUND_HOLDS' if pa < 0.05 else 'BOUND_FAILS'
    keep = ~rejected
    for arm in pred:
        if (arm, 'ORIG') in H:
            out['drops_after_removal'][arm] = boot((H[(arm, 'ORIG')] - H[(arm, 'P3')])[keep])
    s = json.dumps(out, indent=1)
    if a.out:
        open(a.out, 'w').write(s)
    print(s)
    return out


if __name__ == '__main__':
    main()
