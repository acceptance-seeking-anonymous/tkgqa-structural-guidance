"""rescore_R2_after_review.py -- the single scoring pass of the confirmatory robustness probe R2
(multitq_robustness_rule_R2.json v1.0 + amendment A1, REGISTERED AMENDMENT). Run ONCE, after the audit sheet is returned.

PRIMARY set   all judge-accepted R2 rewrites (mechanical checks + LLM judge) = every row of the key (amendment A1).
hits          official multitq_eval.hit_at_k (Hits@1); S and P = mean over their 3 S-seed files; G and GU = seed 0;
              ORIG = synthetic quid 10*quid + 0, rewrite (RW) = 10*quid + 3 (same layout as R1's P3).
drop          Hits@1(ORIG) - Hits@1(RW) per question.
H1-H3         d_i = drop(G_k)_i - drop(P_k)_i, k = 1, 5, 100; paired Poisson(1) bootstrap of mean(d), B = 10,000,
              seed 20261029 (multitq_eval stream); one-sided p = share of bootstrap means <= 0; Holm over the three;
              CONFIRMED iff Holm-adjusted one-sided p < 0.05, else NOT_CONFIRMED. LOW_N flag if n < 200.
audit (A1)    single annotator, rows 1-100 of the sheet (the first 100 review ids in key order = a simple random
              sample): (a) acceptance (Y/Y; 'unsure', N, blank = not accepted) with Wilson 95% CIs;
              (b) tipping point m* per k: remove the PRIMARY questions with the largest d_i first (ties by quid), from
              the k-th test only, until the Holm-adjusted p of H_k is >= 0.05; AUDIT_ROBUST iff m*/n > upper Wilson
              bound of the audit rejection rate, else AUDIT_SENSITIVE (NOT_APPLICABLE if H_k is NOT_CONFIRMED);
              (c) H1-H3 on the audited rows minus rejects (descriptive).
secondary     (descriptive, PRIMARY set) rewrite Hits@1 P_k - G_k; drop(S_k) - drop(P_k); P_k - GU_k at k = 1, 100
              (rewrite gap and drop difference); per-arm drops with R1 labels (B = 2000, seed 0); R1+R2 pooled
              drop(G_k) - drop(P_k) (optional R1 inputs).
The registered R1 results are not changed by this script.

Usage:
  python rescore_R2_after_review.py --sheet paraphrase_review_R2_<initials>.xlsx --key paraphrase_review_R2_key.csv \
     --gold paraphrase_review_R2_gold.json --pred-dir <extracted multitq_robustness_R2_predictions.tgz> \
     --scorer multitq_eval.py --freeze multitq_robustness_R2_freeze.json \
     [--r1-pred-dir <extracted multitq_robustness_predictions.tgz> --r1-key paraphrase_review_key.csv \
      --r1-gold paraphrase_review_gold.json [--r1-sheet R1_reviewer.xlsx ...] --r1-freeze multitq_robustness_freeze.json] \
     [--out rescore_R2_result.json]
Options: --no-sheet (PRIMARY + secondaries only, no audit), --B / --seed (defaults 10000 / 20261029 = registered).
"""
import argparse, csv, gzip, hashlib, importlib.util, json, os
import numpy as np

KS = [('k01_s0', '1'), ('k05_s0', '5'), ('k100', '100')]
VIDX = {'ORIG': 0, 'RW': 3}
DROP_ARMS = ['S_1', 'G_1', 'P_1', 'GU_1', 'S_5', 'G_5', 'P_5', 'S_100', 'G_100', 'P_100', 'GU_100']
B_REG, SEED_REG, ALPHA, LOW_N, N_AUDIT, Z = 10000, 20261029, 0.05, 200, 100, 1.959964


def arm_files(with_gu=True):
    a = {}
    for k, K in KS:
        a[f'S_{K}'] = [f'preds/{k}/variants_S_seed{s}.jsonl' for s in range(3)]
        a[f'G_{K}'] = [f'G/{k}/pred_G_variants_G_{k}.jsonl']
        a[f'P_{K}'] = [f'P/{k}/pred_P_{k}_variants_P_{k}_tauoff_seed{s}.jsonl' for s in range(3)]
    if with_gu:
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


def poisson_boot(d, B, seed):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    for i in range(B):
        w = rng.poisson(1.0, size=d.shape[0])
        sw = w.sum()
        out[i] = (w * d).sum() / sw if sw > 0 else 0.0
    return out


def summarise(d, B, seed):
    D = poisson_boot(d, B, seed)
    lo, hi = np.quantile(D, [0.025, 0.975])
    return dict(delta=float(np.mean(d)), ci95=[float(lo), float(hi)], lower_one_sided_95=float(np.quantile(D, 0.05)),
                p_one_sided=float((D <= 0).mean()), p_two=float(min(1.0, 2 * min((D <= 0).mean(), (D >= 0).mean()))),
                B=int(B), seed=int(seed), n=int(len(d)))


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


def kept_quids(sheets, key):
    keep = set(key.values())
    for s in sheets:
        ids = {r['review_id'] for r in s}
        assert ids == set(key), f'sheet does not cover exactly the {len(key)} review ids'
        keep &= {key[r['review_id']] for r in s if yes(r.get('same_answer_set', '')) and yes(r.get('same_constraint_time', ''))}
    return sorted(keep)


class Data:
    """per-question Hits@1 of every arm on ORIG and RW for one probe (R1 or R2)"""

    def __init__(self, pred_dir, gold, ME, freeze=None, with_gu=True):
        files = arm_files(with_gu)
        if freeze:
            fz = json.load(open(freeze))['files']
            bad = [f for fs in files.values() for f in fs if sha256_file(os.path.join(pred_dir, f)) != fz[f]]
            assert not bad, f'prediction files differ from the freeze: {bad}'
        self.pred = {arm: [load_pred(os.path.join(pred_dir, f)) for f in fs] for arm, fs in files.items()}
        self.gold, self.ME = gold, ME

    def hits(self, arm, var, quids):
        vi = VIDX[var]
        return np.array([float(np.mean([self.ME.hit_at_k(self.gold[q], P_.get(10 * q + vi), 1) for P_ in self.pred[arm]])) for q in quids])

    def drop(self, arm, quids):
        return self.hits(arm, 'ORIG', quids) - self.hits(arm, 'RW', quids)


def analyse(D, quids, B, seed, r1=None):
    """H1-H3 + secondaries on one question set. r1 = (Data, quids) for the pooled estimate (optional)."""
    out = dict(n=len(quids), low_n=len(quids) < LOW_N, hypotheses={}, secondary={})
    dG = {K: D.drop(f'G_{K}', quids) for _, K in KS}
    dP = {K: D.drop(f'P_{K}', quids) for _, K in KS}
    H = {K: summarise(dG[K] - dP[K], B, seed) for _, K in KS}
    for (_, K), pa, h in zip(KS, holm([H[K]['p_one_sided'] for _, K in KS]), ('H1', 'H2', 'H3')):
        H[K].update(p_holm_one_sided=pa, label=('CONFIRMED' if pa < ALPHA else 'NOT_CONFIRMED'),
                    hypothesis=f'{h}: drop(G_{K}) - drop(P_{K}) > 0')
        out['hypotheses'][h] = H[K]
    sec = out['secondary']
    for _, K in KS:
        sec[f'RW: P_{K} - G_{K}'] = summarise(D.hits(f'P_{K}', 'RW', quids) - D.hits(f'G_{K}', 'RW', quids), B, seed)
        sec[f'drop(S_{K}) - drop(P_{K})'] = summarise(D.drop(f'S_{K}', quids) - dP[K], B, seed)
    for K in ('1', '100'):
        if f'GU_{K}' in D.pred:
            sec[f'RW: P_{K} - GU_{K}'] = summarise(D.hits(f'P_{K}', 'RW', quids) - D.hits(f'GU_{K}', 'RW', quids), B, seed)
            sec[f'drop(GU_{K}) - drop(P_{K})'] = summarise(D.drop(f'GU_{K}', quids) - dP[K], B, seed)
    drops = {}
    for arm in DROP_ARMS:
        if arm not in D.pred:
            continue
        o, rw = D.hits(arm, 'ORIG', quids), D.hits(arm, 'RW', quids)
        s = summarise(o - rw, 2000, 0)
        lab = 'BRITTLE' if (s['delta'] >= 0.02 and s['ci95'][0] > 0) else ('ROBUST' if s['ci95'][1] < 0.02 else 'INCONCLUSIVE')
        drops[arm] = dict(orig_hits1=float(o.mean()), rw_hits1=float(rw.mean()), drop=s['delta'], ci95=s['ci95'], label=lab)
    sec['per_arm_drops_R1_labels'] = drops
    if r1 is not None:
        D1, q1 = r1
        for _, K in KS:
            d = np.concatenate([D1.drop(f'G_{K}', q1) - D1.drop(f'P_{K}', q1), dG[K] - dP[K]])
            sec[f'pooled R1+R2: drop(G_{K}) - drop(P_{K})'] = dict(summarise(d, B, seed), n_R1=len(q1), n_R2=len(quids))
    return out


def wilson(x, n, z=Z):
    if n == 0:
        return [0.0, 1.0]
    p = x / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(max(0.0, c - h)), float(min(1.0, c + h))]


def tipping_point(D, quids, B, seed, full_p):
    """amendment A1 (b): per k, smallest m such that removing the m questions with the largest d_i (ties by quid)
    from the k-th test only makes the Holm-adjusted p of H_k >= ALPHA; full_p = {K: one-sided p on the full set}"""
    out = {}
    Ks = [K for _, K in KS]
    for j, K in enumerate(Ks):
        d = D.drop(f'G_{K}', quids) - D.drop(f'P_{K}', quids)
        order = sorted(range(len(quids)), key=lambda i: (-d[i], quids[i]))
        adj0 = holm([full_p[x] for x in Ks])[j]
        if adj0 >= ALPHA:
            out[K] = dict(m=0, n=len(quids), m_over_n=0.0, label='NOT_APPLICABLE', note='H not CONFIRMED on the full PRIMARY set'); continue
        m_star, trace = None, []
        for m in range(1, len(quids)):
            keep = sorted(order[m:])
            p_k = float((poisson_boot(d[keep], B, seed) <= 0).mean())
            ps = [full_p[x] if x != K else p_k for x in Ks]
            adj = holm(ps)[j]
            trace.append((m, p_k, adj))
            if adj >= ALPHA:
                m_star = m; break
        out[K] = dict(m=m_star, n=len(quids), m_over_n=(m_star / len(quids)) if m_star else None,
                      removed_quids=[quids[i] for i in order[:m_star]] if m_star else None, d_of_removed=[float(d[i]) for i in order[:m_star]] if m_star else None,
                      last_steps=[dict(m=t[0], p_one_sided=t[1], p_holm=t[2]) for t in trace[-3:]])
    return out


def audit(D, key, sheets, B, seed, prim_H):
    ids = list(key)                                   # key order = sheet order
    aud_ids = ids[:min(N_AUDIT, len(ids))]
    acc = set(aud_ids)
    for s in sheets:
        rows = {r['review_id']: r for r in s}
        assert set(aud_ids) <= set(rows), 'the audit sheet must contain rows S001-S100'
        acc &= {i for i in aud_ids if yes(rows[i].get('same_answer_set', '')) and yes(rows[i].get('same_constraint_time', ''))}
    n_a, x = len(aud_ids), len(acc)
    rej_ci = wilson(n_a - x, n_a)
    out = dict(n_audited=n_a, n_accepted=x, acceptance_rate=x / n_a, acceptance_wilson95=wilson(x, n_a),
               rejection_rate=(n_a - x) / n_a, rejection_wilson95=rej_ci, n_annotators=len(sheets),
               rejected_review_ids=sorted(set(aud_ids) - acc), rejected_quids=sorted(key[i] for i in set(aud_ids) - acc))
    prim = sorted(key.values())
    tp = tipping_point(D, prim, B, seed, {K: prim_H[h]['p_one_sided'] for h, K in (('H1', '1'), ('H2', '5'), ('H3', '100'))})
    for K, v in tp.items():
        if v.get('label') == 'NOT_APPLICABLE':
            continue
        v['label'] = 'AUDIT_ROBUST' if (v['m_over_n'] is not None and v['m_over_n'] > rej_ci[1]) else 'AUDIT_SENSITIVE'
        v['rule'] = f'AUDIT_ROBUST iff m*/n > upper Wilson 95% bound of the audit rejection rate ({rej_ci[1]:.4f})'
    out['tipping_point'] = tp
    sub = sorted(key[i] for i in acc)
    out['H_on_audited_accepted_descriptive'] = analyse(D, sub, B, seed)['hypotheses'] if sub else None
    out['n_audited_accepted'] = len(sub)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--sheet', action='append', default=[])
    ap.add_argument('--no-sheet', action='store_true', help='PRIMARY + secondaries only (no audit sheet)')
    ap.add_argument('--key', required=True); ap.add_argument('--gold', required=True)
    ap.add_argument('--pred-dir', required=True); ap.add_argument('--scorer', required=True); ap.add_argument('--freeze')
    ap.add_argument('--r1-pred-dir'); ap.add_argument('--r1-key'); ap.add_argument('--r1-gold'); ap.add_argument('--r1-sheet', action='append', default=[])
    ap.add_argument('--r1-freeze')
    ap.add_argument('--B', type=int, default=B_REG); ap.add_argument('--seed', type=int, default=SEED_REG)
    ap.add_argument('--no-gu', action='store_true'); ap.add_argument('--out')
    a = ap.parse_args(argv)
    assert a.sheet or a.no_sheet, 'give --sheet (audit) or --no-sheet (PRIMARY and secondaries only)'
    spec = importlib.util.spec_from_file_location('multitq_eval', a.scorer)
    ME = importlib.util.module_from_spec(spec); spec.loader.exec_module(ME)
    key = {r['review_id']: int(r['quid']) for r in csv.DictReader(open(a.key, encoding='utf-8'))}
    gold = {int(q): v for q, v in json.load(open(a.gold))['questions'].items()}
    assert set(key.values()) == set(gold), 'key and gold cover different questions'
    D = Data(a.pred_dir, gold, ME, a.freeze, with_gu=not a.no_gu)
    r1 = None
    if a.r1_pred_dir:
        k1 = {r['review_id']: int(r['quid']) for r in csv.DictReader(open(a.r1_key, encoding='utf-8'))}
        g1 = {int(q): v for q, v in json.load(open(a.r1_gold))['questions'].items()}
        D1 = Data(a.r1_pred_dir, g1, ME, a.r1_freeze, with_gu=False)
        q1 = kept_quids([read_sheet(p) for p in a.r1_sheet], k1) if a.r1_sheet else sorted(k1.values())
        r1 = (D1, q1)
    all_q = sorted(key.values())
    res = dict(label='REGISTERED_AMENDMENT_R2', rule='multitq_robustness_rule_R2.json v1.0 + amendment A1', B=a.B, seed=a.seed,
               registered_bootstrap=(a.B == B_REG and a.seed == SEED_REG),
               r1_pooled_component=(None if r1 is None else ('R1 human-confirmed rows' if a.r1_sheet else 'R1 judge-accepted rewrites (no R1 sheets supplied)')))
    res['PRIMARY'] = dict(definition='all judge-accepted R2 rewrites', **analyse(D, all_q, a.B, a.seed, r1))
    if a.sheet:
        res['AUDIT'] = audit(D, key, [read_sheet(p) for p in a.sheet], a.B, a.seed, res['PRIMARY']['hypotheses'])
        if len(a.sheet) > 1:
            res['AUDIT']['note'] = 'more than one sheet supplied: a row counts as accepted only if every sheet accepts it (the registered design has one annotator)'
    for part in ('PRIMARY',):
        if part not in res:
            continue
        p = res[part]
        print(f"{part}: n = {p['n']}{' (LOW_N)' if p['low_n'] else ''}; B = {a.B}, seed = {a.seed}")
        print('  %-36s%10s%22s%10s%10s  label' % ('hypothesis', 'pts', '95% CI', 'p_1s', 'Holm p'))
        for h, v in p['hypotheses'].items():
            print('  %-36s%+10.2f%22s%10.4f%10.4f  %s' % (v['hypothesis'], 100 * v['delta'], '[%+.2f, %+.2f]' % (100 * v['ci95'][0], 100 * v['ci95'][1]),
                                                        v['p_one_sided'], v['p_holm_one_sided'], v['label']))
        for name, v in p['secondary'].items():
            if name != 'per_arm_drops_R1_labels':
                print('  %-36s%+10.2f%22s' % (name, 100 * v['delta'], '[%+.2f, %+.2f]' % (100 * v['ci95'][0], 100 * v['ci95'][1])))
    if 'AUDIT' in res:
        A = res['AUDIT']
        print(f"AUDIT: {A['n_accepted']}/{A['n_audited']} accepted ({100 * A['acceptance_rate']:.1f}%, Wilson 95% "
              f"[{100 * A['acceptance_wilson95'][0]:.1f}, {100 * A['acceptance_wilson95'][1]:.1f}]); rejection upper bound {100 * A['rejection_wilson95'][1]:.1f}%")
        for K, v in A['tipping_point'].items():
            print(f"  tipping point k = {K}: m* = {v['m']} of n = {v['n']} ({'' if v['m_over_n'] is None else '%.1f%%' % (100 * v['m_over_n'])}) -> {v['label']}")
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)
    return res


if __name__ == '__main__':
    main()
