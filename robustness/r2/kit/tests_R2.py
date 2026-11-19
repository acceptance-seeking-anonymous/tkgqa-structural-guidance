"""tests_R2.py -- tests of the R2 scoring and audit code (rule R2 + amendments A1, A2). No R2 data are needed for T1-T2.

T1  R1 reproduction: score_R2.py on the R1 frozen predictions (freeze-checked), the R1 kit key and gold, with the R1
    bootstrap (B = 2000, seed 0): the 11 deltas and 95% CIs must equal the post-hoc table (analysis/rewording_contrasts.py)
    robustness_posthoc_parser_vs_reader.csv (0.01 point). The A2 targeted-set rule applied to R1 must give the same rows
    as the R1 audit selection key (in_targeted). The pre-review tipping point on R1 is reported (reference: 26/23/35).
T2  synthetic data with known effects (no MultiTQ data): G loses 30% and P 10% of rewrites at k = 1 and 5, both lose
    the same 20% at k = 100 -> H1, H2 CONFIRMED, H3 NOT_CONFIRMED. Audit kit built on the synthetic set: selection =
    targeted | random 100, fresh order; an all-Y sheet gives BOUND == PRIMARY; rejecting every targeted row at k = 1
    gives BOUND_FAILS at k = 1; a sheet missing a row is refused; Wilson interval values.
T3  real R2 kit structure (gold-free): sheet columns carry no prediction / judge / generator / selection fields; sheet
    ids == selection-key ids; selection == the A2 rule recomputed from the contrasts file; order == the A2 seed; no R2
    quid in the R1 sample; review columns blank; AUDIT_R2_INSTRUCTIONS.md names the single annotator and the row count.
Usage: python tests_R2.py --r1-pred-dir DIR --r1-key K --r1-gold G --r1-freeze F --r1-csv CSV --r1-selection SEL
                          --scorer multitq_eval.py --r1-sample-rule multitq_robustness_rule.json --out tests_R2_result.json
                          [--kit-dir DIR --contrasts R2_contrasts.csv]
"""
import argparse, csv, io, json, os, sys, tempfile, contextlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_R2 as SR
import audit_score_R2 as AS
import build_audit_kit_R2 as BK


def quiet(f, *x):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*x)


def t1(a):
    with tempfile.TemporaryDirectory() as td:
        cp = os.path.join(td, 'c.csv')
        res = quiet(SR.main, ['--key', a.r1_key, '--gold', a.r1_gold, '--pred-dir', a.r1_pred_dir, '--scorer', a.scorer,
                              '--freeze', a.r1_freeze, '--B', '2000', '--seed', '0', '--contrasts-out', cp])
        C = {int(r['quid']): (float(r['c_k1']), float(r['c_k5']), float(r['c_k100'])) for r in csv.DictReader(open(cp))}
    S = res['PRIMARY']
    got = {f'drop(G_{K})-drop(P_{K})': S['hypotheses'][h] for h, K in (('H1', '1'), ('H2', '5'), ('H3', '100'))}
    for K in ('1', '5', '100'):
        got[f'drop(S_{K})-drop(P_{K})'] = S['secondary'][f'drop(S_{K}) - drop(P_{K})']
        got[f'P3: P_{K}-G_{K}'] = S['secondary'][f'RW: P_{K} - G_{K}']
    for K in ('1', '100'):
        got[f'P3: P_{K}-GU_{K}'] = S['secondary'][f'RW: P_{K} - GU_{K}']
    cmp = []
    for r in csv.DictReader(open(a.r1_csv)):
        if r['contrast'] not in got:
            continue
        g = got[r['contrast']]
        lo, hi = [float(x) for x in r['ci95_pts'].strip('[]').split(',')]
        ok = (round(100 * g['delta'], 2) == float(r['delta_pts']) and round(100 * g['ci95'][0], 2) == lo and round(100 * g['ci95'][1], 2) == hi)
        cmp.append(dict(contrast=r['contrast'], csv=[float(r['delta_pts']), lo, hi], script=[round(100 * g['delta'], 2), round(100 * g['ci95'][0], 2), round(100 * g['ci95'][1], 2)], match=ok))
    T, R, _ = BK.select(sorted(C), C)
    lead_T = {int(r['quid']) for r in csv.DictReader(open(a.r1_selection, encoding='utf-8')) if r['in_targeted'] == 'True'}
    return dict(n=S['n'], comparisons=cmp, all_match=all(c['match'] for c in cmp) and len(cmp) == 11,
                targeted_rule_matches_lead_R1=(T == lead_T), n_targeted_R1=len(T), n_targeted_lead=len(lead_T),
                tipping_point_R1_B2000_seed0={K: v['m'] for K, v in res['tipping_point_pre_review_descriptive'].items()})


def write_pred(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for q, rk in rows:
            f.write(json.dumps({'quid': q, 'ranked': rk}) + '\n')


def synth(td, n=300):
    rng = np.random.default_rng(7)
    quids = list(range(5000000, 5000000 + n))
    miss = {('G', '1'): .30, ('G', '5'): .30, ('G', '100'): .20, ('P', '1'): .10, ('P', '5'): .10, ('P', '100'): .20,
            ('S', '1'): .30, ('S', '5'): .30, ('S', '100'): .30, ('GU', '1'): .0, ('GU', '100'): .0}
    gold = {str(q): dict(answers=['Alpha'], answer_type='entity', qtype='equal') for q in quids}
    json.dump(dict(questions=gold), open(os.path.join(td, 'gold.json'), 'w'))
    test = [dict(quid=q, question=f'Who visited Country {q}?', qtype='equal', answers=['Alpha'], answer_type='entity') for q in quids]
    json.dump(test, open(os.path.join(td, 'test.json'), 'w'))
    pset = dict(variants=[dict(quid=q, variant='ORIG', squid=10 * q, question=f'Who visited Country {q}?') for q in quids]
                + [dict(quid=q, variant='RW', squid=10 * q + 3, question=f'Which actor visited Country {q}?') for q in quids])
    json.dump(pset, open(os.path.join(td, 'pset.json'), 'w'))
    lost = {}
    for (arm, K), p in miss.items():
        m = rng.random(n) < p
        if arm == 'P' and K == '100':
            m = lost[('G', '100')]                # identical losses -> drop difference exactly 0
        lost[(arm, K)] = m
    for (arm, K), m in lost.items():
        k = {'1': 'k01_s0', '5': 'k05_s0', '100': 'k100'}[K]
        rows = [(10 * q, ['Alpha', 'Beta']) for q in quids] + [(10 * q + 3, (['Beta', 'Alpha'] if mm else ['Alpha', 'Beta'])) for q, mm in zip(quids, m)]
        if arm in ('S', 'P'):
            for s in range(3):
                write_pred(os.path.join(td, f'preds/{k}/variants_S_seed{s}.jsonl' if arm == 'S' else f'P/{k}/pred_P_{k}_variants_P_{k}_tauoff_seed{s}.jsonl'), rows)
        elif arm == 'G':
            write_pred(os.path.join(td, f'G/{k}/pred_G_variants_G_{k}.jsonl'), rows)
        else:
            write_pred(os.path.join(td, f'GU/{k}/pred_GU_variants_GU_{k}.jsonl'), rows)
    return quids


def write_sheet(path, ids, val):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['audit_id', 'same_answer_set', 'same_constraint_time']); w.writeheader()
        w.writerows(dict(audit_id=i, same_answer_set=val(i), same_constraint_time='Y') for i in ids)


def t2(a):
    with tempfile.TemporaryDirectory() as td:
        quids = synth(td)
        g, cp = os.path.join(td, 'gold.json'), os.path.join(td, 'c.csv')
        base = ['--gold', g, '--pred-dir', td, '--scorer', a.scorer, '--no-freeze-ok', '--B', '2000', '--seed', '0']
        res = quiet(SR.main, base + ['--contrasts-out', cp, '--no-tipping'])
        summ = quiet(BK.main, [os.path.join(td, 'pset.json'), os.path.join(td, 'test.json'), cp, g, os.path.join(td, 'kit')])
        kd = os.path.join(td, 'kit', 'key_do_not_open_before_review')
        sel = list(csv.DictReader(open(os.path.join(kd, 'audit_R2_selection_key.csv'))))
        ids = [r['audit_id'] for r in sel]
        sargs = ['--selection', os.path.join(kd, 'audit_R2_selection_key.csv')] + base
        sy = os.path.join(td, 'Y.csv'); write_sheet(sy, ids, lambda i: 'Y')
        rY = quiet(AS.main, ['--sheet', sy] + sargs)
        k1 = {r['audit_id'] for r in sel if r['favourP_k1'] == 'True'}
        sn = os.path.join(td, 'N.csv'); write_sheet(sn, ids, lambda i: 'N' if i in k1 else 'Y')
        rN = quiet(AS.main, ['--sheet', sn] + sargs)
        sm = os.path.join(td, 'M.csv'); write_sheet(sm, ids[1:], lambda i: 'Y')
        try:
            quiet(AS.main, ['--sheet', sm] + sargs); refused = False
        except AssertionError:
            refused = True
    P = res['PRIMARY']['hypotheses']
    labels = {h: v['label'] for h, v in P.items()}
    allY_eq = all(abs(rY['bound'][K]['delta'] - P[h]['delta']) < 1e-12 and rY['bound'][K]['p_one_sided'] == P[h]['p_one_sided']
                  for h, K in (('H1', '1'), ('H2', '5'), ('H3', '100')))
    w = SR.wilson(3, 100)
    wilson_ok = abs(w[0] - 0.010254523944281516) < 1e-9 and abs(w[1] - 0.08451936489857499) < 1e-9
    ok = (labels == dict(H1='CONFIRMED', H2='CONFIRMED', H3='NOT_CONFIRMED') and allY_eq and rY['random100']['accepted'] == 100
          and rN['bound']['1']['label'] == 'BOUND_FAILS' and rN['bound']['1']['delta'] <= 0 and refused and wilson_ok
          and summ['n_audit'] == summ['n_targeted'] + summ['n_random'] - summ['n_overlap'] and summ['n_random'] == 100)
    return dict(labels=labels, kit=summ, allY_bound_equals_primary=allY_eq, reject_k1_targeted=dict(delta=rN['bound']['1']['delta'], label=rN['bound']['1']['label']),
                missing_row_refused=refused, wilson_3_of_100=w, pass_=ok)


def t3(a):
    kd = a.kit_dir
    sheets = [f for f in os.listdir(kd) if f.startswith('paraphrase_audit_sheet_R2_') and f.endswith('.csv')]
    assert len(sheets) == 1, sheets
    sheet = list(csv.DictReader(open(os.path.join(kd, sheets[0]), encoding='utf-8')))
    key = list(csv.DictReader(open(os.path.join(kd, 'key_do_not_open_before_review', 'audit_R2_selection_key.csv'), encoding='utf-8')))
    C = {int(r['quid']): (float(r['c_k1']), float(r['c_k5']), float(r['c_k100'])) for r in csv.DictReader(open(a.contrasts))}
    T, R, order = BK.select(sorted(C), C)
    cols = list(sheet[0].keys())
    banned = [c for c in cols if any(b in c.lower() for b in ('pred', 'judge', 'model', 'generator', 'hit', 'score', 'arm', 'random', 'target', 'favour', 'contrast'))]
    r1 = set(json.load(open(a.r1_sample_rule))['sample']['quids'])
    txt = open(os.path.join(kd, 'AUDIT_R2_INSTRUCTIONS.md'), encoding='utf-8').read()
    checks = dict(columns=cols, banned_columns=banned, n=len(sheet), ids_match=[r['audit_id'] for r in sheet] == [r['audit_id'] for r in key],
                  ids_sequential=[r['audit_id'] for r in sheet] == [f'A{i + 1:03d}' for i in range(len(sheet))],
                  order_matches_A2=[int(r['quid']) for r in key] == order,
                  flags_match=all((r['in_targeted'] == 'True') == (int(r['quid']) in T) and (r['in_random100'] == 'True') == (int(r['quid']) in R) for r in key),
                  disjoint_from_R1=not ({int(r['quid']) for r in key} & r1), blank_review_columns=all(not r['same_answer_set'] and not r['same_constraint_time'] for r in sheet),
                  instructions=('single annotator' in txt.lower() and f'{len(sheet)} rows' in txt and sheets[0].replace('.csv', '') in txt))
    checks['pass_'] = (not banned) and all(checks[k] for k in ('ids_match', 'ids_sequential', 'order_matches_A2', 'flags_match', 'disjoint_from_R1', 'blank_review_columns', 'instructions'))
    return checks


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for k in ('r1-pred-dir', 'r1-key', 'r1-gold', 'r1-freeze', 'r1-csv', 'r1-selection', 'scorer', 'r1-sample-rule', 'out'):
        ap.add_argument('--' + k, required=True)
    ap.add_argument('--kit-dir'); ap.add_argument('--contrasts')
    a = ap.parse_args()
    out = dict(T1_R1_reproduction=t1(a), T2_synthetic=t2(a))
    if a.kit_dir:
        out['T3_kit_structure'] = t3(a)
    out['all_pass'] = (out['T1_R1_reproduction']['all_match'] and out['T1_R1_reproduction']['targeted_rule_matches_lead_R1']
                       and out['T2_synthetic']['pass_'] and (out['T3_kit_structure']['pass_'] if a.kit_dir else True))
    json.dump(out, open(a.out, 'w'), indent=1)
    print('ALL PASS' if out['all_pass'] else 'FAIL', json.dumps({k: (v.get('all_match', v.get('pass_')) if isinstance(v, dict) else v) for k, v in out.items()}))
