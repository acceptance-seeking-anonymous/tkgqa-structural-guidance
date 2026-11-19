"""a1_rewrite.py -- A1 of protocol/multitq_output_analyses_plan.json: absolute rewrite accuracy on R1 / R2.
Descriptive; frozen predictions only. Run from a directory that holds R1/ and R2/ (the frozen probe prediction trees:
preds/, G/, P/, GU/, checks/freeze_sha256.txt). score_R2 is imported from robustness/r2/kit/.

  python a1_rewrite.py --scorer src/multitq_eval.py --r1-gold <R1 gold json> --r1-key <R1 review key csv> \
      --r1-sheet <R1 annotated sheet xlsx> --r1-sel <R1 selection key csv> --r1-existing <audit_score_R1 result json> \
      --r2-gold <R2 gold json> --r2-sheet <R2 annotated sheet xlsx> --r2-sel <R2 selection key csv> \
      --r2-freeze <R2 freeze json> --r2-existing <audit_score_R2 result json> --outdir out
Writes A1_rewrite_accuracy.csv, A1_rewrite_contrasts.csv, A1_rewrite_per_question.csv and A1_checks.json.
"""
import argparse, csv, hashlib, importlib.util, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, '..', '..', 'robustness', 'r2', 'kit'))
import score_R2 as SR          # frozen R2 scorer helpers (load_pred, read_sheet, yes, sha256_file)

KS = [('k01_s0', '1'), ('k05_s0', '5'), ('k100', '100')]
VIDX = {'ORIG': 0, 'RW': 3}
B, SEED = 2000, 0


def arm_files(probe):
    a = {}
    for k, K in KS:
        a[f'RULE_{K}'] = [f'preds/{k}/variants_RULE.jsonl']
        a[f'S_{K}'] = [f'preds/{k}/variants_S_seed{s}.jsonl' for s in range(3)]
        a[f'G_{K}'] = [f'G/{k}/pred_G_variants_G_{k}.jsonl']
        a[f'P_{K}'] = [f'P/{k}/pred_P_{k}_variants_P_{k}_tauoff_seed{s}.jsonl' for s in range(3)]
    for k, K in (KS[0], KS[2]):
        a[f'GU_{K}'] = [f'GU/{k}/pred_GU_variants_GU_{k}.jsonl']
    return a


def freeze_check(pred_dir, files):
    fz = {}
    for line in open(os.path.join(pred_dir, 'checks', 'freeze_sha256.txt')):
        h, p = line.split(None, 1)
        fz[p.strip()] = h
    bad = [f for fs in files.values() for f in fs if SR.sha256_file(os.path.join(pred_dir, f)) != fz.get(f)]
    assert not bad, f'{pred_dir}: files differ from checks/freeze_sha256.txt: {bad}'
    return len([f for fs in files.values() for f in fs])


def weights(n):
    rng = np.random.default_rng(SEED)
    W = np.empty((B, n), dtype=np.int64)
    for i in range(B):
        W[i] = rng.poisson(1.0, size=n)
    return W


def wmean(W, x):
    # row-wise (w * x).sum() / w.sum(): the same arithmetic (and summation order) as audit_score_R1.boot / score_R2.poisson_boot
    out = np.empty(W.shape[0])
    for i in range(W.shape[0]):
        w = W[i]
        sw = w.sum()
        out[i] = (w * x).sum() / sw if sw > 0 else 0.0
    return out


def ci(W, x):
    lo, hi = np.quantile(wmean(W, x), [0.025, 0.975])
    return float(lo), float(hi)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--scorer', required=True)
    ap.add_argument('--r1-gold', required=True); ap.add_argument('--r1-key', required=True)
    ap.add_argument('--r1-sheet', required=True); ap.add_argument('--r1-sel', required=True)
    ap.add_argument('--r1-existing', required=True)
    ap.add_argument('--r2-gold', required=True); ap.add_argument('--r2-sheet', required=True)
    ap.add_argument('--r2-sel', required=True); ap.add_argument('--r2-freeze', required=True)
    ap.add_argument('--r2-existing', required=True)
    ap.add_argument('--outdir', required=True)
    a = ap.parse_args(argv)
    spec = importlib.util.spec_from_file_location('multitq_eval', a.scorer)
    ME = importlib.util.module_from_spec(spec); spec.loader.exec_module(ME)
    os.makedirs(a.outdir, exist_ok=True)

    probes = {}
    for probe, gold_p, sheet_p, sel_p in (('R1', a.r1_gold, a.r1_sheet, a.r1_sel), ('R2', a.r2_gold, a.r2_sheet, a.r2_sel)):
        gold = {int(q): v for q, v in json.load(open(gold_p))['questions'].items()}
        sel = {r['audit_id']: r for r in csv.DictReader(open(sel_p, encoding='utf-8'))}
        rows = SR.read_sheet(sheet_p)
        assert {r['audit_id'] for r in rows} == set(sel)
        acc = {int(sel[r['audit_id']]['quid']): SR.yes(r.get('same_answer_set', '')) and SR.yes(r.get('same_constraint_time', '')) for r in rows}
        rnd = {int(r['quid']) for r in sel.values() if r['in_random100'] == 'True'}
        tgt = {int(r['quid']) for r in sel.values() if r['in_targeted'] == 'True'}
        files = arm_files(probe)
        nfiles = freeze_check(probe, files)
        if probe == 'R2':      # also against the registered R2 freeze json (all arms incl. RULE are listed there)
            fz = json.load(open(a.r2_freeze))['files']
            bad = [f for fs in files.values() for f in fs if SR.sha256_file(os.path.join(probe, f)) != fz[f]]
            assert not bad, bad
        pred = {arm: [SR.load_pred(os.path.join(probe, f)) for f in fs] for arm, fs in files.items()}
        quids = sorted(gold)
        H = {}
        for arm in pred:
            for var, vi in VIDX.items():
                H[(arm, var)] = np.array([float(np.mean([ME.hit_at_k(gold[q], P_.get(10 * q + vi), 1) for P_ in pred[arm]])) for q in quids])
            # every question has a prediction for both variants in every file
            for P_ in pred[arm]:
                assert all((10 * q) in P_ and (10 * q + 3) in P_ for q in quids), (probe, arm)
        probes[probe] = dict(gold=gold, acc=acc, rnd=rnd, tgt=tgt, quids=quids, H=H, arms=list(pred), nfiles=nfiles)

    arm_order = [f'{x}_{K}' for _, K in KS for x in ('RULE', 'S', 'G', 'P', 'GU')]
    rows_out, contr_out, checks = [], [], {}
    pq_rows = []
    for probe, D in probes.items():
        quids = np.array(D['quids'])
        reviewed = np.array([q in D['acc'] for q in quids])
        rejected = np.array([q in D['acc'] and not D['acc'][q] for q in quids])
        inrnd = np.array([q in D['rnd'] for q in quids])
        pops = {'ALL': np.ones(len(quids), bool), 'HUMAN_FILTERED': ~rejected, 'RANDOM100': inrnd,
                'RANDOM100_ACCEPTED': inrnd & reviewed & ~rejected}
        for i, q in enumerate(quids):
            r = dict(probe=probe, quid=int(q), qtype=D['gold'][q].get('qtype'), answer_type=D['gold'][q].get('answer_type'),
                     reviewed=int(reviewed[i]), human_rejected=int(rejected[i]), in_random100=int(inrnd[i]), in_targeted=int(q in D['tgt']))
            for arm in arm_order:
                if arm in D['arms']:
                    r[f'{arm}_orig@1'] = round(float(D['H'][(arm, 'ORIG')][i]), 6)
                    r[f'{arm}_rw@1'] = round(float(D['H'][(arm, 'RW')][i]), 6)
            pq_rows.append(r)
        for pop, m in pops.items():
            W = weights(int(m.sum()))
            for arm in arm_order:
                if arm not in D['arms']:
                    continue
                o, w = D['H'][(arm, 'ORIG')][m], D['H'][(arm, 'RW')][m]
                d = o - w
                rows_out.append(dict(probe=probe, population=pop, arm=arm.split('_')[0], k=arm.split('_')[1], n=int(m.sum()),
                                     orig_hits1=float(o.mean()), orig_lo=ci(W, o)[0], orig_hi=ci(W, o)[1],
                                     rw_hits1=float(w.mean()), rw_lo=ci(W, w)[0], rw_hi=ci(W, w)[1],
                                     drop=float(d.mean()), drop_lo=ci(W, d)[0], drop_hi=ci(W, d)[1]))
            pairs = []
            for _, K in KS:
                pairs += [(f'P_{K}', f'G_{K}'), (f'P_{K}', f'S_{K}'), (f'G_{K}', f'S_{K}'), (f'P_{K}', f'RULE_{K}')]
                if f'GU_{K}' in D['arms']:
                    pairs += [(f'P_{K}', f'GU_{K}'), (f'G_{K}', f'GU_{K}')]
            for x, y in pairs:
                for qty in ('rw', 'drop', 'orig'):
                    if qty == 'rw':
                        dx = D['H'][(x, 'RW')][m] - D['H'][(y, 'RW')][m]
                    elif qty == 'orig':
                        dx = D['H'][(x, 'ORIG')][m] - D['H'][(y, 'ORIG')][m]
                    else:
                        dx = (D['H'][(x, 'ORIG')][m] - D['H'][(x, 'RW')][m]) - (D['H'][(y, 'ORIG')][m] - D['H'][(y, 'RW')][m])
                    lo, hi = ci(W, dx)
                    contr_out.append(dict(probe=probe, population=pop, quantity=qty, contrast=f'{x} - {y}', n=int(m.sum()),
                                          delta=float(dx.mean()), lo=lo, hi=hi))
        # reproduction check against the human-audit results of the probes (drops_after_removal, B = 2000, seed 0)
        ex = json.load(open(a.r1_existing if probe == 'R1' else a.r2_existing))
        chk = {}
        for arm, v in ex['drops_after_removal'].items():
            mine = [r for r in rows_out if r['probe'] == probe and r['population'] == 'HUMAN_FILTERED' and f"{r['arm']}_{r['k']}" == arm][0]
            dv, civ = (v['delta'], v['ci']) if probe == 'R1' else (v['drop'], v['ci95'])
            same = (mine['drop'] == dv and mine['drop_lo'] == civ[0] and mine['drop_hi'] == civ[1] and mine['n'] == v['n'])
            chk[arm] = dict(identical=bool(same), mine=[mine['drop'], mine['drop_lo'], mine['drop_hi'], mine['n']], existing=[dv, civ[0], civ[1], v['n']])
        checks[probe] = dict(drops_after_removal_identical=all(c['identical'] for c in chk.values()), per_arm=chk,
                             n_all=int(len(quids)), n_reviewed=int(reviewed.sum()), n_rejected=int(rejected.sum()),
                             n_random100=int(inrnd.sum()), n_random100_accepted=int(pops['RANDOM100_ACCEPTED'].sum()),
                             prediction_files_freeze_checked=D['nfiles'])
    return rows_out, contr_out, checks, pq_rows


def write(outdir, rows_out, contr_out, checks, pq_rows):
    import pandas as pd
    pd.DataFrame(rows_out).to_csv(os.path.join(outdir, 'A1_rewrite_accuracy.csv'), index=False, float_format='%.6f')
    pd.DataFrame(contr_out).to_csv(os.path.join(outdir, 'A1_rewrite_contrasts.csv'), index=False, float_format='%.6f')
    pd.DataFrame(pq_rows).to_csv(os.path.join(outdir, 'A1_rewrite_per_question.csv'), index=False)   # values already rounded to 6 digits
    json.dump(checks, open(os.path.join(outdir, 'A1_checks.json'), 'w'), indent=1)


if __name__ == '__main__':
    rows_out, contr_out, checks, pq_rows = main()
    write(next(sys.argv[i + 1] for i, x in enumerate(sys.argv) if x == '--outdir'), rows_out, contr_out, checks, pq_rows)
    print(json.dumps({p: {k: v for k, v in c.items() if k != 'per_arm'} for p, c in checks.items()}))
