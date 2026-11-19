"""r2_identity.py -- gold-free integrity checks of the R2 predictions, run at freeze time (no answers are read).
I2: top-1 agreement of each arm's ORIG re-run with the frozen test predictions on the same questions (R1 thresholds:
    PIPELINE_DRIFT if < 0.98 for G / P / GU, < 0.995 for RULE / S). I3: GU retrieval on ORIG vs the frozen gu_test lists.
Also: invalid / fallback rates per arm and variant, share of questions whose top-1 changes from ORIG to RW.
Usage: python r2_identity.py <pred_dir> <frozen_orig_rows.json> <R2_perturbed_set.json> <out.json>
"""
import collections, gzip, json, os, sys
import numpy as np

KS = ['k01_s0', 'k05_s0', 'k100']; KLAB = {'k01_s0': '1', 'k05_s0': '5', 'k100': '100'}
VAR = {'ORIG': 0, 'RW': 3}


def rows(p):
    op = gzip.open if p.endswith('.gz') else open
    with op(p, 'rt') as f:
        for l in f:
            if l.strip():
                yield json.loads(l)


def load_pred(p):
    out = {}
    for r in rows(p):
        rk = r['ranked']
        out[int(r['quid'])] = json.loads(rk.replace("'", '"')) if isinstance(rk, str) else rk
    return out


def main(pred_dir, frozen_rows, pset_p, out_p):
    pset = json.load(open(pset_p))
    qv = {v: sorted(r['quid'] for r in pset['variants'] if r['variant'] == v) for v in VAR}
    ARMS = {}
    for k in KS:
        K_ = KLAB[k]
        ARMS[f'RULE_{K_}'] = [f'preds/{k}/variants_RULE.jsonl']
        ARMS[f'S_{K_}'] = [f'preds/{k}/variants_S_seed{s}.jsonl' for s in range(3)]
        ARMS[f'G_{K_}'] = [f'G/{k}/pred_G_variants_G_{k}.jsonl']
        ARMS[f'P_{K_}'] = [f'P/{k}/pred_P_{k}_variants_P_{k}_tauoff_seed{s}.jsonl' for s in range(3)]
    for k in ('k01_s0', 'k100'):
        ARMS[f'GU_{KLAB[k]}'] = [f'GU/{k}/pred_GU_variants_GU_{k}.jsonl']
    PRED = {a: [load_pred(os.path.join(pred_dir, f)) for f in fs] for a, fs in ARMS.items()}
    FR = json.load(open(frozen_rows)); fp = FR['preds']
    fz = lambda path: {int(q): v for q, v in fp[path].items()}
    GFZ = {'k01_s0': 'G_k01_s0/pred_G_test_ordS_ep2.jsonl', 'k05_s0': 'G_k05_s0/pred_G_test_main_ep2.jsonl', 'k100': 'G_k100/pred_G_test_main_ep1.jsonl'}
    GUFZ = {'k01_s0': 'GU_k01_s0/pred_GU_test_main_ep3.jsonl', 'k100': 'GU_k100/pred_GU_test_main_ep1.jsonl'}
    PFZ = {'k01_s0': ('k01_s0', 'ep3'), 'k05_s0': ('k05_s0', 'ep1'), 'k100': ('k100', 'ep0.5')}
    FROZEN = {}
    for k in KS:
        K_ = KLAB[k]
        sp = (lambda arm: f'/kpct/preds/{k}/test_{arm}.jsonl') if k != 'k100' else (lambda arm: f'/mtq/phase1_structural/preds/test_{arm}.jsonl')
        FROZEN[f'RULE_{K_}'] = [fz(sp('RULE'))]
        FROZEN[f'S_{K_}'] = [fz(sp(f'S_seed{s}')) for s in range(3)]
        FROZEN[f'G_{K_}'] = [fz(f'/kpct/guided/preds/{GFZ[k]}')]
        sub, ck = PFZ[k]
        FROZEN[f'P_{K_}'] = [fz(f'/kpct/parser/test/{sub}/pred_P_{sub}_test_{ck}_tauoff_seed{s}.jsonl') for s in range(3)]
    for k in ('k01_s0', 'k100'):
        FROZEN[f'GU_{KLAB[k]}'] = [fz(f'/kpct/guided/preds/{GUFZ[k]}')]
    res = dict(n=dict(ORIG=len(qv['ORIG']), RW=len(qv['RW'])), coverage={}, identity={}, rates={}, top1_changed_RW={})
    for arm, Ps in PRED.items():
        res['coverage'][arm] = [all((10 * q + VAR[v]) in P_ for v in VAR for q in qv[v]) for P_ in Ps]
        ag = [float(np.mean([(Ps[s].get(10 * q) or [None])[0] == (FROZEN[arm][s].get(q) or [None])[0] for q in qv['ORIG']])) for s in range(len(Ps))]
        res['identity'][arm] = dict(top1_agreement_orig_rerun_vs_frozen=ag, min=min(ag), flag=('PIPELINE_DRIFT' if min(ag) < (0.98 if arm[0] in 'GP' else 0.995) else 'OK'))
        res['top1_changed_RW'][arm] = float(np.mean([np.mean([(P_.get(10 * q + 3) or [None])[0] != (P_.get(10 * q) or [None])[0] for P_ in Ps]) for q in qv['RW']]))
    RAW = {}
    for k in KS:
        RAW[f'G_{KLAB[k]}'] = {int(r['quid']): r for r in rows(os.path.join(pred_dir, f'G/{k}/raw_G_variants_G_{k}.jsonl.gz'))}
        RAW[f'P_{KLAB[k]}'] = {int(r['quid']): r for r in rows(os.path.join(pred_dir, f'P/{k}/exec_P_{k}_variants_P_{k}.jsonl.gz'))}
    for k in ('k01_s0', 'k100'):
        RAW[f'GU_{KLAB[k]}'] = {int(r['quid']): r for r in rows(os.path.join(pred_dir, f'GU/{k}/raw_GU_variants_GU_{k}.jsonl.gz'))}
    for arm, R_ in RAW.items():
        res['rates'][arm] = {}
        for v, vi in VAR.items():
            rr = [R_[10 * q + vi] for q in qv[v]]
            if arm.startswith('P_'):
                res['rates'][arm][v] = dict(n=len(rr), fallback_rate=sum(r['status'] != 'ok' for r in rr) / len(rr),
                                            invalid_rate=sum(r['reason'] in ('invalid_json', 'unknown_name') for r in rr) / len(rr),
                                            reasons=dict(collections.Counter(r['reason'] for r in rr if r['status'] != 'ok')))
            else:
                res['rates'][arm][v] = dict(n=len(rr), invalid_rate=sum(not r['valid'] for r in rr) / len(rr))
    GUV = {int(r['quid']): r['fact_ids'] for r in rows(os.path.join(pred_dir, 'gu/gu_variants.jsonl.gz'))}
    gfr = list(FR['gu'].values())[0]
    res['identity']['I3_gu_retrieval'] = dict(identical_top20=float(np.mean([GUV[10 * q] == gfr[str(q)] for q in qv['ORIG']])),
                                              identical_top1=float(np.mean([GUV[10 * q][:1] == gfr[str(q)][:1] for q in qv['ORIG']])),
                                              changed_top20_RW=float(np.mean([GUV[10 * q + 3] != GUV[10 * q] for q in qv['RW']])))
    res['I2_pass'] = all(v['flag'] == 'OK' for a, v in res['identity'].items() if a != 'I3_gu_retrieval')
    res['coverage_pass'] = all(all(v) for v in res['coverage'].values())
    json.dump(res, open(out_p, 'w'), indent=1)
    print(json.dumps(dict(I2_pass=res['I2_pass'], coverage_pass=res['coverage_pass'], I2_min={a: round(v['min'], 4) for a, v in res['identity'].items() if a != 'I3_gu_retrieval'},
                          I3=res['identity']['I3_gu_retrieval'])))
    return res


if __name__ == '__main__':
    main(*sys.argv[1:5])
