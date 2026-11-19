"""rob_score.py -- the single scoring pass of the MultiTQ robustness probe (rule multitq_robustness_rule.json v1.0).
Run once, after multitq_robustness_freeze.json exists and every prediction file matches it. Reads gold answers only
here. Writes multitq_robustness_results.json.
Usage: python rob_score.py --scorer src/multitq_eval.py --pred-dir <extracted predictions tarball>
         --frozen-rows frozen_orig_rows.json --template-flags template_flags.json --test-json test.json
         --pset multitq_robustness_perturbed_set.json --rule protocol/multitq_robustness_rule.json
         --freeze multitq_robustness_freeze.json [--out multitq_robustness_results.json]
(equivalently, exec the module with SCORER, PRED_DIR, FROZEN_ROWS, TEMPLATE_FLAGS, TEST_JSON, PSET, RULE, FREEZE defined).
"""
import json, gzip, os, hashlib, collections
import numpy as np
import importlib.util

if __name__ == '__main__':
    import argparse
    _ap = argparse.ArgumentParser(description='single scoring pass of the R1 robustness probe')
    for _k in ('SCORER', 'PRED_DIR', 'FROZEN_ROWS', 'TEMPLATE_FLAGS', 'TEST_JSON', 'PSET', 'RULE', 'FREEZE'):
        _ap.add_argument('--' + _k.lower().replace('_', '-'), dest=_k, required=True)
    _ap.add_argument('--out', dest='OUT', default='multitq_robustness_results.json')
    globals().update(vars(_ap.parse_args()))

spec = importlib.util.spec_from_file_location('ME', SCORER)
ME = importlib.util.module_from_spec(spec); spec.loader.exec_module(ME)

KS = ['k01_s0', 'k05_s0', 'k100']
KLAB = {'k01_s0': '1', 'k05_s0': '5', 'k100': '100'}
VAR = ['ORIG', 'P1', 'P2', 'P3']


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


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


def boot(a, b, B=2000, seed=0):
    """identical to multitq_eval.paired_poisson_bootstrap (same rng stream); also returns the two-sided p"""
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    deltas = np.empty(B)
    for i in range(B):
        w = rng.poisson(1.0, size=d.shape[0])
        sw = w.sum()
        deltas[i] = (w * d).sum() / sw if sw > 0 else 0.0
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    p_le = float((deltas <= 0).mean()); p_ge = float((deltas >= 0).mean())
    return dict(delta=float(d.mean()), ci=[float(lo), float(hi)], p_le0=p_le, p_ge0=p_ge,
                p_two=float(min(1.0, 2 * min(p_le, p_ge))), B=B, n=int(d.shape[0]))


def holm(ps):
    m = len(ps); order = np.argsort(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj.tolist()


gold = {q['quid']: q for q in json.load(open(TEST_JSON))}
pset = json.load(open(PSET))
var_rows = pset['variants']
quids_by_var = {v: sorted(r['quid'] for r in var_rows if r['variant'] == v) for v in VAR}
qtext = {(r['quid'], r['variant']): r['question'] for r in var_rows}
freeze = json.load(open(FREEZE))
# ------------------------------------------------------------------ integrity: every file vs the freeze
bad = []
for rel, s in freeze['files'].items():
    p = os.path.join(PRED_DIR, rel)
    if not os.path.exists(p) or sha256_file(p) != s:
        bad.append(rel)
assert not bad, f'STOP_TEST_INTEGRITY: {bad[:5]}'

# ------------------------------------------------------------------ prediction files per arm (lists of seed files)
ARMS = {}
for k in KS:
    K_ = KLAB[k]
    ARMS[f'RULE_{K_}'] = [f'preds/{k}/variants_RULE.jsonl']
    ARMS[f'S_{K_}'] = [f'preds/{k}/variants_S_seed{s}.jsonl' for s in range(3)]
    ARMS[f'G_{K_}'] = [f'G/{k}/pred_G_variants_G_{k}.jsonl']
    ARMS[f'P_{K_}'] = [f'P/{k}/pred_P_{k}_variants_P_{k}_tauoff_seed{s}.jsonl' for s in range(3)]
for k in ('k01_s0', 'k100'):
    ARMS[f'GU_{KLAB[k]}'] = [f'GU/{k}/pred_GU_variants_GU_{k}.jsonl']
ARM_ORDER = ['RULE_1', 'S_1', 'G_1', 'P_1', 'GU_1', 'RULE_5', 'S_5', 'G_5', 'P_5', 'RULE_100', 'S_100', 'G_100', 'P_100', 'GU_100']
PRED = {a: [load_pred(os.path.join(PRED_DIR, f)) for f in fs] for a, fs in ARMS.items()}

# frozen ORIG (frozen test predictions on the same quids; top-10 rows)
FR = json.load(open(FROZEN_ROWS))
fp = FR['preds']


def fz(path):
    return {int(q): v for q, v in fp[path].items()}


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


def hits(arm, var, quids, kk):
    vi = VAR.index(var)
    out = []
    for q in quids:
        h = [ME.hit_at_k(gold[q], P_.get(10 * q + vi), kk) for P_ in PRED[arm]]
        out.append(float(np.mean(h)))
    return np.array(out)


def hits_frozen(arm, quids, kk):
    return np.array([float(np.mean([ME.hit_at_k(gold[q], F_.get(q), kk) for F_ in FROZEN[arm]])) for q in quids])


def top1(arm, var, q, s=0):
    r = PRED[arm][s].get(10 * q + VAR.index(var))
    return r[0] if r else None


res = dict(name='multitq_robustness_results', rule_file=os.path.basename(RULE), rule_sha256=json.load(open(RULE))['sha256'],
           freeze_sha256=freeze['sha256'], scorer_sha256=sha256_file(SCORER), n=dict((v, len(quids_by_var[v])) for v in VAR),
           cells={}, drops={}, contrasts={}, gaps={}, labels={}, rates={}, strata={}, identity={}, flips={})
# ------------------------------------------------------------------ cells
H = {}
for arm in ARM_ORDER:
    for v in VAR:
        qs = quids_by_var[v]
        for kk in (1, 10):
            H[(arm, v, kk)] = hits(arm, v, qs, kk)
    for kk in (1, 10):
        H[(arm, 'ORIG@P3', kk)] = hits(arm, 'ORIG', quids_by_var['P3'], kk)
        H[(arm, 'FROZEN', kk)] = hits_frozen(arm, quids_by_var['ORIG'], kk)
        H[(arm, 'FROZEN@P3', kk)] = hits_frozen(arm, quids_by_var['P3'], kk)
    res['cells'][arm] = {}
    for key in ('ORIG', 'P1', 'P2', 'ORIG@P3', 'P3', 'FROZEN', 'FROZEN@P3'):
        h1, h10 = H[(arm, key, 1)], H[(arm, key, 10)]
        ci = boot(h1, np.zeros_like(h1))['ci']
        res['cells'][arm][key] = dict(n=len(h1), hits1=float(h1.mean()), hits1_ci95=ci, hits10=float(h10.mean()))


def ref_of(v):
    return 'ORIG@P3' if v == 'P3' else 'ORIG'


# ------------------------------------------------------------------ drops (ORIG re-run - variant) and vs FROZEN
for arm in ARM_ORDER:
    res['drops'][arm] = {}
    for v in ('P1', 'P2', 'P3'):
        d1 = boot(H[(arm, ref_of(v), 1)], H[(arm, v, 1)])
        d10 = boot(H[(arm, ref_of(v), 10)], H[(arm, v, 10)])
        dfz = boot(H[(arm, 'FROZEN@P3' if v == 'P3' else 'FROZEN', 1)], H[(arm, v, 1)])
        lab = 'BRITTLE' if (d1['delta'] >= 0.02 and d1['ci'][0] > 0) else ('ROBUST' if d1['ci'][1] < 0.02 else 'INCONCLUSIVE')
        a_, b_ = H[(arm, ref_of(v), 1)], H[(arm, v, 1)]
        res['drops'][arm][v] = dict(drop_hits1=d1, drop_hits10=d10, drop_vs_frozen_hits1=dfz, label=lab,
                                    lost=float(np.clip(a_ - b_, 0, None).sum()), gained=float(np.clip(b_ - a_, 0, None).sum()))
# P2 on the changed subset (all 2,000 changed by construction if every question ends with '?')
# ------------------------------------------------------------------ contrasts
fam = []
for v in ('P3', 'P1', 'P2'):
    res['contrasts'][v] = {}
    res['gaps'][v] = {}
    for k in KS:
        K_ = KLAB[k]
        S, refq = f'S_{K_}', ref_of(v)
        for L_ in ('G', 'P', 'GU'):
            arm = f'{L_}_{K_}'
            if arm not in PRED:
                continue
            gv = H[(arm, v, 1)] - H[(S, v, 1)]
            go = H[(arm, refq, 1)] - H[(S, refq, 1)]
            c = boot(gv, go)
            res['contrasts'][v][f'Delta_{arm}'] = dict(c, definition=f'({arm} - {S}) on {v} minus on ORIG = drop({S}) - drop({arm})')
            res['gaps'][v][f'{arm}-{S}'] = dict(var=boot(H[(arm, v, 1)], H[(S, v, 1)]), orig=boot(H[(arm, refq, 1)], H[(S, refq, 1)]))
            if v == 'P3' and L_ in ('G', 'P'):
                fam.append((f'Delta_{arm}', c['p_two']))
        if f'GU_{K_}' in PRED:
            gg = boot(H[(f'G_{K_}', v, 1)] - H[(f'GU_{K_}', v, 1)], H[(f'G_{K_}', refq, 1)] - H[(f'GU_{K_}', refq, 1)])
            res['contrasts'][v][f'Delta_G_{K_}_vs_GU_{K_}'] = dict(gg, definition=f'(G_{K_} - GU_{K_}) on {v} minus on ORIG')
adj = holm([p for _, p in fam])
res['labels']['primary_family'] = []
for (name, p), pa in zip(fam, adj):
    c = res['contrasts']['P3'][name]
    lab = 'REPAIRS' if (c['delta'] >= 0.01 and pa < 0.05) else ('AMPLIFIES' if (c['delta'] <= -0.01 and pa < 0.05) else 'NO_DETECTABLE_DIFFERENCE')
    c['p_holm'] = pa; c['label'] = lab
    res['labels']['primary_family'].append(dict(contrast=name, delta=c['delta'], ci=c['ci'], p_two=p, p_holm=pa, label=lab))
for key, g in res['gaps']['P3'].items():
    ci = g['var']['ci']
    g['label_P3'] = 'GAP_HOLDS' if ci[0] > 0 else ('GAP_LOST' if ci[1] < 0 else 'GAP_UNRESOLVED')
# C_R3 under its own name (identical numbers to Delta_G / Delta_P / Delta_GU of P3)
res['contrasts']['C_R3'] = {f'drop(S_{KLAB[k]})-drop({L_}_{KLAB[k]})': res['contrasts']['P3'][f'Delta_{L_}_{KLAB[k]}']['delta']
                            for k in KS for L_ in ('G', 'P', 'GU') if f'Delta_{L_}_{KLAB[k]}' in res['contrasts']['P3']}
# ------------------------------------------------------------------ rates (invalid / fallback), identity, flips
RAW = {}
for k in KS:
    RAW[f'G_{KLAB[k]}'] = {int(r['quid']): r for r in rows(os.path.join(PRED_DIR, f'G/{k}/raw_G_variants_G_{k}.jsonl.gz'))}
    RAW[f'P_{KLAB[k]}'] = {int(r['quid']): r for r in rows(os.path.join(PRED_DIR, f'P/{k}/exec_P_{k}_variants_P_{k}.jsonl.gz'))}
for k in ('k01_s0', 'k100'):
    RAW[f'GU_{KLAB[k]}'] = {int(r['quid']): r for r in rows(os.path.join(PRED_DIR, f'GU/{k}/raw_GU_variants_GU_{k}.jsonl.gz'))}
for arm, R_ in RAW.items():
    res['rates'][arm] = {}
    for v in VAR:
        vi = VAR.index(v)
        rr = [R_[10 * q + vi] for q in quids_by_var[v]]
        if arm.startswith('P_'):
            reasons = collections.Counter(r['reason'] for r in rr if r['status'] != 'ok')
            res['rates'][arm][v] = dict(n=len(rr), fallback_rate=sum(r['status'] != 'ok' for r in rr) / len(rr),
                                        invalid_rate=sum(r['reason'] in ('invalid_json', 'unknown_name') for r in rr) / len(rr), reasons=dict(reasons))
        else:
            res['rates'][arm][v] = dict(n=len(rr), invalid_rate=sum(not r['valid'] for r in rr) / len(rr))
# frozen ORIG fallback / invalid rates on the same quids
res['rates']['frozen_orig'] = {}
for path, d in FR['exec'].items():
    st = [d[str(q)] if str(q) in d else d.get(q) for q in quids_by_var['ORIG']]
    st = [s for s in st if s is not None]
    res['rates']['frozen_orig'][os.path.basename(path)] = dict(n=len(st), fallback_rate=sum(s['status'] != 'ok' for s in st) / max(1, len(st)))
for path, d in FR['raw'].items():
    st = [d.get(str(q)) for q in quids_by_var['ORIG']]
    st = [s for s in st if s is not None]
    res['rates']['frozen_orig'][os.path.basename(path)] = dict(n=len(st), invalid_rate=sum(not s['valid'] for s in st) / max(1, len(st)))
# I2: ORIG re-run top-1 vs frozen top-1 (gold-free)
for arm in ARM_ORDER:
    ag = []
    for s in range(len(PRED[arm])):
        ag.append(float(np.mean([(PRED[arm][s].get(10 * q) or [None])[0] == (FROZEN[arm][s].get(q) or [None])[0] for q in quids_by_var['ORIG']])))
    res['identity'][arm] = dict(top1_agreement_orig_rerun_vs_frozen=ag, min=min(ag), flag=('PIPELINE_DRIFT' if min(ag) < (0.98 if arm[0] in 'GP' else 0.995) else 'OK'))
# I3: GU retrieval ORIG vs frozen gu_test (identical top-20 fact-id lists)
GUV = {int(r['quid']): r['fact_ids'] for r in rows(os.path.join(PRED_DIR, 'gu/gu_variants.jsonl.gz'))}
gfr = list(FR['gu'].values())[0]
res['identity']['I3_gu_retrieval'] = dict(identical_top20=float(np.mean([GUV[10 * q] == gfr[str(q)] for q in quids_by_var['ORIG']])),
                                          identical_top1=float(np.mean([GUV[10 * q][:1] == gfr[str(q)][:1] for q in quids_by_var['ORIG']])),
                                          changed_top20_P1=float(np.mean([GUV[10 * q + 1] != GUV[10 * q] for q in quids_by_var['P1']])),
                                          changed_top20_P2=float(np.mean([GUV[10 * q + 2] != GUV[10 * q] for q in quids_by_var['P2']])),
                                          changed_top20_P3=float(np.mean([GUV[10 * q + 3] != GUV[10 * q] for q in quids_by_var['P3']])))
# top-1 changed share per variant (gold-free)
for arm in ARM_ORDER:
    res['flips'][arm] = {v: float(np.mean([top1(arm, v, q) != top1(arm, ref_of(v).split('@')[0], q) for q in quids_by_var[v]])) for v in ('P1', 'P2', 'P3')}
# ------------------------------------------------------------------ strata (P3): qtype, template seen/unseen (B = 1000)
TF = json.load(open(TEMPLATE_FLAGS))
for arm in ARM_ORDER:
    k = {'1': 'k01_s0', '5': 'k05_s0', '100': 'k100'}[arm.split('_')[1]]
    qs = quids_by_var['P3']
    a_, b_ = H[(arm, 'ORIG@P3', 1)], H[(arm, 'P3', 1)]
    st = {}
    for qt in sorted({gold[q]['qtype'] for q in qs}):
        m = np.array([gold[q]['qtype'] == qt for q in qs])
        st[f'qtype={qt}'] = boot(a_[m], b_[m], B=1000)
    for lab, want in (('paraphrase_template_seen', True), ('paraphrase_template_unseen', False)):
        m = np.array([TF[str(10 * q + 3)]['seen'][k] == want for q in qs])
        if m.sum():
            st[lab] = boot(a_[m], b_[m], B=1000)
    res['strata'][arm] = st
res['template_seen_share'] = {k: dict(orig=float(np.mean([TF[str(10 * q)]['seen'][k] for q in quids_by_var['P3']])),
                                      paraphrase=float(np.mean([TF[str(10 * q + 3)]['seen'][k] for q in quids_by_var['P3']])))
                              for k in KS}
RESULTS = res
if __name__ == '__main__':
    json.dump(RESULTS, open(OUT, 'w'), indent=1, default=float)
