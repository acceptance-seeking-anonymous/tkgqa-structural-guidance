#!/usr/bin/env python3
"""sa_score_A.py -- the single scoring pass of Part A (parser replay), run after the Part A freeze file exists.
Reads gold (official test.json) for the first time for these files. Official evaluator multitq_eval.hit_at_k.
usage: python sa_score_A.py <freeze_A.json> <out_dir>
"""
import collections, gzip, hashlib, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, '..', '..', 'src'))   # src/multitq_eval.py
import multitq_eval as ME
import sa_stats as SS

W = os.environ['SA_WORK']
TEST_JSON = os.environ['MTQ_TEST_JSON']
HITS_CSV = os.environ['KPCT_HITS_CSV']
freeze_p, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
assert hashlib.sha256(open(ME.__file__, 'rb').read()).hexdigest() == '1f85bbdcd623f1810e9df2910e042c24d6f5c269d278e82b30ef05387a1e2d69'
FZ = json.load(open(freeze_p))
FZI = {(f['dir'], f['file']): f['sha256'] for f in FZ['files']}


def checked(d, fn):
    p = os.path.join(W, 'out', d, fn)
    h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    assert FZI[(d, fn)] == h, f'freeze mismatch {d}/{fn}'
    return p


KS = [('k01_s0', '1'), ('k05_s0', '5'), ('k10_s0', '10'), ('k100', '100')]
MODES = ['PK', 'PALONE', 'PREL', 'PENT']
VARS = ['fb', 'strict']
QS = json.load(open(TEST_JSON))
assert hashlib.sha256(open(TEST_JSON, 'rb').read()).hexdigest() == 'd9bf3bf23b489ecc76c7940510c718260a42350d934a6edc1c1c7efcd5238fee'
Q = {int(q['quid']): q for q in QS}
TQ = sorted(Q)
assert len(TQ) == 54584

# check the shared-weight bootstrap equals the evaluator's loop on one random pair
_rng = np.random.default_rng(1); _a = _rng.integers(0, 2, 3000); _b = _rng.integers(0, 2, 3000)
_o = ME.paired_poisson_bootstrap(_a, _b); _m = SS.boot_diff(_a, _b)
assert abs(_o['ci_low'] - _m['ci_low']) < 1e-12 and abs(_o['ci_high'] - _m['ci_high']) < 1e-12, (_o, _m)


def score_split(d, split, k, quids, qof):
    """returns dict of per-question arrays and status info for every mode / var / seed."""
    R = {}
    for m in MODES:
        st, rs, inpool, psize, dt = {}, {}, {}, {}, {}
        with gzip.open(checked(d, f'pool_{m}_{k}_{split}.jsonl.gz'), 'rt') as f:
            for l in f:
                x = json.loads(l); q = int(x['quid'])
                st[q] = x['status']; rs[q] = x['reason']; psize[q] = x['pool_size'] or 0; dt[q] = str(x.get('detail'))
                inpool[q] = ME.hit_at_k(qof(q), x['pool'], len(x['pool'])) if x['status'] == 'ok' else 0
        assert set(st) == set(quids)
        R[(m, 'status')] = np.array([st[q] for q in quids]); R[(m, 'reason')] = np.array([str(rs[q]) for q in quids])
        # validation-stage failure (pk_common.validate_compact / check_names; identical for every mode): detail has no ':'
        R[(m, 'valfail')] = np.array([st[q] != 'ok' and rs[q] in ('invalid_json', 'unknown_name') and ':' not in dt[q] for q in quids])
        R[(m, 'inpool')] = np.array([inpool[q] for q in quids], np.int8); R[(m, 'pool_size')] = np.array([psize[q] for q in quids])
        for v in VARS:
            h1s, h10s, h50s = [], [], []
            for s in range(3):
                rk = SS.read_ranked(checked(d, f'pred_{m}_{k}_{split}_{v}_seed{s}.jsonl'))
                assert set(rk) == set(quids)
                h1s.append(np.array([ME.hit_at_k(qof(q), rk[q], 1) for q in quids], np.int8))
                h10s.append(np.array([ME.hit_at_k(qof(q), rk[q], 10) for q in quids], np.int8))
                h50s.append(np.array([ME.hit_at_k(qof(q), rk[q], 50) for q in quids], np.int8))
            R[(m, v, 'h1s')] = h1s; R[(m, v, 'h10s')] = h10s; R[(m, v, 'h50s')] = h50s
            R[(m, v, 'h1')] = np.mean(h1s, 0); R[(m, v, 'h10')] = np.mean(h10s, 0); R[(m, v, 'h50')] = np.mean(h50s, 0)
    return R


def cell_summary(R, m, v, B=2000):
    st = R[(m, 'status')]; rs = R[(m, 'reason')]
    n = len(st)
    out = dict(hits1=SS.boot_mean(R[(m, v, 'h1')], B), hits1_per_seed=[float(x.mean()) for x in R[(m, v, 'h1s')]],
               hits10=float(R[(m, v, 'h10')].mean()), recall_at50_final=float(R[(m, v, 'h50')].mean()))
    if v == 'fb':
        out.update(parse_validity=float(1 - R[(m, 'valfail')].mean()),
                   executability=float((st == 'ok').mean()), fallback_frequency=float((st != 'ok').mean()),
                   fallback_by_reason={r: int(c) for r, c in collections.Counter(rs[st != 'ok']).items()},
                   candidate_recall=SS.boot_mean(R[(m, 'inpool')].astype(float), B),
                   mean_pool_size_ok=float(R[(m, 'pool_size')][st == 'ok'].mean()), n=n)
    return out


results = dict(track='multitq_structural_ablation', part='A', label='test split reused (official MultiTQ test, 54,584 questions)',
               freeze=dict(file=os.path.basename(freeze_p), sha256=FZ['sha256'], frozen_utc=FZ['frozen_utc']),
               evaluator_sha256='1f85bbdcd623f1810e9df2910e042c24d6f5c269d278e82b30ef05387a1e2d69', test={}, rewrites={})
# ---- registered per-question hits (frozen P_k, S_k) for identity / descriptive contrasts
import csv
REG = {}
with gzip.open(HITS_CSV, 'rt') as f:
    rd = csv.DictReader(f)
    want = [f'{a}_{kk}_seed{s}@1' for a in ('P', 'S') for _, kk in KS for s in range(3)]
    for row in rd:
        REG[int(row['quid'])] = [float(row[c]) for c in want]
REGC = {c: i for i, c in enumerate(want)}
reg = lambda a, kk: np.mean([[REG[q][REGC[f'{a}_{kk}_seed{s}@1']] for q in TQ] for s in range(3)], 0)

cols = collections.OrderedDict(quid=TQ, qtype=[Q[q]['qtype'] for q in TQ], answer_type=[Q[q]['answer_type'] for q in TQ])
prim, strict = {}, {}
for k, kk in KS:
    R = score_split('A_test', 'test', k, TQ, lambda q: Q[q])
    res = dict(cells={f'{m}_{v}': cell_summary(R, m, v) for m in MODES for v in VARS})
    both_ok = (R[('PK', 'status')] == 'ok') & (R[('PALONE', 'status')] == 'ok')
    res['candidate_recall_both_executable'] = dict(n=int(both_ok.sum()), PK=float(R[('PK', 'inpool')][both_ok].mean()),
                                                   PALONE=float(R[('PALONE', 'inpool')][both_ok].mean()),
                                                   diff=SS.boot_diff(R[('PK', 'inpool')][both_ok], R[('PALONE', 'inpool')][both_ok]))
    res['pool_size_both_executable'] = dict(PK_mean=float(R[('PK', 'pool_size')][both_ok].mean()), PALONE_mean=float(R[('PALONE', 'pool_size')][both_ok].mean()),
                                            PK_median=float(np.median(R[('PK', 'pool_size')][both_ok])), PALONE_median=float(np.median(R[('PALONE', 'pool_size')][both_ok])))
    c = SS.boot_diff(R[('PK', 'fb', 'h1')], R[('PALONE', 'fb', 'h1')]); prim[kk] = c
    cs = SS.boot_diff(R[('PK', 'strict', 'h1')], R[('PALONE', 'strict', 'h1')]); strict[kk] = cs
    pkreg = reg('P', kk); sreg = reg('S', kk)
    res['I_A2_replay_vs_registered_P'] = dict(registered_hits1=float(pkreg.mean()), replay_hits1=float(R[('PK', 'fb', 'h1')].mean()),
                                              per_question_identical=float((np.abs(pkreg - R[('PK', 'fb', 'h1')]) < 1e-9).mean()),
                                              per_seed_identical=[float((np.array([REG[q][REGC[f'P_{kk}_seed{s}@1']] for q in TQ]) == R[('PK', 'fb', 'h1s')][s]).mean()) for s in range(3)])
    res['secondary'] = {
        'PK_fb_minus_PK_strict': SS.boot_diff(R[('PK', 'fb', 'h1')], R[('PK', 'strict', 'h1')]),
        'PALONE_fb_minus_PALONE_strict': SS.boot_diff(R[('PALONE', 'fb', 'h1')], R[('PALONE', 'strict', 'h1')]),
        'PALONE_fb_minus_S_k_frozen': dict(S_k_hits1=float(sreg.mean()), **SS.boot_diff(R[('PALONE', 'fb', 'h1')], sreg)),
        'PK_fb_minus_S_k_frozen': dict(S_k_hits1=float(sreg.mean()), **SS.boot_diff(R[('PK', 'fb', 'h1')], sreg)),
        'PREL_fb_minus_PALONE_fb': SS.boot_diff(R[('PREL', 'fb', 'h1')], R[('PALONE', 'fb', 'h1')]),
        'PENT_fb_minus_PALONE_fb': SS.boot_diff(R[('PENT', 'fb', 'h1')], R[('PALONE', 'fb', 'h1')]),
        'PK_fb_minus_PREL_fb': SS.boot_diff(R[('PK', 'fb', 'h1')], R[('PREL', 'fb', 'h1')]),
        'PK_fb_minus_PENT_fb': SS.boot_diff(R[('PK', 'fb', 'h1')], R[('PENT', 'fb', 'h1')]),
        'candidate_recall_PK_minus_PALONE_all': SS.boot_diff(R[('PK', 'inpool')], R[('PALONE', 'inpool')]),
        'hits10_PK_fb_minus_PALONE_fb': SS.boot_diff(R[('PK', 'fb', 'h10')], R[('PALONE', 'fb', 'h10')]),
    }
    # strata (descriptive, B = 1000)
    strata = {}
    for key in ('qtype', 'answer_type'):
        vals = np.array(cols[key])
        for sv in sorted(set(vals)):
            msk = vals == sv
            strata[f'{key}={sv}'] = dict(n=int(msk.sum()), PK_fb=float(R[('PK', 'fb', 'h1')][msk].mean()), PALONE_fb=float(R[('PALONE', 'fb', 'h1')][msk].mean()),
                                        PK_inpool=float(R[('PK', 'inpool')][msk].mean()), PALONE_inpool=float(R[('PALONE', 'inpool')][msk].mean()),
                                        diff=SS.boot_diff(R[('PK', 'fb', 'h1')][msk], R[('PALONE', 'fb', 'h1')][msk], B=1000))
    res['strata'] = strata
    results['test'][kk] = res
    for m in MODES:
        cols[f'{m}_{kk}_status'] = [s if s == 'ok' else f'fb:{r}' for s, r in zip(R[(m, 'status')], R[(m, 'reason')])]
        cols[f'{m}_{kk}_inpool'] = R[(m, 'inpool')].tolist()
        for v in VARS:
            for s in range(3):
                cols[f'{m}_{kk}_{v}_seed{s}@1'] = R[(m, v, 'h1s')][s].tolist()
            cols[f'{m}_{kk}_{v}@1'] = np.round(R[(m, v, 'h1')], 6).tolist()
            cols[f'{m}_{kk}_{v}@10'] = np.round(R[(m, v, 'h10')], 6).tolist()
    print(k, 'PK_fb', round(R[('PK', 'fb', 'h1')].mean(), 5), 'PALONE_fb', round(R[('PALONE', 'fb', 'h1')].mean(), 5), 'D', round(c['delta'], 5), flush=True)
adj = SS.holm({kk: c['p_two_sided'] for kk, c in prim.items()})
adjs = SS.holm({kk: c['p_two_sided'] for kk, c in strict.items()})
lab = {kk: SS.label(prim[kk], adj[kk], 'ALTERNATIVES_HELP', 'ALTERNATIVES_HURT', 'NO_DETECTABLE_DIFFERENCE') for kk in prim}
labs = {kk: SS.label(strict[kk], adjs[kk], 'ALTERNATIVES_HELP', 'ALTERNATIVES_HURT', 'NO_DETECTABLE_DIFFERENCE') for kk in strict}
summ = lambda L: 'SUPPORTED' if all(v == 'ALTERNATIVES_HELP' for v in L.values()) else ('PARTIAL' if any(v == 'ALTERNATIVES_HELP' for v in L.values()) else 'NOT_SUPPORTED')
results['primary_D_A'] = {kk: dict(**prim[kk], p_holm=adj[kk], label=lab[kk]) for kk in prim}
results['primary_summary'] = summ(lab)
results['strict_D_A'] = {kk: dict(**strict[kk], p_holm=adjs[kk], label=labs[kk]) for kk in strict}
results['strict_summary'] = summ(labs)

# ---------------------------------------------------------------- rewrites (descriptive)
import pandas as pd
pd.DataFrame(cols).to_csv(os.path.join(OUT, 'multitq_structural_ablation_A_test_hits.csv.gz'), index=False, compression='gzip')
RW = {'R1': {0: 'ORIG', 1: 'P1_prefix', 2: 'P2_qmark', 3: 'P3_paraphrase'}, 'R2': {0: 'ORIG', 3: 'RW_rewrite'}}
rcols = collections.OrderedDict()
for R_, vmap in RW.items():
    out = {}
    sq = None
    for k, kk in [x for x in KS if x[0] != 'k10_s0']:
        if sq is None:
            sq = sorted(int(json.loads(l)['quid']) for l in gzip.open(checked(f'A_{R_}', f'pool_PK_{k}_{R_}.jsonl.gz'), 'rt'))
            rcols[R_] = collections.OrderedDict(squid=sq, quid=[s // 10 for s in sq], variant=[vmap[s % 10] for s in sq])
        R = score_split(f'A_{R_}', R_, k, sq, lambda s: Q[s // 10])
        idx = {s: i for i, s in enumerate(sq)}
        per = {}
        for vi, vn in vmap.items():
            sel = [i for i, s in enumerate(sq) if s % 10 == vi]
            orig = [idx[(s // 10) * 10] for s in np.array(sq)[sel]]
            d = dict(n=len(sel))
            for m in MODES:
                for v in VARS:
                    h = R[(m, v, 'h1')]
                    d[f'{m}_{v}'] = SS.boot_mean(h[sel])
                d[f'{m}_fallback_frequency'] = float((R[(m, 'status')][sel] != 'ok').mean())
                d[f'{m}_candidate_recall'] = float(R[(m, 'inpool')][sel].mean())
            d['D_A_fb'] = SS.boot_diff(R[('PK', 'fb', 'h1')][sel], R[('PALONE', 'fb', 'h1')][sel])
            d['D_A_strict'] = SS.boot_diff(R[('PK', 'strict', 'h1')][sel], R[('PALONE', 'strict', 'h1')][sel])
            if vi != 0:
                for v in VARS:
                    for m in ('PK', 'PALONE'):
                        d[f'drop_{m}_{v}'] = SS.boot_diff(R[(m, v, 'h1')][sel], R[(m, v, 'h1')][orig])
                    dd_pk = R[('PK', v, 'h1')][sel] - R[('PK', v, 'h1')][orig]
                    dd_pa = R[('PALONE', v, 'h1')][sel] - R[('PALONE', v, 'h1')][orig]
                    d[f'drop_PK_minus_drop_PALONE_{v}'] = SS.boot_diff(dd_pk, dd_pa)
                d['ORIG_same_quids'] = {f'{m}_{v}': float(R[(m, v, 'h1')][orig].mean()) for m in ('PK', 'PALONE') for v in VARS}
            per[vn] = d
        out[kk] = per
        for m in MODES:
            rcols[R_][f'{m}_{kk}_status'] = [s if s == 'ok' else f'fb:{r}' for s, r in zip(R[(m, 'status')], R[(m, 'reason')])]
            rcols[R_][f'{m}_{kk}_inpool'] = R[(m, 'inpool')].tolist()
            for v in VARS:
                rcols[R_][f'{m}_{kk}_{v}@1'] = np.round(R[(m, v, 'h1')], 6).tolist()
                for s in range(3):
                    rcols[R_][f'{m}_{kk}_{v}_seed{s}@1'] = R[(m, v, 'h1s')][s].tolist()
        print(R_, k, {vn: (round(per[vn]['PK_fb']['mean'], 4), round(per[vn]['PALONE_fb']['mean'], 4)) for vn in per}, flush=True)
    results['rewrites'][R_] = out
    pd.DataFrame(rcols[R_]).to_csv(os.path.join(OUT, f'multitq_structural_ablation_A_{R_}_hits.csv.gz'), index=False, compression='gzip')
json.dump(results, open(os.path.join(OUT, 'multitq_structural_ablation_A_results.json'), 'w'), indent=1)
print(json.dumps({kk: (round(v['delta'], 5), round(v['ci_low'], 5), round(v['ci_high'], 5), v['label']) for kk, v in results['primary_D_A'].items()}))
