"""pf_score.py -- Parser-first track: the SINGLE registered scoring pass (rule v1.0), run only after the freeze record
multitq_parser_first_freeze.json exists. Verifies every frozen file by sha256, then reads test.json gold for the first
time for these variants. Official multitq_eval.hit_at_k; statistics r2_common.summarise / r3_common.method_tests
(paired Poisson(1) bootstrap, B = 10,000, seed 20261118).
usage: python pf_score.py <freeze.json> <out_dir>"""
import collections, json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pf_lib as PL
WS = PL.W
sys.path.insert(0, WS)
sys.path.insert(0, os.environ['R3_CODE'])
import multitq_eval as ME
import r2_common as R2C
import r3_common as R3
FZ = json.load(open(sys.argv[1])); OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
for f in FZ['files']:
    p = f['path'] if os.path.isabs(f['path']) else (os.path.join(WS, f['path']) if os.path.exists(os.path.join(WS, f['path'])) else os.path.abspath(f['path']))
    assert PL.sha256_file(p) == f['sha256'], f'STOP_TEST_INTEGRITY freeze mismatch {f["path"]}'
assert PL.sha256_file(os.path.join(WS, 'multitq_eval.py')) == '1f85bbdcd623f1810e9df2910e042c24d6f5c269d278e82b30ef05387a1e2d69'
for n, h in (('r2_common.py', '5622ea2cb19853e4ddcc7e64dfb9872f35e920cdf33d67c583b8ca2359b2be5c'), ('r3_common.py', 'd80d811e3031ebd8f470010d801df9d424793b5185922d9f471d32e6f3a147bc')):
    assert PL.sha256_file(os.path.join(os.environ['R3_CODE'], n)) == h
SEL = json.load(open(os.path.join(WS, FZ['selection_path'])))['selection']
B, SEED = R3.B_SECONDARY, R3.BOOT_SEED
assert (B, SEED) == (10000, 20261118)
TEST_JSON = os.environ['MTQ_TEST_JSON']
assert PL.sha256_file(TEST_JSON) == 'd9bf3bf23b489ecc76c7940510c718260a42350d934a6edc1c1c7efcd5238fee'
DRY = os.environ.get('PF_DRYRUN') == '1'
if DRY:   # gold-free plumbing test: every answer replaced by a sentinel (all hits 0), small B; output never used
    B = 50
    Q = {int(q['quid']): dict(quid=q['quid'], qtype=q['qtype'], answer_type=q['answer_type'], answers=['__dryrun_sentinel__']) for q in json.load(open(TEST_JSON))}
else:
    Q = {int(q['quid']): q for q in json.load(open(TEST_JSON))}
TQ = sorted(Q); assert len(TQ) == 54584
R3Q = sorted(FZ['R3_primary_squids']); assert len(R3Q) == 578


def hits(Ls, quids, gold_of, k=1):
    per = np.array([[ME.hit_at_k(gold_of(q), L[q], k) for q in quids] for L in Ls], float)
    return per.mean(0), per.mean(1)


def summ(x):
    return R2C.summarise(np.asarray(x, float), B, SEED)


def orig_label(c, same):
    if same:
        return 'METHOD_EQUALS_BASELINE'
    lo, hi = c['ci95']
    return 'ORIG_COST' if hi < 0 else ('ORIG_WITHIN_0.3' if lo >= -0.003 else 'ORIG_INCONCLUSIVE')


res = collections.OrderedDict(track='multitq_parser_first', stage='single registered scoring pass',
                              label='reuse of the official MultiTQ test split (test.json sha256 d9bf3bf2...); R3 / R1 / R2 rewrites of official test questions',
                              rule_artifact=FZ['rule_artifact'], rule_sha256=FZ['rule_sha256'], selection_artifact=FZ['selection_artifact'],
                              freeze_sha256=FZ['sha256'], frozen_utc=FZ['frozen_utc'], bootstrap=dict(B=B, seed=SEED, method='paired Poisson(1), r2_common.summarise'),
                              selected={k: SEL[k]['selected'] for k in PL.KS})
tcols = collections.OrderedDict(quid=TQ, qtype=[Q[q]['qtype'] for q in TQ], answer_type=[Q[q]['answer_type'] for q in TQ])
TEST_H = {}        # (k, cand) -> per-question seed-mean Hits@1 over TQ
res['test'] = collections.OrderedDict()
for k in PL.KS:
    lev = PL.LEV[k]; D = PL.load('test', k); assert D['quids'] == TQ
    m = SEL[k]['selected']
    cell = collections.OrderedDict(n=len(TQ), selected=m)
    allc = collections.OrderedDict()
    for c in PL.CANDS:
        Ls = PL.lists(D, c)
        h1, ps = hits(Ls, TQ, lambda q: Q[q]); h10, _ = hits(Ls, TQ, lambda q: Q[q], 10)
        TEST_H[(k, c)] = h1
        allc[c] = dict(hits1=float(h1.mean()), hits1_per_seed=ps.tolist(), hits10=float(h10.mean()), routing=PL.routing_share(D, c, TQ))
        tcols[f'{c}_{lev}@1'] = np.round(h1, 6).tolist()
        if c in (m, 'V0'):
            for s in range(3):
                tcols[f'{c}_{lev}_seed{s}@1'] = [ME.hit_at_k(Q[q], Ls[s][q], 1) for q in TQ]
    for c in PL.CANDS:
        if c != 'V0':
            allc[c]['minus_V0'] = summ(TEST_H[(k, c)] - TEST_H[(k, 'V0')])
    cell['method'] = dict(name=m, **summ(TEST_H[(k, m)]), hits1_per_seed=allc[m]['hits1_per_seed'], hits10=allc[m]['hits10'])
    cell['baseline_V0'] = dict(**summ(TEST_H[(k, 'V0')]), hits1_per_seed=allc['V0']['hits1_per_seed'], hits10=allc['V0']['hits10'])
    con = summ(TEST_H[(k, m)] - TEST_H[(k, 'V0')])
    cell['method_minus_V0'] = dict(**con, label=orig_label(con, m == 'V0'))
    cell['all_candidates_descriptive'] = allc
    res['test'][lev] = cell
    print('test', k, m, round(cell['method']['delta'], 5), round(cell['baseline_V0']['delta'], 5), flush=True)

# ---------------------------------------------------------------- R3 (registered secondary analysis)
tidx = {q: i for i, q in enumerate(TQ)}
r3o = [s // 10 for s in R3Q]
rw_rows = []
res['R3'] = collections.OrderedDict(n=len(R3Q), set_artifact='multitq_robustness_R3_perturbed_set.json', ORIG_reference='full-test predictions of each candidate on the original question')
per_level, descr, R3H = {}, collections.OrderedDict(), {}
for k in PL.KREW:
    lev = PL.LEV[k]; Dv = PL.load('var', k)
    assert set(R3Q) <= set(Dv['quids'])
    dl = collections.OrderedDict()
    for c in PL.CANDS:
        r, ps = hits(PL.lists(Dv, c, R3Q), R3Q, lambda s: Q[s // 10])
        o = TEST_H[(k, c)][[tidx[q] for q in r3o]]
        R3H[(k, c)] = (o, r)
        dl[c] = dict(orig=float(o.mean()), rw=float(r.mean()), drop=float((o - r).mean()), rw_per_seed=ps.tolist(), routing_rw=PL.routing_share(Dv, c, R3Q))
    for c in PL.CANDS:
        if c != 'V0':
            o_m, r_m = R3H[(k, c)]; o_b, r_b = R3H[(k, 'V0')]
            C = R3.contrasts(o_m, r_m, o_b, r_b)
            dl[c]['c_drop_vs_V0'] = summ(C['c_drop']); dl[c]['c_rw_vs_V0'] = summ(C['c_rw'])
    descr[lev] = dl
    m = SEL[k]['selected']
    if m != 'V0':
        o_m, r_m = R3H[(k, m)]; o_b, r_b = R3H[(k, 'V0')]
        per_level[lev] = dict(o_m=o_m, r_m=r_m, o_b=o_b, r_b=r_b)
tests = R3.method_tests(per_level) if per_level else {}
res['R3']['registered_tests'] = collections.OrderedDict()
for k in PL.KREW:
    lev = PL.LEV[k]; m = SEL[k]['selected']
    if m == 'V0':
        o, r = R3H[(k, 'V0')]
        res['R3']['registered_tests'][lev] = dict(method_name='V0', status='METHOD_EQUALS_BASELINE (not a test)', baseline=dict(orig=float(o.mean()), rw=float(r.mean()), drop=float((o - r).mean())))
    else:
        res['R3']['registered_tests'][lev] = dict(method_name=m, **tests[lev])
res['R3']['holm_family'] = [f'{lev}:{cn}' for lev in per_level for cn in ('c_drop', 'c_rw')]
res['R3']['all_candidates_descriptive'] = descr
c3 = collections.OrderedDict(squid=R3Q, quid=r3o)
for k in PL.KREW:
    lev = PL.LEV[k]; m = SEL[k]['selected']
    o_m, r_m = R3H[(k, m)]; o_b, r_b = R3H[(k, 'V0')]
    C = R3.contrasts(o_m, r_m, o_b, r_b)
    c3[f'method_{lev}'] = [m] * len(R3Q)
    c3[f'c_drop_{lev}'] = np.round(C['c_drop'], 6).tolist(); c3[f'c_rw_{lev}'] = np.round(C['c_rw'], 6).tolist()
pd.DataFrame(c3).to_csv(os.path.join(OUT, 'multitq_parser_first_r3_contrasts.csv'), index=False)
for c in PL.CANDS:
    for k in PL.KREW:
        o, r = R3H[(k, c)]
        for i, s in enumerate(R3Q):
            rw_rows.append(dict(probe='R3', squid=s, quid=s // 10, variant='RW3', level=PL.LEV[k], cand=c, orig_h1=round(float(o[i]), 6), rw_h1=round(float(r[i]), 6)))
print('R3 done', {lev: (t.get('method_name'), t.get('status', '')) for lev, t in res['R3']['registered_tests'].items()}, flush=True)

# ---------------------------------------------------------------- R1 / R2 (secondary, descriptive)
VMAP = {'R1': {1: 'P1_prefix', 2: 'P2_qmark', 3: 'P3_paraphrase'}, 'R2': {3: 'RW_rewrite'}}
for probe, vmap in VMAP.items():
    out = collections.OrderedDict(); fam_keys, fam_p = [], []
    for k in PL.KREW:
        lev = PL.LEV[k]; D = PL.load(probe, k); sq = D['quids']; idx = {s: i for i, s in enumerate(sq)}
        H = {c: hits(PL.lists(D, c), sq, lambda s: Q[s // 10])[0] for c in PL.CANDS}
        m = SEL[k]['selected']; lvl = collections.OrderedDict()
        for vi, vn in vmap.items():
            sel = [s for s in sq if s % 10 == vi]; ii = [idx[s] for s in sel]; oi = [idx[(s // 10) * 10] for s in sel]
            d = collections.OrderedDict(n=len(sel))
            for c in PL.CANDS:
                o, r = H[c][oi], H[c][ii]
                d[c] = dict(orig=float(o.mean()), rw=float(r.mean()), drop=float((o - r).mean()))
                for i_, s in enumerate(sel):
                    rw_rows.append(dict(probe=probe, squid=s, quid=s // 10, variant=vn, level=lev, cand=c, orig_h1=round(float(o[i_]), 6), rw_h1=round(float(r[i_]), 6)))
            if m != 'V0':
                C = R3.contrasts(H[m][oi], H[m][ii], H['V0'][oi], H['V0'][ii])
                for cn, cv in C.items():
                    d[f'{cn}_method_vs_V0'] = summ(cv); fam_keys.append((lev, vn, cn)); fam_p.append(d[f'{cn}_method_vs_V0']['p_one_sided'])
            for c in PL.CANDS:
                if c not in ('V0', m):
                    C = R3.contrasts(H[c][oi], H[c][ii], H['V0'][oi], H['V0'][ii])
                    d[c]['c_drop_vs_V0'] = summ(C['c_drop']); d[c]['c_rw_vs_V0'] = summ(C['c_rw'])
            lvl[vn] = d
        out[lev] = dict(method=m, variants=lvl)
    adj = R2C.holm(fam_p) if fam_p else []
    for (lev, vn, cn), pa in zip(fam_keys, adj):
        out[lev]['variants'][vn][f'{cn}_method_vs_V0']['p_holm_one_sided_descriptive'] = pa
        out[lev]['variants'][vn][f'{cn}_method_vs_V0']['label_descriptive'] = 'SUPPORTED' if pa < 0.05 else 'NOT_SUPPORTED'
    res[probe] = out
    print(probe, 'done', flush=True)

pd.DataFrame(tcols).to_csv(os.path.join(OUT, 'multitq_parser_first_test_hits.csv.gz'), index=False, compression='gzip')
pd.DataFrame(rw_rows).to_csv(os.path.join(OUT, 'multitq_parser_first_rewrite_hits.csv.gz'), index=False, compression='gzip')
json.dump(res, open(os.path.join(OUT, 'multitq_parser_first_results.json'), 'w'), indent=1)
print('done', flush=True)
