#!/usr/bin/env python3
"""sa_score_B.py -- the single scoring pass of Part B (Ranker feature-group ablation), run after the Part B freeze file exists.
usage: python sa_score_B.py <freeze_B.json> <out_dir>
"""
import collections, gzip, hashlib, json, os, sys, csv
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, '..', '..', 'src'))   # src/multitq_eval.py
import multitq_eval as ME
import sa_stats as SS

PRED = os.environ['SB_PRED']           # dir with pred_SA_* files
SB = os.environ['SB_ROOT']             # structural tree (out_hs0 / out_hs7)
QDIR = os.environ['MTQ_QDIR']
HITS_CSV = os.environ['KPCT_HITS_CSV']
freeze_p, OUT = sys.argv[1], sys.argv[2]
assert hashlib.sha256(open(ME.__file__, 'rb').read()).hexdigest() == '1f85bbdcd623f1810e9df2910e042c24d6f5c269d278e82b30ef05387a1e2d69'
FZ = json.load(open(freeze_p)); FZI = {f['file']: f['sha256'] for f in FZ['files']}


def checked(fn):
    p = os.path.join(PRED, fn)
    assert FZI[fn] == hashlib.sha256(open(p, 'rb').read()).hexdigest(), fn
    return p


GROUPS = ['G_OPSAT', 'G_TPOS', 'G_RELP', 'G_LINK', 'G_PROV', 'G_PRIOR']
COMPS = ['C_RULEKEY', 'C_TEMPORAL']
ARMS = ['full'] + [f'minus_{g}' for g in GROUPS] + [f'minus_{c}' for c in COMPS] + ['full_hs7']
KS = [('k01_s0', '1'), ('k05_s0', '5')]
Qt = {int(q['quid']): q for q in json.load(open(os.path.join(QDIR, 'test.json')))}
assert hashlib.sha256(open(os.path.join(QDIR, 'test.json'), 'rb').read()).hexdigest() == 'd9bf3bf23b489ecc76c7940510c718260a42350d934a6edc1c1c7efcd5238fee'
Qd_all = {int(q['quid']): q for q in json.load(open(os.path.join(QDIR, 'dev.json')))}
TQ = sorted(Qt)
REG = {}
with gzip.open(HITS_CSV, 'rt') as f:
    for row in csv.DictReader(f):
        REG[int(row['quid'])] = {c: float(row[c]) for c in row if c.startswith('S_') and c.endswith('@1')}

results = dict(track='multitq_structural_ablation', part='B', label='test split reused (official MultiTQ test, 54,584 questions); dev = dev_sel2000 (selection set)',
               freeze=dict(file=os.path.basename(freeze_p), sha256=FZ['sha256'], frozen_utc=FZ['frozen_utc']), levels={})
cols = collections.OrderedDict(quid=TQ, qtype=[Qt[q]['qtype'] for q in TQ], answer_type=[Qt[q]['answer_type'] for q in TQ])
prim = {}
for k, kk in KS:
    res = dict(selection={}, test={}, dev={}, contrasts_test={}, contrasts_dev={}, secondary={}, strata={})
    for arm in ARMS:
        root = 'out_hs7' if arm == 'full_hs7' else 'out_hs0'
        sel = json.load(open(os.path.join(SB, root, k, 'ablate', 'full' if arm == 'full_hs7' else arm, 'selection.json')))
        res['selection'][arm] = dict(selected=sel['selected'], n_features=sel['n_features'], dev_sel2000_hits1_seed_from_training=sel['seed_dev_sel2000_hits1'],
                                     S_train_rows=sel['S_train_rows'], hash_seed=sel['hash_seed'])
    H = {}
    for split, QQ in (('test', Qt), ('dev_sel2000', None)):
        for arm in ARMS:
            h1s, h10s = [], []
            for s in range(3):
                rk = SS.read_ranked(checked(f'pred_SA_{arm}_{k}_{split}_seed{s}.jsonl'))
                if QQ is None:
                    QQ = {q: Qd_all[q] for q in rk}
                    dq = sorted(QQ); assert len(dq) == 2000
                qs = TQ if split == 'test' else dq
                assert set(rk) == set(qs)
                h1s.append(np.array([ME.hit_at_k(QQ[q], rk[q], 1) for q in qs], np.int8))
                h10s.append(np.array([ME.hit_at_k(QQ[q], rk[q], 10) for q in qs], np.int8))
            H[(split, arm)] = (h1s, np.mean(h1s, 0), np.mean(h10s, 0))
            tgt = res['test' if split == 'test' else 'dev']
            tgt[arm] = dict(hits1=SS.boot_mean(H[(split, arm)][1]), hits1_per_seed=[float(x.mean()) for x in h1s], hits10=float(H[(split, arm)][2].mean()))
            if split == 'test':
                for s in range(3):
                    cols[f'SA_{kk}_{arm}_seed{s}@1'] = h1s[s].tolist()
                cols[f'SA_{kk}_{arm}@1'] = np.round(H[(split, arm)][1], 6).tolist()
                cols[f'SA_{kk}_{arm}@10'] = np.round(H[(split, arm)][2], 6).tolist()
    for arm in ARMS[1:]:
        for split, tgt in (('test', res['contrasts_test']), ('dev_sel2000', res['contrasts_dev'])):
            tgt[f'full_minus_{arm}' if arm != 'full_hs7' else 'full_hs0_minus_full_hs7'] = SS.boot_diff(H[(split, 'full')][1], H[(split, arm)][1])
    for g in GROUPS:
        prim[(kk, g)] = res['contrasts_test'][f'full_minus_minus_{g}']
    sreg = np.mean([[REG[q][f'S_{kk}_seed{s}@1'] for q in TQ] for s in range(3)], 0)
    res['secondary']['SA_full_minus_frozen_S_k_test'] = dict(frozen_S_k_hits1=float(sreg.mean()), **SS.boot_diff(H[('test', 'full')][1], sreg))
    res['secondary']['noise_floor_abs_full_hs0_minus_full_hs7_test'] = abs(res['contrasts_test']['full_hs0_minus_full_hs7']['delta'])
    # strata (descriptive, B = 1000): Delta per qtype / answer type
    for key in ('qtype', 'answer_type'):
        vals = np.array(cols[key])
        for sv in sorted(set(vals)):
            msk = vals == sv
            res['strata'][f'{key}={sv}'] = dict(n=int(msk.sum()), full=float(H[('test', 'full')][1][msk].mean()),
                                               **{f'Delta_{g}': round(SS.boot_diff(H[('test', 'full')][1][msk], H[('test', f'minus_{g}')][1][msk], B=1000)['delta'], 6) for g in GROUPS + COMPS})
    results['levels'][kk] = res
    print(k, {a: round(res['test'][a]['hits1']['mean'], 5) for a in ARMS}, flush=True)
adj = SS.holm({f'{kk}|{g}': c['p_two_sided'] for (kk, g), c in prim.items()})
results['primary'] = {}
for (kk, g), c in prim.items():
    lab = SS.label(c, adj[f'{kk}|{g}'], 'GROUP_NEEDED', 'GROUP_HARMFUL', 'NO_DETECTABLE_EFFECT')
    nf = results['levels'][kk]['secondary']['noise_floor_abs_full_hs0_minus_full_hs7_test']
    results['primary'][f'{kk}|{g}'] = dict(k=kk, group=g, **c, p_holm=adj[f'{kk}|{g}'], label=lab,
                                           below_noise_floor=bool(abs(c['delta']) < nf) if lab != 'NO_DETECTABLE_EFFECT' else None)
os.makedirs(OUT, exist_ok=True)
import pandas as pd
pd.DataFrame(cols).to_csv(os.path.join(OUT, 'multitq_structural_ablation_B_test_hits.csv.gz'), index=False, compression='gzip')
json.dump(results, open(os.path.join(OUT, 'multitq_structural_ablation_B_results.json'), 'w'), indent=1)
print(json.dumps({key: (round(v['delta'], 5), round(v['ci_low'], 5), round(v['ci_high'], 5), v['label']) for key, v in results['primary'].items()}))
