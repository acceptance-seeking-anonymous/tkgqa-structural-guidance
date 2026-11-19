"""kpct_subsets.py -- nested qtype-stratified MultiTQ train subsets, dev selection subset, OOF splits, template profile.
Reads question files for quid + qtype + question text only (no answers are read for any purpose).
Run from the workspace root after kp/kpct_keys.py (kp/keys_kpct.pkl).
"""
import json, hashlib, collections, pickle, datetime
import numpy as np

import os
Q = os.path.join(os.environ.get('DATA_DIR', 'data/MultiTQ_94ad61c/data/MultiTQ'), 'questions') + '/'
R60 = os.environ.get('READER_TRAIN60K_PATH', 'struct/reader_train60k_quids.json')
S100 = os.environ.get('S_TRAIN100K_PATH', 'struct/s_train100k_quids.json')
QTYPES = None


def lsha(lst):
    """sha256 recipe for a quid list: sorted ints, compact JSON, utf-8."""
    return hashlib.sha256(json.dumps(sorted(int(x) for x in lst), separators=(',', ':')).encode()).hexdigest()


def fsha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def lr_alloc(counts, tot):
    """proportional allocation with largest-remainder rounding (dev_sel recipe); ties by qtype name."""
    N = sum(counts.values())
    raw = {k: tot * v / N for k, v in counts.items()}
    al = {k: int(np.floor(x)) for k, x in raw.items()}
    rem = tot - sum(al.values())
    for k in sorted(raw, key=lambda k: (-(raw[k] - al[k]), k))[:rem]:
        al[k] += 1
    return al


def meta(split):
    qs = json.load(open(Q + f'{split}.json'))
    return [(int(q['quid']), q['qtype']) for q in qs]


tr = meta('train'); dv = meta('dev'); te = meta('test')
assert (len(tr), len(dv), len(te)) == (386787, 57979, 54584)
by = collections.defaultdict(list)
for qd, qt in tr:
    by[qt].append(qd)
QTYPES = sorted(by)
cnt = {k: len(by[k]) for k in QTYPES}
NTR = len(tr)
qt_of = {qd: qt for qd, qt in tr}

# ---- seed-0 nested chain: one permutation per qtype, subsets are per-qtype prefixes
TOT = {'k01': round(0.01 * NTR), 'k05': round(0.05 * NTR), 'k10': round(0.10 * NTR), 'cap20k_k10': 20000}
AL = {nm: lr_alloc(cnt, t) for nm, t in TOT.items()}
for k in QTYPES:
    assert AL['k01'][k] <= AL['k05'][k] <= AL['cap20k_k10'][k] <= AL['k10'][k], k   # monotone -> nested prefixes


def perms(seed):
    rng = np.random.default_rng(seed)
    return {k: [int(x) for x in rng.permutation(np.array(sorted(by[k]), dtype=np.int64))] for k in QTYPES}


P = {s: perms(s) for s in (0, 1, 2)}
sub = {}
sub['k01_s0'] = sorted(x for k in QTYPES for x in P[0][k][:AL['k01'][k]])
sub['k05_s0'] = sorted(x for k in QTYPES for x in P[0][k][:AL['k05'][k]])
sub['k10_s0'] = sorted(x for k in QTYPES for x in P[0][k][:AL['k10'][k]])
sub['k01_s1'] = sorted(x for k in QTYPES for x in P[1][k][:AL['k01'][k]])
sub['k01_s2'] = sorted(x for k in QTYPES for x in P[2][k][:AL['k01'][k]])
sub['k100'] = sorted(qd for qd, _ in tr)
assert set(sub['k01_s0']) < set(sub['k05_s0']) < set(sub['k10_s0']) < set(sub['k100'])

# ---- LLM training lists
r60 = [int(x) for x in json.load(open(R60))]
assert len(r60) == 60001 and len(set(r60)) == 60001 and set(r60) <= set(sub['k100'])
s100 = [int(x) for x in json.load(open(S100))]
assert len(s100) == 99999 and not (set(s100) & set(r60))
llm = {'k01_s0': sub['k01_s0'], 'k05_s0': sub['k05_s0'], 'k10_s0': sub['k10_s0'], 'k01_s1': sub['k01_s1'],
       'k01_s2': sub['k01_s2'], 'k100': sorted(r60)}
# fallback-3 cap lists (used only if fallback 3 triggers)
cap = {}
cap['cap20k_k10'] = sorted(x for k in QTYPES for x in P[0][k][:AL['cap20k_k10'][k]])
assert set(sub['k05_s0']) <= set(cap['cap20k_k10']) <= set(sub['k10_s0'])
by60 = collections.defaultdict(list)
for x in r60:
    by60[qt_of[x]].append(x)
al60 = lr_alloc({k: len(by60[k]) for k in QTYPES}, 20000)
rng = np.random.default_rng(0)
c100 = []
for k in QTYPES:
    v = sorted(by60[k]); c100 += [int(x) for x in rng.choice(v, al60[k], replace=False)]
cap['cap20k_k100'] = sorted(c100)
assert len(cap['cap20k_k100']) == 20000 and set(cap['cap20k_k100']) <= set(r60)

# ---- dev selection subset: dev_sel recipe (largest remainder, default_rng(0).choice over sorted quids per qtype)
byd = collections.defaultdict(list)
for qd, qt in dv:
    byd[qt].append(qd)
ald = lr_alloc({k: len(byd[k]) for k in sorted(byd)}, 2000)
rng = np.random.default_rng(0)
dsel = []
for k in sorted(byd):
    v = sorted(byd[k]); dsel += [int(x) for x in rng.choice(v, ald[k], replace=False)]
dsel = sorted(dsel)
assert len(dsel) == 2000 and len(set(dsel)) == 2000


# ---- 2-fold OOF split inside each k<100 subset (seed 0), with an inner 2-fold split of each fold
def oof(lst):
    rng = np.random.default_rng(0)
    qbuck = collections.defaultdict(list)
    for x in lst:
        qbuck[qt_of[x]].append(x)
    quarter = {}
    for k in QTYPES:
        m = sorted(qbuck[k]); p = rng.permutation(len(m))
        for j, i in enumerate(p):
            quarter[m[i]] = j % 4
    fold = {f: sorted(x for x, q in quarter.items() if q // 2 == f) for f in (0, 1)}
    inner = {f'{f}{g}': sorted(x for x, q in quarter.items() if q == 2 * f + g) for f in (0, 1) for g in (0, 1)}
    return fold, inner


OOF = {nm: oof(sub[nm]) for nm in ('k01_s0', 'k05_s0', 'k10_s0', 'k01_s1', 'k01_s2')}
for nm, (fold, inner) in OOF.items():
    assert sorted(fold[0] + fold[1]) == sub[nm] and not set(fold[0]) & set(fold[1])
    assert sorted(inner['00'] + inner['01']) == fold[0] and sorted(inner['10'] + inner['11']) == fold[1]

# ---- template profile (template keys only; no answers or model outputs)
K = pickle.load(open('kp/keys_kpct.pkl', 'rb'))
T = {sp: {r[0]: r[1] for r in K[sp]} for sp in K}
assert set(T['train']) == set(sub['k100']) and set(T['test']) == {q for q, _ in te}
te_q = [q for q, _ in te]; te_qt = {q: t for q, t in te}
dv_qt = {q: t for q, t in dv}


def prof(lst):
    s = set(T['train'][x] for x in lst)
    qc = collections.Counter(qt_of[x] for x in lst)
    te_un = [q for q in te_q if T['test'][q] not in s]
    dv_un = [q for q in dsel if T['dev'][q] not in s]
    return dict(n=len(lst), qtype_counts={k: qc.get(k, 0) for k in QTYPES}, distinct_masked_templates=len(s),
                dev_sel2000_unseen_template_n=len(dv_un), dev_sel2000_unseen_template_share=round(len(dv_un) / len(dsel), 4),
                test_unseen_template_n=len(te_un), test_unseen_template_share=round(len(te_un) / len(te_q), 4),
                test_unseen_template_by_qtype={k: sum(te_qt[q] == k for q in te_un) for k in QTYPES},
                test_distinct_templates_unseen=len({T['test'][q] for q in te_q} - s)), te_un


PROF, UNSEEN = {}, {}
for nm, lst in list(sub.items()) + [('llm_k100_reader60k', llm['k100']), ('s_train_k100', s100),
                                     ('cap20k_k10', cap['cap20k_k10']), ('cap20k_k100', cap['cap20k_k100'])]:
    PROF[nm], UNSEEN[nm] = prof(lst)
assert PROF['k100']['test_unseen_template_n'] == 11175 and PROF['k100']['test_distinct_templates_unseen'] == 9922
