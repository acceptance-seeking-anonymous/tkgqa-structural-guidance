"""pf_lib.py -- Parser-first track (rule multitq_parser_first_rule.json v1.0): gold-free loading of the cached Parser
replay outputs and routing of every registered variant (V0 frozen P_k; V1 PALONE; V2 PALONE else PK; V3(t) PK when
mean_logprob < t else V2). Every variant maps each question to the PALONE_fb or the PK_fb list (same seed).
No gold is read in this module."""
import gzip, hashlib, json, os
import numpy as np

W = os.environ.get('PF_WS', os.path.abspath('ws'))
KPCT_PRED = os.environ.get('PF_KPCT_PRED', os.path.join(W, 'kpct_predictions'))   # local copy of the k% predictions
# archive (parser/dev_sel2000/<k>/, parser/test/<k>/: raw Parser generations and frozen P_k predictions)
KS = ['k01_s0', 'k05_s0', 'k10_s0', 'k100']
KREW = ['k01_s0', 'k05_s0', 'k100']
LEV = {'k01_s0': '1', 'k05_s0': '5', 'k10_s0': '10', 'k100': '100'}
CKPT = {'k01_s0': 'ep3', 'k05_s0': 'ep1', 'k10_s0': 'ep0.5', 'k100': 'ep0.5'}
TGRID = [-0.30, -0.10, -0.03, -0.01, -0.003]
CANDS = ['V0', 'V1', 'V2'] + [f'V3({t})' for t in TGRID]     # registered tie order


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read_ranked(p):
    d = {}
    op = gzip.open if p.endswith('.gz') else open
    with op(p, 'rt') as f:
        for l in f:
            if l.strip():
                x = json.loads(l); d[int(x['quid'])] = x['ranked']
    return d


def read_raw_mlp(p):
    out = {}
    with gzip.open(p, 'rt') as f:
        for l in f:
            x = json.loads(l)
            v = x.get('mean_logprob')
            out[int(x['quid'])] = float(v) if v not in (None, '') else float('-inf')
    return out


def read_pool_status(p):
    st = {}
    with gzip.open(p, 'rt') as f:
        for l in f:
            x = json.loads(l); st[int(x['quid'])] = (x['status'], x['reason'])
    return st


def npz_status(p, mode):
    z = np.load(p)
    meta = json.loads(bytes(z[f'{mode}_meta']).decode())
    return {int(r['quid']): (r['status'], r['reason']) for r in meta}


# ------------------------------------------------------------------ sources per (split, k)
def sources(split, k):
    """returns dict(PK=[3 dicts], PALONE=[3 dicts], st_PK, st_PALONE, mlp, V0=[3 dicts], files={role: path})"""
    S = dict(files={})
    rp = os.path.join(W, 'replay', 'out')
    saA = os.path.join(W, 'saA', 'multitq_structural_ablation_predictions_A')
    if split == 'dev':
        d = os.path.join(rp, f'dev_{k}')
        for m in ('PK', 'PALONE'):
            S['files'][f'{m}_pool'] = os.path.join(d, f'pool_{m}_{k}_dev_sel2000.jsonl.gz')
            for s in range(3):
                S['files'][f'{m}_s{s}'] = os.path.join(d, f'pred_{m}_{k}_dev_sel2000_fb_seed{s}.jsonl')
        S['files']['raw'] = os.path.join(KPCT_PRED, 'parser', 'dev_sel2000', k, f'raw_P_dev_sel2000_{CKPT[k]}.jsonl.gz')
        for s in range(3):
            S['files'][f'V0_s{s}'] = os.path.join(KPCT_PRED, 'parser', 'dev_sel2000', k, f'pred_P_{k}_dev_sel2000_{CKPT[k]}_tauoff_seed{s}.jsonl')
    elif split == 'var':      # R3 vidx 4 + DEV_ACCEPTED vidx 5 + DEVPARA_JREJ vidx 6
        d = os.path.join(rp, f'var_{k}')
        for m in ('PK', 'PALONE'):
            S['files'][f'{m}_pool'] = os.path.join(d, f'pool_{m}_{k}_variants.jsonl.gz')
            for s in range(3):
                S['files'][f'{m}_s{s}'] = os.path.join(d, f'pred_{m}_{k}_variants_fb_seed{s}.jsonl')
        S['files']['raw'] = os.path.join(W, 'r3pred', 'llm', 'P', 'raw', f'raw_P_variants_P_{k}.jsonl.gz')
        for s in range(3):
            S['files'][f'V0_s{s}'] = os.path.join(W, 'r3pred', 'llm', 'P', k, f'pred_P_{k}_variants_P_{k}_tauoff_seed{s}.jsonl')
    elif split in ('test', 'R1', 'R2'):
        d = os.path.join(saA, 'preds_x', f'A_{split}')
        for m in ('PK', 'PALONE'):
            for s in range(3):
                S['files'][f'{m}_s{s}'] = os.path.join(d, f'pred_{m}_{k}_{split}_fb_seed{s}.jsonl')
        S['files']['pools_npz'] = os.path.join(saA, 'pools', f'A_{split}_{k}_{split}.npz')
        if split == 'test':
            S['files']['raw'] = os.path.join(KPCT_PRED, 'parser', 'test', k, f'raw_P_test_{CKPT[k]}.jsonl.gz')
            for s in range(3):
                S['files'][f'V0_s{s}'] = os.path.join(KPCT_PRED, 'parser', 'test', k, f'pred_P_{k}_test_{CKPT[k]}_tauoff_seed{s}.jsonl.gz')
        else:
            S['files']['raw'] = os.path.join(W, f'{split.lower()}raw', 'P', 'raw', f'raw_P_variants_P_{k}.jsonl.gz')
            for s in range(3):   # frozen P_k rewrite files of the R1 / R2 archives
                S['files'][f'V0_s{s}'] = os.path.join(W, f'{split.lower()}raw', 'P', k, f'pred_P_{k}_variants_P_{k}_tauoff_seed{s}.jsonl')
    else:
        raise ValueError(split)
    return S


def load(split, k, with_v0=True):
    S = sources(split, k)
    F = S['files']
    out = dict(files=F)
    for m in ('PK', 'PALONE'):
        out[m] = [read_ranked(F[f'{m}_s{s}']) for s in range(3)]
    if 'pools_npz' in F:
        out['st_PK'] = npz_status(F['pools_npz'], 'PK'); out['st_PALONE'] = npz_status(F['pools_npz'], 'PALONE')
    else:
        out['st_PK'] = read_pool_status(F['PK_pool']); out['st_PALONE'] = read_pool_status(F['PALONE_pool'])
    out['mlp'] = read_raw_mlp(F['raw'])
    if with_v0 and 'V0_s0' in F:
        out['V0'] = [read_ranked(F[f'V0_s{s}']) for s in range(3)]
    else:
        out['V0'] = out['PK']
    q = sorted(out['PK'][0])
    assert set(out['PALONE'][0]) == set(q) and set(out['st_PK']) == set(q) and set(out['st_PALONE']) == set(q), split
    assert set(q) <= set(out['mlp']), f'{split} {k}: quids without raw'
    out['quids'] = q
    return out


def route(D, cand, q):
    """'PK' or 'PALONE' (V0 handled separately: frozen files)."""
    pal_ok = D['st_PALONE'][q][0] == 'ok'
    v2 = 'PALONE' if pal_ok else 'PK'
    if cand == 'V1':
        return 'PALONE'
    if cand == 'V2':
        return v2
    if cand.startswith('V3('):
        t = float(cand[3:-1])
        return 'PK' if D['mlp'][q] < t else v2
    raise ValueError(cand)


def lists(D, cand, quids=None):
    """per-seed ranked lists {quid: list} of a candidate"""
    quids = D['quids'] if quids is None else quids
    if cand == 'V0':
        return [{q: D['V0'][s][q] for q in quids} for s in range(3)]
    R = {q: route(D, cand, q) for q in quids}
    return [{q: D[R[q]][s][q] for q in quids} for s in range(3)]


def routing_share(D, cand, quids):
    if cand == 'V0':
        return dict(PK=1.0, PALONE=0.0)
    r = [route(D, cand, q) for q in quids]
    return dict(PK=float(np.mean([x == 'PK' for x in r])), PALONE=float(np.mean([x == 'PALONE' for x in r])))
