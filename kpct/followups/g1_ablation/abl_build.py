#!/usr/bin/env python3
"""abl_build.py -- G_1 ablation (multitq_g1_ablation_rule.json): rebuild k01_s0 guidance bundles for dev questions.

Recipe = robustness/code/rob_build.py (= kpct_run.stage_infer for kind 'test'), frozen k01_s0 executor models:
linker.json -> K.link_rec ; K.parser_probs (parser.pt, one batch over the file) ; K.MC.parse_info ; K.build_info ;
S_seed0-2 ; K.make_ranked with FB.json ; bundle = parse_json + K.bundle_core (S seed-0 order).
Input: {quid, question} rows (question text only; no answer is read).
Identity gate S2 on the dev_sel2000 quids: bundle fields, rendered user turns of every non-GU variant, S seed-0 lists.
CPU only (no GPU requested).
"""
import os, sys, json, time, argparse, hashlib
os.environ.setdefault('OMP_NUM_THREADS', '1')
EXEC = os.environ.get('KPCT_EXECUTOR', '/kpct/executor')
os.environ.setdefault('MTQ_CODE', os.path.join(EXEC, 'struct'))
os.environ.setdefault('MTQ_ROOT', os.path.join(EXEC, 'mtqdata'))
sys.path.insert(0, EXEC)
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch  # noqa: F401  (torch's libgomp before lightgbm)
import kpct_core as K

_G = {}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def w_link(items):
    return [K.link_rec(_G['LK'], q, t) for q, t in items]


def _bst():
    if 'bst' not in _G:
        import lightgbm as lgb
        _G['bst'] = [lgb.Booster(model_file=os.path.join(_G['mdir'], f'S_seed{s}.txt')) for s in range(3)]
    return _G['bst']


def w_infer(idx):
    bst = _bst()
    builds = []
    for i in idx:
        rec = _G['recs'][i]
        info = K.MC.parse_info(rec, {k: v[i] for k, v in _G['probs'].items()}, _G['qtext'][i])
        builds.append(K.build_info(rec, info))
    sc = K.predict_chunk(bst, builds)
    out = []
    FB = _G['FB']
    for n, (i, o) in enumerate(zip(idx, builds)):
        rec = _G['recs'][i]
        res = {'i': i, 'quid': rec['quid']}
        res['S0'] = K.make_ranked([o['names'][j] for j in K.s_order(sc[0][n])], o, FB)
        try:
            pj = K.parse_json(rec, o['info'])
            core = K.bundle_core(o, rec, sc[0][n])
            res['bundle'] = dict(parse=pj, **core)
        except Exception as e:                 # frozen builder raises on this question: excluded (listed)
            res['bundle'] = None
            res['error'] = f'{type(e).__name__}: {e}'
        out.append(res)
    return out


def read_jsonl(p):
    with open(p) as f:
        for l in f:
            if l.strip():
                yield json.loads(l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', default='k01_s0')
    ap.add_argument('--questions', required=True, help='jsonl {quid, question}')
    ap.add_argument('--sel_quids', required=True, help='json list: dev_sel2000 quids (identity gate)')
    ap.add_argument('--frozen_bundles', required=True)
    ap.add_argument('--frozen_s0', required=True)
    ap.add_argument('--root', required=True, help='output dir (persistent)')
    ap.add_argument('--out', default='.', help='small check files (job cwd)')
    ap.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()
    t0 = time.time()
    k = a.k
    mdir = os.path.join(EXEC, 'models', k)
    os.makedirs(a.root, exist_ok=True)
    rows = sorted(read_jsonl(a.questions), key=lambda r: int(r['quid']))
    assert all(set(r) == {'quid', 'question'} for r in rows), 'question text only'
    K.kg()
    lk = json.load(open(os.path.join(mdir, 'linker.json')))
    _G['LK'] = K.make_linker(lk['alias_rules'], lk['ent_block'])
    _G['mdir'] = mdir
    _G['FB'] = json.load(open(os.path.join(mdir, 'FB.json')))
    import multiprocessing as mp
    items = [(int(r['quid']), r['question']) for r in rows]
    chunks = [items[i:i + 200] for i in range(0, len(items), 200)]
    with mp.get_context('fork').Pool(a.workers) as pool:
        recs = [x for ch in pool.map(w_link, chunks) for x in ch]
    log(k, 'linked', len(recs), round(time.time() - t0), 's')
    voc, net = K.load_parser(os.path.join(mdir, 'parser.pt'))
    torch.set_num_threads(1)
    probs = K.parser_probs(voc, net, recs)
    _G.update(recs=recs, probs=probs, qtext=[r['question'] for r in rows])
    idx = list(range(len(recs)))
    ich = [idx[i:i + 50] for i in range(0, len(idx), 50)]
    res = [None] * len(recs)
    with mp.get_context('fork').Pool(a.workers) as pool:
        for out in pool.imap(w_infer, ich):
            for r in out:
                res[r['i']] = r
    log(k, 'built', len(res), round(time.time() - t0), 's')
    bf = os.path.join(a.root, f'dev_abl_{k}.jsonl')
    errors = [dict(quid=r['quid'], error=r['error']) for r in res if r['bundle'] is None]
    log('builder errors', len(errors))
    with open(bf + '.tmp', 'w') as f:
        for r, row in zip(res, rows):
            if r['bundle'] is None:
                continue
            f.write(json.dumps(dict(quid=r['quid'], question=row['question'], split='dev', k_subset=k,
                                    template_unseen=None, **r['bundle']), ensure_ascii=False) + '\n')
    os.replace(bf + '.tmp', bf)
    sf = os.path.join(a.root, f'dev_abl_{k}_S_seed0.jsonl')
    with open(sf + '.tmp', 'w') as f:
        for r in res:
            f.write(json.dumps({'quid': r['quid'], 'ranked': r['S0']}) + '\n')
    os.replace(sf + '.tmp', sf)
    # ------------------------------------------------------------- S2 identity gate on dev_sel2000
    import gk_common as GC, gk_ablation as AB
    sel = set(int(q) for q in json.load(open(a.sel_quids)))
    mine = {int(r['quid']): r for r in res if int(r['quid']) in sel and r['bundle'] is not None}
    frz = {int(x['quid']): x for x in read_jsonl(a.frozen_bundles)}
    fs0 = {}
    for x in read_jsonl(a.frozen_s0):
        rk = x['ranked']
        fs0[int(x['quid'])] = json.loads(rk.replace("'", '"')) if isinstance(rk, str) else rk
    fields = dict(candidates=0, facts=0, parse=0, trace=0, s0_list=0)
    var = {v: 0 for v in AB.VARIANTS if v != 'A_GUfacts'}
    mism = []
    for q in sorted(sel):
        if q not in mine:
            mism.append(dict(quid=q, variant='builder_error')); continue
        mb, fb = mine[q]['bundle'], frz[q]
        fields['candidates'] += mb['candidates'] == fb['candidates']
        fields['facts'] += mb['facts'] == fb['facts']
        fields['parse'] += {kk: v for kk, v in mb['parse'].items()} == fb['parse']
        fields['trace'] += mb['operator_trace'] == fb['operator_trace']
        fields['s0_list'] += mine[q]['S0'] == fs0[q]
        bm = GC.normalize_bundle(dict(quid=q, question=fb['question'], **mb))
        bz = GC.normalize_bundle(fb)
        for v in var:
            same = AB.ab_user(bm, v) == AB.ab_user(bz, v)
            var[v] += same
            if not same and len(mism) < 20:
                mism.append(dict(quid=q, variant=v))
    n = len(sel)
    ident = dict(k=k, n=n, builder_errors=dict(n=len(errors), rows=errors), field_identity={kk: vv / n for kk, vv in fields.items()},
                 rendered_user_turn_identity={v: c / n for v, c in var.items()}, mismatch_examples=mism,
                 S2_pass=bool(min(var.values()) / n >= 0.999 and fields['s0_list'] / n >= 0.999),
                 outputs=dict(bundles=dict(file=bf, n=len(res) - len(errors), sha256=sha256_file(bf)),
                              S_seed0=dict(file=sf, n=len(res), sha256=sha256_file(sf))),
                 inputs=dict(questions_sha256=sha256_file(a.questions), frozen_bundles_sha256=sha256_file(a.frozen_bundles),
                             frozen_s0_sha256=sha256_file(a.frozen_s0),
                             models={f: sha256_file(os.path.join(mdir, f)) for f in sorted(os.listdir(mdir))},
                             kpct_core_sha256=sha256_file(os.path.join(EXEC, 'kpct_core.py'))),
                 code_sha256={f: sha256_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), f))
                              for f in ('abl_build.py', 'gk_common.py', 'gk_ablation.py')},
                 secs=round(time.time() - t0, 1))
    json.dump(ident, open(os.path.join(a.out, 'abl_build_identity.json'), 'w'), indent=1)
    log(json.dumps({kk: ident[kk] for kk in ('field_identity', 'rendered_user_turn_identity', 'S2_pass', 'secs')}))


if __name__ == '__main__':
    main()
