#!/usr/bin/env python3
"""rob_build.py -- MultiTQ robustness probe, structural rebuild on perturbed question text (no training, no gold).

For one k subset, every variant row {quid: synthetic quid, question} of inputs/variants.jsonl goes through the frozen
structural pipeline exactly as kpct_run.stage_infer (kind 'test') / kpct_k100.py did for the frozen test files:
  linker (executor/models/<k>/linker.json) -> K.link_rec ; classifier probabilities K.parser_probs (models/<k>/parser.pt,
  one batch over the file) ; K.MC.parse_info ; K.build_info ; S_seed0-2 (models/<k>/S_seed*.txt) ; K.make_ranked with
  models/<k>/FB.json ; bundle = parse_json + K.bundle_core (S seed-0 order).
Outputs (under --root):
  bundles/<k>/variants.jsonl                   bundle rows (frozen test-bundle schema; quid = synthetic quid)
  preds/<k>/variants_{RULE,S_seed0,S_seed1,S_seed2,S_mean}.jsonl   {"quid", "ranked"}
  checks/I1_<k>.json                           identity of the ORIG rows vs the frozen test predictions and bundles
  checks/frozen_bundle_fields_<k>.json          frozen test-bundle fields of the sampled quids (template_unseen, ...)
Never reads an answer.
"""
import os, sys, json, time, argparse, hashlib
os.environ.setdefault('OMP_NUM_THREADS', '1')
EXEC = os.environ.get('KPCT_EXECUTOR', '/kpct/executor')
os.environ.setdefault('MTQ_CODE', os.path.join(EXEC, 'struct'))
os.environ.setdefault('MTQ_ROOT', os.path.join(EXEC, 'mtqdata'))
sys.path.insert(0, EXEC)
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
    """= kpct_run.w_infer for kind 'test'"""
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
        res['rule'] = K.make_ranked(o['names'], o, FB)
        res['S'] = [K.make_ranked([o['names'][j] for j in K.s_order(sc[b][n])], o, FB) for b in range(len(bst))]
        mean = (sc[0][n] + sc[1][n] + sc[2][n]) / 3.0
        res['S_mean'] = K.make_ranked([o['names'][j] for j in K.s_order(mean)], o, FB)
        pj = K.parse_json(rec, o['info'])
        core = K.bundle_core(o, rec, sc[0][n])
        res['bundle'] = dict(parse=pj, **core)
        out.append(res)
    return out


def read_jsonl(p):
    with open(p) as f:
        for l in f:
            if l.strip():
                yield json.loads(l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', required=True)
    ap.add_argument('--root', default='/kpct/robustness')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--frozen_preds', required=True, help='dir with test_{RULE,S_seed0,S_seed1,S_seed2}.jsonl of this k')
    ap.add_argument('--frozen_bundles', required=True, help='frozen bundles/<k>/test.jsonl')
    a = ap.parse_args()
    t0 = time.time()
    R = a.root
    k = a.k
    mdir = os.path.join(EXEC, 'models', k)
    for d in (f'{R}/bundles/{k}', f'{R}/preds/{k}', f'{R}/checks'):
        os.makedirs(d, exist_ok=True)
    rows = list(read_jsonl(f'{R}/inputs/variants.jsonl'))
    rows.sort(key=lambda r: r['quid'])
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
    bf = f'{R}/bundles/{k}/variants.jsonl'
    with open(bf + '.tmp', 'w') as f:
        for r, row in zip(res, rows):
            f.write(json.dumps(dict(quid=r['quid'], question=row['question'], split='test', k_subset=k,
                                    template_unseen=None, **r['bundle']), ensure_ascii=False) + '\n')
    os.replace(bf + '.tmp', bf)
    pf = {}
    for arm, get in (('RULE', lambda r: r['rule']), ('S_seed0', lambda r: r['S'][0]), ('S_seed1', lambda r: r['S'][1]),
                     ('S_seed2', lambda r: r['S'][2]), ('S_mean', lambda r: r['S_mean'])):
        p = f'{R}/preds/{k}/variants_{arm}.jsonl'
        with open(p + '.tmp', 'w') as f:
            for r in res:
                f.write(json.dumps({'quid': r['quid'], 'ranked': get(r)}) + '\n')
        os.replace(p + '.tmp', p)
        pf[arm] = dict(file=os.path.relpath(p, R), sha256=sha256_file(p))
    # ---------------------------------------------------------------- I1: ORIG rows vs frozen files (no gold)
    orig = {r['quid'] // 10: r for r in res if r['quid'] % 10 == 0}
    need = set(orig)
    I1 = dict(k=k, n=len(orig), arms={})
    for arm, ix in (('RULE', None), ('S_seed0', 0), ('S_seed1', 1), ('S_seed2', 2)):
        fp = os.path.join(a.frozen_preds, f'test_{arm}.jsonl')
        fz = {}
        for x in read_jsonl(fp):
            q = int(x['quid'])
            if q in need:
                rk = x['ranked']
                fz[q] = json.loads(rk.replace("'", '"')) if isinstance(rk, str) else rk
        mine = {q: (orig[q]['rule'] if ix is None else orig[q]['S'][ix]) for q in need}
        I1['arms'][arm] = dict(frozen_file=fp, frozen_sha256=sha256_file(fp), n_found=len(fz),
                               top1=sum(mine[q][:1] == fz.get(q, [None])[:1] for q in need) / len(need),
                               top10=sum(mine[q][:10] == fz.get(q, [])[:10] for q in need) / len(need),
                               full=sum(mine[q] == fz.get(q) for q in need) / len(need))
    fb_fields = {}
    cmp = dict(candidates=0, facts=0, relations=0, mentions=0, trace=0, qtype=0, n=0)
    for x in read_jsonl(a.frozen_bundles):
        q = int(x['quid'])
        if q not in need:
            continue
        mb = orig[q]['bundle']
        cmp['n'] += 1
        cmp['candidates'] += [c['answer'] for c in x['candidates']] == [c['answer'] for c in mb['candidates']]
        cmp['facts'] += [f['fid'] for f in x['facts']] == [f['fid'] for f in mb['facts']]
        cmp['relations'] += [(r['relation'], r['direction']) for r in x['parse']['relations']] == [(r['relation'], r['direction']) for r in mb['parse']['relations']]
        cmp['mentions'] += [[al['entity'] for al in m['alternatives']] for m in x['parse']['mentions']] == [[al['entity'] for al in m['alternatives']] for m in mb['parse']['mentions']]
        cmp['trace'] += x['operator_trace'] == mb['operator_trace']
        cmp['qtype'] += x['parse']['qtype']['label'] == mb['parse']['qtype']['label']
        fb_fields[q] = dict(template_unseen=x.get('template_unseen'), pool_size=x.get('pool_size'),
                            cand_answers=[c['answer'] for c in x['candidates']])
    n = max(1, cmp['n'])
    I1['bundles'] = dict(frozen_file=a.frozen_bundles, n_found=cmp['n'], **{f'{kk}_identical': cmp[kk] / n for kk in cmp if kk != 'n'})
    I1['outputs'] = dict(bundles=dict(file=os.path.relpath(bf, R), sha256=sha256_file(bf)), preds=pf)
    I1['secs'] = round(time.time() - t0, 1)
    json.dump(I1, open(f'{R}/checks/I1_{k}.json', 'w'), indent=1)
    json.dump(fb_fields, open(f'{R}/checks/frozen_bundle_fields_{k}.json', 'w'))
    log(k, json.dumps({kk: v['top1'] for kk, v in I1['arms'].items()}), json.dumps(I1['bundles']))


if __name__ == '__main__':
    main()
