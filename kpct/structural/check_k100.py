"""Reproduce frozen RULE / S dev predictions on dev_sel2000 from the frozen artefacts (validates the inference path
and the parse-JSON bridge)."""
import os, sys, json, time, multiprocessing as mp
import numpy as np
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kpct_core as K
import lightgbm as lgb
G = {}

def init():
    K.kg()
    G['LK'] = K.make_linker(json.load(open(f'{K.CODE}/alias_rules.json')), json.load(open(f'{K.CODE}/ent_block.json')))
    G['voc'], G['net'] = K.load_frozen_parser()
    G['B'] = [lgb.Booster(model_file=f'{K.CODE}/S_seed{s}.txt') for s in range(3)]
    G['FB'] = json.load(open('kp/out/FB_k100.json'))

def work(qs):
    import torch; torch.set_num_threads(1)
    recs = [K.link_rec(G['LK'], q['quid'], q['question']) for q in qs]
    pr = K.parser_probs(G['voc'], G['net'], recs)
    outs, eq = [], 0
    for i, (q, r) in enumerate(zip(qs, recs)):
        p = {k: v[i] for k, v in pr.items()}
        o = K.MC.build(r, p, q['question'])
        pj = K.parse_json(r, o['info'])
        r2, info2 = K.rec_info_from_parse(json.loads(json.dumps(pj)))
        o2 = K.build_info(r2, info2)
        same = (o['names'] == o2['names'] and np.array_equal(o['X'], o2['X']) and o['facts'] == o2['facts'] and o['rinfo'] == o2['rinfo'])
        eq += same
        outs.append(o)
    sc = K.predict_chunk(G['B'], outs)
    res = []
    for i, (q, o) in enumerate(zip(qs, outs)):
        rule = K.make_ranked(o['names'], o, G['FB'])
        S = [K.make_ranked([o['names'][j] for j in K.s_order(sc[b][i])], o, G['FB']) for b in range(3)]
        res.append((q['quid'], rule, S))
    return res, eq

if __name__ == '__main__':
    mp.set_start_method('fork')
    sub = json.load(open('kp/subsets/multitq_kpct_subsets.json'))
    dsel = set(sub['dev_selection']['quids'])
    tr = K.load_questions('train')
    json.dump(K.build_FB(tr), open('kp/out/FB_k100.json', 'w')); del tr
    dev = [q for q in K.load_questions('dev') if q['quid'] in dsel]
    init()
    t0 = time.time()
    chunks = [dev[i:i + 100] for i in range(0, len(dev), 100)]
    with mp.Pool(8) as pool:
        out = pool.map(work, chunks)
    res = [x for r, _ in out for x in r]; eq = sum(e for _, e in out)
    print('build+bridge identical', eq, len(dev), 'secs', round(time.time() - t0, 1))
    frozen = {}
    for nm in ('RULE', 'S_seed0', 'S_seed1', 'S_seed2'):
        frozen[nm] = {}
        for line in open(f'exp/preds/dev_{nm}.jsonl'):
            d = json.loads(line)
            if d['quid'] in dsel:
                frozen[nm][d['quid']] = d['ranked']
    m = {'RULE': [0, 0], 'S_seed0': [0, 0], 'S_seed1': [0, 0], 'S_seed2': [0, 0]}
    for quid, rule, S in res:
        for nm, mine in (('RULE', rule), ('S_seed0', S[0]), ('S_seed1', S[1]), ('S_seed2', S[2])):
            f = frozen[nm][quid]
            m[nm][0] += mine == f; m[nm][1] += mine[:1] == f[:1]
    print({k: (v[0], v[1]) for k, v in m.items()})
