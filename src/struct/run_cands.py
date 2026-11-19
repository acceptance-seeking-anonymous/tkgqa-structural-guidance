import json, pickle, sys, time, numpy as np, multiprocessing as mp
import mtq_lib as L, mtq_cand as MC
G = {}
def work(rng_):
    a, b = rng_
    out = []
    for i in range(a, b):
        r = G['P'][i]
        rec = {k: r[k] for k in ('quid', 'ments', 'ws', 'tp')}   # no silver / gold fields
        pr = {k: v[i] for k, v in G['probs'].items()}
        o = MC.build(rec, pr, G['Q'][i]['question'])
        if G['cap']:
            o['names'] = o['names'][:G['cap']]; o['X'] = o['X'][:G['cap']]; o['rinfo'] = o['rinfo'][:G['cap']]; o['topics'] = o['topics'][:G['cap']]
        o['quid'] = r['quid']
        out.append(o)
    return out
if __name__ == '__main__':
    mp.set_start_method('fork')
    split, probs_file, idx_file, cap, outf = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    kg = L.KG(); MC.init(kg)
    Qs = json.load(open(f'{L.DATA}/questions/{split}.json'))
    Ps = pickle.load(open(f'parse_{split}_v3.pkl', 'rb'))
    pz = np.load(probs_file)
    probs = {k: pz[k] for k in pz.files}
    if idx_file != '-':
        keep = set(json.load(open(idx_file)))
        ids = [i for i, q in enumerate(Qs) if q['quid'] in keep]
    else:
        ids = list(range(len(Qs)))
    G['Q'] = [Qs[i] for i in ids]; G['P'] = [Ps[i] for i in ids]; G['probs'] = {k: v[ids] for k, v in probs.items()}; G['cap'] = cap
    n = len(ids); t0 = time.time()
    chunks = [(a, min(n, a + 1000)) for a in range(0, n, 1000)]
    res = []
    with mp.Pool(10) as pool:
        for r in pool.imap(work, chunks):
            res.extend(r)
    pickle.dump(res, open(outf, 'wb'), protocol=5)
    print(split, n, round(time.time() - t0), flush=True)
