"""kpct_keys.py -- masked-template keys for MultiTQ train/dev/test questions (text only).
Definition:
template = frozen-linker tokens with each linked mention span -> 'ENT' and the parsed time expression -> 'timex'
(link_question blanks the time expression to ' timex ' before linking). Frozen v3 linker artefacts
(alias_rules.json, ent_block.json) are used for every split so the stratum key is identical across k.
"""
import os, sys, json, pickle, time, multiprocessing as mp
CODE = os.path.abspath('struct')
sys.path.insert(0, CODE)
import mtq_lib as L
ROOT = os.path.abspath(os.environ.get('DATA_DIR', 'data/MultiTQ_94ad61c/data/MultiTQ'))
G = {}

def work(chunk):
    out = []
    for quid, qtext in chunk:
        (ms, ws), tp = G['LK'].link_question(qtext)
        toks = list(ws)
        for m in ms:
            for j in range(m['lo'], m['hi'] + 1):
                toks[j] = None
            toks[m['lo']] = 'ENT'
        tmpl = ' '.join(t for t in toks if t is not None)
        ents = tuple(m['alts'][0]['e'] for m in ms)
        exact = tuple(int(m['alts'][0]['full'] and not m['alts'][0].get('alias')) for m in ms)
        alias = tuple(int(bool(m['alts'][0].get('alias'))) for m in ms)
        out.append((quid, tmpl, ents, tp[1] if tp else None, exact, alias))
    return out

if __name__ == '__main__':
    mp.set_start_method('fork')
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else None
    kg = L.KG(root=ROOT)
    G['LK'] = L.Linker(kg, alias=json.load(open(f'{CODE}/alias_rules.json')), ent_block=json.load(open(f'{CODE}/ent_block.json')))
    res = {}
    for split in ('train', 'dev', 'test'):
        qs = json.load(open(f'{ROOT}/questions/{split}.json'))
        items = [(q['quid'], q['question']) for q in qs][:nmax]   # question text only
        del qs
        chunks = [items[i:i + 2000] for i in range(0, len(items), 2000)]
        rows = []; t0 = time.time()
        with mp.Pool(9) as pool:
            for r in pool.imap(work, chunks):
                rows.extend(r)
        res[split] = rows
        print(split, len(rows), round(time.time() - t0, 1), 's', flush=True)
    if nmax is None:
        pickle.dump(res, open('kp/keys_kpct.pkl', 'wb'))
