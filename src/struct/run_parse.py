"""Pass 2: link every question of train/dev/test with the frozen linker; silver labels on train/dev."""
import json, pickle, time, sys, os, multiprocessing as mp
import mtq_lib as L, mtq_align as A

def _init():
    global KG, LK
    KG = L.KG()
    block = json.load(open('ent_block.json')) if os.path.exists('ent_block.json') else None
    LK = L.Linker(KG, alias=json.load(open('alias_rules.json')), ent_block=block)
    A.init(KG, LK)

def pack(ms, ws):
    return [(m['lo'], m['hi'], [(a['e'], round(a['eff'], 3), int(bool(a.get('alias'))), int(a['full'])) for a in m['alts']]) for m in ms]

def work(args):
    split, chunk = args
    out = []
    for q in chunk:
        if split == 'test':
            (ms, ws), tp = LK.link_question(q['question'])
            rec = dict(quid=q['quid'], ments=pack(ms, ws), ws=ws, tp=tp[:2] if tp else None)
        else:
            s, ws, ms = A.silver(q)
            tp = s['tp']
            rec = dict(quid=q['quid'], ments=pack(ms, ws), ws=ws, tp=tp[:2] if tp else None,
                       topic=s['res']['topic'], partner=s['res']['partner'],
                       topic_linked=s['res']['topic_linked'], partner_linked=s['res']['partner_linked'],
                       rel_labels=s['rel_labels'], strong=s['strong'], fl=s['fl'], ba=s['ba'])
        out.append(rec)
    return out

if __name__ == '__main__':
    mp.set_start_method('fork')
    _init()
    tag = sys.argv[1] if len(sys.argv) > 1 else 'v1'
    for split in ['train', 'dev', 'test']:
        qs = json.load(open(f'{L.DATA}/questions/{split}.json'))
        chunks = [(split, qs[i:i + 2000]) for i in range(0, len(qs), 2000)]
        t0 = time.time(); res = []
        with mp.Pool(10) as pool:
            for r in pool.imap(work, chunks):
                res.extend(r)
        assert [r['quid'] for r in res] == [q['quid'] for q in qs]
        pickle.dump(res, open(f'parse_{split}_{tag}.pkl', 'wb'))
        print(split, len(res), round(time.time() - t0), flush=True)
