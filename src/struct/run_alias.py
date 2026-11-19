import json, pickle, collections, time, sys, multiprocessing as mp
import mtq_lib as L, mtq_align as A

KG = None


def _init():
    global KG
    KG = L.KG()
    import os; lk = L.Linker(KG, alias=json.load(open("alias_rules.json")) if os.environ.get("USE_RULES") else None)
    A.init(KG, lk)


def work(chunk):
    den = collections.Counter(); num = collections.Counter(); stats = collections.Counter()
    for q in chunk:
        res, d, n = A.alias_evidence(q)
        den.update(d); num.update(n)
        stats['n'] += 1
        stats['topic'] += res['topic'] is not None
        stats['topic_unlinked'] += res['topic_linked'] is False
        stats['partner'] += res['partner'] is not None
        stats['partner_unlinked'] += res['partner_linked'] is False
    return den, num, stats


if __name__ == '__main__':
    mp.set_start_method('fork')
    _init()
    tr = json.load(open(f'{L.DATA}/questions/train.json'))
    chunks = [tr[i:i + 2000] for i in range(0, len(tr), 2000)]
    t0 = time.time()
    den = collections.Counter(); num = collections.Counter(); stats = collections.Counter()
    with mp.Pool(10) as pool:
        for k, (d, n, s) in enumerate(pool.imap_unordered(work, chunks)):
            den.update(d); num.update(n); stats.update(s)
            if k % 40 == 0:
                print(k, len(chunks), round(time.time() - t0), flush=True)
    import os; pickle.dump(dict(den=den, num=num, stats=stats), open('alias_counts_rules.pkl' if os.environ.get('USE_RULES') else 'alias_counts.pkl', 'wb'))
    print(dict(stats), round(time.time() - t0))
