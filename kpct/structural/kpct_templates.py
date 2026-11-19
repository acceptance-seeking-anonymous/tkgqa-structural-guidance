"""Template keys (kpct_keys.py definition, frozen v3 linker) for all train, dev_sel2000 and test
questions; question TEXT only. Verifies the registered profile counts."""
import os, sys, json, pickle, time, multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import kpct_core as K
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
        out.append((quid, ' '.join(t for t in toks if t is not None)))
    return out

if __name__ == '__main__':
    out = os.environ.get('KPCT_OUT', os.path.join(HERE, 'out'))
    sub = json.load(open(os.path.join(HERE, 'subsets', 'multitq_kpct_subsets.json')))
    prof = json.load(open(os.path.join(HERE, 'subsets', 'multitq_kpct_subset_profile.json')))
    K.kg()
    G['LK'] = K.make_linker(json.load(open(f'{K.CODE}/alias_rules.json')), json.load(open(f'{K.CODE}/ent_block.json')))
    dsel = set(sub['dev_selection']['quids'])
    res = {}
    for split in ('train', 'dev', 'test'):
        qs = K.load_questions(split)
        items = [(q['quid'], q['question']) for q in qs if split != 'dev' or q['quid'] in dsel]
        del qs
        chunks = [items[i:i + 2000] for i in range(0, len(items), 2000)]
        with mp.get_context('fork').Pool(int(os.environ.get('KPCT_NPROC', '8'))) as pool:
            rows = [x for r in pool.map(work, chunks) for x in r]
        res['dev_sel2000' if split == 'dev' else split] = dict(rows)
        print(split, len(rows), flush=True)
    chk = {}
    for nm in ('k01_s0', 'k05_s0', 'k10_s0', 'k01_s1', 'k01_s2', 'k100'):
        keep = set(sub['subsets'][nm]['quids']) if nm != 'k100' else None
        seen = {t for q, t in res['train'].items() if keep is None or q in keep}
        du = sum(t not in seen for t in res['dev_sel2000'].values())
        tu = sum(t not in seen for t in res['test'].values())
        p = prof['profiles'][nm]
        chk[nm] = dict(distinct=len(seen), dev_unseen=du, test_unseen=tu,
                       match=(len(seen) == p['distinct_masked_templates'] and du == p['dev_sel2000_unseen_template_n'] and tu == p['test_unseen_template_n']))
    print(json.dumps(chk))
    assert all(v['match'] for v in chk.values()), 'template keys do not reproduce the registered profile'
    pickle.dump(res, open(os.path.join(out, 'template_keys.pkl'), 'wb'))
    json.dump(chk, open(os.path.join(out, 'template_keys_check.json'), 'w'), indent=1)
