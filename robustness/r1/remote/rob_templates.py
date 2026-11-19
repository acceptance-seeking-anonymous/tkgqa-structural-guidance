#!/usr/bin/env python3
"""rob_templates.py -- masked-template keys (kpct_keys.py / kpct_templates.py definition: frozen v3 linker =
executor/models/k100/linker.json; linked mention spans -> 'ENT', time expression -> 'timex') for all train questions and
every probe variant; train template sets of the k lists k01_s0, k05_s0 (manifest quids) and k100 (all train).
Verifies the registered distinct-template counts (template_keys_check.json: 3140 / 11608 / 94889).
Question TEXT only. Output: checks/template_flags.json {squid: {tmpl, seen: {k: bool}}} + checks/template_check.json
"""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
EXEC = os.environ.get('KPCT_EXECUTOR', '/kpct/executor')
os.environ.setdefault('MTQ_CODE', os.path.join(EXEC, 'struct'))
os.environ.setdefault('MTQ_ROOT', os.path.join(EXEC, 'mtqdata'))
sys.path.insert(0, EXEC)
import kpct_core as K
G = {}
EXPECT = {'k01_s0': 3140, 'k05_s0': 11608, 'k100': 94889}


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


def main():
    R = sys.argv[1] if len(sys.argv) > 1 else '/kpct/robustness'
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    t0 = time.time()
    K.kg()
    lk = json.load(open(os.path.join(EXEC, 'models', 'k100', 'linker.json')))
    G['LK'] = K.make_linker(lk['alias_rules'], lk['ent_block'])
    sub = json.load(open('/kpct/guided/code/multitq_kpct_subsets.json'))
    import multiprocessing as mp
    tr = json.load(open('/data/questions/train.json'))
    items = [(q['quid'], q['question']) for q in tr]
    del tr
    var = [json.loads(l) for l in open(f'{R}/inputs/variants.jsonl')]
    vitems = [(int(r['quid']), r['question']) for r in var]
    with mp.get_context('fork').Pool(nproc) as pool:
        trk = dict(x for ch in pool.map(work, [items[i:i + 2000] for i in range(0, len(items), 2000)]) for x in ch)
        vk = dict(x for ch in pool.map(work, [vitems[i:i + 500] for i in range(0, len(vitems), 500)]) for x in ch)
    seen = {}
    for nm in EXPECT:
        keep = set(sub['subsets'][nm]['quids']) if nm != 'k100' else None
        seen[nm] = {t for q, t in trk.items() if keep is None or q in keep}
    chk = {nm: dict(distinct=len(seen[nm]), expected=EXPECT[nm], match=len(seen[nm]) == EXPECT[nm]) for nm in EXPECT}
    flags = {str(q): dict(tmpl=t, seen={nm: (t in seen[nm]) for nm in EXPECT}) for q, t in vk.items()}
    os.makedirs(f'{R}/checks', exist_ok=True)
    json.dump(flags, open(f'{R}/checks/template_flags.json', 'w'))
    json.dump(dict(check=chk, n_train=len(trk), n_variants=len(vk), secs=round(time.time() - t0, 1)), open(f'{R}/checks/template_check.json', 'w'), indent=1)
    print(json.dumps(chk), flush=True)


if __name__ == '__main__':
    main()
