#!/usr/bin/env python3
"""r3_combine.py -- rule R3: build the LLM-arm input root from the structural rebuilds of the R3 root and the DEV root.
Keeps only the REWRITE rows (synthetic quid % 10 in --vidx: 4 = R3 rewrite, 5 = dev paraphrase); ORIG rows of the LLM
arms are the frozen predictions (rule R3, arms.orig_reference). Copies rows verbatim (no recomputation):
  <out>/inputs/variants.jsonl, <out>/inputs/variants_questions.json
  <out>/bundles/<k>/variants.jsonl, <out>/preds/<k>/variants_{RULE,S_seed0,S_seed1,S_seed2,S_mean}.jsonl
and writes <out>/checks/combine.json (row counts, sha256 of every output). Never reads an answer.
"""
import argparse, hashlib, json, os


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def rows(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def write(p, rs):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p + '.tmp', 'w') as f:
        for r in rs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    os.replace(p + '.tmp', p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roots', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--vidx', default='4,5'); ap.add_argument('--ks', default='k01_s0,k05_s0,k100')
    a = ap.parse_args()
    keep = {int(x) for x in a.vidx.split(',')}
    roots = a.roots.split(','); ks = a.ks.split(',')
    rec = dict(roots=roots, vidx=sorted(keep), files={}, counts={})
    sel = lambda rs: sorted([r for r in rs if int(r['quid']) % 10 in keep], key=lambda r: int(r['quid']))
    inp = sel([r for R in roots for r in rows(f'{R}/inputs/variants.jsonl')])
    assert len({int(r['quid']) for r in inp}) == len(inp), 'duplicate synthetic quid'
    write(f'{a.out}/inputs/variants.jsonl', inp)
    json.dump([{'quid': r['quid'], 'question': r['question']} for r in inp], open(f'{a.out}/inputs/variants_questions.json', 'w'), ensure_ascii=False)
    rec['counts']['inputs'] = len(inp)
    want = [int(r['quid']) for r in inp]
    for k in ks:
        b = sel([r for R in roots for r in rows(f'{R}/bundles/{k}/variants.jsonl')])
        assert [int(r['quid']) for r in b] == want, f'bundle rows of {k} do not match the inputs'
        write(f'{a.out}/bundles/{k}/variants.jsonl', b)
        for arm in ('RULE', 'S_seed0', 'S_seed1', 'S_seed2', 'S_mean'):
            p = sel([r for R in roots for r in rows(f'{R}/preds/{k}/variants_{arm}.jsonl')])
            assert [int(r['quid']) for r in p] == want, f'{k} {arm} rows do not match the inputs'
            write(f'{a.out}/preds/{k}/variants_{arm}.jsonl', p)
        rec['counts'][k] = len(b)
    for dp, _, fs in os.walk(a.out):
        for f in fs:
            p = os.path.join(dp, f)
            if '/checks/' not in p and not f.endswith('.tmp'):
                rec['files'][os.path.relpath(p, a.out)] = sha(p)
    os.makedirs(f'{a.out}/checks', exist_ok=True)
    json.dump(rec, open(f'{a.out}/checks/combine.json', 'w'), indent=1)
    print(json.dumps(rec['counts']), flush=True)


if __name__ == '__main__':
    main()
