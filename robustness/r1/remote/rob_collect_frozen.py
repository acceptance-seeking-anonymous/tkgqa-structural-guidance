#!/usr/bin/env python3
"""rob_collect_frozen.py -- verify sha256 of the frozen test prediction files, adapters and frozen code against the
freezes (expected_sha.json), and extract the rows of the 2,000 probe quids (top-10 lists; P exec statuses; G/GU raw
answers; frozen GU retrieval fact ids). Predictions only, never gold. Exit code 3 on any mismatch (STOP_TEST_INTEGRITY).
"""
import os, sys, json, gzip, hashlib, glob


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def rows(p):
    op = gzip.open if p.endswith('.gz') else open
    with op(p, 'rt') as f:
        for l in f:
            if l.strip():
                yield json.loads(l)


def main():
    R = sys.argv[1] if len(sys.argv) > 1 else '/kpct/robustness'
    E = json.load(open(f'{R}/code/expected_sha.json'))
    need = {int(json.loads(l)['quid']) // 10 for l in open(f'{R}/inputs/variants.jsonl')}
    ver = dict(pred_files={}, adapters={}, code={})
    bad = []
    for grp in ('pred_files', 'adapters', 'code'):
        for p, s in E[grp].items():
            got = sha256_file(p) if os.path.exists(p) else None
            ver[grp][p] = dict(expected=s, got=got, ok=(got == s))
            if got != s:
                bad.append(p)
    out = dict(preds={}, raw={}, exec={}, gu={})
    for p in E['pred_files']:
        if not os.path.exists(p):
            continue
        if os.path.basename(p).startswith('raw_P'):
            continue
        if os.path.basename(p).startswith('raw_'):
            out['raw'][p] = {int(r['quid']): dict(answer=r.get('answer'), valid=r.get('valid')) for r in rows(p) if int(r['quid']) in need}
            continue
        d = {}
        for r in rows(p):
            q = int(r['quid'])
            if q in need:
                rk = r['ranked']
                rk = json.loads(rk.replace("'", '"')) if isinstance(rk, str) else rk
                d[q] = rk[:10]
        out['preds'][p] = d
    for p in sorted(glob.glob('/kpct/parser/test/*/exec_P_*_test_*.jsonl.gz')):
        out['exec'][p] = {int(r['quid']): dict(status=r.get('status'), reason=r.get('reason')) for r in rows(p) if int(r['quid']) in need}
    gp = E['gu_test']
    out['gu'][gp] = {int(r['quid']): r['fact_ids'] for r in rows(gp) if int(r['quid']) in need}
    ver['gu_test_sha256'] = sha256_file(gp)
    ver['gu_emb_sha256'] = sha256_file(E['gu_emb'])
    ver['n_bad'] = len(bad)
    ver['bad'] = bad
    os.makedirs(f'{R}/checks', exist_ok=True)
    json.dump(ver, open(f'{R}/checks/verify_frozen.json', 'w'), indent=1)
    json.dump(out, open(f'{R}/checks/frozen_orig_rows.json', 'w'))
    print(json.dumps(dict(n_bad=len(bad), bad=bad[:10], n_pred_files=len(out['preds']), n_exec=len(out['exec']))), flush=True)
    sys.exit(3 if bad else 0)


if __name__ == '__main__':
    main()
