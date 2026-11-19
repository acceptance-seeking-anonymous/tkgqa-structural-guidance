#!/usr/bin/env python3
"""fill_expected_sha.py -- write expected_sha.json (the integrity list read by rob_collect_frozen.py) from
expected_sha.template.json by hashing your own frozen files. Template paths are container paths (/kpct/... =
$WORK_DIR, /mtq/... = $MTQ_DIR); pass --kpct / --mtq to map them on the host. Relative template paths (the seed-0
reference kpct/followups/seeds_P/host_a/ref_seed0.template.json) are resolved against --root.
Normally called once by scripts/freeze_integrity_refs.sh right after the test pass, so that later checks detect any
change of the frozen files. --strict exits non-zero when a listed file is missing.

  python fill_expected_sha.py r1/remote/expected_sha.template.json expected_sha_R1.json --kpct $WORK_DIR --mtq $MTQ_DIR --strict
"""
import argparse, hashlib, json, os


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('template'); ap.add_argument('out')
    ap.add_argument('--kpct', default=os.environ.get('WORK_DIR')); ap.add_argument('--mtq', default=os.environ.get('MTQ_DIR'))
    ap.add_argument('--root', default=None, help='base directory for relative template paths')
    ap.add_argument('--strict', action='store_true', help='fail if any listed file is missing')
    a = ap.parse_args()
    T = json.load(open(a.template))

    def host(p):
        if p.startswith('/kpct/') and a.kpct:
            return os.path.join(a.kpct, p[len('/kpct/'):])
        if p.startswith('/mtq/') and a.mtq:
            return os.path.join(a.mtq, p[len('/mtq/'):])
        if not os.path.isabs(p) and a.root:
            return os.path.join(a.root, p)
        return p
    out, missing = {}, []
    for grp, entries in T.items():
        if not isinstance(entries, dict):
            out[grp] = entries
            continue
        out[grp] = {}
        for p in entries:
            hp = host(p)
            if os.path.exists(hp):
                out[grp][p] = sha256_file(hp)
            else:
                out[grp][p] = None; missing.append(p)
    json.dump(out, open(a.out, 'w'), indent=1)
    print(f'{sum(len(v) for v in out.values() if isinstance(v, dict))} entries, {len(missing)} missing')
    if missing and a.strict:
        raise SystemExit('missing: ' + ', '.join(missing[:10]) + (' ...' if len(missing) > 10 else ''))


if __name__ == '__main__':
    main()
