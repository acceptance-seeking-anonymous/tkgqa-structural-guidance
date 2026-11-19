#!/usr/bin/env python3
"""gr_verify.py <expected.json> <out.json> -- grounded_reader STOP_TEST_INTEGRITY check: sha256 of every frozen input
listed in expected.json ({path: sha256 or null}) inside the container; a listed sha256 that differs -> exit 3.
Entries with null are recorded only (no registered reference)."""
import hashlib, json, os, sys
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()
E = json.load(open(sys.argv[1])); out = dict(checked=0, recorded=0, mismatches=[], missing=[], sha256={})
for p, want in sorted(E.items()):
    if not os.path.exists(p):
        out['missing'].append(p); continue
    s = sha(p); out['sha256'][p] = s
    if want:
        out['checked'] += 1
        if s != want:
            out['mismatches'].append([p, want, s])
    else:
        out['recorded'] += 1
json.dump(out, open(sys.argv[2], 'w'), indent=1)
print(json.dumps({k: (v if not isinstance(v, dict) else len(v)) for k, v in out.items()}))
sys.exit(3 if (out['mismatches'] or out['missing']) else 0)
