#!/usr/bin/env python3
"""sa_freeze.py -- sha256 freeze of every new prediction file of one part, written BEFORE any scoring.
usage: python sa_freeze.py <part A|B> <out.json> <rule_id> <dir> [<dir> ...]   (rule_id: e.g. the sha256 of the rule file)
Lists every pred_* / pool_* / RULE_* / *stats* file below the dirs with sha256, bytes and line count. Never reads gold.
"""
import datetime, gzip, hashlib, json, os, sys


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def nlines(p):
    op = gzip.open if p.endswith('.gz') else open
    with op(p, 'rb') as f:
        return sum(1 for _ in f)


part, out, rule_vid, dirs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
files = []
for d in dirs:
    for fn in sorted(os.listdir(d)):
        p = os.path.join(d, fn)
        if os.path.isfile(p) and (fn.startswith(('pred_', 'pool_', 'RULE_')) or 'stats' in fn):
            files.append(dict(dir=os.path.basename(os.path.normpath(d)), file=fn, sha256=sha256_file(p), bytes=os.path.getsize(p),
                              lines=nlines(p) if fn.endswith(('.jsonl', '.jsonl.gz')) else None))
inputs = [dict(file=p, sha256=sha256_file(p), bytes=os.path.getsize(p)) for p in os.environ.get('FREEZE_INPUTS', '').split(',') if p]
body = dict(track='multitq_structural_ablation', part=part, rule_version_id=rule_vid, inputs=inputs,
            frozen_utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            note='sha256 of every new prediction / pool / stats file of this part, recorded before the scoring pass reads any gold answer. Test split reused (official MultiTQ test, 54,584 questions).',
            n_files=len(files), files=files)
raw = json.dumps(body, indent=1, sort_keys=True)
body['sha256'] = hashlib.sha256(raw.encode()).hexdigest()
json.dump(body, open(out, 'w'), indent=1)
print(json.dumps({'part': part, 'n_files': len(files), 'frozen_utc': body['frozen_utc'], 'sha256': body['sha256']}))
