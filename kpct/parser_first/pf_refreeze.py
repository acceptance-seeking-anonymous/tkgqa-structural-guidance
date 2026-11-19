"""pf_refreeze.py -- update the code sha256 entries of an existing freeze (no prediction file regenerated) and log why."""
import datetime, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pf_lib as PL
src, dst, note = sys.argv[1], sys.argv[2], sys.argv[3]
fz = json.load(open(src)); prev = fz.pop('sha256')
log = fz.get('freeze_log', [])
HERE = os.path.dirname(os.path.abspath(__file__))
changed = []
for f in fz['files']:
    if f['role'].startswith('code:'):
        p = f['path'] if os.path.isabs(f['path']) else (os.path.join(PL.W, f['path']) if os.path.exists(os.path.join(PL.W, f['path'])) else os.path.abspath(f['path']))
        h = PL.sha256_file(p)
        if h != f['sha256']:
            changed.append(dict(file=f['role'], old=f['sha256'], new=h)); f['sha256'] = h
log.append(dict(previous_freeze_sha256=prev, previous_frozen_utc=fz['frozen_utc'], changed_code=changed, note=note))
fz['freeze_log'] = log
fz['frozen_utc'] = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
fz['sha256'] = hashlib.sha256(json.dumps(fz, indent=1, sort_keys=True).encode()).hexdigest()
json.dump(fz, open(dst, 'w'), indent=1)
print(fz['frozen_utc'], fz['sha256'], changed)
