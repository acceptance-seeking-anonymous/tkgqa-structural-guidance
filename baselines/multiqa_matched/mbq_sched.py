#!/usr/bin/env python3
"""mbq_sched.py -- run a list of mbq_qa.py runs, at most P concurrently on the one visible GPU.
Usage: python3 mbq_sched.py plan.json  (plan = {"parallel": P, "prep_levels": "...", "runs": [{k, emb, seed, max_epochs, patience, smoke_epochs}]})
Writes sched_log.jsonl; each run writes runs/<tag>/run.json + predictions (*.jsonl.gz).
"""
import json, os, subprocess, sys, time

W = os.path.abspath(os.getcwd())
plan = json.load(open(sys.argv[1]))
logp = os.path.join(W, 'sched_log.jsonl')


def log(d):
    d['utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    with open(logp, 'a') as f:
        f.write(json.dumps(d) + '\n')
    print(json.dumps(d), flush=True)


if plan.get('prep_levels'):
    t = time.time()
    r = subprocess.run([sys.executable, 'mbq_qa.py', 'prep', '--levels', plan['prep_levels']], cwd=W)
    log({'event': 'prep', 'rc': r.returncode, 's': round(time.time() - t, 1)})
    if r.returncode:
        sys.exit(1)
queue = list(plan['runs']); live = []; P = int(plan.get('parallel', 3))
env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
while queue or live:
    while queue and len(live) < P:
        r = queue.pop(0)
        cmd = [sys.executable, 'mbq_qa.py', 'run', '--k', r['k'], '--emb', r['emb'], '--seed', str(r['seed']),
               '--max_epochs', str(r['max_epochs']), '--patience', str(r.get('patience', 0)),
               '--official_budget', str(r.get('official_budget', 10)), '--smoke_epochs', str(r.get('smoke_epochs', 0))]
        tag = f"{r['k']}_{r['emb']}_seed{r['seed']}"
        os.makedirs(os.path.join(W, 'runs', tag), exist_ok=True)
        fo = open(os.path.join(W, 'runs', tag, 'driver.log'), 'w')
        live.append((subprocess.Popen(cmd, cwd=W, env=env, stdout=fo, stderr=subprocess.STDOUT), tag, time.time(), fo))
        log({'event': 'start', 'tag': tag})
    time.sleep(5)
    for item in list(live):
        p, tag, t0, fo = item
        if p.poll() is not None:
            fo.close(); live.remove(item)
            log({'event': 'end', 'tag': tag, 'rc': p.returncode, 's': round(time.time() - t0, 1)})
log({'event': 'all_done'})
