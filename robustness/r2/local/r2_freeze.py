"""r2_freeze.py -- build multitq_robustness_R2_freeze.json from the jobD2 outputs (no gold, no scoring).
Usage: from r2_freeze import build_freeze; build_freeze(...)
"""
import datetime, hashlib, json, os


def _sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def build_freeze(freeze_txt, tgz, pred_dir, ident, I1, tmpl, vf, code_sha_txt, usage_log_tsv, pset_path, meta, out='multitq_robustness_R2_freeze.json'):
    files = {}
    for l in open(freeze_txt):
        if l.strip():
            s, p = l.split(None, 1)
            files[p.strip()] = s
    bad = [p for p, s in files.items() if not p.startswith('bundles/') and _sha(os.path.join(pred_dir, p)) != s]
    assert not bad, f'local copies differ from the remote freeze list: {bad[:5]}'
    led = [l.rstrip('\n').split('\t') for l in open(usage_log_tsv) if l.startswith('robustness_r2')]
    fz = dict(name='multitq_robustness_R2_freeze', **meta,
              perturbed_set_sha256=_sha(pset_path),
              frozen_utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
              scored=False, gold_read=False, remote_root='the GPU host $WORK_DIR/robustness_r2',
              files={p: s for p, s in files.items() if not p.startswith('bundles/')},
              bundles_sha256={p: s for p, s in files.items() if p.startswith('bundles/')},
              predictions_tarball=dict(file=os.path.basename(tgz), sha256=_sha(tgz)),
              frozen_inputs_verified=vf, pipeline_identity_I1=I1, I1_pass=all(v['arms'][a]['top1'] >= 0.995 for v in I1.values() for a in v['arms']),
              template_check=tmpl, identity_I2_I3_rates=ident, code_and_inputs_sha256=[l.strip() for l in open(code_sha_txt) if l.strip()],
              gpu_usage_rows=led, gpu_h=round(sum(float(r[6]) for r in led), 4), gpu_cards=sorted({r[1] for r in led}))
    body = json.dumps(fz, sort_keys=True, ensure_ascii=False)
    fz['sha256'] = hashlib.sha256(body.encode()).hexdigest()
    json.dump(fz, open(out, 'w'), indent=1, ensure_ascii=False)
    return fz
