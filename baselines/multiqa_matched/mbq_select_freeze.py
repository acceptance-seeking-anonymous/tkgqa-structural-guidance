"""Dev-only embedding selection per level + sha256 freeze of every test prediction file (no test label is read here).
usage: python mbq_select_freeze.py <run_dir[,run_dir ...]> <out_dir> <rule_id>
  run_dir = the runs/ folders written by mbq_sched.py (one per plan); rule_id = an identifier of the registered rule
  (e.g. the sha256 of protocol/multitq_matched_baseline_rule.json), recorded in the selection and freeze files."""
import datetime, gzip, hashlib, json, os, shutil, sys

LVL = {'k01_s0': '1', 'k05_s0': '5', 'k10_s0': '10', 'k100': '100'}


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def main(run_dirs, out_dir, rule_vid):
    runs = {}
    for d in run_dirs:
        for tag in sorted(os.listdir(d)):
            p = os.path.join(d, tag, 'run.json')
            if os.path.exists(p):
                runs[tag] = (json.load(open(p)), os.path.join(d, tag))
    sel = {'name': 'multitq_matched_baseline_selection', 'rule_version_id': rule_vid,
           'created_utc': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
           'basis': 'dev_sel2000 Hits@1 of the primary checkpoint, mean over the available seeds; ties -> OFF', 'levels': {}, 'selected': {}}
    for k, lv in LVL.items():
        row = {}
        for emb in ['off', 'full']:
            rr = [runs[t][0] for t in runs if runs[t][0]['k'] == k and runs[t][0]['emb'] == emb]
            if not rr:
                continue
            row[emb] = {'seeds': [r['seed'] for r in rr], 'dev_h1_primary': [r['best_dev_h1'] for r in rr],
                        'dev_h1_budget10': [r['best10_dev_h1'] for r in rr], 'best_epoch': [r['best_epoch'] for r in rr],
                        'best10_epoch': [r['best10_epoch'] for r in rr], 'epochs_run': [len(r['epochs']) for r in rr],
                        'max_epochs': [r['max_epochs'] for r in rr], 'early_stopped_at': [r.get('early_stopped_at') for r in rr],
                        'best_at_cap': [r['best_epoch'] == r['max_epochs'] for r in rr],
                        'mean_dev_h1_primary': sum(r['best_dev_h1'] for r in rr) / len(rr),
                        'mean_dev_h1_budget10': sum(r['best10_dev_h1'] for r in rr) / len(rr)}
        if not row:
            continue
        sel['levels'][lv] = row
        if 'off' in row and 'full' in row:
            emb = 'full' if row['full']['mean_dev_h1_primary'] > row['off']['mean_dev_h1_primary'] else 'off'
        else:
            emb = 'off' if 'off' in row else 'full'
        sel['selected'][lv] = {'emb': emb, 'mean_dev_h1': row[emb]['mean_dev_h1_primary']}
    json.dump(sel, open(os.path.join(out_dir, 'multitq_matched_baseline_selection.json'), 'w'), indent=1)
    pdir = os.path.join(out_dir, 'predictions'); os.makedirs(pdir, exist_ok=True)
    files = []
    for tag, (r, d) in sorted(runs.items()):
        lv = LVL[r['k']]; E = r['emb'].upper()
        for ck, suffix in [('primary', ''), ('budget10', '10')]:
            for split in ['test', 'dev_sel2000']:
                src = os.path.join(d, f'{split}_{ck}.jsonl.gz')
                dst_name = f"{split}_MBQ_{E}{suffix}_{lv}_seed{r['seed']}.jsonl.gz"
                dst = os.path.join(pdir, dst_name)
                shutil.copyfile(src, dst)
                raw = gzip.open(dst, 'rb').read()
                raw_sha = hashlib.sha256(raw).hexdigest()
                exp = r['pred'][ck][f"{'test' if split == 'test' else 'dev'}_sha256"]
                assert raw_sha == exp, (tag, ck, split)
                if split == 'test':
                    files.append({'name': dst_name, 'arm': f"MBQ_{E}{suffix}_{lv}_seed{r['seed']}", 'group': f'MBQ_{E}{suffix}_{lv}',
                                  'run': tag, 'checkpoint': ck, 'epoch': r['pred'][ck]['epoch'], 'local_path': dst,
                                  'sha256': sha(dst), 'sha256_uncompressed_jsonl': raw_sha, 'n_lines': raw.count(b'\n'),
                                  'dev_h1': r['best_dev_h1'] if ck == 'primary' else r['best10_dev_h1']})
    freeze = {'name': 'multitq_matched_baseline_freeze', 'rule_version_id': rule_vid,
              'frozen_utc': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
              'statement': ('Every MultiQA test prediction file of this track, frozen before any test scoring. Test files were produced '
                            'on the GPU host from answer-stripped questions; the uncompressed sha256 equals the value the training '
                            'job recorded at write time. Scored once with multitq_eval.py (official eval_multi semantics). '
                            'Label: reuse of the official MultiTQ test split.'),
              'selection_sha256': sha(os.path.join(out_dir, 'multitq_matched_baseline_selection.json')),
              'n_files': len(files), 'files': files}
    json.dump(freeze, open(os.path.join(out_dir, 'multitq_matched_baseline_freeze.json'), 'w'), indent=1)
    return sel, freeze


if __name__ == '__main__':
    main(sys.argv[1].split(','), sys.argv[2], sys.argv[3])
