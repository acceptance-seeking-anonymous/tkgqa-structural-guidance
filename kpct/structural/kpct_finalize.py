"""kpct_finalize.py -- dev_sel2000 results, selection log, OOF-leak check and the structural freeze (no test scoring)."""
import os, sys, json, glob, hashlib, time, datetime, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import kpct_core as K

OUT = os.path.join(HERE, 'out')
SUBS = ['k01_s0', 'k05_s0', 'k10_s0', 'k01_s1', 'k01_s2']
RULE_FILE = os.path.join(HERE, 'rule', 'multitq_kpct_decision_rule.json')
FROZEN_FREEZE = os.path.join(HERE, '..', 'exp', 'multitq_structural_freeze.json')
FROZEN_PREDS = os.path.join(HERE, '..', 'exp', 'preds')
REMOTE = '$WORK_DIR'


def sha(p):
    return K.sha256_file(p)


def nlines(p):
    with open(p, 'rb') as f:
        return sum(1 for _ in f)


def oof_leak_check(nm):
    sub = json.load(open(os.path.join(HERE, 'subsets', 'multitq_kpct_subsets.json')))
    o = sub['oof_splits'][nm]
    F = {0: set(o['fold0']['quids']), 1: set(o['fold1']['quids'])}
    n = bad = 0
    per = collections.Counter()
    for l in open(f'{OUT}/{nm}/bundles/train.jsonl'):
        x = json.loads(l)
        f = x['oof']['outer_fold']
        n += 1; per[f] += 1
        # the producing stack (linker, silver labels, vocabulary, classifier, S incl. its inner-OOF training rows) was fit on fold 1-f only
        if x['quid'] in F[1 - f] or x['quid'] not in F[f] or x['oof']['stack_trained_on'] != f'{nm}/fold{1 - f}':
            bad += 1
    fd = {f: json.load(open(f'{OUT}/{nm}/fold{f}/fold_done.json')) for f in (0, 1)}
    lists_ok = all(fd[f]['train_quids_sha256'] == o[f'fold{1 - f}']['sha256'] and fd[f]['heldout_quids_sha256'] == o[f'fold{f}']['sha256'] for f in (0, 1))
    return dict(n_rows=n, per_fold=dict(per), violations=bad, stack_lists_match_manifest=lists_ok,
                status='PASS' if bad == 0 and lists_ok and n == sub['subsets'][nm]['n'] else 'FAIL')


def main():
    now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    rule = json.load(open(RULE_FILE))
    sub = json.load(open(os.path.join(HERE, 'subsets', 'multitq_kpct_subsets.json')))
    dev, sel, freeze_arms, leak = {}, {}, {}, {}
    for nm in SUBS + ['k100']:
        d = f'{OUT}/{nm}'
        s = json.load(open(f'{d}/dev_summary.json'))
        dev[nm] = s
    for nm in SUBS:
        d = f'{OUT}/{nm}'
        sE = json.load(open(f'{d}/selection_E.json')); sS = json.load(open(f'{d}/selection_S.json'))
        fa = json.load(open(f'{d}/fitA_done.json'))
        fds = {f: json.load(open(f'{d}/fold{f}/fold_done.json')) for f in (0, 1)}
        lk = json.load(open(f'{d}/linker_full.json'))
        sel[nm] = dict(classifier_epochs=sE, S=sS, S_training=dict(rows=fa['S_rows'], groups=fa['S_groups'], dropped_no_positive=fa['S_dropped_no_positive']),
                       linker=dict(alias_rules=len(lk['alias_rules']), block_list=len(lk['ent_block'])),
                       fold_stacks={f: {k: fds[f][k] for k in ('n_train', 'n_heldout', 'E', 'S_rows', 'S_groups', 'S_dropped_no_positive')} for f in (0, 1)})
        leak[nm] = oof_leak_check(nm)
        E = sE['E_selected']
        files = {}
        for a in ('RULE', 'S_seed0', 'S_seed1', 'S_seed2', 'S_mean'):
            p = f'{d}/preds/test_{a}.jsonl'
            files[f'test_{a}'] = dict(file=f'preds/{nm}/test_{a}.jsonl', sha256=sha(p), bytes=os.path.getsize(p), rows=nlines(p))
            p = f'{d}/preds/dev_sel2000_{a}.jsonl'
            files[f'dev_sel2000_{a}'] = dict(file=f'preds/{nm}/dev_sel2000_{a}.jsonl', sha256=sha(p), rows=nlines(p))
        models = {f'S_seed{s}.txt': sha(f'{d}/S_seed{s}.txt') for s in range(3)}
        models[f'parser_full_E{E}.pt'] = sha(f'{d}/parser_full_E{E}.pt')
        models['linker_full.json'] = sha(f'{d}/linker_full.json'); models['FB.json'] = sha(f'{d}/FB.json')
        for f in (0, 1):
            models[f'fold{f}/parser.pt'] = sha(f'{d}/fold{f}/parser.pt'); models[f'fold{f}/S_seed0.txt'] = sha(f'{d}/fold{f}/S_seed0.txt')
            models[f'fold{f}/linker.json'] = sha(f'{d}/fold{f}/linker.json')
        bundles = {sp: dict(file=f'bundles/{nm}/{sp}.jsonl', sha256=sha(f'{d}/bundles/{sp}.jsonl'), rows=nlines(f'{d}/bundles/{sp}.jsonl'))
                   for sp in ('train', 'dev_sel2000', 'test')}
        freeze_arms[nm] = dict(train_list=dict(name=nm, n=sub['subsets'][nm]['n'], sha256=sub['subsets'][nm]['sha256']),
                               classifier_epochs=E,
                               S_config=dict(sel[nm]['S']['selected'], min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
                                             bagging_freq=1, lambdarank_truncation_level=20, seeds=[0, 1, 2]),
                               prediction_files=files, models=models, bundles=bundles,
                               arms={'RULE_k': f'preds/{nm}/test_RULE.jsonl',
                                     'S_k': [f'preds/{nm}/test_S_seed{s}.jsonl' for s in range(3)],
                                     'S_k_statistic': 'per-question hit averaged over the 3 seed files (registered)',
                                     'S_mean_file': f'preds/{nm}/test_S_mean.jsonl (ranked by the mean of the 3 seed scores; descriptive, not a registered arm)'})
    # k100: frozen arms reused; hash-verify
    ff = json.load(open(FROZEN_FREEZE))
    k100v = {}
    for a in ('RULE', 'S_seed0', 'S_seed1', 'S_seed2'):
        p = f'{FROZEN_PREDS}/test_{a}.jsonl'
        h = sha(p); exp = ff['sha256']['test_predictions'][f'preds/test_{a}.jsonl']
        k100v[f'test_{a}'] = dict(file=f'phase1_structural preds/test_{a}.jsonl', sha256=h, frozen_sha256=exp, match=h == exp)
    d = f'{OUT}/k100'
    freeze_arms['k100'] = dict(status='REUSED frozen full-data arms (RULE_100, S_100); previously scored (RULE 0.9273, S 0.9310); not new findings',
                               prediction_files=k100v,
                               regenerated_identity=dev['k100'].get('identity_with_frozen_prediction_files'),
                               bundles={sp: dict(file=f'bundles/k100/{sp}.jsonl', sha256=sha(f'{d}/bundles/{sp}.jsonl'), rows=nlines(f'{d}/bundles/{sp}.jsonl'))
                                        for sp in ('train', 'dev_sel2000', 'test')},
                               dev_sel2000_prediction_files={a: dict(file=f'preds/k100/dev_sel2000_{a}.jsonl', sha256=sha(f'{d}/preds/dev_sel2000_{a}.jsonl'))
                                                             for a in ('RULE', 'S_seed0', 'S_seed1', 'S_seed2', 'S_mean')})
    code = {f: sha(os.path.join(HERE, f)) for f in ('kpct_core.py', 'kpct_run.py', 'kpct_executor.py', 'kpct_k100.py', 'kpct_templates.py',
                                                     'kpct_stage.py', 'kpct_finalize.py', 'check_k100.py')}
    code.update({f'struct/{f}': sha(os.path.join(HERE, 'struct', f)) for f in ('mtq_lib.py', 'mtq_align.py', 'mtq_clf.py', 'mtq_cand.py', 'mtq_export.py', 'multitq_eval.py')})
    freeze = dict(
        track='multitq_kpct structural (RULE_k, S_k; guidance bundles; executor)',
        frozen_utc=now,
        state='All structural k<100 arms frozen from models trained on their k% list only. Test prediction files written once and NOT scored; no test answer, qtype, answer_type, time_level or qlabel was loaded (test questions read as quid + question).',
        rule=dict(file='multitq_kpct_decision_rule.json', sha256_recorded=rule['sha256'], file_sha256=sha(RULE_FILE), rule_version=rule['rule_version']),
        subsets_manifest_sha256=sha(os.path.join(HERE, 'subsets', 'multitq_kpct_subsets.json')),
        scorer_sha256=rule['data']['scorer']['sha256'],
        prediction_format='JSONL {"quid": int, "ranked": [>= 10 distinct strings, top-50]}; entity names with spaces (KG name form); time strings at the predicted granularity (as the frozen Phase-1 files)',
        arms=freeze_arms,
        oof_leak_check=leak,
        remote_root=f'the GPU host {REMOTE}',
        code_sha256=code,
        test_scored=False,
    )
    # executor self-tests (run on the GPU host) and staging checksum verification (remote sha256 == local sha256)
    st = os.path.join(OUT, 'selftest_results.json')
    freeze['executor_selftest'] = json.load(open(st)) if os.path.exists(st) else None
    import re
    stg = collections.defaultdict(lambda: [0, 0])
    for p in glob.glob(os.path.join(HERE, '..', 'hpc', '*', 'remote_sha_*.txt')):
        for l in open(p):
            h, f = l.split()
            m = re.search(r'(k\d\d_s\d|k100)', f)
            nm, base = m.group(1), os.path.basename(f)
            kind = 'bundles' if (f.startswith('bundles/') or '/bundles/' in f) else 'preds'
            loc = os.path.join(OUT, nm, kind, base)
            stg[nm][1] += 1
            stg[nm][0] += os.path.exists(loc) and sha(loc) == h
    freeze['staging_verification'] = {k: {'files': v[1], 'sha256_match': v[0]} for k, v in sorted(stg.items())}
    rb = {k: v for k, v in rule.items() if k != 'sha256'}
    rule_sha_ok = hashlib.sha256(json.dumps(rb, indent=1, sort_keys=True, default=str).encode()).hexdigest() == rule['sha256'] \
        and rule['sha256'] == '0f5f760450356349a7bf353301520d76de3617f1dfec0b4e4b1430c97b1d6cd7'
    freeze['rule']['amendment_log_entries'] = len(rule.get('amendment_log', []))
    freeze['rule']['sha256_matches_registration'] = rule_sha_ok
    body = {k: v for k, v in freeze.items()}
    freeze['sha256'] = hashlib.sha256(json.dumps(body, indent=1, sort_keys=True, default=str).encode()).hexdigest()
    json.dump(freeze, open(os.path.join(HERE, 'multitq_kpct_structural_freeze.json'), 'w'), indent=1)
    # dev results
    rows = {}
    for nm in SUBS + ['k100']:
        s = dev[nm]
        rows[nm] = dict(RULE_hits1=s['RULE_hits1'], RULE_hits1_ci95=s['RULE_hits1_ci95'], RULE_hits10=s['RULE_hits10'],
                        S_hits1_per_seed=s['S_hits1_per_seed'], S_hits1_mean=s['S_hits1_mean'], S_hits1_mean_ci95=s['S_hits1_mean_ci95'],
                        S_hits10_per_seed=s['S_hits10_per_seed'], S_hits10_mean=s['S_hits10_mean'],
                        S_seedmean_score_hits1=s['S_seedmean_score_hits1'],
                        S_seed0_recall20=s['S_seed0_recall20'], S_seed0_headroom20=s['S_seed0_headroom20'],
                        template_unseen_share=s['template_unseen_share'],
                        unseen=dict(n=s['unseen_n'], RULE_hits1=s.get('unseen_RULE_hits1'), S_hits1_mean=s.get('unseen_S_hits1_mean')),
                        seen=dict(n=s['seen_n'], RULE_hits1=s.get('seen_RULE_hits1'), S_hits1_mean=s.get('seen_S_hits1_mean')),
                        by_qtype=s['by_qtype'], train_in_sample_silver=s.get('train_in_sample'))
    k1 = ['k01_s0', 'k01_s1', 'k01_s2']
    spread = {m: dict(mean=float(np.mean([rows[x][m] for x in k1])), sd=float(np.std([rows[x][m] for x in k1], ddof=1)),
                      min=float(min(rows[x][m] for x in k1)), max=float(max(rows[x][m] for x in k1))) for m in ('RULE_hits1', 'S_hits1_mean')}
    devres = dict(name='multitq_kpct_structural_dev_results', created_utc=now, split='dev_sel2000 (n = 2000; selection split; dev answers used only here)',
                  statistic='Hits@1 / Hits@10 official matching (multitq_eval.hit_at_k); 95% CI = Poisson(1) bootstrap B = 1000 seed 0; S_k = per-question hit averaged over 3 LightGBM seeds',
                  rows=rows, k1_subset_spread=spread, selection=sel,
                  note='k100 rows re-execute the frozen full-data system on dev_sel2000 (RULE_100 / S_100; already known arms).')
    json.dump(devres, open(os.path.join(HERE, 'multitq_kpct_structural_dev_results.json'), 'w'), indent=1)
    json.dump(dict(name='multitq_kpct_structural_selection_log', created_utc=now,
                   note='every dev_sel2000 selection value computed by the structural track (classifier epochs E grid -> RULE_k Hits@1; S grid -> ndcg@1 / Hits@1 / Hits@10 at LightGBM seed 0)',
                   selection=sel), open(os.path.join(HERE, 'multitq_kpct_structural_selection_log.json'), 'w'), indent=1)
    print(json.dumps({nm: (round(rows[nm]['RULE_hits1'], 4), round(rows[nm]['S_hits1_mean'], 4)) for nm in rows}))
    print(json.dumps(leak)); print(json.dumps(k100v))


if __name__ == '__main__':
    main()
