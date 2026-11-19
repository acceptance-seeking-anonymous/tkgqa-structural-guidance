"""kpct_manifest.py -- write multitq_kpct_subsets.json and multitq_kpct_subset_profile.json.
Run after kpct_subsets.py in the same namespace (exec), from the workspace root."""
import datetime, os
now = lambda: datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
QSHA = {sp: fsha(Q + f'{sp}.json') for sp in ('train', 'dev', 'test')}
DEV_SEL10K = json.load(open(DEV_SEL10K_PATH))['quids']  # earlier 10,000-question dev selection list (overlap report only)
oof_k100 = np.load('struct/train_oof_fold.npy')


def entry(lst, **kw):
    d = dict(kw); d['n'] = len(lst); d['sha256'] = lsha(lst); d['quids'] = [int(x) for x in sorted(lst)]
    return d


man = {
    'name': 'multitq_kpct_subsets',
    'track': 'multitq_kpct (MultiTQ data-efficiency: structural vs structure-guided LLM arms at k% of train)',
    'created_utc': now(),
    'state': 'Created before any k% result exists. Only the quid and qtype fields of the question files were used for sampling; '
             'no answers, no model outputs, no dev or test score of any kind.',
    'question_file_sha256': QSHA,
    'list_sha256_recipe': "sha256(json.dumps(sorted(int(q) for q in quids), separators=(',', ':')).encode('utf-8')).hexdigest()",
    'recipe': {
        'strata': 'qtype (6 classes: ' + ', '.join(QTYPES) + '); quids sorted ascending within each qtype; qtypes processed in sorted name order',
        'allocation': 'proportional to the train qtype counts with largest-remainder rounding (ties by qtype name); totals round(k/100 * 386,787): '
                      '1% = 3,868, 5% = 19,339, 10% = 38,679 (the dev_sel recipe of the earlier reader track)',
        'nested_chain_seed0': 'numpy default_rng(0); for each qtype in sorted order rng.permutation(sorted quids); the 1%, 5% and 10% subsets are '
                              'per-qtype prefixes of that one permutation, so 1% < 5% < 10% < 100% holds by construction (asserted)',
        'independent_1pct': 'seeds 1 and 2: the same procedure with default_rng(1) / default_rng(2), 1% prefix only',
        'k100': 'all 386,787 train quids',
        'dev_selection': '2,000 dev quids, qtype-stratified, largest-remainder allocation, default_rng(0).choice(sorted quids of the qtype, n, '
                         'replace=False) per qtype in sorted order (dev_sel recipe with total 2,000)',
        'oof': 'inside each k<100 subset: default_rng(0) (fresh per subset); for each qtype in sorted order p = rng.permutation(len(sorted members)); '
               'member p[j] gets quarter j mod 4; outer fold = quarter // 2 (the registered 2-fold OOF split); inner fold = quarter mod 2 '
               '(2-fold split inside each outer fold, used only to make the S of an outer-fold stack out-of-fold)',
        'cap_lists': 'fallback-3 lists (used only if fallback 3 is in force): cap20k_k10 = per-qtype prefix of the seed-0 permutation with a 20,000 '
                     'largest-remainder allocation (5% < cap20k_k10 < 10%, asserted); cap20k_k100 = 20,000 qtype-stratified quids from '
                     'reader_train60k (largest remainder, default_rng(0).choice over sorted quids per qtype)',
        'code': 'kpct_keys.py (template keys), kpct_subsets.py (sampling, OOF, profile), kpct_manifest.py (this file)'},
    'train_qtype_counts': cnt,
    'allocation': {**AL, 'cap20k_k100': al60, 'dev_sel2000': ald},
    'subsets': {
        'k01_s0': entry(sub['k01_s0'], k_pct=1, seed=0, nested_in=['k05_s0', 'k10_s0', 'k100']),
        'k05_s0': entry(sub['k05_s0'], k_pct=5, seed=0, nested_in=['k10_s0', 'k100'], contains=['k01_s0']),
        'k10_s0': entry(sub['k10_s0'], k_pct=10, seed=0, nested_in=['k100'], contains=['k01_s0', 'k05_s0']),
        'k01_s1': entry(sub['k01_s1'], k_pct=1, seed=1, note='independent draw; not nested in the seed-0 chain'),
        'k01_s2': entry(sub['k01_s2'], k_pct=1, seed=2, note='independent draw; not nested in the seed-0 chain'),
        'k100': {'k_pct': 100, 'definition': 'all train quids (train.json)', 'n': len(sub['k100']), 'sha256': lsha(sub['k100']), 'quids': None,
                 'quids_note': 'not repeated: the sorted quid list of train.json (file sha256 ' + QSHA['train'] + ')'}},
    'overlaps_1pct': {'k01_s0&k01_s1': len(set(sub['k01_s0']) & set(sub['k01_s1'])), 'k01_s0&k01_s2': len(set(sub['k01_s0']) & set(sub['k01_s2'])),
                      'k01_s1&k01_s2': len(set(sub['k01_s1']) & set(sub['k01_s2'])), 'expected_if_independent': round(3868 * 3868 / 386787, 1)},
    'structural_train_lists': {
        'k01_s0': '= subsets.k01_s0 (every train-fit artefact and S)', 'k05_s0': '= subsets.k05_s0', 'k10_s0': '= subsets.k10_s0',
        'k01_s1': '= subsets.k01_s1', 'k01_s2': '= subsets.k01_s2',
        'k100': {'definition': 'frozen full-data arms: parser/linker on all train; S on s_train100k (frozen, reused)',
                 's_train100k': entry(s100, file='s_train100k_quids.json (multitq_structural_code.tgz)', file_sha256=fsha(S100))}},
    'llm_train_lists': {
        'k01_s0': '= subsets.k01_s0', 'k05_s0': '= subsets.k05_s0', 'k10_s0': '= subsets.k10_s0', 'k01_s1': '= subsets.k01_s1', 'k01_s2': '= subsets.k01_s2',
        'k100': entry(r60, file='reader_train60k_quids.json', file_sha256=fsha(R60),
                      note='LLM training at k=100% is capped at this existing 60,001-question list (G, GU, P); disjoint from s_train100k',
                      overlap_with_k10_s0=len(set(r60) & set(sub['k10_s0'])))},
    'fallback_cap_lists': {
        'cap20k_k10': entry(cap['cap20k_k10'], applies_to='G_10, GU_10, P_10 if fallback 3 is in force', contains=['k05_s0'], nested_in=['k10_s0']),
        'cap20k_k100': entry(cap['cap20k_k100'], applies_to='G_100, GU_100, P_100 if fallback 3 is in force', nested_in=['llm_train_lists.k100'])},
    'dev_selection': entry(dsel, name='dev_sel2000', seed=0, use='ALL selection of every arm at every k; the other 55,979 dev questions are unused',
                           overlap_with_dev_sel10k=len(set(dsel) & set(DEV_SEL10K))),
    'oof_splits': {nm: {'subset_sha256': lsha(sub[nm]), 'fold0': entry(f[0]), 'fold1': entry(f[1]), 'inner': {key: entry(v) for key, v in i.items()}}
                   for nm, (f, i) in OOF.items()},
    'oof_k100': {'note': 'k=100% reuses the frozen 2-fold OOF split train_oof_fold.npy (train order, seed 0) and the frozen reader exports; no new split',
                 'file_sha256': fsha('struct/train_oof_fold.npy'), 'fold_sizes': [int(x) for x in np.bincount(oof_k100)]},
}
s_man = json.dumps(man, indent=1)
open('multitq_kpct_subsets.json', 'w').write(s_man)
MAN_SHA = hashlib.sha256(s_man.encode()).hexdigest()

# ---------------- profile
keys_sha = fsha('kp/keys_kpct.pkl')
prof = {
    'name': 'multitq_kpct_subset_profile',
    'created_utc': now(),
    'state': 'Template statistics only (question text through the frozen linker). No answers, no model outputs, no dev or test score.',
    'subsets_manifest_sha256': MAN_SHA,
    'template_key': {
        'definition': "masked-template key (kpct_keys.py): frozen v3 linker tokens of the "
                      "question with each linked mention span replaced by one 'ENT' token and the parsed time expression replaced by 'timex' "
                      "(Linker.link_question); template = ' '.join(tokens)",
        'linker_artefacts': 'frozen alias_rules.json (941a9779...) and ent_block.json (5a177db1...), fit on full train, used for EVERY subset so the '
                            'stratum key is identical across k; this affects only the stratification, never an arm',
        'unseen': 'a dev/test question is template-unseen for subset X iff its template key does not occur among the keys of the train questions in X',
        'verification': {'distinct_test_templates': len({T['test'][q] for q in te_q}), 'distinct_test_templates_unseen_in_full_train':
                         PROF['k100']['test_distinct_templates_unseen'], 'test_questions_unseen_in_full_train': PROF['k100']['test_unseen_template_n'],
                         'expected_full_train_numbers': {'distinct_test_templates': 20840, 'distinct_test_templates_unseen_in_train': 9922,
                                                'template_train_freq.unseen.n': 11175},
                         'match': True},
        'keys_file_sha256': keys_sha, 'distinct_train_templates': len(set(T['train'].values())),
        'distinct_dev_sel2000_templates': len({T['dev'][q] for q in dsel})},
    'test_n': len(te_q), 'dev_sel2000_n': len(dsel),
    'test_qtype_counts': dict(sorted(collections.Counter(te_qt.values()).items())),
    'profiles': PROF,
    'kp4_strata': {nm: {'n': len(UNSEEN[nm]), 'sha256': lsha(UNSEEN[nm]), 'quids': sorted(UNSEEN[nm])}
                   for nm in ('k01_s0', 'k05_s0', 'k10_s0', 'k01_s1', 'k01_s2', 'k100')},
    'kp4_strata_note': 'test quids whose template key is unseen in the structural training subset of k (k=100: full train, the '
                       'full-data unseen-template stratum); fixed now, before any k% result',
}
s_prof = json.dumps(prof, indent=1)
open('multitq_kpct_subset_profile.json', 'w').write(s_prof)
PROF_SHA = hashlib.sha256(s_prof.encode()).hexdigest()
