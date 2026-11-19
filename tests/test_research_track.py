"""test_research_track.py -- data-free checks of the research-track code (matched baseline, structural ablation,
output analyses). Run: python tests/test_research_track.py (or pytest)."""
import ast, importlib.util, json, os, sys, tempfile
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, rel))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


M = load('multitq_eval', 'src/multitq_eval.py')


def test_R1_structural_ablation_bootstrap_equals_evaluator():
    """sa_stats.boot_diff draws the evaluator's Poisson(1) stream: same CI as multitq_eval.paired_poisson_bootstrap"""
    SS = load('sa_stats', 'analysis/structural_ablation/sa_stats.py')
    rng = np.random.default_rng(3)
    a, b = rng.integers(0, 2, 1500), rng.integers(0, 2, 1500)
    o, m = M.paired_poisson_bootstrap(a, b), SS.boot_diff(a, b)
    assert abs(o['ci_low'] - m['ci_low']) < 1e-12 and abs(o['ci_high'] - m['ci_high']) < 1e-12 and abs(o['delta'] - m['delta']) < 1e-12
    adj = SS.holm({'x': 0.01, 'y': 0.04, 'z': 0.03})
    assert abs(adj['x'] - 0.03) < 1e-12 and abs(adj['z'] - 0.06) < 1e-12 and abs(adj['y'] - 0.06) < 1e-12


def test_R2_ranker_feature_groups_partition_the_34_features():
    """the six registered groups of sb_ablate.py partition mtq_cand.FEATS exactly"""
    sys.path.insert(0, os.path.join(REPO, 'src', 'struct'))
    MC = load('mtq_cand', 'src/struct/mtq_cand.py')
    tree = ast.parse(open(os.path.join(REPO, 'analysis/structural_ablation/sb_ablate.py')).read())
    groups = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'GROUPS')
    flat = [f for g in groups.values() for f in g]
    assert len(MC.FEATS) == 34 and sorted(flat) == sorted(MC.FEATS) and len(set(flat)) == 34


def test_R3_coverage_tables_paired_bootstrap():
    """a2_tables.cell_rows: contrast row = difference of the arm means; empty cell -> NaN"""
    AT = load('a2_tables', 'analysis/coverage/a2_tables.py')
    import pandas as pd
    rng = np.random.default_rng(0)
    df = pd.DataFrame({'A': rng.integers(0, 2, 400).astype(float), 'B': rng.integers(0, 2, 400).astype(float)})
    rows = AT.cell_rows(df, np.ones(400, bool), ['A', 'B'], 'ALL', contrasts=[('A', 'B')])
    v = {r['quantity']: r for r in rows}
    assert abs(v['A - B']['value'] - (v['A']['value'] - v['B']['value'])) < 1e-12
    assert v['A']['lo'] <= v['A']['value'] <= v['A']['hi']
    e = AT.cell_rows(df, np.zeros(400, bool), ['A'], 'EMPTY')
    assert e[0]['n'] == 0 and np.isnan(e[0]['value'])


def test_R4_matched_baseline_bundle_inputs():
    """mbq_make_bundle.py: answer-stripped test file, dev_sel2000 subset in dev order, sha256-checked lists"""
    GC = load('gk_common', 'kpct/reader/gk_common.py')
    with tempfile.TemporaryDirectory() as d:
        qd = os.path.join(d, 'questions'); os.makedirs(qd)
        mk = lambda base, n: [dict(quid=base + i, question=f'q{i}', answers=['a'], answer_type='entity', time_level='day',
                                   qtype='equal', qlabel='Single') for i in range(n)]
        train, dev, test = mk(1000000, 50), mk(2000000, 2500), mk(3000000, 7)
        for n, x in (('train', train), ('dev', dev), ('test', test)):
            json.dump(x, open(os.path.join(qd, f'{n}.json'), 'w'))
        sel = [q['quid'] for q in dev[::-1][:2000]]
        lists = {'k01_s0': [1000001, 1000003], 'k05_s0': list(range(1000000, 1000010)), 'k10_s0': list(range(1000000, 1000020))}
        man = {'dev_selection': dict(quids=sel, n=len(sel), sha256=GC.list_sha256(sel)), 'fallback_cap_lists': {},
               'subsets': {k: dict(quids=v, n=len(v), sha256=GC.list_sha256(v)) for k, v in lists.items()}}
        json.dump(man, open(os.path.join(d, 'manifest.json'), 'w'))
        import subprocess
        r = subprocess.run([sys.executable, os.path.join(REPO, 'baselines/multiqa_matched/mbq_make_bundle.py'), '--questions', qd,
                            '--manifest', os.path.join(d, 'manifest.json'), '--out', os.path.join(d, 'job')], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        ts = json.load(open(os.path.join(d, 'job/qin/test_stripped.json')))
        assert all(set(q) == {'quid', 'question'} for q in ts) and len(ts) == 7
        dv = json.load(open(os.path.join(d, 'job/qin/dev_sel2000.json')))
        assert [q['quid'] for q in dv] == sorted(sel) and 'answers' in dv[0]
        assert json.load(open(os.path.join(d, 'job/lists/k01_s0.json'))) == lists['k01_s0']


def test_R5_matched_baseline_scoring_statistics():
    """mbq_score.paired: delta = mean difference, n_a_only / n_b_only count the discordant questions"""
    MS = load('mbq_score', 'baselines/multiqa_matched/mbq_score.py')
    a = np.array([1, 1, 0, 1, 0, 0, 1, 1] * 25, float); b = np.array([1, 0, 0, 1, 1, 0, 0, 1] * 25, float)
    r = MS.paired(a, b)
    assert abs(r['delta'] - (a.mean() - b.mean())) < 1e-12 and r['n_a_only'] == 50 and r['n_b_only'] == 25
    m, ci = MS.boot_mean(a)
    assert ci[0] <= m <= ci[1]


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS {name}')
