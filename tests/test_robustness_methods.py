"""test_robustness_methods.py -- data-free checks of the rewording-robustness method-track code (R3 probe sets and
secondary analysis, Parser-first routing, question-normalizer selection). Run: python tests/test_robustness_methods.py
(or pytest)."""
import importlib.util, os, sys
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
for _p in ('robustness/r1/local', 'robustness/r2/local', 'robustness/r3', 'src/struct'):
    sys.path.insert(0, os.path.join(REPO, _p))


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, rel))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


R3 = load('r3_common', 'robustness/r3/r3_common.py')


def test_RM1_training_order_is_k01_first_then_permutation():
    """r3_common.training_order: the nested 1% list (ascending) first, then a permutation of the rest of the 5% list"""
    k05 = list(range(1000, 1300)); k01 = [1290, 1010, 1100, 1005]
    o = R3.training_order(k01, k05)
    assert o[:4] == sorted(k01) and sorted(o) == k05 and len(set(o)) == len(o)
    assert o == R3.training_order(list(reversed(k01)), list(reversed(k05)))      # input order does not matter


def test_RM2_contrasts_and_audit_selection():
    """c_drop = drop(b) - drop(m), c_rw = RW(m) - RW(b); audit set = targeted rows union a seeded random 100"""
    C = R3.contrasts([1, 1, 0], [0, 1, 1], [1, 1, 1], [0, 0, 1])
    assert list(C['c_drop']) == [0.0, 1.0, 1.0] and list(C['c_rw']) == [0.0, 1.0, 0.0]
    quids = list(range(5000, 5300))
    pos = np.zeros(300); pos[[3, 50, 299]] = 1.0
    rows, T, R = R3.audit_selection(quids, {'m@5:c_drop': pos, 'm@5:c_rw': -np.ones(300)})
    assert T == {5003, 5050, 5299} and len(R) == 100 and {r['quid'] for r in rows} == T | R
    assert [r['audit_id'] for r in rows] == [f'A{i + 1:03d}' for i in range(len(rows))]
    rows2, _, _ = R3.audit_selection(list(reversed(quids)), {'m@5:c_drop': pos, 'm@5:c_rw': -np.ones(300)})
    assert rows == rows2


def test_RM3_bound_analysis_requires_reviewed_positive_rows():
    """bound_analysis refuses unreviewed rows with a positive contrast and labels every test after removal"""
    rng = np.random.default_rng(0)
    n = 400; quids = list(range(n))
    o_b = np.ones(n); r_b = rng.integers(0, 2, n).astype(float); o_m = np.ones(n); r_m = np.maximum(r_b, rng.integers(0, 2, n))
    lev = {'5': dict(o_m=o_m, r_m=r_m, o_b=o_b, r_b=r_b)}
    pos = [q for q in quids if r_m[q] > r_b[q]]
    try:
        R3.bound_analysis({'m': lev}, quids, rejected=[], reviewed=pos[:-1], B=200)
        raised = False
    except AssertionError as e:
        raised = 'unreviewed rows' in str(e)
    assert raised
    out = R3.bound_analysis({'m': lev}, quids, rejected=pos[:5], reviewed=pos, B=200)
    assert out['m']['5']['c_rw']['bound_label'] in ('BOUND_HOLDS', 'BOUND_FAILS') and out['m']['5']['n'] == n - 5


def test_RM4_wilson_interval():
    lo, hi = R3.wilson(50, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3 and np.isnan(R3.wilson(0, 0)[0])


def test_RM5_parser_first_routing():
    """pf_lib.route: V1 = PALONE; V2 = PALONE unless its parse failed; V3(t) = PK when mean log-prob < t, else V2"""
    PL = load('pf_lib', 'kpct/parser_first/pf_lib.py')
    D = dict(st_PALONE={1: ('ok', ''), 2: ('fallback', 'empty'), 3: ('ok', '')}, mlp={1: -0.5, 2: -0.01, 3: -0.001})
    assert [PL.route(D, 'V1', q) for q in (1, 2, 3)] == ['PALONE'] * 3
    assert [PL.route(D, 'V2', q) for q in (1, 2, 3)] == ['PALONE', 'PK', 'PALONE']
    assert [PL.route(D, 'V3(-0.03)', q) for q in (1, 2, 3)] == ['PK', 'PK', 'PALONE']


def test_RM6_normalizer_objective_and_selection():
    """nm_common.objective / select: max J among configurations within the dev-original loss cap; registered tie order"""
    NC = load('nm_common', 'kpct/normalizer/nm_common.py')
    base = {'P_5': {'orig': 0.90}, 'G_5': {'orig': 0.92}}
    h = {'P_5': {'orig': 0.899, 'para': 0.70}, 'G_5': {'orig': 0.921, 'para': 0.74}, 'base': base}
    J, ok, loss = NC.objective(h)
    assert abs(J - 0.72) < 1e-12 and ok and abs(loss - 0.001) < 1e-12
    c = [dict(ckpt_epoch=1.0, guard='A', tau=0.0, J=0.720, feasible=True, loss=0.001),
         dict(ckpt_epoch=0.5, guard='A', tau=0.2, J=0.7202, feasible=True, loss=0.002),
         dict(ckpt_epoch=0.5, guard='B', tau=0.0, J=0.7203, feasible=True, loss=0.002),
         dict(ckpt_epoch=2.0, guard='B', tau=0.2, J=0.800, feasible=False, loss=0.010)]
    best, status = NC.select(c)
    assert status == 'SELECTED' and best['ckpt_epoch'] == 0.5 and best['guard'] == 'B'
    best2, status2 = NC.select([dict(x, feasible=False) for x in c])
    assert status2 == 'CONSTRAINT_UNMET' and best2['loss'] == 0.001
    assert NC.reuse_source('a', 'a', 'o') == 'input' and NC.reuse_source('a', 'o', 'o') == 'original' and NC.reuse_source('a', 'b', 'o') == 'run'


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS {name}')
