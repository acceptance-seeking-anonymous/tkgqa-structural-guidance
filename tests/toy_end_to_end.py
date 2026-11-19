#!/usr/bin/env python3
"""toy_end_to_end.py -- data-free smoke test of the scoring path used by the test pass: 20 synthetic MultiTQ-style
questions and three synthetic arms are written to a temporary directory, scored with the official scorer
(multitq_eval.load_predictions / hit_at_k / score), compared with the paired Poisson(1) bootstrap (B = 2000, seed 0)
and Holm-adjusted with the test-pass implementation. Checks hand-computable values. Nothing is written to the repo."""
import importlib.util
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(REPO, 'src'))
import multitq_eval as M  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


TP = load('multitq_kpct_test_pass', os.path.join(REPO, 'kpct', 'test_pass', 'multitq_kpct_test_pass.py'))
ENTS = ['China', 'Japan', 'Iraq', 'Malaysia', 'Party Member (Iraq)']


def main():
    qs = []
    for i in range(20):
        ent = i % 2 == 0
        qs.append(dict(quid=100 + i, question=f'Synthetic question {i}?', answers=[ENTS[i % 5]] if ent else [f'20{10 + i % 5}-0{1 + i % 9}'],
                       answer_type='entity' if ent else 'time', time_level='day' if ent else 'month',
                       qtype='equal' if i < 12 else 'before_after', qlabel='Single' if i < 12 else 'Multiple'))
    gold = {q['quid']: q for q in qs}
    # arm A: correct on the first 15 questions; arm B: correct on the first 10; arm C = arm B
    def ranked(q, ok):
        a = q['answers'][0]
        return [a, 'Japan'] if ok else ['Nowhere', a]
    arms = {'A': {q['quid']: ranked(q, i < 15) for i, q in enumerate(qs)},
            'B': {q['quid']: ranked(q, i < 10) for i, q in enumerate(qs)}}
    arms['C'] = dict(arms['B'])
    with tempfile.TemporaryDirectory() as d:
        json.dump(qs, open(os.path.join(d, 'test.json'), 'w'))
        H = {}
        for a, P in arms.items():
            p = os.path.join(d, f'pred_{a}.jsonl')
            with open(p, 'w') as f:
                for q, r in P.items():
                    f.write(json.dumps(dict(quid=q, ranked=r)) + '\n')
            L = M.load_predictions(p)
            H[a] = [M.hit_at_k(gold[q], L.get(q), 1) for q in sorted(gold)]
            h10 = [M.hit_at_k(gold[q], L.get(q), 10) for q in sorted(gold)]
            summary, _ = M.score(qs, L, ks=(1, 10))
            assert abs(summary['hits@1']['overall']['hits'] - sum(H[a]) / 20) < 1e-12
            assert sum(h10) == 20, (a, sum(h10))
    assert sum(H['A']) == 15 and sum(H['B']) == 10 and H['B'] == H['C']
    ab = M.paired_poisson_bootstrap(H['A'], H['B'], B=2000, seed=0)
    bc = M.paired_poisson_bootstrap(H['B'], H['C'], B=2000, seed=0)
    # A wins 5 discordant questions and loses none, so a resample has delta <= 0 only when all five Poisson(1)
    # weights are 0 (probability e^-5 = 0.0067); B = 2000 with seed 0 gives p = 10 / 2000 = 0.005.
    wins = sum(x > y for x, y in zip(H['A'], H['B'])); losses = sum(x < y for x, y in zip(H['A'], H['B']))
    assert (wins, losses) == (5, 0), (wins, losses)
    assert abs(ab['delta'] - 0.25) < 1e-12 and ab['ci_high'] > ab['ci_low'] > 0, ab
    assert 0 < ab['p_one_sided_le0'] < 0.02, ab
    assert bc['delta'] == 0.0 and bc['ci_low'] == bc['ci_high'] == 0.0 and bc['p_one_sided_le0'] == 1.0
    adj = TP.holm({'KP2': 0.01, 'KP3': 0.04, 'KP4': 0.03})
    assert abs(adj['KP2'] - 0.03) < 1e-12 and abs(adj['KP4'] - 0.06) < 1e-12 and abs(adj['KP3'] - 0.06) < 1e-12
    assert TP.kp1_label(0.90, 0.92, 0.887) == 'ABOVE' and TP.kp1_label(0.80, 0.82, 0.887) == 'BELOW'
    print(json.dumps(dict(toy_hits1={a: sum(v) / 20 for a, v in H.items()}, A_minus_B=ab, holm=adj)))
    print('toy end-to-end ok')


if __name__ == '__main__':
    main()
