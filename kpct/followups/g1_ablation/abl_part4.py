#!/usr/bin/env python3
"""abl_part4.py -- G_1 vs S_1 seed-0 mechanism counts from EXISTING frozen predictions (no new inference).

Rule multitq_g1_ablation_rule.json, existing_prediction_analysis. Per split (dev_sel2000, test):
copy rate (G_1 rank-1 == S0 rank-1, exact string), fixes (S0 miss, G_1 hit at Hits@1), losses, net, fixes whose G_1
answer is in S0's top 20 (two readings: S0 ranked list[:20] incl. padding; the bundle's <= 20 candidates), plus the
gold-string coverage of the structural provenance facts and of the GU facts (all 20, and the first n = number of
structural facts). Test predictions were scored once in the k% pass; this is a descriptive re-analysis.
"""
import argparse, gzip, json, os, sys, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gk_common as GC
import gk_ablation as AB


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def ranked_file(p):
    out = {}
    for x in GC.read_jsonl(p):
        rk = x['ranked']
        out[int(x['quid'])] = json.loads(rk.replace("'", '"')) if isinstance(rk, str) else rk
    return out


def analyse(name, g_pred, s0_pred, bundles, gu, questions, ME):
    G, S = ranked_file(g_pred), ranked_file(s0_pred)
    Q = {q['quid']: q for q in json.load(open(questions)) if q['quid'] in G}
    assert set(G) == set(S) == set(Q), (len(G), len(S), len(Q))
    B = {}
    for r in GC.read_jsonl(bundles):
        b = GC.normalize_bundle(r)
        B[b['quid']] = b
    U = {}
    for x in GC.read_jsonl(gu):
        U[int(x['quid'])] = [tuple(f) for f in x['facts']]
    assert set(B) == set(G) and set(U) == set(G)
    rows = []
    for q in sorted(G):
        g, s, qq, b = G[q], S[q], Q[q], B[q]
        hg, hs = ME.hit_at_k(qq, g, 1), ME.hit_at_k(qq, s, 1)
        cands = [c['candidate'] for c in b['candidates']]
        gold = qq['answers']
        struct_groups = GC.group_facts(b['facts'])[0]
        gu_all = GC.group_facts(U[q])[0]
        gu_n = GC.group_facts(U[q][:len(b['facts'])])[0]
        rows.append(dict(quid=q, qtype=qq['qtype'], answer_type=qq['answer_type'], g_hit=bool(hg), s0_hit=bool(hs),
                         copy=g[0] == s[0], g_in_s0top20=g[0] in s[:20], g_in_bundle_cands=g[0] in cands,
                         g_hit_equiv_copy=bool(hg) == bool(hs),
                         gold_struct_facts=AB.gold_in_groups(gold, struct_groups), gold_gu20=AB.gold_in_groups(gold, gu_all),
                         gold_gu_n=AB.gold_in_groups(gold, gu_n), gold_in_cands=any(GC.gold_hit(c, gold) for c in cands),
                         n_struct_facts=len(b['facts'])))
    n = len(rows)
    fx = [r for r in rows if r['g_hit'] and not r['s0_hit']]
    ls = [r for r in rows if r['s0_hit'] and not r['g_hit']]
    out = dict(split=name, n=n,
               inputs={k: dict(file=v, sha256=sha(v)) for k, v in (('G_1', g_pred), ('S0', s0_pred), ('bundles', bundles), ('gu', gu))},
               G1_hits1=float(np.mean([r['g_hit'] for r in rows])), S0_hits1=float(np.mean([r['s0_hit'] for r in rows])),
               copy_rate=float(np.mean([r['copy'] for r in rows])), n_copy=int(sum(r['copy'] for r in rows)),
               fixes=len(fx), losses=len(ls), net=len(fx) - len(ls),
               fixes_in_S0_top20_list=int(sum(r['g_in_s0top20'] for r in fx)),
               fixes_in_bundle_candidates=int(sum(r['g_in_bundle_cands'] for r in fx)),
               fixes_copy=int(sum(r['copy'] for r in fx)),
               noncopy_n=int(sum(not r['copy'] for r in rows)),
               noncopy_in_S0_top20_list=int(sum(r['g_in_s0top20'] for r in rows if not r['copy'])),
               noncopy_in_bundle_candidates=int(sum(r['g_in_bundle_cands'] for r in rows if not r['copy'])),
               copy_but_hit_differs=int(sum(r['copy'] and r['g_hit'] != r['s0_hit'] for r in rows)),
               gold_coverage=dict(structural_facts=float(np.mean([r['gold_struct_facts'] for r in rows])),
                                  gu_facts_top20=float(np.mean([r['gold_gu20'] for r in rows])),
                                  gu_facts_same_count=float(np.mean([r['gold_gu_n'] for r in rows])),
                                  candidates_top20=float(np.mean([r['gold_in_cands'] for r in rows]))),
               struct_facts_count={str(k): int(v) for k, v in zip(*np.unique([r['n_struct_facts'] for r in rows], return_counts=True))},
               by_qtype={})
    for qt in sorted({r['qtype'] for r in rows}):
        rr = [r for r in rows if r['qtype'] == qt]
        out['by_qtype'][qt] = dict(n=len(rr), copy_rate=float(np.mean([r['copy'] for r in rr])),
                                   fixes=int(sum(r['g_hit'] and not r['s0_hit'] for r in rr)),
                                   losses=int(sum(r['s0_hit'] and not r['g_hit'] for r in rr)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--K', default='/kpct'); ap.add_argument('--data', default='/data')
    ap.add_argument('--eval', required=True); ap.add_argument('--out', default='.')
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(a.eval)))
    import multitq_eval as ME
    K = a.K
    res = dict(rule='multitq_g1_ablation_rule.json existing_prediction_analysis', eval_sha256=sha(a.eval), splits={})
    res['splits']['dev_sel2000'] = analyse('dev_sel2000', f'{K}/guided/preds/G_k01_s0/pred_G_dev_sel2000_ordS_ep2.jsonl',
                                           f'{K}/preds/k01_s0/dev_sel2000_S_seed0.jsonl', f'{K}/bundles/k01_s0/dev_sel2000.jsonl',
                                           f'{K}/gu_retrieval/gu_dev_sel2000.jsonl.gz', f'{a.data}/questions/dev.json', ME)
    res['splits']['test'] = analyse('test', f'{K}/guided/preds/G_k01_s0/pred_G_test_ordS_ep2.jsonl',
                                    f'{K}/preds/k01_s0/test_S_seed0.jsonl', f'{K}/bundles/k01_s0/test.jsonl',
                                    f'{K}/gu_retrieval/gu_test.jsonl.gz', f'{a.data}/questions/test.json', ME)
    json.dump(res, open(os.path.join(a.out, 'abl_part4.json'), 'w'), indent=1)
    print(json.dumps({s: {k: v for k, v in d.items() if k not in ('inputs', 'by_qtype')} for s, d in res['splits'].items()}, indent=1))


if __name__ == '__main__':
    main()
