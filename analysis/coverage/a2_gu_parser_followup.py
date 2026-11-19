"""a2_gu_parser_followup.py -- Dense-RAG / Parser part of A2 (protocol/multitq_output_analyses_plan.json). --fetch-dir holds
gu_test.jsonl.gz (Dense-RAG test retrieval), the 9 Parser test prediction files (.gz) and the istats__GU_* infer_stats
files of the Reader runs.

Usage (workspace with multitq_eval.py, a2_coverage.py, a2_tables.py, A2_coverage_per_question.csv, MultiTQ test.json):
  python a2_gu_parser_followup.py --fetch-dir <input dir> --per-question A2_coverage_per_question.csv \
      --test-json test.json --scorer multitq_eval.py --out out_followup
Checks: Parser Hits@1 recomputed from the files == the per-question hits columns (P_k_seed<s>) (always fatal);
gu_test.jsonl.gz and the Parser files against the sha256 of the frozen files of the reported run (GU_SHA, P_FILES):
a mismatch is reported, and is fatal only with EXPECT_FROZEN_HASHES=1 (re-run arms give other files).
Outputs: A2_followup_tables.csv / .md, A2_followup_per_question.csv, A2_followup_summary.json.
"""
import argparse, gzip, hashlib, importlib.util, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a2_coverage as AC
import a2_tables as AT

GU_SHA = '23fef2753162398be1f00c3a0e4793d77311a2608dcd74c6cfc6cfdf0de2cf99'
STRICT = os.environ.get('EXPECT_FROZEN_HASHES') == '1'
P_FILES = {  # (k label, seed) -> (file name, sha256 of the reported run's frozen file (parser freeze))
    ('1', 0): ('pred_P_k01_s0_test_ep3_tauoff_seed0.jsonl', 'ac754e6052dde9e2ae7ff1d4b89c25867e08d96aaa134c24eb09e189f50a65f7'),
    ('1', 1): ('pred_P_k01_s0_test_ep3_tauoff_seed1.jsonl', '4cb872234bc0ae7a35d7ca33567c9d7ab35a9e950006bf713f23520cf1d13307'),
    ('1', 2): ('pred_P_k01_s0_test_ep3_tauoff_seed2.jsonl', '504befde10443231db5cbdcef882763e9c4a228a97f5738b627abc92d5b91e6e'),
    ('5', 0): ('pred_P_k05_s0_test_ep1_tauoff_seed0.jsonl', 'b71dd3c7176f60424dddd9da4ee8e9b2f6b094b17b5231eeba27f28c29efed74'),
    ('5', 1): ('pred_P_k05_s0_test_ep1_tauoff_seed1.jsonl', 'f737a13df950d1b4e6169005229666df16cb79abd283b1dfdfd769f3e8539996'),
    ('5', 2): ('pred_P_k05_s0_test_ep1_tauoff_seed2.jsonl', '86e7de5ed5e8050aca72c34e8534c12d3232e997a61647a353b2aa8c36fe1858'),
    ('100', 0): ('pred_P_k100_test_ep0.5_tauoff_seed0.jsonl', 'dcf083e0258f7133c48216d110100cce162d704d0b643a1eb040e3c79ea03364'),
    ('100', 1): ('pred_P_k100_test_ep0.5_tauoff_seed1.jsonl', 'a961445bd14b658c04f34f2a98e23f2ec58e2b8f3561828feefa765f0e242801'),
    ('100', 2): ('pred_P_k100_test_ep0.5_tauoff_seed2.jsonl', '0e91acd3322d065d819c8818613e94ddafb355028058ef2da8fd622cdc622e5b'),
}


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def gu_truncation(fetch_dir):
    """n_truncated recorded by the GU test inference (infer_stats files fetched from guided/preds)"""
    out = {}
    for fn in os.listdir(fetch_dir):
        if fn.startswith('istats__GU_') and 'test' in fn:
            st = json.load(open(os.path.join(fetch_dir, fn)))
            out[fn] = st.get('n_truncated')
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    for x in ('--fetch-dir', '--per-question', '--test-json', '--scorer', '--out'):
        ap.add_argument(x, required=True)
    ap.add_argument('--gu-keep', help='optional CSV quid,facts_dropped,n_facts from tok_trunc.py --gu-facts')
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    spec = importlib.util.spec_from_file_location('multitq_eval', a.scorer)
    ME = importlib.util.module_from_spec(spec); spec.loader.exec_module(ME)
    gold = {int(q['quid']): q for q in json.load(open(a.test_json))}
    D = pd.read_csv(a.per_question)
    summ = dict(plan='protocol/multitq_output_analyses_plan.json', label='DESCRIPTIVE / POST HOC; reuse of the official test split')
    # ---- GU retrieved-fact coverage
    gp = os.path.join(a.fetch_dir, 'gu_test.jsonl.gz')
    raw = open(gp, 'rb').read()
    summ['gu_test_sha256_ok'] = sha256_bytes(raw) == GU_SHA
    if not summ['gu_test_sha256_ok']:
        msg = 'gu_test.jsonl.gz differs from the frozen file of the reported run'
        if STRICT:
            raise SystemExit('ERROR: ' + msg)
        print('NOTE: ' + msg)
    rows = [json.loads(l) for l in gzip.decompress(raw).decode().splitlines() if l.strip()]
    keep = None
    if a.gu_keep:
        g = pd.read_csv(a.gu_keep); keep = dict(zip(g.quid, g.n_facts - g.facts_dropped))
    summ['gu_infer_stats_n_truncated'] = gu_truncation(a.fetch_dir)
    cov = AC.gu_coverage(rows, gold, ME, keep)
    D['GU_F'] = D.quid.map({q: v['GU_F'] for q, v in cov.items()}).astype(int)
    D['GU_n_facts'] = D.quid.map({q: v['GU_n_facts'] for q, v in cov.items()})
    # ---- Parser own top-20 coverage
    pchk = {}
    for (K, s), (fn, sh) in P_FILES.items():
        b = gzip.open(os.path.join(a.fetch_dir, fn + '.gz')).read()
        pchk[fn] = sha256_bytes(b) == sh
        P = {}
        for l in b.decode().splitlines():
            if l.strip():
                r = json.loads(l); rk = r['ranked']
                P[int(r['quid'])] = json.loads(rk.replace("'", '"')) if isinstance(rk, str) else rk
        h1 = np.array([ME.hit_at_k(gold[q], P.get(q), 1) for q in D.quid])
        pchk[fn + ':hits_equal_frozen_column'] = bool((h1 == D[f'P_{K}_seed{s}'].values).all())
        D[f'Pown20_{K}_seed{s}'] = [ME.hit_at_k(gold[q], P.get(q), 20) for q in D.quid]
    summ['parser_file_checks'] = pchk
    assert all(v for k, v in pchk.items() if k.endswith(':hits_equal_frozen_column')), pchk
    if not all(pchk.values()):
        msg = 'Parser files differ from the frozen files of the reported run: ' + ', '.join(k for k, v in pchk.items() if not v)
        if STRICT:
            raise SystemExit('ERROR: ' + msg)
        print('NOTE: ' + msg)
    for K in ('1', '5', '100'):
        D[f'Pown20_{K}'] = D[[f'Pown20_{K}_seed{s}' for s in range(3)]].mean(1)
    # ---- tables
    T = []
    for K in ('1', '100'):
        cols = [f'GU_{K}', f'G_{K}', f'P_{K}']
        cons = [(f'G_{K}', f'GU_{K}'), (f'P_{K}', f'GU_{K}'), (f'G_{K}', f'P_{K}')]
        for lab, m in (('GU_F1', D.GU_F.values == 1), ('GU_F0', D.GU_F.values == 0)):
            T += AT.cell_rows(D, m, cols, lab, extra=dict(k=K, table='GU_F'), contrasts=cons)
        FC = (D[f'F_{K}'].values == 1) & (D[f'C_{K}'].values == 1)
        for lab, m in (('GU_F1_bundleFC1', (D.GU_F.values == 1) & FC), ('GU_F1_bundleFC0', (D.GU_F.values == 1) & ~FC),
                       ('GU_F0_bundleFC1', (D.GU_F.values == 0) & FC), ('GU_F0_bundleFC0', (D.GU_F.values == 0) & ~FC)):
            T += AT.cell_rows(D, m, cols, lab, extra=dict(k=K, table='GU_F_x_bundleFC'), contrasts=cons)
        T += AT.cell_rows(D, (D.GU_F.values == 1) & FC, cols + [f'S_{K}', f'RULE_{K}'], f'common_covered_k{K}',
                          extra=dict(k=K, table='common_covered_with_GU'), contrasts=cons)
    allk = np.ones(len(D), bool)
    for K in ('1', '5', '100'):
        allk &= (D[f'F_{K}'].values == 1) & (D[f'C_{K}'].values == 1)
    allk &= D.GU_F.values == 1
    colsall = [f'{x}_{K}' for x in ('G', 'P', 'S', 'RULE') for K in ('1', '5', '100')] + ['GU_1', 'GU_100']
    T += AT.cell_rows(D, allk, colsall, 'common_covered_all_k_and_GU_F', extra=dict(k='all', table='common_covered_with_GU'),
                      contrasts=[('G_1', 'GU_1'), ('G_100', 'GU_100'), ('P_1', 'GU_1'), ('P_100', 'GU_100'), ('G_1', 'P_1'), ('G_5', 'P_5'), ('G_100', 'P_100')])
    for K in ('1', '5', '100'):
        po = D[f'Pown20_{K}'].values
        for lab, m in (('Pown20_all_seeds', po == 1), ('Pown20_no_seed', po == 0), ('Pown20_some_seeds', (po > 0) & (po < 1))):
            T += AT.cell_rows(D, m, [f'P_{K}', f'G_{K}'], lab, extra=dict(k=K, table='Parser_own_top20'), contrasts=[(f'G_{K}', f'P_{K}')])
    T = pd.DataFrame(T)
    succ = []
    for K in ('1', '100'):
        h = D[f'GU_{K}'].values; m = D.GU_F.values == 0
        succ.append(dict(k=K, arm='GU', cell='GU_F0', n_cell=int(m.sum()), successes=float(h[m].sum()), rate_in_cell=float(h[m].mean()),
                         share_of_arm_successes=float(h[m].sum() / h.sum())))
    summ['coverage_rates'] = dict(GU_F=float(D.GU_F.mean()), **{f'Pown20_{K}_mean_over_seeds': float(D[f'Pown20_{K}'].mean()) for K in ('1', '5', '100')})
    summ['GU_successes_without_retrieved_coverage'] = succ
    T.to_csv(os.path.join(a.out, 'A2_followup_tables.csv'), index=False, float_format='%.6f')
    D[['quid', 'GU_F', 'GU_n_facts'] + [c for c in D.columns if c.startswith('Pown20_')]].to_csv(os.path.join(a.out, 'A2_followup_per_question.csv'), index=False, float_format='%.6f')
    json.dump(summ, open(os.path.join(a.out, 'A2_followup_summary.json'), 'w'), indent=1)
    L = ['# A2 follow-up: Dense-RAG retrieved-fact coverage and Parser own-list coverage', '',
         'Descriptive; reuse of the official MultiTQ test split. Values in percent, 95% Poisson-bootstrap CI (B = 2,000, seed 0). String presence does not certify semantic support.', '',
         '| Table | k | Cell | n | Quantity | Hits@1 / contrast |', '|---|---|---|---|---|---|']
    for _, r in T.iterrows():
        L.append(f"| {r.table} | {r.k} | {r.cell} | {r.n:,} | {r.quantity} | {100 * r.value:.1f} [{100 * r.lo:.1f}, {100 * r.hi:.1f}] |")
    L += ['', f"GU_F rate: {100 * summ['coverage_rates']['GU_F']:.2f}% of test questions."]
    for s in succ:
        L.append(f"GU_{s['k']} successes without retrieved coverage: {s['successes']:.0f} of {s['n_cell']:,} uncovered questions ({100 * s['rate_in_cell']:.1f}%), {100 * s['share_of_arm_successes']:.2f}% of GU_{s['k']} successes.")
    open(os.path.join(a.out, 'A2_followup_tables.md'), 'w').write('\n'.join(L) + '\n')
    print(json.dumps({k: v for k, v in summ.items() if k != 'parser_file_checks'}))


if __name__ == '__main__':
    main()
