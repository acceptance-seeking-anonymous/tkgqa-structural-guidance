"""reader_mechanism.py -- post-hoc Reader_1 (G_1) mechanism numbers: copy / fix / loss against Ranker seed 0 (S_1 seed 0).

Reads only frozen k% artifacts (no test answers beyond the stored per-question hits):
  multitq_kpct_test_per_question_hits.csv.gz  (official per-question Hits@1 of the k% test pass)
  multitq_kpct_guided_predictions.tgz         (G_k01_s0/pred_G_test_ordS_ep2.jsonl = frozen G_1 test ranked lists)
  multitq_kpct_structural_bundles.tar         (preds/k01_s0/test_S_seed0.jsonl.xz = S_1 seed-0 ranked lists;
                                               bundles/k01_s0/test.jsonl.xz = the <= 20 structural candidates shown to G)
Usage: python reader_mechanism.py HITS_CSV_GZ GUIDED_TGZ BUNDLES_TAR OUT_JSON
"""
import json, lzma, os, sys, tarfile
import pandas as pd

hits_p, guided_p, bundles_p, out_p = sys.argv[1:5]
ph = pd.read_csv(hits_p).set_index('quid')
with tarfile.open(bundles_p) as t:
    S0 = {r['quid']: r['ranked'] for r in map(json.loads, lzma.decompress(
        t.extractfile('preds/k01_s0/test_S_seed0.jsonl.xz').read()).decode().splitlines())}
    BC = {r['quid']: [c['answer'] for c in r['candidates']] for r in map(json.loads, lzma.decompress(
        t.extractfile('bundles/k01_s0/test.jsonl.xz').read()).decode().splitlines())}
with tarfile.open(guided_p) as t:
    G1 = {r['quid']: r['ranked'] for r in map(json.loads, t.extractfile('G_k01_s0/pred_G_test_ordS_ep2.jsonl'))}
q = list(ph.index)
assert set(q) == set(S0) == set(G1) == set(BC) and len(q) == 54584
g, s = ph['G_1@1'].astype(int), ph['S_1_seed0@1'].astype(int)
fix = [x for x in q if g[x] == 1 and s[x] == 0]
loss = [x for x in q if g[x] == 0 and s[x] == 1]
out = dict(
    n=len(q), G_1_hits1=float(g.mean()), S_1_seed0_hits1=float(s.mean()),
    copy_S_top1_share=sum(G1[x][0] == S0[x][0] for x in q) / len(q),
    fixes=len(fix), losses=len(loss), net=len(fix) - len(loss),
    fixes_answer_in_bundle_candidates=sum(G1[x][0] in BC[x] for x in fix),
    fixes_answer_in_S_seed0_top20=sum(G1[x][0] in S0[x][:20] for x in fix),
    questions_with_empty_bundle_candidates=sum(len(BC[x]) == 0 for x in q),
    definitions=dict(copy='G_1 rank-1 string equals S_1 seed-0 rank-1 string',
                     fix='G_1 Hits@1 = 1 and S_1 seed-0 Hits@1 = 0 (official per-question hits)',
                     loss='G_1 Hits@1 = 0 and S_1 seed-0 Hits@1 = 1',
                     in_bundle='G_1 rank-1 answer among the <= 20 structural candidates of its bundle',
                     in_top20='G_1 rank-1 answer among the first 20 entries of the S_1 seed-0 ranked list (incl. padding)'),
    status='post hoc, descriptive; not a registered criterion')
if abs(out['G_1_hits1'] - 0.799171918510919) > 1e-9:   # frozen G_1 test Hits@1
    msg = f"G_1 Hits@1 = {out['G_1_hits1']:.6f} differs from the frozen G_1 predictions (0.799172)"
    if os.environ.get('EXPECT_FROZEN_G1') == '1':
        raise SystemExit('ERROR: ' + msg)
    print('NOTE: ' + msg + ' (expected for re-trained arms; set EXPECT_FROZEN_G1=1 to make this fatal)')
json.dump(out, open(out_p, 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'definitions'}))
