"""pf_select.py -- Parser-first track, registered dev selection pass (rule v1.0, sha256 41a4f38c...).
1) gold-free integrity checks of every source file (sha256 vs the replay stats / freeze_A / R3 freeze / R1-R2 archives);
2) dev scoring of every candidate on dev_sel2000 ORIG (2,000) and DEV_ACCEPTED paraphrases (1,762) at k = 1/5/100 and
   dev ORIG at k = 10; 3) registered selection + k10 transfer; writes multitq_parser_first_selection.json.
Reads ONLY dev.json gold. Official multitq_eval.hit_at_k."""
import collections, glob, hashlib, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pf_lib as PL
WS = PL.W
sys.path.insert(0, WS)
import multitq_eval as ME
assert PL.sha256_file(os.path.join(WS, 'multitq_eval.py')) == '1f85bbdcd623f1810e9df2910e042c24d6f5c269d278e82b30ef05387a1e2d69'
OUT = sys.argv[1]
RULE_SHA = '41a4f38cdebd768b298109c00b9038eb16763a6ff08286bd13fd6c12eaa1e191'
integ = collections.OrderedDict()

# ---------------------------------------------------------------- 1) integrity (gold-free)
def chk(name, path, expect):
    h = PL.sha256_file(path); ok = (h == expect)
    integ[name] = dict(path=os.path.relpath(path, WS) if path.startswith(WS) else path, sha256=h, expected=expect, ok=ok)
    assert ok, f'STOP_TEST_INTEGRITY {name}'

rp = os.path.join(WS, 'replay', 'out')
inputs_sha = {}
for l in open(os.path.join(WS, 'replay', 'inputs_sha.txt')):
    h, p = l.split(); inputs_sha[p] = h
for d in sorted(os.listdir(rp)):
    st = json.load(open(glob.glob(os.path.join(rp, d, 'replay_stats_*.json'))[0]))
    for fn, h in st['files'].items():
        p = os.path.join(rp, d, fn)
        if os.path.exists(p):
            chk(f'replay/{d}/{fn}', p, h)
    for s, v in st['identity_PK_vs_frozen'].items():
        assert v['top1'] >= 0.995 and v['n_found'] == st['n'], f'STOP_PIPELINE_DRIFT {d} {s}'
    integ[f'replay/{d}/identity_PK_vs_frozen'] = st['identity_PK_vs_frozen']
    k = st['k']
    if st['split'] == 'dev_sel2000':
        chk(f'raw/dev/{k}', PL.sources('dev', k)['files']['raw'], st['raw_sha256'])
        for s in range(3):
            chk(f'V0/dev/{k}/s{s}', PL.sources('dev', k)['files'][f'V0_s{s}'], inputs_sha[f'/kpct/parser/exec/{k}/pred_P_{k}_dev_sel2000_{PL.CKPT[k]}_tauoff_seed{s}.jsonl'])
    else:
        chk(f'raw/var/{k}', PL.sources('var', k)['files']['raw'], st['raw_sha256'])
        for s in range(3):
            chk(f'V0/var/{k}/s{s}', PL.sources('var', k)['files'][f'V0_s{s}'], inputs_sha[f'/kpct/robustness_r3/llm/P/{k}/pred_P_{k}_variants_P_{k}_tauoff_seed{s}.jsonl'])
# structural-ablation sources (test / R1 / R2): PK / PALONE fb files vs freeze_A; raw vs its replay stats
FZA = json.load(open(os.environ['PF_FREEZE_A']))
FZI = {(f['dir'], f['file']): f['sha256'] for f in FZA['files']}
saA = os.path.join(WS, 'saA', 'multitq_structural_ablation_predictions_A')
for split in ('test', 'R1', 'R2'):
    for k in (PL.KS if split == 'test' else PL.KREW):
        S = PL.sources(split, k)['files']
        for m in ('PK', 'PALONE'):
            for s in range(3):
                chk(f'{m}/{split}/{k}/s{s}', S[f'{m}_s{s}'], FZI[(f'A_{split}', os.path.basename(S[f'{m}_s{s}']))])
        st = json.load(open(os.path.join(saA, 'stats', f'A_{split}_replay_stats_{k}_{split}.json')))
        chk(f'raw/{split}/{k}', S['raw'], st['raw_sha256'])
        integ[f'pools/{split}/{k}'] = dict(path=os.path.relpath(S['pools_npz'], WS), sha256=PL.sha256_file(S['pools_npz']), note='structural-ablation predictions_A archive member; status read from PK_meta / PALONE_meta')
        if split == 'test':
            for s in range(3):
                chk(f'V0/test/{k}/s{s}', S[f'V0_s{s}'], st['identity_PK_vs_frozen'][f'seed{s}']['frozen_sha256'])
        else:
            for s in range(3):   # frozen R1 / R2 P_k rewrite files: identity record of the structural-ablation replay
                chk(f'V0/{split}/{k}/s{s}', S[f'V0_s{s}'], st['identity_PK_vs_frozen'][f'seed{s}']['frozen_sha256'])
print('integrity ok', len(integ), flush=True)

# ---------------------------------------------------------------- 2) dev scoring (dev gold only)
DEVJ = os.environ['MTQ_DEV_JSON']
assert PL.sha256_file(DEVJ) == 'ad0dfe7c94dfdd7c00b4df22896d8cb3df3b860f1e5e10cd1ae110ab9791bbb7'
DQ = {int(q['quid']): q for q in json.load(open(DEVJ))}
def h_at(Ls, gold_of, quids, k):
    return np.mean([[ME.hit_at_k(gold_of(q), L[q], k) for q in quids] for L in Ls], 0)

dev_tab = collections.OrderedDict(); perq = collections.OrderedDict()
for k in PL.KS:
    Do = PL.load('dev', k)
    oq = Do['quids']; assert len(oq) == 2000
    Dv = PL.load('var', k) if k in PL.KREW else None
    if Dv is not None:
        pq = [q for q in Dv['quids'] if q % 10 == 5]; jq = [q for q in Dv['quids'] if q % 10 == 6]
        assert len(pq) == 1762 and len(jq) == 192
        acc_o = [q // 10 for q in pq]
    perq.setdefault('quid', oq)
    rows = collections.OrderedDict()
    for c in PL.CANDS:
        Lo = PL.lists(Do, c)
        h1o = h_at(Lo, lambda q: DQ[q], oq, 1); h10o = h_at(Lo, lambda q: DQ[q], oq, 10)
        r = dict(dev_orig_h1=float(h1o.mean()), dev_orig_h1_per_seed=[float(np.mean([ME.hit_at_k(DQ[q], Lo[s][q], 1) for q in oq])) for s in range(3)],
                 dev_orig_h10=float(h10o.mean()), routing_dev_orig=PL.routing_share(Do, c, oq))
        perq[f'{c}_{PL.LEV[k]}_orig_h1'] = np.round(h1o, 6).tolist()
        if Dv is not None:
            Lp = PL.lists(Dv, c, pq + jq)
            h1p = h_at(Lp, lambda s: DQ[s // 10], pq, 1); h10p = h_at(Lp, lambda s: DQ[s // 10], pq, 10)
            h1j = h_at(Lp, lambda s: DQ[s // 10], jq, 1)
            idx = {q: i for i, q in enumerate(oq)}
            h1ao = h1o[[idx[q] for q in acc_o]]
            r.update(para_h1=float(h1p.mean()), para_h1_per_seed=[float(np.mean([ME.hit_at_k(DQ[s_ // 10], Lp[s][s_], 1) for s_ in pq])) for s in range(3)],
                     para_h10=float(h10p.mean()), acc_orig_h1=float(h1ao.mean()), drop_acc=float((h1ao - h1p).mean()),
                     mech_ok_para_h1=float(np.concatenate([h1p, h1j]).mean()), n_para=len(pq), n_mech_ok=len(pq) + len(jq),
                     routing_dev_para=PL.routing_share(Dv, c, pq))
            m = {q: v for q, v in zip(acc_o, h1p)}
            perq[f'{c}_{PL.LEV[k]}_para_h1'] = [round(float(m[q]), 6) if q in m else None for q in oq]
        rows[c] = r
    dev_tab[k] = rows
    print(k, {c: (round(v['dev_orig_h1'], 4), round(v.get('para_h1', float('nan')), 4)) for c, v in rows.items()}, flush=True)

# ---------------------------------------------------------------- 3) registered selection
sel = collections.OrderedDict()
for k in PL.KREW:
    rows = dev_tab[k]; base = rows['V0']['dev_orig_h1']
    feas = [c for c in PL.CANDS if rows[c]['dev_orig_h1'] >= base - 0.003 - 1e-12]
    best = max(rows[c]['para_h1'] for c in feas)
    tied = [c for c in feas if rows[c]['para_h1'] >= best - 0.0005 - 1e-12]
    choice = tied[0]   # CANDS is the registered tie order
    sel[k] = dict(level=PL.LEV[k], feasible=feas, infeasible=[c for c in PL.CANDS if c not in feas], best_para_h1=best, tied=tied, selected=choice,
                  selected_dev_orig_h1=rows[choice]['dev_orig_h1'], V0_dev_orig_h1=base, selected_para_h1=rows[choice]['para_h1'], V0_para_h1=rows['V0']['para_h1'])
fam = lambda c: 'V3' if c.startswith('V3') else c
fams = [fam(sel[k]['selected']) for k in PL.KREW]
cnt = collections.Counter(fams)
top = [f for f, n in cnt.items() if n >= 2]
if top:
    f10 = top[0]
else:
    order = ['V0', 'V1', 'V2', 'V3']; f10 = sorted(fams, key=order.index)[0]
if f10 == 'V3':
    ts = sorted(float(sel[k]['selected'][3:-1]) for k in PL.KREW if sel[k]['selected'].startswith('V3'))
    t10 = ts[(len(ts) - 1) // 2]          # median; the lower of the two middle values when even
    c10 = f'V3({t10})'
else:
    c10 = f10
r10 = dev_tab['k10_s0']
ok10 = r10[c10]['dev_orig_h1'] >= r10['V0']['dev_orig_h1'] - 0.003 - 1e-12
sel['k10_s0'] = dict(level='10', rule='k10_transfer', families_at_1_5_100=fams, transferred=c10, constraint_ok=bool(ok10),
                     selected=c10 if ok10 else 'V0', selected_dev_orig_h1=r10[c10 if ok10 else 'V0']['dev_orig_h1'], V0_dev_orig_h1=r10['V0']['dev_orig_h1'])
print('SELECTED', {k: v['selected'] for k, v in sel.items()}, flush=True)

res = dict(track='multitq_parser_first', stage='dev selection (registered)', rule_sha256=RULE_SHA,
           rule_artifact='protocol/multitq_parser_first_rule.json', evaluator_sha256='1f85bbdcd623f1810e9df2910e042c24d6f5c269d278e82b30ef05387a1e2d69',
           dev_json_sha256='ad0dfe7c94dfdd7c00b4df22896d8cb3df3b860f1e5e10cd1ae110ab9791bbb7', candidates=PL.CANDS,
           dev_table=dev_tab, selection=sel, integrity=integ,
           note='Hits@1 = mean over the 3 S_k seeds; dev ORIG = dev_sel2000 (2,000); PARA = DEV_ACCEPTED (1,762, vidx 5); MECH_OK = vidx 5 union 6 (descriptive); no test / R1 / R2 / R3 gold read.')
json.dump(res, open(os.path.join(OUT, 'multitq_parser_first_selection.json'), 'w'), indent=1)
import pandas as pd
pd.DataFrame(perq).to_csv(os.path.join(OUT, 'multitq_parser_first_dev_hits.csv.gz'), index=False, compression='gzip')
