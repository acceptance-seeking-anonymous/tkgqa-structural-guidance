"""pf_freeze.py -- Parser-first track: gold-free routing table of every candidate, materialised prediction files of the
SELECTED variant at each level (test, R3 / dev rewrite root, R1, R2), and the sha256 freeze written BEFORE the single
scoring pass. Reads no gold.  usage: python pf_freeze.py <selection.json> <out_dir>"""
import collections, datetime, gzip, hashlib, json, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pf_lib as PL
WS = PL.W
sel_p, OUT = sys.argv[1], sys.argv[2]
SEL = json.load(open(sel_p))
files = []
def add(path, role):
    files.append(dict(path=os.path.relpath(path, WS) if path.startswith(WS) else path, role=role, sha256=PL.sha256_file(path)))

# R3 PRIMARY squids from the R3 set file
R3SET = json.load(open(os.environ['PF_R3_SET']))
r3sq = sorted(int(v['squid']) for v in R3SET['variants'] if v['variant'] == 'RW3' and int(v['squid']) % 10 == 4)
assert len(r3sq) == 578

rows = []
mat_dir = os.path.join(OUT, 'selected_preds'); os.makedirs(mat_dir, exist_ok=True)
for split, ks in (('test', PL.KS), ('var', PL.KREW), ('R1', PL.KREW), ('R2', PL.KREW)):
    for k in ks:
        D = PL.load(split, k)
        for role, p in D['files'].items():
            add(p, f'source:{split}:{k}:{role}')
        assert all(set(D['quids']) <= set(D['V0'][s]) for s in range(3)), f'V0 coverage {split} {k}'
        if split == 'var':
            assert set(r3sq) <= set(D['quids'])
        R = {c: [PL.route(D, c, q) for q in D['quids']] for c in PL.CANDS if c != 'V0'}
        for i, q in enumerate(D['quids']):
            rows.append(dict(split=split, k=k, quid=q, st_PK=D['st_PK'][q][0], st_PALONE=D['st_PALONE'][q][0],
                             reason_PALONE=D['st_PALONE'][q][1], mean_logprob=D['mlp'][q], **{f'route_{c}': R[c][i] for c in R}))
        m = SEL['selection'][k]['selected']
        if m != 'V0':
            Ls = PL.lists(D, m)
            for s in range(3):
                p = os.path.join(mat_dir, f'pred_PF_{m}_{k}_{split}_seed{s}.jsonl.gz')
                with gzip.open(p, 'wt', compresslevel=6) as f:
                    for q in D['quids']:
                        f.write(json.dumps({'quid': q, 'ranked': Ls[s][q]}, ensure_ascii=False) + '\n')
                add(p, f'selected:{split}:{k}:{m}:seed{s}')
        print(split, k, m, len(D['quids']), flush=True)
rt = os.path.join(OUT, 'multitq_parser_first_routing.csv.gz')
pd.DataFrame(rows).to_csv(rt, index=False, compression='gzip')
add(rt, 'routing_table')
add(sel_p, 'selection')
HERE = os.path.dirname(os.path.abspath(__file__))
for fn in ('pf_lib.py', 'pf_select.py', 'pf_freeze.py', 'pf_score.py'):
    add(os.path.join(HERE, fn), f'code:{fn}')
add(os.path.join(WS, 'multitq_eval.py'), 'code:multitq_eval.py')
fz = collections.OrderedDict(track='multitq_parser_first', stage='freeze before the single scoring pass',
     rule_artifact='protocol/multitq_parser_first_rule.json', rule_sha256='41a4f38cdebd768b298109c00b9038eb16763a6ff08286bd13fd6c12eaa1e191',
     selection_artifact='multitq_parser_first_selection.json', selection_path=os.path.relpath(sel_p, WS) if sel_p.startswith(WS) else sel_p,
     selected={k: SEL['selection'][k]['selected'] for k in PL.KS}, R3_primary_squids=r3sq,
     note='Levels whose selected variant is V0 use the frozen P_k files (listed as sources). No test / R1 / R2 / R3 gold has been read by this track at freeze time.',
     frozen_utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), n_files=len(files), files=files)
raw = json.dumps(fz, indent=1, sort_keys=True)
fz['sha256'] = hashlib.sha256(raw.encode()).hexdigest()
json.dump(fz, open(os.path.join(OUT, 'multitq_parser_first_freeze.json'), 'w'), indent=1)
print('frozen', fz['frozen_utc'], fz['sha256'], len(files))
