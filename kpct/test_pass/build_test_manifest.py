#!/usr/bin/env python3
"""build_test_manifest.py -- build kpct_test_manifest.json (the list of every registered test file and its sha256)
verbatim from the five freeze records. The manifest is the input of `multitq_kpct_test_pass.py remote`; paths are
relative to the project root mounted at /mtq (kpct/... = $WORK_DIR, phase1_structural/... = full-data exports).

usage: python build_test_manifest.py --structural multitq_kpct_structural_freeze.json \
         --phase1 multitq_structural_freeze.json --guided multitq_kpct_guided_freeze.json \
         --parser multitq_kpct_parser_freeze.json --k0 multitq_kpct_k0_freeze.json --out kpct_test_manifest.json
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    for k in ('structural', 'phase1', 'guided', 'parser', 'k0'):
        ap.add_argument(f'--{k}', required=True)
    ap.add_argument('--out', default='kpct_test_manifest.json')
    a = ap.parse_args()
    F_struct = json.load(open(a.structural))
    F_p1 = json.load(open(a.phase1))
    F_g = json.load(open(a.guided))
    F_p = json.load(open(a.parser))
    F_k0 = json.load(open(a.k0))

    KTAG = {'k01_s0': '1', 'k01_s1': '1_s1', 'k01_s2': '1_s2', 'k05_s0': '5', 'k10_s0': '10', 'k100': '100'}
    M = []  # entries: arm, role, path (relative to the project root), sha256, source

    for k in ['k01_s0', 'k05_s0', 'k10_s0', 'k01_s1', 'k01_s2']:
        pf = F_struct['arms'][k]['prediction_files']
        t = KTAG[k]
        M.append(dict(arm=f'RULE_{t}', role='pred', path=f"kpct/{pf['test_RULE']['file']}", sha256=pf['test_RULE']['sha256'], src='structural_freeze'))
        for s in range(3):
            e = pf[f'test_S_seed{s}']
            M.append(dict(arm=f'S_{t}_seed{s}', role='pred', path=f"kpct/{e['file']}", sha256=e['sha256'], src='structural_freeze'))
        e = pf['test_S_mean']
        M.append(dict(arm=f'SMEANFILE_{t}', role='pred_descriptive', path=f"kpct/{e['file']}", sha256=e['sha256'], src='structural_freeze'))

    # k100: full-data (phase-1) freeze, cross-checked against the k% structural freeze k100 entries
    p1 = F_p1['sha256']['test_predictions']
    s100 = F_struct['arms']['k100']['prediction_files']
    for nm, arm in [('test_RULE', 'RULE_100'), ('test_S_seed0', 'S_100_seed0'), ('test_S_seed1', 'S_100_seed1'), ('test_S_seed2', 'S_100_seed2')]:
        h = p1[f'preds/{nm}.jsonl']
        assert h == s100[nm]['sha256'] == s100[nm]['frozen_sha256'], nm
        M.append(dict(arm=arm, role='pred', path=f'phase1_structural/preds/{nm}.jsonl', sha256=h, src='phase1_structural_freeze (= kpct structural freeze k100)'))

    # Reader (G) / Dense-RAG (GU)
    GMAP = {'G_k01_s0': 'G_1', 'G_k01_s1': 'G_1_s1', 'G_k01_s2': 'G_1_s2', 'G_k05_s0': 'G_5', 'G_k10_s0': 'G_10', 'G_k100': 'G_100',
            'GU_k01_s0': 'GU_1', 'GU_k10_s0': 'GU_10', 'GU_k100': 'GU_100'}
    for key, arm in GMAP.items():
        tp = F_g['arms'][key]['test_predictions']
        for fn, h in tp.items():
            M.append(dict(arm=arm, role='pred' if fn.startswith('pred_') else 'raw', path=f'kpct/guided/preds/{key}/{fn}', sha256=h, src='guided_freeze'))
    for key, arm in [('SHUF_G1', 'SHUF_G1'), ('QONLY_G1', 'QONLY_G1')]:
        for fn, h in F_g['controls'][key]['test_predictions'].items():
            M.append(dict(arm=arm, role='pred' if fn.startswith('pred_') else 'raw', path=f'kpct/guided/preds/G_k01_s0/{fn}', sha256=h, src='guided_freeze'))

    # Parser (P)
    for key, t in {'P_1': '1', 'P_5': '5', 'P_10': '10', 'P_100': '100'}.items():
        te = F_p['arms'][key]['test']
        for fp, h in te['pred_files'].items():
            s = fp.split('seed')[-1].split('.')[0]
            M.append(dict(arm=f'P_{t}_seed{s}', role='pred', path=f'kpct/{fp}', sha256=h, src='parser_freeze'))
        M.append(dict(arm=f'P_{t}', role='raw', path=f"kpct/{te['raw_file']}", sha256=te['raw_sha256'], src='parser_freeze'))

    # k = 0
    for fn, h in F_k0['arms']['P_0']['test_predictions'].items():
        sub = {'raw': 'k0/p0/raw', 'exec': 'k0/p0/exec', 'pred': 'k0/p0/exec'}[fn.split('_')[0]]
        M.append(dict(arm='P_0', role='pred' if fn.startswith('pred_') else 'raw', path=f'kpct/{sub}/{fn}', sha256=h, src='k0_freeze'))
    for fn, h in F_k0['arms']['G_0']['test_predictions'].items():
        sub = {'raw': 'k0/g0/raw', 'post': 'k0/g0/pred', 'pred': 'k0/g0/pred'}[fn.split('_')[0]]
        M.append(dict(arm='G_0', role='pred' if fn.startswith('pred_') else 'raw', path=f'kpct/{sub}/{fn}', sha256=h, src='k0_freeze'))
    assert F_k0['test_prediction_files']['P_0']['sha256'] == F_k0['arms']['P_0']['test_predictions']['pred_P0_test_d3.jsonl']
    assert F_k0['test_prediction_files']['G_0']['sha256'] == F_k0['arms']['G_0']['test_predictions']['pred_G0_test_g1.jsonl']

    # auxiliary: the G_1 test bundle (NOVEL_GOLD_G1 flag)
    b = F_struct['arms']['k01_s0']['bundles']['test']
    assert b['sha256'] == F_g['arms']['G_k01_s0']['bundle_inputs_sha256']['test.jsonl']
    M.append(dict(arm='BUNDLE_k01_s0_test', role='aux', path=f"kpct/{b['file']}", sha256=b['sha256'], src='structural_freeze (= guided freeze bundle_inputs)'))
    assert len(set(m['path'] for m in M)) == len(M)
    for fp, h in F_p['fallback_preds_k100_test'].items():
        M.append(dict(arm='P_100_fallback_copy', role='aux', path=f'kpct/{fp}', sha256=h, src='parser_freeze (fallback copies of frozen S_100)'))

    json.dump(M, open(a.out, 'w'), indent=1)
    print(len(M))


if __name__ == '__main__':
    main()
