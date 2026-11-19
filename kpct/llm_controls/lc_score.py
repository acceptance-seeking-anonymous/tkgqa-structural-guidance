#!/usr/bin/env python3
"""lc_score.py -- scoring of the LLM controls (protocol/multitq_kpct_amendment_LC1_llm_controls.json, with LC1-A1).
Loads the frozen inputs, checks the new prediction files against their freeze records, runs lc_analyze.py once and
writes the results. CPU only; reads test gold (run it only after the freeze records exist).

  python kpct/llm_controls/lc_score.py --scorer src/multitq_eval.py --test $DATA_DIR/questions/test.json \
      --hits multitq_kpct_test_per_question_hits.csv.gz --lc $WORK_DIR/llm_controls \
      --r1-set <R1 perturbed set json> --r1-pred <R1 frozen probe prediction tree> \
      --r2-set <R2 perturbed set json> --r2-pred <R2 frozen probe prediction tree> --outdir out

--lc        output root of kpct/llm_controls/lc_job.sh (preds/<ARM>/, robust/robustness*/<ARM>/, tokens/)
--hits      per-question hits of the k% test pass (kpct/test_pass/multitq_kpct_test_pass.py --out-csv)
--r1-pred / --r2-pred
            frozen R1 / R2 probe prediction trees (preds/, G/, P/, GU/, checks/freeze_sha256.txt), as for
            analysis/rewrite_tables/a1_rewrite.py
--rejected  JSON with the human-rejected rewrites; default: the lists in the LC1 rule copy in protocol/
            (robustness_analyses.human_audit_removal.R1_rejected_quids / R2_rejected_quids). Alternatively
            --r1-sheet/--r1-sel/--r2-sheet/--r2-sel (annotated review sheets + selection keys): a rewrite is rejected
            iff same_answer_set or same_constraint_time is not 'Y'.
Writes multitq_llm_controls_results.json (arms, contrasts, strata, robustness, input sha256),
multitq_llm_controls_test_per_question_hits.csv.gz and multitq_llm_controls_rewrite_per_variant_hits.csv.gz.
"""
import argparse, csv, hashlib, importlib.util, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, '..', '..'))
LC1_RULE = os.path.join(REPO, 'protocol', 'multitq_kpct_amendment_LC1_llm_controls.json')
# new arms: arm -> (prompt style, run name, selection file, freeze file, test prediction prefix)
NEW = {'GPLAIN_1': ('G', 'GPLAIN_k01_s0', 'selection_G_k01_s0_plain.json', 'freeze_G_k01_s0.json', 'pred_G_test_plain_'),
       'GU_5': ('GU', 'GU_k05_s0', 'selection_GU_k05_s0_main.json', 'freeze_GU_k05_s0.json', 'pred_GU_test_main_')}
SETS = {'R1': 'robustness', 'R2': 'robustness_r2'}
# frozen arms needed by the robustness contrasts of lc_analyze.py (files inside the probe prediction trees)
PROBE_ARMS = {'G_1': ['G/k01_s0/pred_G_variants_G_k01_s0.jsonl'], 'G_5': ['G/k05_s0/pred_G_variants_G_k05_s0.jsonl'],
              'GU_1': ['GU/k01_s0/pred_GU_variants_GU_k01_s0.jsonl'], 'GU_100': ['GU/k100/pred_GU_variants_GU_k100.jsonl'],
              'P_1': [f'P/k01_s0/pred_P_k01_s0_variants_P_k01_s0_tauoff_seed{s}.jsonl' for s in range(3)],
              'P_5': [f'P/k05_s0/pred_P_k05_s0_variants_P_k05_s0_tauoff_seed{s}.jsonl' for s in range(3)],
              'S_1': [f'preds/k01_s0/variants_S_seed{s}.jsonl' for s in range(3)],
              'S_5': [f'preds/k05_s0/variants_S_seed{s}.jsonl' for s in range(3)]}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_pred(p):
    out = {}
    for line in open(p):
        if line.strip():
            r = json.loads(line); rk = r['ranked']
            out[int(r['quid'])] = json.loads(rk.replace("'", '"')) if isinstance(rk, str) else rk
    return out


def probe_hits(ME, TESTQ, tree, variants):
    """Hits@1 per variant prompt (squid) of the frozen arms; multi-seed arms (P, S) as the per-prompt seed mean."""
    fz = {}
    for line in open(os.path.join(tree, 'checks', 'freeze_sha256.txt')):
        h, p = line.split(None, 1)
        fz[p.strip()] = h
    bad = [f for fs in PROBE_ARMS.values() for f in fs if sha256_file(os.path.join(tree, f)) != fz.get(f)]
    assert not bad, f'{tree}: files differ from checks/freeze_sha256.txt: {bad}'
    hits = {}
    for arm, fs in PROBE_ARMS.items():
        hs = []
        for f in fs:
            P = load_pred(os.path.join(tree, f))
            hs.append({r['squid']: ME.hit_at_k(TESTQ[r['quid']], P.get(r['squid']), 1) for r in variants})
        hits[arm] = {sq: float(np.mean([h[sq] for h in hs])) for sq in hs[0]}
    return hits


def rejected_from_sheets(sheet, sel):
    sys.path.insert(0, os.path.join(REPO, 'robustness', 'r2', 'kit'))
    import score_R2 as SR
    key = {r['audit_id']: int(r['quid']) for r in csv.DictReader(open(sel, encoding='utf-8'))}
    rows = SR.read_sheet(sheet)
    assert {r['audit_id'] for r in rows} == set(key)
    return sorted(key[r['audit_id']] for r in rows
                  if not (SR.yes(r.get('same_answer_set', '')) and SR.yes(r.get('same_constraint_time', ''))))


def new_arm_files(lc, arm):
    style, run, sel_f, frz_f, prefix = NEW[arm]
    pdir = os.path.join(lc, 'preds', run)
    sel = json.load(open(os.path.join(pdir, sel_f)))
    frz = json.load(open(os.path.join(pdir, frz_f)))
    test = os.path.join(pdir, f"{prefix}{sel['selected']}.jsonl")
    files = {test: frz['test_predictions'][os.path.basename(test)]}      # freeze record written before the test pass
    var = {}
    for S, sub in SETS.items():
        d = os.path.join(lc, 'robust', sub, run)
        p = os.path.join(d, f'pred_{style}_variants_{run}.jsonl')
        var[S] = p
        st_p = os.path.join(d, f'infer_stats_{style}_variants.json')
        if os.path.exists(st_p):                                         # adapter and prediction hashes of the pass
            a = json.load(open(st_p))['adapters'][run]
            assert a['adapter_sha256'] == frz['adapter_sha256']['adapter_model.safetensors'], f'{arm} {S}: adapter differs from the freeze'
            ps = a['styles'][style].get('pred_sha256')
            if ps:
                files[p] = ps
    return test, var, files, dict(selected=sel['selected'], candidates=sel.get('candidates'))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--scorer', default=os.path.join(REPO, 'src', 'multitq_eval.py'))
    ap.add_argument('--test', required=True); ap.add_argument('--hits', required=True); ap.add_argument('--lc', required=True)
    for S in ('r1', 'r2'):
        ap.add_argument(f'--{S}-set', required=True); ap.add_argument(f'--{S}-pred', required=True)
        ap.add_argument(f'--{S}-sheet'); ap.add_argument(f'--{S}-sel')
    ap.add_argument('--rejected', default=LC1_RULE)
    ap.add_argument('--arms', nargs='+', default=['GPLAIN_1', 'GU_5'], choices=list(NEW))
    ap.add_argument('--tokens', help='per-question prompt tokens of lc_tokens.py (default: <lc>/tokens/tokens_gplain_vs_g_per_question.csv.gz if present)')
    ap.add_argument('--outdir', required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)

    spec = importlib.util.spec_from_file_location('multitq_eval', a.scorer)
    ME = importlib.util.module_from_spec(spec); spec.loader.exec_module(ME)
    TESTQ = {q['quid']: q for q in json.load(open(a.test))}
    H = pd.read_csv(a.hits)

    ROB = {}
    for S, set_p, tree in (('R1', a.r1_set, a.r1_pred), ('R2', a.r2_set, a.r2_pred)):
        vr = json.load(open(set_p))['variants']
        ROB[S] = dict(variants=vr, hits=probe_hits(ME, TESTQ, tree, vr))

    if a.r1_sheet and a.r2_sheet:
        REJ = {'R1': rejected_from_sheets(a.r1_sheet, a.r1_sel), 'R2': rejected_from_sheets(a.r2_sheet, a.r2_sel)}
    else:
        d = json.load(open(a.rejected))
        d = d.get('robustness_analyses', {}).get('human_audit_removal', d)
        REJ = {S: sorted(int(q) for q in d[f'{S}_rejected_quids']) for S in ('R1', 'R2')}

    NEW_TEST, NEW_VAR, FILES, SEL = {}, {}, {}, {}
    for arm in a.arms:
        test, var, files, sel = new_arm_files(a.lc, arm)
        NEW_TEST[arm] = test; FILES.update(files); SEL[arm] = sel
        for S, p in var.items():
            NEW_VAR[(S, arm)] = p

    g = dict(ME=ME, TESTQ=TESTQ, H=H, ROB=ROB, REJ=REJ, NEW_TEST=NEW_TEST, NEW_VAR=NEW_VAR, FREEZE={'files_local': FILES},
             __name__='lc_analyze')
    src = os.path.join(HERE, 'lc_analyze.py')
    exec(compile(open(src).read(), src, 'exec'), g)
    RES, PQ, PV = g['RES'], g['PQ'], g['PV']

    tok = a.tokens or os.path.join(a.lc, 'tokens', 'tokens_gplain_vs_g_per_question.csv.gz')
    cols = ['quid'] + [f'{x}@{k}' for x in a.arms for k in (1, 10)] + ['qlabel', 'answer_type']
    out_pq = PQ[cols]
    if os.path.exists(tok):
        tq = pd.read_csv(tok)
        tq = tq[tq.split == 'test'][['quid', 'g_tok', 'plain_tok']].rename(columns={'g_tok': 'G_1_prompt_tokens', 'plain_tok': 'GPLAIN_1_prompt_tokens'})
        out_pq = out_pq.merge(tq, on='quid', how='left')
    out_pq.to_csv(os.path.join(a.outdir, 'multitq_llm_controls_test_per_question_hits.csv.gz'), index=False)
    rej = {S: set(REJ[S]) for S in REJ}
    pv = PV.copy()
    pv['human_rejected'] = [(r.variant in ('P3', 'RW')) and (r.quid in rej[r.set]) for r in pv.itertuples()]
    pv.to_csv(os.path.join(a.outdir, 'multitq_llm_controls_rewrite_per_variant_hits.csv.gz'), index=False)

    res = dict(name='multitq_llm_controls_results',
               label='reuse of the official MultiTQ test split (post hoc control arms); descriptive',
               evaluator_sha256=sha256_file(a.scorer), test_json_sha256=sha256_file(a.test),
               frozen_hits_csv_sha256=sha256_file(a.hits), n_test=int(len(PQ)),
               checked_prediction_sha256={os.path.relpath(p, a.lc): s for p, s in FILES.items()},
               n_rejected={S: len(REJ[S]) for S in REJ}, dev_sel2000=SEL, **RES)
    json.dump(res, open(os.path.join(a.outdir, 'multitq_llm_controls_results.json'), 'w'), indent=1)
    for arm in a.arms:
        x = RES['arms'][arm]
        print(arm, 'Hits@1 %.4f [%.4f, %.4f]' % (x['hits1'], *x['ci95']), 'Hits@10 %.4f' % x['hits10'])
    for n, x in RES['contrasts'].items():
        print(n, '%+.4f [%+.4f, %+.4f]' % (x['delta'], *x['ci95']), x['label'])


if __name__ == '__main__':
    main()
