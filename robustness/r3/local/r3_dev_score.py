"""r3_dev_score.py -- rule R3 dev_scoring_by_this_track: the single scoring pass of the frozen arms on dev_sel2000 ORIG and
the dev paraphrases (DEV_ACCEPTED = vidx 5; DEV_MECH_OK = vidx 5 union 6, amendment A2). Never touches R3 rows (vidx 4).
ORIG = frozen dev_sel2000 selection-time predictions (rows extracted by rob_collect_frozen.py into
dev/checks/frozen_orig_rows.json, top-10 lists); PARA = the frozen-arm predictions on the llm/ root.
Statistic: Poisson(1) bootstrap 95% percentile CI (r2_common.summarise, B = 10,000, seed 20261118) for Hits@1 and drop;
R1/R2 drop label (r2_common.drop_label, B = 2000, seed 0). Official multitq_eval.hit_at_k.
"""
import json, os
import numpy as np
import r2_common as R2C
import r3_common as R3

KS = {'1': 'k01_s0', '5': 'k05_s0', '100': 'k100'}
ORIG_DEV = {
    'RULE_1': ['/kpct/preds/k01_s0/dev_sel2000_RULE.jsonl'], 'RULE_5': ['/kpct/preds/k05_s0/dev_sel2000_RULE.jsonl'],
    'RULE_100': ['/mtq/phase1_structural/preds/dev_RULE.jsonl'],
    'S_1': [f'/kpct/preds/k01_s0/dev_sel2000_S_seed{s}.jsonl' for s in range(3)], 'S_5': [f'/kpct/preds/k05_s0/dev_sel2000_S_seed{s}.jsonl' for s in range(3)],
    'S_100': [f'/mtq/phase1_structural/preds/dev_S_seed{s}.jsonl' for s in range(3)],
    'G_1': ['/kpct/guided/preds/G_k01_s0/pred_G_dev_sel2000_ordS_ep2.jsonl'], 'G_5': ['/kpct/guided/preds/G_k05_s0/pred_G_dev_sel2000_main_ep2.jsonl'],
    'G_100': ['/kpct/guided/preds/G_k100/pred_G_dev_sel2000_main_ep1.jsonl'],
    'P_1': [f'/kpct/parser/exec/k01_s0/pred_P_k01_s0_dev_sel2000_ep3_tauoff_seed{s}.jsonl' for s in range(3)],
    'P_5': [f'/kpct/parser/exec/k05_s0/pred_P_k05_s0_dev_sel2000_ep1_tauoff_seed{s}.jsonl' for s in range(3)],
    'P_100': [f'/kpct/parser/exec/k100/pred_P_k100_dev_sel2000_ep0.5_tauoff_seed{s}.jsonl' for s in range(3)],
    'GU_1': ['/kpct/guided/preds/GU_k01_s0/pred_GU_dev_sel2000_main_ep3.jsonl'], 'GU_5': ['/kpct/llm_controls/preds/GU_k05_s0/pred_GU_dev_sel2000_main_ep2.jsonl'],
    'GU_100': ['/kpct/guided/preds/GU_k100/pred_GU_dev_sel2000_main_ep1.jsonl'],
    'GPLAIN_1': ['/kpct/llm_controls/preds/GPLAIN_k01_s0/pred_G_dev_sel2000_plain_ep3.jsonl'],
}
ORIG_TEST = {
    'RULE_1': ['/kpct/preds/k01_s0/test_RULE.jsonl'], 'RULE_5': ['/kpct/preds/k05_s0/test_RULE.jsonl'], 'RULE_100': ['/mtq/phase1_structural/preds/test_RULE.jsonl'],
    'S_1': [f'/kpct/preds/k01_s0/test_S_seed{s}.jsonl' for s in range(3)], 'S_5': [f'/kpct/preds/k05_s0/test_S_seed{s}.jsonl' for s in range(3)],
    'S_100': [f'/mtq/phase1_structural/preds/test_S_seed{s}.jsonl' for s in range(3)],
    'G_1': ['/kpct/guided/preds/G_k01_s0/pred_G_test_ordS_ep2.jsonl'], 'G_5': ['/kpct/guided/preds/G_k05_s0/pred_G_test_main_ep2.jsonl'],
    'G_100': ['/kpct/guided/preds/G_k100/pred_G_test_main_ep1.jsonl'],
    'P_1': [f'/kpct/parser/test/k01_s0/pred_P_k01_s0_test_ep3_tauoff_seed{s}.jsonl' for s in range(3)],
    'P_5': [f'/kpct/parser/test/k05_s0/pred_P_k05_s0_test_ep1_tauoff_seed{s}.jsonl' for s in range(3)],
    'P_100': [f'/kpct/parser/test/k100/pred_P_k100_test_ep0.5_tauoff_seed{s}.jsonl' for s in range(3)],
    'GU_1': ['/kpct/guided/preds/GU_k01_s0/pred_GU_test_main_ep3.jsonl'], 'GU_5': ['/kpct/llm_controls/preds/GU_k05_s0/pred_GU_test_main_ep2.jsonl'],
    'GU_100': ['/kpct/guided/preds/GU_k100/pred_GU_test_main_ep1.jsonl'],
    'GPLAIN_1': ['/kpct/llm_controls/preds/GPLAIN_k01_s0/pred_G_test_plain_ep3.jsonl'],
}


def rewrite_files(arm):
    a, lev = arm.rsplit('_', 1); k = KS[lev]
    if a == 'RULE':
        return [f'preds/{k}/variants_RULE.jsonl']
    if a == 'S':
        return [f'preds/{k}/variants_S_seed{s}.jsonl' for s in range(3)]
    if a == 'G':
        return [f'G/{k}/pred_G_variants_G_{k}.jsonl']
    if a == 'P':
        return [f'P/{k}/pred_P_{k}_variants_P_{k}_tauoff_seed{s}.jsonl' for s in range(3)]
    if a == 'GU':
        return [f'GU/{k}/pred_GU_variants_GU_{k}.jsonl']
    if a == 'GPLAIN':
        return [f'GPLAIN/{k}/pred_G_variants_GPLAIN_{k}.jsonl']
    raise KeyError(arm)


ARMS = list(ORIG_DEV)


def load_rw(path):
    out = {}
    with open(path) as f:
        for l in f:
            if l.strip():
                r = json.loads(l)
                rk = r['ranked']
                out[int(r['quid'])] = json.loads(rk.replace("'", '"')) if isinstance(rk, str) else rk
    return out


def score_dev(frozen_rows, llm_dir, gold, acc_quids, jrej_quids, ME, B=R3.B_SECONDARY, seed=R3.BOOT_SEED):
    """frozen_rows: dev/checks/frozen_orig_rows.json (preds: {path: {quid: top10}}); llm_dir: local copy of the llm/
    root; gold: {quid: {answers, answer_type}} for dev_sel2000; acc_quids: DEV_ACCEPTED quids; jrej_quids: judge-rejected"""
    allq = sorted(gold)
    acc = sorted(acc_quids); mok = sorted(set(acc_quids) | set(jrej_quids))
    res = {}
    for arm in ARMS:
        O = [{int(q): v for q, v in frozen_rows['preds'][p].items()} for p in ORIG_DEV[arm]]
        Rw = [load_rw(os.path.join(llm_dir, p)) for p in rewrite_files(arm)]
        def h(P_, q, sq, k):
            return float(np.mean([ME.hit_at_k(gold[q], P.get(sq, []), k) for P in P_]))
        o_all = np.array([h(O, q, q, 1) for q in allq])
        out = dict(files_orig=ORIG_DEV[arm], files_rw=rewrite_files(arm), n_seeds=len(O),
                   orig_all=dict(n=len(allq), hits1=R2C.summarise(o_all, B, seed), hits10=float(np.mean([h(O, q, q, 10) for q in allq]))))
        for name, qs, vix in (('DEV_ACCEPTED', acc, {q: 5 for q in acc}), ('DEV_MECH_OK', mok, {**{q: 5 for q in acc}, **{q: 6 for q in jrej_quids}})):
            missing = sum(1 for q in qs for P in Rw if 10 * q + vix[q] not in P)
            o = np.array([h(O, q, q, 1) for q in qs]); r = np.array([h(Rw, q, 10 * q + vix[q], 1) for q in qs])
            lab, s_lab = R2C.drop_label(o - r)
            out[name] = dict(n=len(qs), missing_rw_predictions=missing, orig_hits1=R2C.summarise(o, B, seed), para_hits1=R2C.summarise(r, B, seed),
                             drop=R2C.summarise(o - r, B, seed), drop_label=lab, drop_label_ci95_B2000_seed0=s_lab['ci95'],
                             orig_hits10=float(np.mean([h(O, q, q, 10) for q in qs])), para_hits10=float(np.mean([h(Rw, q, 10 * q + vix[q], 10) for q in qs])),
                             top1_changed=float(np.mean([np.mean([(O[i].get(q, [None])[:1] != Rw[i].get(10 * q + vix[q], [None])[:1]) for i in range(len(O))]) for q in qs])))
        res[arm] = out
    return res
