"""nm_select.py -- normalizer track, registered dev scoring and selection (rule multitq_normalizer_rule.json, dev_selection).
Dev only: dev_sel2000 ORIG (2,000) and DEV_ACCEPTED paraphrases (1,762). For each of the 12 configurations:
  per-item prediction source (nm_build_root decisions): 'input' -> frozen dev hits of that input (dev per-question hits
  file of the R3 probe track: <arm>_orig_h1 for ORIG items, <arm>_para_h1 for paraphrase items); 'original' -> the frozen
  ORIG hits of the quid; 'run' -> the new prediction of the unchanged pipeline on the normalized text (official
  multitq_eval.hit_at_k, dev gold); P_5 = mean over its 3 S-seed files.
  J and the dev-original constraint via nm_common.objective; choice via nm_common.select.
Also: rates by input class, exact-match of normalized paraphrases vs the original question, identity-control agreement.
"""
import json, os, re
import numpy as np
import nm_common as NC


def load_ranked(p):
    out = {}
    with open(p) as f:
        for l in f:
            if l.strip():
                r = json.loads(l); rk = r["ranked"]
                out[int(r["quid"])] = json.loads(rk.replace("'", '"')) if isinstance(rk, str) else rk
    return out


def norm_cmp(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def score_dev(items, dec, frozen_hits, gold, pred_root, ME, configs):
    """items: dev item rows; dec: {config_key: {iid: decision}}; frozen_hits: DataFrame of the dev per-question hits
    (index quid); gold: {quid: question dict}; pred_root: local copy of the dev root predictions."""
    P = [load_ranked(os.path.join(pred_root, f"P/k05_s0/pred_P_k05_s0_variants_P_k05_s0_tauoff_seed{s}.jsonl")) for s in range(3)]
    G = load_ranked(os.path.join(pred_root, "G/k05_s0/pred_G_variants_G_k05_s0.jsonl"))
    orig_items = [it for it in items if it["vidx"] == 0]
    para_items = [it for it in items if it["vidx"] == 5]
    fh = frozen_hits
    base = {"P_5": {"orig": float(fh["P_5_orig_h1"].mean())}, "G_5": {"orig": float(fh["G_5_orig_h1"].mean())}}
    acc = sorted(it["quid"] for it in para_items)
    base["P_5"]["para"] = float(fh.loc[acc, "P_5_para_h1"].mean()); base["G_5"]["para"] = float(fh.loc[acc, "G_5_para_h1"].mean())
    base["P_5"]["orig_on_accepted"] = float(fh.loc[acc, "P_5_orig_h1"].mean()); base["G_5"]["orig_on_accepted"] = float(fh.loc[acc, "G_5_orig_h1"].mean())
    out = dict(baseline=base, configs=[])
    per_item = {}
    for ck, g, tau in configs:
        key = f"{ck}|{g}|{tau:g}"; D = dec[key]
        H = {}
        missing = 0
        for it in items:
            d = D[it["iid"]]; q = it["quid"]; cls = "orig" if it["vidx"] == 0 else "para"
            if d["source"] == "input":
                hp, hg = fh.at[q, f"P_5_{cls}_h1"], fh.at[q, f"G_5_{cls}_h1"]
            elif d["source"] == "original":
                hp, hg = fh.at[q, "P_5_orig_h1"], fh.at[q, "G_5_orig_h1"]
            else:
                s = d["squid"]
                if any(s not in Ps for Ps in P) or s not in G:
                    missing += 1
                hp = float(np.mean([ME.hit_at_k(gold[q], Ps.get(s, []), 1) for Ps in P]))
                hg = float(ME.hit_at_k(gold[q], G.get(s, []), 1))
            H[it["iid"]] = (float(hp), float(hg))
        per_item[key] = H
        h = {"P_5": {"orig": float(np.mean([H[it["iid"]][0] for it in orig_items])), "para": float(np.mean([H[it["iid"]][0] for it in para_items]))},
             "G_5": {"orig": float(np.mean([H[it["iid"]][1] for it in orig_items])), "para": float(np.mean([H[it["iid"]][1] for it in para_items]))},
             "base": base}
        J, feas, loss = NC.objective(h)
        from nm_build_root import rates
        oi = [it["iid"] for it in orig_items]; pi = [it["iid"] for it in para_items]
        otext = {it["quid"]: it["text"] for it in orig_items}
        ex_final = float(np.mean([D[i["iid"]]["text"] == otext[i["quid"]] for i in para_items]))
        ex_final_norm = float(np.mean([norm_cmp(D[i["iid"]]["text"]) == norm_cmp(otext[i["quid"]]) for i in para_items]))
        out["configs"].append(dict(key=key, ckpt=ck, ckpt_epoch=float(ck.replace("ep", "")), guard=g, tau=tau, J=J, feasible=feas, loss=loss,
                                   hits=dict(P_5=h["P_5"], G_5=h["G_5"]), missing_run_predictions=missing,
                                   rates_orig=rates(D, oi), rates_para=rates(D, pi),
                                   exact_match_para_vs_original=dict(final_exact=ex_final, final_normcmp=ex_final_norm)))
    sel, flag = NC.select(out["configs"])
    out["selected"] = dict(key=sel["key"], flag=flag)
    return out, per_item
