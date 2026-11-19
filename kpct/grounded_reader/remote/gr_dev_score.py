#!/usr/bin/env python3
"""gr_dev_score.py -- grounded_reader track: the registered DEV selection pass (dev split only; never reads test).

Candidates per level k: frozen G_k (baseline, structural bundle), GR-A_k (PALONE bundles), GR-K_k (PK bundles).
Data: dev_sel2000 ORIG (n = 2,000; frozen G_k = its selection-time dev file) and DEV_ACCEPTED paraphrases (vidx 5 of the
R3-track llm root, n = 1,762; frozen G_k = its frozen variants file). Official multitq_eval.hit_at_k, dev.json answers.
Registered rule: admissible iff hits_orig(c) - hits_orig(G_k) >= -0.003 * 2000 (= at most 0.3 points dev-original loss);
select the admissible candidate with the highest DEV_ACCEPTED Hits@1; ties (|diff| < 0.0005) -> G_k, then GR-K_k, then
GR-A_k (closest to the baseline first). G_k is always admissible; selecting G_k = NOT_SELECTED (no final pass at k).
Descriptive: Hits@1/@10 with Poisson(1) bootstrap 95% CIs (r2_common.summarise, B = 10,000, seed 20261118); I2
reuse-check agreement (gold-free). Writes --out (selection json).
"""
import argparse, gzip, json, os, sys
import numpy as np
sys.path.insert(0, "/kpct/guided/code"); sys.path.insert(0, "/kpct/robustness_r3/code")
import multitq_eval as ME       # frozen official evaluator
import r2_common as R2C         # frozen bootstrap

GT = {"k01_s0": "ordS_ep2", "k05_s0": "main_ep2", "k100": "main_ep1"}
GR = "/kpct/grounded_reader"
PREF = {"G_k": 0, "PK": 1, "PALONE": 2}


def read_ranked(p):
    d = {}
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                x = json.loads(l); rk = x["ranked"]
                d[int(x["quid"])] = json.loads(rk.replace("'", '"')) if isinstance(rk, str) else rk
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = {int(q["quid"]): q for q in json.load(open("/data/questions/dev.json"))}
    sel = json.load(open("/kpct/parser/dev_sel2000_quids.json"))
    sel = sorted(int(q) for q in (sel["quids"] if isinstance(sel, dict) else sel))
    assert len(sel) == 2000
    res = dict(levels={}, selection={}, rule="admissible iff hits_orig(c) - hits_orig(G_k) >= -6 (of 2,000); argmax DEV_ACCEPTED Hits@1; ties < 0.0005 -> G_k > PK > PALONE")
    for k in a.levels.split(","):
        fz_dev = read_ranked(f"/kpct/guided/preds/G_{k}/pred_G_dev_sel2000_{GT[k]}.jsonl")
        fz_llm = read_ranked(f"/kpct/robustness_r3/llm/G/{k}/pred_G_variants_G_{k}.jsonl")
        acc = sorted(q // 10 for q in fz_llm if q % 10 == 5)
        C = {"G_k": (fz_dev, fz_llm)}
        for m in ("PALONE", "PK"):
            C[m] = (read_ranked(f"{GR}/gen/dev/assembled/{k}/{m}/dev.jsonl.gz"), read_ranked(f"{GR}/gen/dev/assembled/{k}/{m}/devpara.jsonl.gz"))
        L = {}
        for c, (Pd, Pp) in C.items():
            ho = np.array([ME.hit_at_k(dev[q], Pd.get(q, []), 1) for q in sel], float)
            hp = np.array([ME.hit_at_k(dev[q], Pp.get(10 * q + 5, []), 1) for q in acc], float)
            ho10 = np.array([ME.hit_at_k(dev[q], Pd.get(q, []), 10) for q in sel], float)
            hp10 = np.array([ME.hit_at_k(dev[q], Pp.get(10 * q + 5, []), 10) for q in acc], float)
            L[c] = dict(n_orig=len(sel), n_para=len(acc), hits_orig=int(ho.sum()), hits_para=int(hp.sum()),
                        orig_hits1=R2C.summarise(ho, 10000, 20261118), para_hits1=R2C.summarise(hp, 10000, 20261118),
                        orig_hits10=float(ho10.mean()), para_hits10=float(hp10.mean()),
                        missing_orig=sum(q not in Pd for q in sel), missing_para=sum(10 * q + 5 not in Pp for q in acc),
                        _ho=ho, _hp=hp)
        base = L["G_k"]
        for c in L:
            L[c]["admissible"] = bool(L[c]["hits_orig"] - base["hits_orig"] >= -6)
            L[c]["delta_orig_vs_G"] = (L[c]["hits_orig"] - base["hits_orig"]) / len(sel)
            L[c]["delta_para_vs_G"] = (L[c]["hits_para"] - base["hits_para"]) / len(acc)
            if c != "G_k":
                L[c]["paired_orig_vs_G"] = R2C.summarise(L[c]["_ho"] - base["_ho"], 10000, 20261118)
                L[c]["paired_para_vs_G"] = R2C.summarise(L[c]["_hp"] - base["_hp"], 10000, 20261118)
        adm = [c for c in L if L[c]["admissible"]]
        best = max(L[c]["hits_para"] / len(acc) for c in adm)
        tied = [c for c in adm if best - L[c]["hits_para"] / len(acc) < 0.0005]
        choice = sorted(tied, key=lambda c: PREF[c])[0]
        for c in L:
            L[c].pop("_ho"); L[c].pop("_hp")
        # I2 reuse check (gold-free): regenerated frozen-bundle answers vs frozen G_k answers
        I2 = {}
        for lab, fz in (("dev", fz_dev), ("devpara", fz_llm)):
            p = f"{GR}/gen/dev/{k}/REUSECHECK/pred_G_reuse{lab}_G_{k}.jsonl"
            if os.path.exists(p):
                g = read_ranked(p)
                I2[lab] = dict(n=len(g), top1_agree=float(np.mean([g[q][:1] == fz[q][:1] for q in g])) if g else None)
        res["levels"][k] = dict(candidates=L, admissible=adm, selected=choice, I2_reuse_check=I2)
        res["selection"][k] = choice
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: (v["selected"], {c: (round(v["candidates"][c]["orig_hits1"]["delta"], 4), round(v["candidates"][c]["para_hits1"]["delta"], 4), v["candidates"][c]["admissible"]) for c in v["candidates"]}) for k, v in res["levels"].items()}))


if __name__ == "__main__":
    main()
