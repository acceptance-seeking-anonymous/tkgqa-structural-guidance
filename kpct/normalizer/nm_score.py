"""nm_score.py -- normalizer track, the ONE registered scoring pass (rule multitq_normalizer_rule.json: test_pass,
statistics, outcome_labels). Official test split (reuse) + R3 PRIMARY (secondary, registered R3 analysis plan) + R1 P3 /
R2 RW (descriptive; R2 GENERATOR_MATCHED).

Per item and arm, method hits (N + arm): decision source 'input' -> the frozen prediction of the input (official test
per-question hits for test ORIG items; frozen rewrite predictions for R3 / R1 / R2 items), 'original' -> frozen official
test hits of the quid, 'run' -> new prediction of the unchanged pipeline on the normalized text. Baseline hits: frozen
official test hits (ORIG) and frozen rewrite predictions (rewrites). S, P: mean over 3 S-seed files.
Statistics: r2_common.summarise (paired Poisson(1) bootstrap, B = 10,000, seed 20261118); r3_common.method_tests
(c_drop, c_rw, one-sided, Holm within method over 2 levels x 2 contrasts).
"""
import json, os
import numpy as np
import r2_common as R2C
import r3_common as R3

B, SEED = 10000, 20261118
LEVELS = {"5": "k05_s0", "100": "k100"}
METHODS = {"NR": "RULE", "NS": "S", "NP": "P", "NG": "G"}


def arm_files(arm, k):
    if arm == "RULE":
        return [f"preds/{k}/variants_RULE.jsonl"]
    if arm == "S":
        return [f"preds/{k}/variants_S_seed{s}.jsonl" for s in range(3)]
    if arm == "P":
        return [f"P/{k}/pred_P_{k}_variants_P_{k}_tauoff_seed{s}.jsonl" for s in range(3)]
    if arm == "G":
        return [f"G/{k}/pred_G_variants_G_{k}.jsonl"]
    raise KeyError(arm)


def frozen_cols(arm, lev, at=1):
    if arm == "RULE":
        return [f"RULE_{lev}@{at}"]
    if arm in ("S", "P"):
        return [f"{arm}_{lev}_seed{s}@{at}" for s in range(3)]
    return [f"G_{lev}@{at}"]


def load_ranked(p):
    out = {}
    with open(p) as f:
        for l in f:
            if l.strip():
                r = json.loads(l); rk = r["ranked"]
                out[int(r["quid"])] = json.loads(rk.replace("'", '"')) if isinstance(rk, str) else rk
    return out


def hits_from(files, squid, gq, ME, at):
    return float(np.mean([ME.hit_at_k(gq, F.get(squid, []), at) for F in files]))


def score(items, dec, gold, test_hits, new_root, frozen_rw_roots, ME, run_arms):
    """items: test item rows (sets TEST, R3, R1, R2); dec: {iid: decision} of the selected configuration;
    gold: {quid: test question dict}; test_hits: DataFrame of the official per-question test hits (index quid);
    new_root: local dir of the normalized-root predictions; frozen_rw_roots: {'R3': dir, 'R1': dir, 'R2': dir};
    run_arms: arms whose new predictions exist (e.g. ['RULE_5', ..., 'G_100'])."""
    res = dict(arms={}, not_run=[], n=dict())
    test_items = [it for it in items if it["set"] == "TEST"]
    test_q = [it["quid"] for it in test_items]
    th = test_hits.loc[test_q]
    rows = {}            # per-question output rows
    for mname, arm in METHODS.items():
        for lev, k in LEVELS.items():
            name = f"{arm}_{lev}"
            if name not in run_arms:
                res["not_run"].append(name); continue
            NEW = [load_ranked(os.path.join(new_root, f)) for f in arm_files(arm, k)]
            FRW = {s: [load_ranked(os.path.join(d, f)) for f in arm_files(arm, k)] for s, d in frozen_rw_roots.items()}
            A = {}
            for at in (1, 10):
                fz = th[frozen_cols(arm, lev, at)].mean(axis=1).to_numpy()
                fz_by_q = dict(zip(test_q, fz))
                m = np.empty(len(test_items)); missing = 0
                for i, it in enumerate(test_items):
                    d = dec[it["iid"]]
                    if d["source"] == "run":
                        if any(d["squid"] not in F for F in NEW):
                            missing += 1
                        m[i] = hits_from(NEW, d["squid"], gold[it["quid"]], ME, at)
                    else:
                        m[i] = fz_by_q[it["quid"]]
                A[f"test@{at}"] = dict(base=fz, method=m, missing_new=missing)
                for s in ("R3", "R1", "R2"):
                    its = sorted([it for it in items if it["set"] == s], key=lambda r: r["quid"])
                    ob, rb, om, rm = [], [], [], []
                    miss_rw = 0
                    for it in its:
                        q = it["quid"]; sq_in = 10 * q + it["vidx"]
                        if any(sq_in not in F for F in FRW[s]):
                            miss_rw += 1
                        b_rw = hits_from(FRW[s], sq_in, gold[q], ME, at)
                        d = dec[it["iid"]]
                        if d["source"] == "input":
                            m_rw = b_rw
                        elif d["source"] == "original":
                            m_rw = fz_by_q[q]
                        else:
                            m_rw = hits_from(NEW, d["squid"], gold[q], ME, at)
                        ob.append(fz_by_q[q]); rb.append(b_rw); rm.append(m_rw)
                    # method on the original question = N applied to the original (full-test pass)
                    idx = {q: i for i, q in enumerate(test_q)}
                    om = [m[idx[it["quid"]]] for it in its]
                    A[f"{s}@{at}"] = dict(quids=[it["quid"] for it in its], o_b=np.array(ob), r_b=np.array(rb), o_m=np.array(om), r_m=np.array(rm), missing_frozen_rw=miss_rw)
            res["arms"][name] = A
    return res


def analyse(res):
    out = dict(full_test={}, R3={}, R1={}, R2={}, labels={})
    for name, A in res["arms"].items():
        t = A["test@1"]; d = t["method"] - t["base"]
        s = R2C.summarise(d, B, SEED)
        lab = "NONINFERIOR" if s["ci95"][0] >= -0.003 else ("INFERIOR" if s["ci95"][1] < -0.003 else "INCONCLUSIVE")
        out["full_test"][name] = dict(n=int(len(d)), base_hits1=float(t["base"].mean()), method_hits1=float(t["method"].mean()),
                                      base_hits1_ci=R2C.summarise(t["base"], B, SEED)["ci95"], method_hits1_ci=R2C.summarise(t["method"], B, SEED)["ci95"],
                                      delta=s, cost_label=lab, base_hits10=float(A["test@10"]["base"].mean()), method_hits10=float(A["test@10"]["method"].mean()),
                                      missing_new_predictions=t["missing_new"])
    for s in ("R3", "R1", "R2"):
        for mname, arm in METHODS.items():
            per_level = {}
            for lev in LEVELS:
                name = f"{arm}_{lev}"
                if name in res["arms"]:
                    x = res["arms"][name][f"{s}@1"]
                    per_level[lev] = dict(o_m=x["o_m"], r_m=x["r_m"], o_b=x["o_b"], r_b=x["r_b"])
            if not per_level:
                continue
            T = R3.method_tests(per_level)
            for lev in T:
                x = res["arms"][f"{arm}_{lev}"]
                T[lev]["hits_ci"] = {k: R2C.summarise(x[f"{s}@1"][k], B, SEED)["ci95"] for k in ("o_b", "r_b", "o_m", "r_m")}
                T[lev]["drop_ci"] = dict(base=R2C.summarise(x[f"{s}@1"]["o_b"] - x[f"{s}@1"]["r_b"], B, SEED)["ci95"],
                                         method=R2C.summarise(x[f"{s}@1"]["o_m"] - x[f"{s}@1"]["r_m"], B, SEED)["ci95"])
                T[lev]["hits10"] = {k: float(np.mean(x[f"{s}@10"][k])) for k in ("o_b", "r_b", "o_m", "r_m")}
                T[lev]["missing_frozen_rw"] = x[f"{s}@1"]["missing_frozen_rw"]
            if s != "R3":
                for lev in T:
                    for cn in ("c_drop", "c_rw"):
                        T[lev][cn]["label"] = "DESCRIPTIVE"
                        T[lev][cn]["note"] = "descriptive (not registered as a test); Holm value shown for reference only"
            out[s][mname] = T
    for mname, arm in METHODS.items():
        for lev in LEVELS:
            name = f"{arm}_{lev}"
            if name not in res["arms"] or mname not in out["R3"]:
                continue
            rw = out["R3"][mname][lev]["c_rw"]["label"]; cost = out["full_test"][name]["cost_label"]
            out["labels"][f"{mname}_{lev}"] = ("ROBUSTNESS_GAIN" if (rw == "SUPPORTED" and cost == "NONINFERIOR") else
                                               "GAIN_WITH_COST" if rw == "SUPPORTED" else "NO_GAIN")
    return out
