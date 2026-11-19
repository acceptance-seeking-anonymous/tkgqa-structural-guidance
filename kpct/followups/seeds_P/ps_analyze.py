"""ps_analyze.py -- registered analyses of multitq_kpct_seed_amendment_P (A1-A10) from frozen, scored per-question hits.
Inputs: SCORE_CSV (ps_score.py output: <arm>_S<b>@1/@10 for P_k_lora0/1/2), REG_CSV (registered per-question hits,
a93215cd), SELECT (dev selection files per arm), TR (registered test results). Bootstrap: multitq_eval.paired_poisson_bootstrap
(B = 2000, seed 0; single arm = against a zero vector; strata B = 1000)."""
import json, sys, numpy as np, pandas as pd, importlib.util, hashlib, datetime
def load_me(p):
    spec = importlib.util.spec_from_file_location("multitq_eval", p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
COMP = {"AT2QA": 0.887, "ORACLE-T": 0.869, "TKG-Thinker": 0.855, "SCoP": 0.825}
def kp1(lo, hi, v): return "ABOVE" if lo > v else ("BELOW" if hi < v else "TIE")
def holm(ps):
    idx = sorted(range(len(ps)), key=lambda i: ps[i]); m = len(ps); adj = [0.0] * m; run = 0.0
    for r, i in enumerate(idx):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj
def run(SC, REG, ME, sel_files, TR, out_prefix):
    assert (SC["quid"].values == REG["quid"].values).all(), "quid order"
    R = {"name": "multitq_kpct_seed_results_P", "amendment": "multitq_kpct_seed_amendment_P v1.0",
         "label": "registered amendment (seed replication); secondary analyses, original registered results unchanged", "n_test": int(len(SC))}
    ppb = ME.paired_poisson_bootstrap
    z = np.zeros(len(SC))
    def ci1(x, B=2000):
        r = ppb(x, np.zeros(len(x)), B=B); return dict(hits=r["delta"], ci95=[r["ci_low"], r["ci_high"]])
    def ctr(a, b):
        r = ppb(a, b); return dict(delta=r["delta"], ci95=[r["ci_low"], r["ci_high"]], p_one_sided=r["p_one_sided_le0"], B=2000, seed=0, n=int(len(a)))
    S = {}
    for kk in (1, 10):
        for k in (1, 5):
            S[(k, kk)] = REG[[f"S_{k}_seed{b}@{kk}" for b in range(3)]].values.mean(1)
    G5 = REG["G_5@1"].values.astype(float)
    # identity check: re-scored seed-0 files == registered per-question hits
    ident = {}
    for k in (1, 5):
        for b in range(3):
            for kk in (1, 10):
                c = f"P_{k}_lora0_S{b}@{kk}"
                if c in SC: ident[c] = bool((SC[c].values == REG[f"P_{k}_seed{b}@{kk}"].values).all())
    R["identity_check_seed0_rescored_vs_registered"] = ident
    assert all(ident.values()), ident
    arms = {}
    PQOUT = pd.DataFrame({"quid": SC["quid"], "qlabel_bucket": SC["qlabel_bucket"], "answer_type": SC["answer_type"]})
    for k in (1, 5):
        seeds = [s for s in range(3) if f"P_{k}_lora{s}_S0@1" in SC]
        if len(seeds) < 2: continue
        H = {}
        for s in seeds:
            for kk in (1, 10):
                H[(s, kk)] = SC[[f"P_{k}_lora{s}_S{b}@{kk}" for b in range(3)]].values.mean(1)
                PQOUT[f"P_{k}_lora{s}@{kk}"] = H[(s, kk)]
                for b in range(3): PQOUT[f"P_{k}_lora{s}_S{b}@{kk}"] = SC[f"P_{k}_lora{s}_S{b}@{kk}"].values
        full = len(seeds) == 3
        if full:
            for kk in (1, 10): PQOUT[f"P_{k}_loramean@{kk}"] = np.mean([H[(s, kk)] for s in seeds], 0)
        A = {"lora_seeds": seeds, "per_seed": {}}
        names = [f"lora{s}" for s in seeds] + (["loramean"] if full else [])
        for nm in names:
            h1 = PQOUT[f"P_{k}_{nm}@1"].values; h10 = PQOUT[f"P_{k}_{nm}@10"].values
            e = {"hits1": ci1(h1), "hits10": ci1(h10)}
            if nm != "loramean": e["hits1_per_S_seed"] = [float(SC[f"P_{k}_{nm}_S{b}@1"].mean()) for b in range(3)]
            e["KP1"] = {c: kp1(*e["hits1"]["ci95"], v) for c, v in COMP.items()}
            e[f"P_{k}_minus_S_{k}"] = ctr(h1, S[(k, 1)])
            Ucol = "U_k05_s0" if k == 5 else "U_k01_s0"; U = REG[Ucol].values == 1
            e[f"P_{k}_minus_S_{k}_on_U"] = ctr(h1[U], S[(k, 1)][U]); e[f"P_{k}_minus_S_{k}_on_seen"] = ctr(h1[~U], S[(k, 1)][~U])
            e["hits1_on_U"] = ci1(h1[U]); e["hits1_on_seen"] = ci1(h1[~U])
            if k == 5:
                e["P_5_minus_G_5"] = ctr(h1, G5)
            st = {}
            for grp, col in (("qlabel", "qlabel_bucket"), ("answer_type", "answer_type")):
                for v in sorted(SC[col].unique()):
                    m = SC[col].values == v; st[f"{grp}={v}"] = dict(n=int(m.sum()), **ci1(h1[m], B=1000))
            e["strata_B1000"] = st
            if k == 1:
                d = e["P_1_minus_S_1_on_U"]; p = [0.0, 0.0, d["p_one_sided"]]   # KP2, KP3 registered seed-0 one-sided p = 0
                hp = holm(p)[2]
                e["KP4"] = dict(d, threshold=0.03, holm_family="{KP2 (registered G_1 - GU_1, p 0), KP3 (registered G_1 - S_1, p 0), KP4(seed)}", holm_p=hp,
                                margin_met=bool(d["delta"] >= 0.03), pass_=bool(d["delta"] >= 0.03 and hp < 0.05),
                                label=("UNSEEN_TEMPLATE_GAIN" if (d["delta"] >= 0.03 and hp < 0.05) else "NO_UNSEEN_GAIN"))
            A["per_seed"][nm] = e
        v = [A["per_seed"][f"lora{s}"]["hits1"]["hits"] for s in seeds]
        A["seed_summary_hits1"] = dict(values=v, mean=float(np.mean(v)), sd_ddof1=float(np.std(v, ddof=1)), range=[float(min(v)), float(max(v))], spread=float(max(v) - min(v)))
        v10 = [A["per_seed"][f"lora{s}"]["hits10"]["hits"] for s in seeds]
        A["seed_summary_hits10"] = dict(values=v10, mean=float(np.mean(v10)), sd_ddof1=float(np.std(v10, ddof=1)), range=[float(min(v10)), float(max(v10))])
        # seed labels
        def stab(key):
            ds = [A["per_seed"][f"lora{s}"][key] for s in seeds]; sgn0 = np.sign(ds[0]["delta"])
            ok = all((d["ci95"][0] > 0) if sgn0 > 0 else (d["ci95"][1] < 0) for d in ds)
            return dict(label=("SEED_STABLE" if ok and full else ("SEED_SENSITIVE" if full else "NOT_TESTED (seeds missing)")), deltas=[d["delta"] for d in ds], ci95=[d["ci95"] for d in ds])
        L = {}
        for key in [f"P_{k}_minus_S_{k}", f"P_{k}_minus_S_{k}_on_U", f"P_{k}_minus_S_{k}_on_seen"] + (["P_5_minus_G_5"] if k == 5 else []):
            L[key] = stab(key)
        for c in COMP:
            labs = [A["per_seed"][f"lora{s}"]["KP1"][c] for s in seeds]
            L[f"KP1_{c}"] = dict(labels=labs, label=("LABEL_STABLE" if len(set(labs)) == 1 and full else "LABEL_SENSITIVE"))
        if k == 1:
            L["KP4"] = dict(stab("KP4"), test_labels=[A["per_seed"][f"lora{s}"]["KP4"]["label"] for s in seeds])
            L["KP4"]["test_label_stability"] = "LABEL_STABLE" if len(set(L["KP4"]["test_labels"])) == 1 and full else "LABEL_SENSITIVE"
        A["seed_labels"] = L
        if k == 5:   # k_min_above re-derived per seed (registered primaries: k0 G_0, k1 G_1, k5 P_5, k10 P_10, k100 G_100)
            reg = TR["KP1"]["table"]; kma = {}
            for nm in names:
                kma[nm] = {}
                for c in COMP:
                    if A["per_seed"][nm]["KP1"][c] == "ABOVE": kma[nm][c] = 5
                    else:
                        nxt = [kk_ for kk_, arm in (("10", "P_10"), ("100", "G_100")) if reg.get(arm, {}).get(c) == "ABOVE"]
                        kma[nm][c] = int(nxt[0]) if nxt else None
            A["k_min_above_primary_llm_arm"] = dict(registered=TR["KP1"]["k_min_above_primary_llm_arm"], per_seed=kma)
        # dev transparency
        A["dev_sel2000"] = {}
        for s in seeds:
            f = sel_files.get((k, s))
            if f:
                d = json.load(open(f)); A["dev_sel2000"][f"lora{s}"] = dict(chosen=d["chosen"]["checkpoint"] + "/" + d["chosen"]["tau"], hits1=d["dev_sel2000_hits1"],
                    ci95=d["dev_sel2000_hits1_ci95"], unseen_hits1=d["unseen_hits1"], grid=[(g["checkpoint"], g["tau"], g["hits1"]) for g in d["grid"]])
        if k == 1:
            A["primary_declaration_check"] = {nm: dict(dev=A["dev_sel2000"].get(nm, {}).get("hits1"), G_1_dev=0.799,
                                              would_exceed_G1=(A["dev_sel2000"].get(nm, {}).get("hits1", 0) > 0.799)) for nm in A["dev_sel2000"]}
        arms[f"P_{k}"] = A
    R["arms"] = arms
    R["references"] = {"S_5_test_hits1": float(S[(5, 1)].mean()), "S_1_test_hits1": float(S[(1, 1)].mean()), "G_5_test_hits1": float(G5.mean()), "published": COMP,
                       "registered_seed0": {"P_5_minus_S_5": TR["KP3_descriptive"]["k5_primary_P_5_minus_S_5"], "KP4": TR["primary_tests_k1"]["KP4"]}}
    R["computed_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return R, PQOUT
