#!/usr/bin/env python3
"""seeds_analyze.py -- registered analyses of multitq_kpct_seed_amendment_G.json (LoRA-seed replication of G_1).

Reads ONLY frozen files: the seed test prediction files (sha256 verified against the saved freeze record first),
the frozen seed-0 G_1 prediction file, the registered per-question hit vectors (multitq_kpct_test_hits.npz) for
GU_1, S_1 (3 LightGBM seeds), P_1 (3 S seeds), the frozen unseen-template stratum U_1, and the official test answers.
Scorer: multitq_eval.py (official semantics; sha256 1f85bbdc...). Bootstrap: multitq_eval.paired_poisson_bootstrap
(B = 2000, seed 0; the same weights for every full-test comparison); strata B = 1000.
"""
import hashlib, json, sys
import numpy as np

COMPARATORS = {"AT2QA": 0.887, "ORACLE-T": 0.869, "TKG-Thinker": 0.855, "SCoP": 0.825}
THR = {"KP2": 0.020, "KP3": 0.010, "KP4": 0.030}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def holm(ps, alpha=0.05):
    """Holm step-down adjusted p-values (dict name -> p)"""
    items = sorted(ps.items(), key=lambda kv: kv[1]); m = len(items); adj, run = {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p)); adj[k] = run
    return adj


def kp1_label(ci, v):
    return "ABOVE" if ci[0] > v else ("BELOW" if ci[1] < v else "TIE")


def run(cfg):
    sys.path.insert(0, cfg["eval_dir"])
    import multitq_eval as ME
    assert sha(cfg["eval"]) == "1f85bbdcd623f1810e9df2910e042c24d6f5c269d278e82b30ef05387a1e2d69"
    assert sha(cfg["test_json"]) == "d9bf3bf23b489ecc76c7940510c718260a42350d934a6edc1c1c7efcd5238fee"
    boot = ME.paired_poisson_bootstrap
    Q = {q["quid"]: q for q in json.load(open(cfg["test_json"]))}
    z = np.load(cfg["npz"], allow_pickle=True)
    quids = [int(q) for q in z["quids"]]; arms = [str(a) for a in z["arms"]]; ix = {a: i for i, a in enumerate(arms)}
    A1 = lambda n: z["H1"][ix[n]].astype(float)
    S1 = (A1("S_1_seed0") + A1("S_1_seed1") + A1("S_1_seed2")) / 3
    P1 = (A1("P_1_seed0") + A1("P_1_seed1") + A1("P_1_seed2")) / 3
    GU1 = A1("GU_1")
    U1 = np.array([cfg["U1"][q] for q in quids], dtype=bool)
    assert U1.sum() == 40205
    # ---------------------------------------------------------------- frozen prediction files: verify sha, score
    hits, files = {}, {}
    for name, spec in cfg["preds"].items():
        got = sha(spec["path"])
        assert got == spec["sha256"], f"{name}: sha256 {got} != frozen {spec['sha256']}"
        P = ME.load_predictions(spec["path"])
        assert set(P) == set(quids), name
        h1 = np.array([ME.hit_at_k(Q[q], P[q], 1) for q in quids], float)
        h10 = np.array([ME.hit_at_k(Q[q], P[q], 10) for q in quids], float)
        hits[name] = (h1, h10); files[name] = dict(path=spec["path"], sha256=got, verified=True)
    assert np.array_equal(hits["G_1_lora0"][0], A1("G_1")), "seed-0 rescoring != registered G_1 vector"
    seeds = [n for n in ("G_1_lora0", "G_1_lora1", "G_1_lora2") if n in hits]
    mean1 = np.mean([hits[n][0] for n in seeds], axis=0); mean10 = np.mean([hits[n][1] for n in seeds], axis=0)
    hits["G_1_loramean"] = (mean1, mean10)
    zero = np.zeros(len(quids))
    res = dict(n_test=len(quids), files=files, seed0_rescore_identical_to_registered=True, arms={}, KP={}, seed_labels={}, strata={})
    # ---------------------------------------------------------------- A1, A5
    for n in seeds + ["G_1_loramean"]:
        h1, h10 = hits[n]
        b1, b10 = boot(h1, zero), boot(h10, zero)
        res["arms"][n] = dict(hits1=b1["delta"], hits1_ci95=[b1["ci_low"], b1["ci_high"]], hits10=b10["delta"], hits10_ci95=[b10["ci_low"], b10["ci_high"]],
                              KP1={c: kp1_label([b1["ci_low"], b1["ci_high"]], v) for c, v in COMPARATORS.items()})
    # ---------------------------------------------------------------- A2
    v = np.array([res["arms"][n]["hits1"] for n in seeds])
    v10 = np.array([res["arms"][n]["hits10"] for n in seeds])
    res["seed_summary"] = dict(seeds=seeds, hits1=dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), min=float(v.min()), max=float(v.max()), range=float(v.max() - v.min())),
                               hits10=dict(mean=float(v10.mean()), sd=float(v10.std(ddof=1)), min=float(v10.min()), max=float(v10.max()), range=float(v10.max() - v10.min())))
    # pairwise agreement between seeds (rank-1 hit agreement)
    res["seed_summary"]["pairwise_hit_disagreement"] = {f"{a} vs {b}": float(np.mean(hits[a][0] != hits[b][0])) for i, a in enumerate(seeds) for b in seeds[i + 1:]}
    # ---------------------------------------------------------------- A3 KP2-KP4 + Holm
    k4 = boot(P1[U1], S1[U1])
    for n in seeds + ["G_1_loramean"]:
        g = hits[n][0]
        kp = {}
        b = boot(g, GU1); kp["KP2"] = dict(contrast=f"{n} - GU_1", delta=b["delta"], ci95=[b["ci_low"], b["ci_high"]], p_one_sided=b["p_one_sided_le0"], n=len(g))
        b = boot(g, S1); kp["KP3"] = dict(contrast=f"{n} - S_1 (3-seed mean)", delta=b["delta"], ci95=[b["ci_low"], b["ci_high"]], p_one_sided=b["p_one_sided_le0"], n=len(g))
        kp["KP4"] = dict(contrast="P_1 - S_1 on U_1 (registered test; independent of the G LoRA seed)", delta=k4["delta"], ci95=[k4["ci_low"], k4["ci_high"]],
                         p_one_sided=k4["p_one_sided_le0"], n=int(U1.sum()))
        b = boot(g[U1], S1[U1]); kp["KP4_G_companion"] = dict(contrast=f"{n} - S_1 on U_1 (registered descriptive)", delta=b["delta"], ci95=[b["ci_low"], b["ci_high"]],
                                                               p_one_sided=b["p_one_sided_le0"], n=int(U1.sum()))
        adj = holm({k: kp[k]["p_one_sided"] for k in ("KP2", "KP3", "KP4")})
        for k in ("KP2", "KP3", "KP4"):
            kp[k].update(threshold=THR[k], holm_p=adj[k], margin_met=kp[k]["delta"] >= THR[k], pass_=bool(kp[k]["delta"] >= THR[k] and adj[k] < 0.05))
        kp["KP2"]["label"] = "GUIDANCE_VALUE" if kp["KP2"]["pass_"] else "NO_GUIDANCE_VALUE"
        kp["KP3"]["label"] = "LLM_OVER_STRUCTURE" if kp["KP3"]["pass_"] else "NO_LLM_GAIN"
        kp["KP4"]["label"] = "UNSEEN_TEMPLATE_GAIN" if kp["KP4"]["pass_"] else "NO_UNSEEN_GAIN"
        if n == "G_1_loramean":
            kp["note"] = "seed-mean arm: descriptive (per-question mean hit over LoRA seeds 0-2)"
        res["KP"][n] = kp
    # ---------------------------------------------------------------- A4 seed labels
    for k in ("KP2", "KP3", "KP4_G_companion"):
        d0 = np.sign(res["KP"]["G_1_lora0"][k]["delta"])
        ok = all((res["KP"][n][k]["ci95"][0] > 0) if d0 > 0 else (res["KP"][n][k]["ci95"][1] < 0) for n in seeds)
        res["seed_labels"][k] = dict(label="SEED_STABLE" if ok else "SEED_SENSITIVE", direction_seed0="+" if d0 > 0 else "-",
                                     ci95_per_seed={n: res["KP"][n][k]["ci95"] for n in seeds},
                                     pass_label_identical_across_seeds=(len({res["KP"][n][k].get("pass_") for n in seeds}) == 1) if k != "KP4_G_companion" else None)
    res["seed_labels"]["KP4"] = dict(label="NOT_APPLICABLE (registered KP4 test is P_1 - S_1; identical for every G seed)", pass_=res["KP"]["G_1_lora0"]["KP4"]["pass_"])
    res["seed_labels"]["KP1_AT2QA"] = dict(labels={n: res["arms"][n]["KP1"]["AT2QA"] for n in seeds + ["G_1_loramean"]},
                                           identical=len({res["arms"][n]["KP1"]["AT2QA"] for n in seeds}) == 1)
    # ---------------------------------------------------------------- A7 strata
    ql = np.array([ME.qlabel_bucket(Q[q]) for q in quids]); at = np.array([Q[q]["answer_type"] for q in quids])
    for col, vec in (("qlabel", ql), ("answer_type", at)):
        res["strata"][col] = {}
        for lv in sorted(set(vec)):
            m = vec == lv
            row = dict(n=int(m.sum()))
            for n in seeds + ["G_1_loramean"]:
                bb = boot(hits[n][0][m], zero[m], B=1000)
                row[n] = dict(hits1=float(hits[n][0][m].mean()), ci95=[bb["ci_low"], bb["ci_high"]])
            res["strata"][col][lv] = row
    return res, hits, quids
