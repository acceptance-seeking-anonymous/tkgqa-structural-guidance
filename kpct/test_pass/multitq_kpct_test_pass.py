#!/usr/bin/env python3
"""multitq_kpct_test_pass.py -- the single registered test pass of the MultiTQ k% data-efficiency experiment.

Rule: protocol/multitq_kpct_decision_rule.json v1.0 (recipe sha256 0f5f7604...), amendments B1, B2, guided B3,
primary declaration (k1 G_1, k5 P_5, k10 P_10, k100 G_100; k0 G_0 from the k0 freeze).

Stage 1  `remote`  (the GPU host, inside tkgqa-gpu:latest, frozen files mounted read-only)
  1. sha256 of EVERY registered test file (prediction + raw/exec/post generation files + the G_1 test bundle)
     against kpct_test_manifest.json, which is built verbatim from the five freeze files. Any mismatch or
     missing file -> write hash_check.json, exit 3, and NEVER open test.json (STOP_TEST_INTEGRITY).
  2. sha256 of test.json (d9bf3bf2...) and multitq_eval.py (1f85bbdc...).
  3. Official scoring: multitq_eval.load_predictions + multitq_eval.hit_at_k (k = 1, 10) for every prediction
     file, in test.json order -> per-question hit vectors (uint8) + coverage statistics.
  4. NOVEL_GOLD_G1 input: for every test question, does G_1's rank-1 answer occur in its own k01_s0 test bundle
     (no gold needed for this flag; combined with the hit locally).
Stage 2  `analyze` (local): arm hits (S_k / P_k = per-question mean over 3 seeds), single-arm CIs, KP1-KP4,
  Holm, qualifiers, descriptive strata, tables -> multitq_kpct_test_results.json and per-question CSV.

Statistic (rule `statistic`): multitq_eval.paired_poisson_bootstrap, B = 2000, seed 0, two-sided 95% percentile
CI, one-sided p = share of bootstrap deltas <= 0; single-arm CI = the same bootstrap of the mean hit
(paired against a zero vector); KP4 over the U_k questions with B = 2000; descriptive strata B = 1000.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import sys
import time

TEST_JSON_SHA = "d9bf3bf23b489ecc76c7940510c718260a42350d934a6edc1c1c7efcd5238fee"
SCORER_SHA = "1f85bbdcd623f1810e9df2910e042c24d6f5c269d278e82b30ef05387a1e2d69"


def sha256_file(p, bs=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(bs)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_scorer(path):
    assert sha256_file(path) == SCORER_SHA, "scorer sha256 mismatch"
    spec = importlib.util.spec_from_file_location("multitq_eval", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --------------------------------------------------------------------------------------------------------------
# stage 1: remote
# --------------------------------------------------------------------------------------------------------------
def _score_one(args):
    arm, path, scorer_path, test_path = args
    ME = load_scorer(scorer_path)
    Q = json.load(open(test_path, encoding="utf-8"))
    import numpy as np
    preds = ME.load_predictions(path)
    gq = [int(q["quid"]) for q in Q]
    gs = set(gq)
    h1 = np.zeros(len(Q), dtype=np.uint8)
    h10 = np.zeros(len(Q), dtype=np.uint8)
    n_short10 = 0
    n_empty = 0
    for i, q in enumerate(Q):
        r = preds.get(int(q["quid"]))
        if r is not None:
            if len(r) < 10:
                n_short10 += 1
            if len(r) == 0:
                n_empty += 1
        h1[i] = ME.hit_at_k(q, r, 1)
        h10[i] = ME.hit_at_k(q, r, 10)
    cov = {"n_pred": len(preds), "n_missing": sum(1 for x in gq if x not in preds),
           "n_extra": sum(1 for x in preds if x not in gs), "n_short_lt10": n_short10, "n_empty_list": n_empty,
           "hits1": float(h1.mean()), "hits10": float(h10.mean())}
    return arm, h1, h10, cov


def _norm(s):
    return " ".join(str(s).replace("_", " ").split()).casefold()


def bundle_occurrence(bundle_path, g1_pred_path, scorer_path):
    """For every test quid: is G_1's rank-1 answer string present in its own bundle?
    field: exact (normalised) match against bundle strings -- candidate answers, fact heads/tails, fact dates
           and their year / month prefixes, linked-entity alternatives, the parse time value;
    text : normalised substring of the serialised bundle (looser; sensitivity)."""
    ME = load_scorer(scorer_path)
    preds = ME.load_predictions(g1_pred_path)
    out = {}
    with open(bundle_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            b = json.loads(line)
            quid = int(b["quid"])
            r = preds.get(quid) or []
            ans = _norm(r[0]) if r else ""
            S = set()
            for c in b.get("candidates") or []:
                S.add(_norm(c.get("answer", "")))
            for fa in b.get("facts") or []:
                for key in ("head", "tail"):
                    if fa.get(key) is not None:
                        S.add(_norm(fa[key]))
                d = fa.get("date")
                if d:
                    d = str(d)
                    S.update({d, d[:4], d[:7]})
            p = b.get("parse") or {}
            for m in p.get("mentions") or []:
                for a in m.get("alternatives") or []:
                    if a.get("entity"):
                        S.add(_norm(a["entity"]))
            t = p.get("time")
            if isinstance(t, dict) and t.get("value"):
                S.add(_norm(t["value"]))
            txt = _norm(json.dumps({k: b.get(k) for k in ("parse", "candidates", "facts", "operator_trace")},
                                   ensure_ascii=False))
            out[quid] = (int(bool(ans) and ans in S), int(bool(ans) and ans in txt))
    return out


def remote(a):
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    M = json.load(open(a.manifest))
    rows = []
    for m in M:
        p = os.path.join(a.root, m["path"])
        if not os.path.exists(p):
            rows.append(dict(m, observed=None, ok=False, bytes=None, note="missing"))
            continue
        h = sha256_file(p)
        rows.append(dict(m, observed=h, ok=(h == m["sha256"]), bytes=os.path.getsize(p)))
    all_ok = all(r["ok"] for r in rows)
    hc = {"checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n_files": len(rows),
          "n_ok": sum(r["ok"] for r in rows), "all_ok": all_ok, "files": rows}
    json.dump(hc, open(os.path.join(a.out, "hash_check.json"), "w"), indent=1)
    print(f"hash check: {hc['n_ok']}/{hc['n_files']} ok", flush=True)
    if not all_ok:
        print("STOP_TEST_INTEGRITY: hash mismatch -- nothing scored, test.json not opened", flush=True)
        sys.exit(3)
    # only now: gold file
    assert sha256_file(a.test_json) == TEST_JSON_SHA, "test.json sha256 mismatch"
    load_scorer(a.scorer)
    jobs = [(m["arm"], os.path.join(a.root, m["path"]), a.scorer, a.test_json)
            for m in M if m["role"] in ("pred", "pred_descriptive")]
    import numpy as np
    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        res = pool.map(_score_one, jobs)
    arms = [r[0] for r in res]
    assert len(set(arms)) == len(arms)
    H1 = np.stack([r[1] for r in res])
    H10 = np.stack([r[2] for r in res])
    cov = {r[0]: r[3] for r in res}
    Q = json.load(open(a.test_json, encoding="utf-8"))
    quids = np.array([int(q["quid"]) for q in Q], dtype=np.int64)
    # NOVEL_GOLD_G1 occurrence flags
    bm = [m for m in M if m["arm"] == "BUNDLE_k01_s0_test"][0]
    gm = [m for m in M if m["arm"] == "G_1" and m["role"] == "pred"][0]
    occ = bundle_occurrence(os.path.join(a.root, bm["path"]), os.path.join(a.root, gm["path"]), a.scorer)
    occ_field = np.array([occ.get(int(q), (0, 0))[0] for q in quids], dtype=np.uint8)
    occ_text = np.array([occ.get(int(q), (0, 0))[1] for q in quids], dtype=np.uint8)
    n_bundle_missing = int(sum(1 for q in quids if int(q) not in occ))
    # G-family invalid rates recomputed from the raw generation files ('valid' field)
    inv = {}
    for m in M:
        if m["role"] == "raw" and m["src"] == "guided_freeze":
            n = bad = 0
            with gzip.open(os.path.join(a.root, m["path"]), "rt", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        n += 1
                        bad += 0 if r.get("valid") else 1
            inv[m["arm"]] = {"n": n, "invalid": bad, "invalid_rate": bad / max(n, 1)}
    np.savez_compressed(os.path.join(a.out, "kpct_test_hits.npz"), arms=np.array(arms), quids=quids, H1=H1, H10=H10,
                        g1_answer_in_bundle_field=occ_field, g1_answer_in_bundle_text=occ_text)
    meta = {"scored_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n_test": len(Q),
            "test_json_sha256": TEST_JSON_SHA, "scorer_sha256": SCORER_SHA, "arms": arms, "coverage": cov,
            "bundle_rows_missing": n_bundle_missing, "g_invalid_from_raw": inv, "secs": round(time.time() - t0, 1)}
    json.dump(meta, open(os.path.join(a.out, "kpct_test_scoring_meta.json"), "w"), indent=1)
    print(json.dumps({k: meta[k] for k in ("scored_utc", "n_test", "secs", "bundle_rows_missing")}), flush=True)


# --------------------------------------------------------------------------------------------------------------
# stage 2: analyze (local)
# --------------------------------------------------------------------------------------------------------------
COMPARATORS = [("AT2QA", 0.887), ("ORACLE-T", 0.869), ("TKG-Thinker", 0.855), ("SCoP", 0.825)]
QTYPES = ["equal", "before_after", "first_last", "equal_multi", "after_first", "before_last"]
KLISTS = ["k01_s0", "k01_s1", "k01_s2", "k05_s0", "k10_s0", "k100"]
KTAG = {"k01_s0": "1", "k01_s1": "1_s1", "k01_s2": "1_s2", "k05_s0": "5", "k10_s0": "10", "k100": "100"}
PRIMARY = {"0": "G_0", "1": "G_1", "5": "P_5", "10": "P_10", "100": "G_100"}   # declaration 13da45f4 + k0 freeze
NONPRIMARY = {"0": "P_0", "1": "P_1", "5": "G_5", "10": "G_10", "100": "P_100"}


def kp1_label(lo, hi, c):
    return "ABOVE" if lo > c else ("BELOW" if hi < c else "TIE")


def holm(pvals):
    """Holm step-down adjusted p-values {name: p} -> {name: p_adj}."""
    names = sorted(pvals, key=lambda n: pvals[n])
    m = len(names)
    adj, prev = {}, 0.0
    for i, n in enumerate(names):
        a = min(1.0, (m - i) * pvals[n])
        prev = max(prev, a)
        adj[n] = prev
    return adj


def analyze(a):
    import numpy as np
    ME = load_scorer(a.scorer)
    hc = json.load(open(a.hash_check))
    assert hc["all_ok"] and hc["n_ok"] == hc["n_files"], "hash check not clean -- refusing to analyse"
    D = np.load(a.hits)
    meta = json.load(open(a.meta))
    Q = json.load(open(a.test_json, encoding="utf-8"))
    assert sha256_file(a.test_json) == TEST_JSON_SHA
    quids = D["quids"]
    assert [int(q["quid"]) for q in Q] == [int(x) for x in quids]
    n = len(Q)
    arms_file = [str(x) for x in D["arms"]]
    H = {1: {}, 10: {}}
    for i, arm in enumerate(arms_file):
        H[1][arm] = D["H1"][i].astype(float)
        H[10][arm] = D["H10"][i].astype(float)
    # registered derived arms: per-question mean over the 3 seeds
    for t in ["1", "1_s1", "1_s2", "5", "10", "100"]:
        for k in (1, 10):
            H[k][f"S_{t}"] = np.mean([H[k][f"S_{t}_seed{s}"] for s in range(3)], axis=0)
    for t in ["1", "5", "10", "100"]:
        for k in (1, 10):
            H[k][f"P_{t}"] = np.mean([H[k][f"P_{t}_seed{s}"] for s in range(3)], axis=0)
    cfg = json.load(open(a.config))
    prof = json.load(open(cfg["profile"]))
    U = {}
    qpos = {int(q): i for i, q in enumerate(quids)}
    for kl in KLISTS:
        st = prof["kp4_strata"][kl]
        qs = sorted(int(x) for x in st["quids"])
        h = hashlib.sha256(json.dumps(qs, separators=(",", ":")).encode("utf-8")).hexdigest()
        assert h == st["sha256"] and len(qs) == st["n"], kl
        m = np.zeros(n, dtype=bool)
        m[[qpos[x] for x in qs]] = True
        U[kl] = m
    qt = np.array([q["qtype"] for q in Q])
    ql = np.array(["Single" if "Single" in q["qlabel"] else "Multiple" for q in Q])
    at = np.array([q["answer_type"] for q in Q])
    zeros = np.zeros(n)
    # The Poisson(1) weights depend only on (n, B, seed) (rule `statistic`), so for full-test comparisons the
    # 2000 x 54,584 weight matrix is drawn once (same generator stream as multitq_eval.paired_poisson_bootstrap)
    # and re-used; every registered decision is additionally recomputed with the official function and asserted
    # bit-identical.
    _W = {}

    def PB(x, y, B=2000, seed=0, alpha=0.05):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) != n or B != 2000 or seed != 0:
            return ME.paired_poisson_bootstrap(x, y, B=B, seed=seed, alpha=alpha)
        if "W" not in _W:
            rng = np.random.default_rng(seed)
            W = np.empty((B, n), dtype=np.uint8)
            for i in range(B):
                w = rng.poisson(1.0, size=n)
                assert w.max() < 256
                W[i] = w
            _W["W"] = W
            _W["sw"] = W.sum(axis=1, dtype=np.int64)
        W, sw = _W["W"], _W["sw"]
        d = x - y
        deltas = np.empty(B)
        for i in range(0, B, 100):
            Wc = W[i:i + 100].astype(np.int64)
            num = (Wc * d).sum(axis=1)
            s = sw[i:i + 100]
            deltas[i:i + 100] = np.where(s > 0, num / np.where(s > 0, s, 1), 0.0)
        lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
        return {"delta": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
                "p_one_sided_le0": float((deltas <= 0).mean()), "B": B, "seed": seed}

    def single(h, B=2000):
        r = PB(h, np.zeros(len(h)), B=B, seed=0)
        return {"hits": r["delta"], "ci95": [r["ci_low"], r["ci_high"]], "n": int(len(h))}

    def pair(x, y, B=2000):
        r = PB(x, y, B=B, seed=0)
        return {"delta": r["delta"], "ci95": [r["ci_low"], r["ci_high"]], "p_one_sided": r["p_one_sided_le0"],
                "B": B, "seed": 0, "n": int(len(x))}

    reg_arms = {
        "0": ["P_0", "G_0"],
        "1": ["RULE_1", "S_1", "G_1", "GU_1", "P_1"],
        "1_s1": ["RULE_1_s1", "S_1_s1", "G_1_s1"],
        "1_s2": ["RULE_1_s2", "S_1_s2", "G_1_s2"],
        "5": ["RULE_5", "S_5", "G_5", "P_5"],
        "10": ["RULE_10", "S_10", "G_10", "GU_10", "P_10"],
        "100": ["RULE_100", "S_100", "G_100", "GU_100", "P_100"],
    }
    not_run = {"GU_1_s1": "not in design (GU at k = 1 on subset seed 0 only)",
               "GU_1_s2": "not in design (GU at k = 1 on subset seed 0 only)",
               "P_1_s1": "NOT_RUN (fallback 1, amendments B1/B2)", "P_1_s2": "NOT_RUN (fallback 1, amendments B1/B2)",
               "GU_5": "NOT_RUN (fallback 2, amendments B1/B2)"}
    descriptive_arms = [f"S_{t}_seed{s}" for t in ["1", "1_s1", "1_s2", "5", "10", "100"] for s in range(3)] + \
                       [f"P_{t}_seed{s}" for t in ["1", "5", "10", "100"] for s in range(3)] + \
                       [f"SMEANFILE_{t}" for t in ["1", "1_s1", "1_s2", "5", "10"]] + ["SHUF_G1", "QONLY_G1"]
    all_arms = [x for v in reg_arms.values() for x in v] + descriptive_arms
    R = {"name": "multitq_kpct_test_results", "rule_file": "protocol/multitq_kpct_decision_rule.json",
         "rule_sha256": "0f5f760450356349a7bf353301520d76de3617f1dfec0b4e4b1430c97b1d6cd7",
         "scored_utc": meta["scored_utc"], "n_test": n, "scorer_sha256": SCORER_SHA, "test_json_sha256": TEST_JSON_SHA,
         "hash_check": {"n_files": hc["n_files"], "n_ok": hc["n_ok"], "all_ok": hc["all_ok"],
                        "checked_utc": hc["checked_utc"],
                        "local_copies_checked": cfg.get("local_copies_checked")},
         "coverage": meta["coverage"], "statistic": "multitq_eval.paired_poisson_bootstrap, B = 2000, seed 0 (full test and KP4); strata B = 1000; single-arm CI = bootstrap of the mean hit; one-sided p = share of bootstrap deltas <= 0",
         "primary_llm_arm": PRIMARY, "not_run": not_run}
    # ---- per-arm overall --------------------------------------------------------------------------------------
    arms = {}
    for arm in all_arms:
        e = {"hits1": float(H[1][arm].mean()), "hits10": float(H[10][arm].mean())}
        s1 = single(H[1][arm])
        s10 = single(H[10][arm])
        e["hits1_ci95"] = s1["ci95"]
        e["hits10_ci95"] = s10["ci95"]
        e["registered"] = arm not in descriptive_arms
        if e["registered"]:
            e["KP1"] = {c: kp1_label(s1["ci95"][0], s1["ci95"][1], v) for c, v in COMPARATORS}
        b = {}
        for k in (1, 10):
            h = H[k][arm]
            d = {"Overall": float(h.mean()), "Single": float(h[ql == "Single"].mean()),
                 "Multiple": float(h[ql == "Multiple"].mean()), "Entity": float(h[at == "entity"].mean()),
                 "Time": float(h[at == "time"].mean())}
            d.update({t: float(h[qt == t].mean()) for t in QTYPES})
            b[f"hits{k}"] = d
        e["breakdown"] = b
        arms[arm] = e
    # descriptive stratum CIs (B = 1000), Hits@1, registered arms + controls
    for arm in [x for v in reg_arms.values() for x in v] + ["SHUF_G1", "QONLY_G1"]:
        h = H[1][arm]
        ci = {}
        for nm, msk in [("Single", ql == "Single"), ("Multiple", ql == "Multiple"), ("Entity", at == "entity"),
                        ("Time", at == "time")] + [(t, qt == t) for t in QTYPES]:
            ci[nm] = single(h[msk], B=1000)["ci95"]
        arms[arm]["breakdown"]["hits1_ci95_B1000"] = ci
    R["arms"] = arms
    # ---- KP1 ----------------------------------------------------------------------------------------------------
    g1 = arms["G_1"]
    kp1 = {"primary_statement": {"arm": "G_1", "comparator": "AT2QA", "published": 0.887, "hits1": g1["hits1"],
                                 "ci95": g1["hits1_ci95"], "label": g1["KP1"]["AT2QA"]}}
    labs = [arms[x]["KP1"]["AT2QA"] for x in ("G_1", "G_1_s1", "G_1_s2")]
    kp1["primary_statement"]["subset_seed_labels"] = {"s0": labs[0], "s1": labs[1], "s2": labs[2]}
    kp1["primary_statement"]["qualifier"] = "SUBSET_ROBUST" if len(set(labs)) == 1 else "SUBSET_FRAGILE"
    kmin = {}
    for c, v in COMPARATORS:
        kk = [k for k in ["0", "1", "5", "10", "100"] if arms[PRIMARY[k]]["KP1"][c] == "ABOVE"]
        kmin[c] = (int(kk[0]) if kk else None)
    kp1["k_min_above_primary_llm_arm"] = kmin
    kp1["table"] = {arm: {"hits1": arms[arm]["hits1"], "ci95": arms[arm]["hits1_ci95"], **arms[arm]["KP1"]}
                    for v in reg_arms.values() for arm in v}
    R["KP1"] = kp1
    # ---- KP2-KP4 primary (k = 1) + Holm -------------------------------------------------------------------------
    kp2 = pair(H[1]["G_1"], H[1]["GU_1"])
    kp3 = pair(H[1]["G_1"], H[1]["S_1"])
    u1 = U["k01_s0"]
    kp4 = pair(H[1]["P_1"][u1], H[1]["S_1"][u1])
    adj = holm({"KP2": kp2["p_one_sided"], "KP3": kp3["p_one_sided"], "KP4": kp4["p_one_sided"]})
    thr = {"KP2": 0.020, "KP3": 0.010, "KP4": 0.030}
    lab = {"KP2": ("GUIDANCE_VALUE", "NO_GUIDANCE_VALUE"), "KP3": ("LLM_OVER_STRUCTURE", "NO_LLM_GAIN"),
           "KP4": ("UNSEEN_TEMPLATE_GAIN", "NO_UNSEEN_GAIN")}
    prim = {}
    for nm, r, what in [("KP2", kp2, "G_1 - GU_1, test Hits@1, all 54,584 questions"),
                        ("KP3", kp3, "G_1 - S_1 (3-seed mean), test Hits@1, all 54,584 questions"),
                        ("KP4", kp4, "P_1 - S_1 on U_1 (k01_s0 unseen-template stratum, n = 40,205)")]:
        ok = (r["delta"] >= thr[nm]) and (adj[nm] < 0.05)
        prim[nm] = dict(r, contrast=what, threshold=thr[nm], holm_p=adj[nm], margin_met=r["delta"] >= thr[nm],
                        pass_=ok, label=lab[nm][0] if ok else lab[nm][1])
    # identity check of the cached-weight path against the official function on the registered decisions
    chk = {"KP2": ME.paired_poisson_bootstrap(H[1]["G_1"], H[1]["GU_1"], B=2000, seed=0),
           "KP3": ME.paired_poisson_bootstrap(H[1]["G_1"], H[1]["S_1"], B=2000, seed=0),
           "KP4": ME.paired_poisson_bootstrap(H[1]["P_1"][u1], H[1]["S_1"][u1], B=2000, seed=0),
           "KP1_G_1": ME.paired_poisson_bootstrap(H[1]["G_1"], zeros, B=2000, seed=0)}
    for nm, r in [("KP2", kp2), ("KP3", kp3), ("KP4", kp4)]:
        assert (chk[nm]["delta"], chk[nm]["ci_low"], chk[nm]["ci_high"], chk[nm]["p_one_sided_le0"]) == \
               (r["delta"], r["ci95"][0], r["ci95"][1], r["p_one_sided"]), nm
    assert [chk["KP1_G_1"]["ci_low"], chk["KP1_G_1"]["ci_high"]] == arms["G_1"]["hits1_ci95"]
    R["bootstrap_identity_check"] = "cached-weight bootstrap == multitq_eval.paired_poisson_bootstrap (bit-identical) on KP1 primary, KP2, KP3, KP4"
    R["primary_tests_k1"] = prim
    R["multiplicity"] = "Holm step-down over KP2, KP3, KP4 at k = 1 (one-sided bootstrap p), family alpha 0.05"
    # ---- KP2 descriptive ----------------------------------------------------------------------------------------
    kp2d = {}
    for t in ["10", "100"]:
        r = pair(H[1][f"G_{t}"], H[1][f"GU_{t}"])
        r["descriptive_criterion_met"] = bool(r["delta"] >= 0.020 and r["ci95"][0] > 0)
        kp2d[f"k{t}"] = r
    kp2d["k5"] = "NOT_RUN (GU_5 dropped by fallback 2)"
    R["KP2_descriptive"] = kp2d
    # ---- KP3 descriptive ----------------------------------------------------------------------------------------
    kp3d = {}
    for t in ["5", "10", "100"]:
        kp3d[f"k{t}_primary_{PRIMARY[t]}_minus_S_{t}"] = pair(H[1][PRIMARY[t]], H[1][f"S_{t}"])
    for t in ["1", "5", "10", "100"]:
        kp3d[f"k{t}_primary_{PRIMARY[t]}_minus_S_{t}_seed0"] = pair(H[1][PRIMARY[t]], H[1][f"S_{t}_seed0"])
        kp3d[f"k{t}_nonprimary_{NONPRIMARY[t]}_minus_S_{t}"] = pair(H[1][NONPRIMARY[t]], H[1][f"S_{t}"])
    sub = {"s0": kp3}
    for s in ["s1", "s2"]:
        sub[s] = pair(H[1][f"G_1_{s}"], H[1][f"S_1_{s}"])
        kp3d[f"k1_{s}_G_1_{s}_minus_S_1_{s}"] = sub[s]
    kp3d["k1_subset_qualifier"] = "SUBSET_ROBUST" if all(sub[s]["delta"] >= 0.010 for s in sub) else "SUBSET_FRAGILE"
    kp3d["k1_subset_deltas"] = {s: sub[s]["delta"] for s in sub}
    # k = 0: no S_0 exists; descriptive G_0 - P_0
    kp3d["k0_G_0_minus_P_0"] = pair(H[1]["G_0"], H[1]["P_0"])
    R["KP3_descriptive"] = kp3d
    # ---- KP4 / table 4 ------------------------------------------------------------------------------------------
    t4 = {}
    for kl in KLISTS:
        t = KTAG[kl]
        m = U[kl]
        row = {"U_n": int(m.sum()), "seen_n": int((~m).sum()), "arms_on_U": {}, "arms_on_seen": {}}
        cand = [x for x in [f"RULE_{t}", f"S_{t}", f"G_{t}", f"GU_{t}", f"P_{t}"] if x in H[1]]
        if kl == "k01_s0":
            cand += ["P_0", "G_0", "SHUF_G1", "QONLY_G1"]
        if kl == "k100":
            cand += ["P_0", "G_0"]
        for arm in cand:
            row["arms_on_U"][arm] = single(H[1][arm][m], B=1000)
            row["arms_on_seen"][arm] = single(H[1][arm][~m], B=1000)
        if f"P_{t}" in H[1]:
            row["P_minus_S_on_U"] = pair(H[1][f"P_{t}"][m], H[1][f"S_{t}"][m])
            row["P_minus_S_on_seen"] = pair(H[1][f"P_{t}"][~m], H[1][f"S_{t}"][~m])
        else:
            row["P_minus_S_on_U"] = "NOT_RUN (fallback 1)" if kl in ("k01_s1", "k01_s2") else None
        row["G_minus_S_on_U"] = pair(H[1][f"G_{t}"][m], H[1][f"S_{t}"][m])
        row["G_minus_S_on_seen"] = pair(H[1][f"G_{t}"][~m], H[1][f"S_{t}"][~m])
        t4[kl] = row
    t4["k0_note"] = "k = 0 arms read no train question beyond 8 exemplars (P_0); U_0 = the whole test split (54,584)"
    R["KP4_and_table4_unseen_templates"] = t4
    R["KP4_subset_qualifier"] = "NOT_TESTED (P_1 subset seeds 1-2 NOT_RUN, fallback 1)"
    # ---- table 5 subset variance (k = 1) ------------------------------------------------------------------------
    t5 = {}
    for fam in ["RULE", "S", "G"]:
        v = [arms[f"{fam}_1"]["hits1"], arms[f"{fam}_1_s1"]["hits1"], arms[f"{fam}_1_s2"]["hits1"]]
        t5[fam] = {"s0": v[0], "s1": v[1], "s2": v[2], "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                   "range": [float(min(v)), float(max(v))]}
    t5["P"] = {"s0": arms["P_1"]["hits1"], "s1": "NOT_RUN", "s2": "NOT_RUN", "note": "fallback 1"}
    for fam in ["S", "G"]:
        v = [float(H[1][f"{fam}_1{x}"][U[kl]].mean()) for x, kl in [("", "k01_s0"), ("_s1", "k01_s1"), ("_s2", "k01_s2")]]
        t5[fam + "_on_own_U"] = {"s0": v[0], "s1": v[1], "s2": v[2], "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
    R["table5_subset_variance_k1"] = t5
    # ---- controls / qualifiers ----------------------------------------------------------------------------------
    occ_f = D["g1_answer_in_bundle_field"].astype(bool)
    occ_t = D["g1_answer_in_bundle_text"].astype(bool)
    hitg1 = H[1]["G_1"].astype(bool)
    novel = float((hitg1 & ~occ_f).mean())
    novel_t = float((hitg1 & ~occ_t).mean())
    shuf, qonly, g1h = arms["SHUF_G1"]["hits1"], arms["QONLY_G1"]["hits1"], arms["G_1"]["hits1"]
    ctrl = {"SHUF_G1": {"hits1": shuf, "ci95": arms["SHUF_G1"]["hits1_ci95"], "ratio_to_G_1": shuf / g1h,
                        "qualifier": "SHUF_CLEAN" if shuf < 0.5 * g1h else "SHUF_UNCLEAN",
                        "G_1_minus_SHUF_G1": pair(H[1]["G_1"], H[1]["SHUF_G1"])},
            "QONLY_G1": {"hits1": qonly, "ci95": arms["QONLY_G1"]["hits1_ci95"], "ratio_to_G_1": qonly / g1h,
                         "G_1_minus_QONLY_G1": pair(H[1]["G_1"], H[1]["QONLY_G1"])},
            "NOVEL_GOLD_G1": {"value": novel, "definition": "share of test questions where G_1's rank-1 answer is a Hits@1 hit and the answer string (normalised: case fold, '_' -> ' ', whitespace collapse) is not among the bundle's strings (candidate answers, fact heads/tails, fact dates and their year/month prefixes, linked-entity alternatives, parse time value)",
                              "sensitivity_substring": novel_t,
                              "n_bundle_rows_missing": meta["bundle_rows_missing"],
                              "answer_in_bundle_share": float(occ_f.mean())}}
    mem = (qonly >= 0.5 * g1h) or (novel >= 0.010)
    ctrl["MEMORY_SUSPECT"] = {"flag": bool(mem), "QONLY_ratio": qonly / g1h, "NOVEL_GOLD_G1": novel,
                              "rule": "QONLY_G1 >= 0.5 x G_1, or NOVEL_GOLD_G1 >= 0.010 (qualifies, never overturns)"}
    R["controls_k1"] = ctrl
    R["qualifiers"] = cfg["qualifier_inputs"]
    R["qualifiers"]["SHUF"] = ctrl["SHUF_G1"]["qualifier"]
    R["qualifiers"]["MEMORY_SUSPECT"] = bool(mem)
    R["qualifiers"]["KP1_primary_SUBSET"] = kp1["primary_statement"]["qualifier"]
    R["qualifiers"]["KP3_SUBSET"] = kp3d["k1_subset_qualifier"]
    R["g_invalid_from_raw"] = meta["g_invalid_from_raw"]
    R["published_comparators"] = {c: v for c, v in COMPARATORS}
    # per-question CSV
    import csv
    cols = [x for x in arms_file] + [x for x in H[1] if x not in arms_file]
    with gzip.open(a.out_csv, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quid"] + [f"{c}@1" for c in cols] + [f"{c}@10" for c in cols] +
                   ["U_" + kl for kl in KLISTS] + ["G_1_answer_in_bundle"])
        for i in range(n):
            w.writerow([int(quids[i])] + [_fmt(H[1][c][i]) for c in cols] + [_fmt(H[10][c][i]) for c in cols] +
                       [int(U[kl][i]) for kl in KLISTS] + [int(occ_f[i])])
    R["per_question_csv_columns"] = {"arms": cols, "note": "<arm>@1 / <arm>@10 per-question hits; S_k / P_k columns are the registered 3-seed per-question means (0, 1/3, 2/3, 1); U_<list> = 1 if the question is in the frozen unseen-template stratum of that k% list"}
    json.dump(R, open(a.out_json, "w"), indent=1, default=float)
    return R


def summarize(a):
    """Adds the headline label block, the table-1 rows (train sizes, dev_sel2000, GPU-h) and the GPU usage log to the
    results JSON. Pure bookkeeping over the analyze output and the freezes; no scoring, no bootstrap."""
    R = json.load(open(a.results))
    cfg = json.load(open(a.config))
    led = cfg["gpu"]
    p = R["primary_tests_k1"]
    R["labels"] = {
        "KP1_primary_statement": f"G_1 vs AT2QA 0.887: {R['KP1']['primary_statement']['label']}",
        "KP1_primary_label": R["KP1"]["primary_statement"]["label"],
        "KP1_primary_subset_qualifier": R["KP1"]["primary_statement"]["qualifier"],
        "KP1_k_min_above_primary_llm_arm": R["KP1"]["k_min_above_primary_llm_arm"],
        "KP2_k1": p["KP2"]["label"], "KP3_k1": p["KP3"]["label"], "KP4_k1": p["KP4"]["label"],
        "KP2_k5": "NOT_RUN", "KP3_k1_subset_qualifier": R["KP3_descriptive"]["k1_subset_qualifier"],
        "KP4_subset_qualifier": "NOT_TESTED",
        "SHUF": R["qualifiers"]["SHUF"], "MEMORY_SUSPECT": R["qualifiers"]["MEMORY_SUSPECT"],
        "LOW_SILVER": {k: v["LOW_SILVER"] for k, v in R["qualifiers"]["LOW_SILVER"].items()},
        "DEV_NO_GAIN": {k: v.get("DEV_NO_GAIN", v.get("qualifier")) for k, v in R["qualifiers"]["DEV_NO_GAIN"].items()},
        "DEV_FAILURE": "none triggered",
        "PROVISIONAL": [x["numbers"] for x in R["qualifiers"]["PROVISIONAL"]],
        "NOT_RUN": sorted(k for k, v in R["not_run"].items() if v.startswith("NOT_RUN")),
        "KP1_all": {arm: R["arms"][arm]["KP1"] for arm in R["KP1"]["table"]},
    }
    rows = []
    for k, arms in [("0", ["P_0", "G_0"]), ("1", ["RULE_1", "S_1", "G_1", "GU_1", "P_1"]),
                    ("1 (subset s1)", ["RULE_1_s1", "S_1_s1", "G_1_s1"]), ("1 (subset s2)", ["RULE_1_s2", "S_1_s2", "G_1_s2"]),
                    ("5", ["RULE_5", "S_5", "G_5", "P_5"]), ("10", ["RULE_10", "S_10", "G_10", "GU_10", "P_10"]),
                    ("100", ["RULE_100", "S_100", "G_100", "GU_100", "P_100"])]:
        for arm in arms:
            e = R["arms"][arm]
            rows.append({"k": k, "arm": arm, "train_structural": cfg["train_sizes"][arm]["structural"],
                         "train_llm": cfg["train_sizes"][arm]["llm"], "test_hits1": e["hits1"],
                         "test_hits1_ci95": e["hits1_ci95"], "test_hits10": e["hits10"],
                         "dev_sel2000_hits1": cfg["dev_sel2000_hits1"].get(arm), "KP1": e["KP1"],
                         "gpu_h": led["per_arm"].get(arm, 0.0), "primary": arm in R["primary_llm_arm"].values()})
    R["table1_data_efficiency"] = rows
    R["gpu_usage"] = led
    R["published_reference_rows"] = cfg["published_rows"]
    R["provenance"] = cfg.get("provenance")
    R["summarized_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(R, open(a.results, "w"), indent=1, default=float)


def _fmt(x):
    x = float(x)
    return str(int(x)) if x in (0.0, 1.0) else f"{x:.6f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    z = sub.add_parser("analyze")
    z.add_argument("--hits", required=True)
    z.add_argument("--meta", required=True)
    z.add_argument("--hash-check", required=True)
    z.add_argument("--test-json", required=True)
    z.add_argument("--scorer", required=True)
    z.add_argument("--config", required=True)
    z.add_argument("--out-json", default="multitq_kpct_test_results.json")
    z.add_argument("--out-csv", default="multitq_kpct_test_per_question_hits.csv.gz")
    s = sub.add_parser("summarize")
    s.add_argument("--results", default="multitq_kpct_test_results.json")
    s.add_argument("--config", required=True)
    r = sub.add_parser("remote")
    r.add_argument("--root", required=True)
    r.add_argument("--manifest", required=True)
    r.add_argument("--test-json", required=True)
    r.add_argument("--scorer", required=True)
    r.add_argument("--out", default="out")
    r.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    if a.cmd == "remote":
        remote(a)
    elif a.cmd == "analyze":
        analyze(a)
    elif a.cmd == "summarize":
        summarize(a)


if __name__ == "__main__":
    main()
