"""lc_analyze.py -- single scoring pass for amendment LC1 (GU_5, GPLAIN_1). Executed once by lc_score.py, which loads
the inputs and sets the globals:
  ME (official multitq_eval module), TESTQ {quid: question}, H (per-question hits DataFrame of the k% test pass),
  ROB {R1/R2: dict(variants, hits{arm: {squid: hit}})} for the frozen arms, REJ {R1: [...], R2: [...]},
  NEW_TEST {arm: pred_path}, NEW_VAR {(set, arm): pred_path}, FREEZE ({"files_local": {path: sha256}} of the new prediction files).
Writes RES (results dict), PQ (per-question test hits DataFrame), PV (per-variant hits DataFrame)."""
import json, hashlib
import numpy as np, pandas as pd


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _load(p):
    out = {}
    for l in open(p):
        if l.strip():
            r = json.loads(l); rk = r["ranked"]
            out[int(r["quid"])] = json.loads(rk.replace("'", '"')) if isinstance(rk, str) else rk
    return out


def boot1(h, B=2000, seed=0):
    """single-arm CI: same Poisson(1) stream as multitq_eval.paired_poisson_bootstrap with b = 0"""
    r = ME.paired_poisson_bootstrap(np.asarray(h, float), np.zeros(len(h)), B=B, seed=seed)
    return dict(hits1=float(np.mean(h)), ci95=[r["ci_low"], r["ci_high"]], n=int(len(h)))


def pair(a, b, B=2000, seed=0):
    r = ME.paired_poisson_bootstrap(np.asarray(a, float), np.asarray(b, float), B=B, seed=seed)
    lab = "POSITIVE" if r["ci_low"] > 0 else ("NEGATIVE" if r["ci_high"] < 0 else "CI_INCLUDES_0")
    return dict(delta=r["delta"], ci95=[r["ci_low"], r["ci_high"]], p_one_sided_le0=r["p_one_sided_le0"], n=int(len(a)), label=lab, B=B, seed=seed)


# ------------------------------------------------------------------ integrity against the freeze
bad = [p for p, s in FREEZE["files_local"].items() if _sha(p) != s]
assert not bad, f"STOP_TEST_INTEGRITY {bad}"
quids = H["quid"].astype(int).tolist()
assert len(quids) == 54584 and set(quids) == set(TESTQ)
PQ = pd.DataFrame({"quid": quids})
PRE = globals().get("PRE_PQ", {})          # {arm: DataFrame(quid, arm@1, arm@10)} from that file's single scoring pass
for arm, p in NEW_TEST.items():
    if arm in PRE:
        d = PRE[arm].set_index("quid").loc[quids]
        PQ[f"{arm}@1"] = d[f"{arm}@1"].to_numpy(); PQ[f"{arm}@10"] = d[f"{arm}@10"].to_numpy(); continue
    P = _load(p)
    assert set(P) == set(quids), f"{arm}: prediction quids != test"
    PQ[f"{arm}@1"] = [ME.hit_at_k(TESTQ[q], P[q], 1) for q in quids]
    PQ[f"{arm}@10"] = [ME.hit_at_k(TESTQ[q], P[q], 10) for q in quids]
PQ["qlabel"] = [ME.qlabel_bucket(TESTQ[q]) for q in quids]
PQ["answer_type"] = [TESTQ[q]["answer_type"] for q in quids]
Hx = H.set_index("quid").loc[quids].reset_index()
REF = {c: Hx[c].to_numpy(float) for c in ("G_1@1", "G_5@1", "P_5@1", "GU_1@1", "GU_10@1", "GU_100@1", "G_1@10", "GU_1@10", "P_1@1", "S_1@1", "S_5@1")}
RES = dict(arms={}, contrasts={}, strata={}, robustness={})
for arm in NEW_TEST:
    RES["arms"][arm] = dict(boot1(PQ[f"{arm}@1"].to_numpy(float)), hits10=float(PQ[f"{arm}@10"].mean()))
for c in ("G_1@1", "G_5@1", "P_5@1", "GU_1@1", "GU_10@1", "GU_100@1"):
    RES["arms"][c.replace("@1", "") + "_frozen"] = boot1(REF[c])
CON = []
if "GU_5" in NEW_TEST:
    CON += [("G_5 - GU_5", REF["G_5@1"], PQ["GU_5@1"].to_numpy(float)), ("P_5 - GU_5", REF["P_5@1"], PQ["GU_5@1"].to_numpy(float)),
            ("GU_5 - GU_1 (descriptive extra)", PQ["GU_5@1"].to_numpy(float), REF["GU_1@1"]),
            ("GU_10 - GU_5 (descriptive extra)", REF["GU_10@1"], PQ["GU_5@1"].to_numpy(float))]
if "GPLAIN_1" in NEW_TEST:
    CON += [("G_1 - GPLAIN_1", REF["G_1@1"], PQ["GPLAIN_1@1"].to_numpy(float)), ("GPLAIN_1 - GU_1", PQ["GPLAIN_1@1"].to_numpy(float), REF["GU_1@1"])]
for name, a, b in CON:
    RES["contrasts"][name] = pair(a, b)
for col in ("qlabel", "answer_type"):
    for lev in sorted(PQ[col].unique()):
        m = (PQ[col] == lev).to_numpy()
        key = f"{col}={lev}"
        RES["strata"][key] = dict(n=int(m.sum()), arms={arm: boot1(PQ.loc[m, f"{arm}@1"].to_numpy(float)) for arm in NEW_TEST},
                                  frozen={c: float(REF[c][m].mean()) for c in ("G_1@1", "G_5@1", "P_5@1", "GU_1@1")},
                                  contrasts={name: pair(a[m], b[m]) for name, a, b in CON})

# ------------------------------------------------------------------ robustness R1 / R2
PV_rows = []
for S in ("R1", "R2"):
    vr = ROB[S]["variants"]; sq = {(r["quid"], r["variant"]): r["squid"] for r in vr}
    rwv = ["P1", "P2", "P3"] if S == "R1" else ["RW"]
    arms_new = [a for (s_, a) in NEW_VAR if s_ == S]
    hits = dict(ROB[S]["hits"])
    PREV = globals().get("PRE_PV", {})     # {(set, arm): {squid: hit}} from that file's single scoring pass
    for a in arms_new:
        P = _load(NEW_VAR[(S, a)])
        assert set(P) == {r["squid"] for r in vr}, f"{S} {a}: variant quids mismatch"
        hits[a] = PREV[(S, a)] if (S, a) in PREV else {r["squid"]: ME.hit_at_k(TESTQ[r["quid"]], P[r["squid"]], 1) for r in vr}
        # identity check (gold-free): ORIG re-run top-1 vs the frozen test prediction top-1
        T = _load(NEW_TEST[a]); Pv = P
        orig = [r for r in vr if r["variant"] == "ORIG"]
        agree = float(np.mean([str(Pv[r["squid"]][0]) == str(T[r["quid"]][0]) for r in orig]))
        RES["robustness"].setdefault(S, {}).setdefault("identity_top1_orig_vs_test", {})[a] = dict(agree=agree, flag="OK" if agree >= 0.98 else "PIPELINE_DRIFT")
    for r in vr:
        PV_rows.append(dict(set=S, squid=r["squid"], quid=r["quid"], variant=r["variant"], **{f"{a}@1": hits[a][r["squid"]] for a in arms_new}))
    out = RES["robustness"].setdefault(S, {})
    origq = sorted(r["quid"] for r in vr if r["variant"] == "ORIG")
    comp = {"GPLAIN_1": ["G_1", "GU_1", "P_1", "S_1"], "GU_5": ["G_5", "P_5", "S_5", "GU_1", "GU_100"]}
    for v in rwv:
        vq = sorted(r["quid"] for r in vr if r["variant"] == v)
        for scope, qs in (("all", vq), ("after_human_removal", [q for q in vq if q not in set(REJ[S])])):
            if v in ("P1", "P2") and scope != "all":
                continue
            cell = out.setdefault(v, {}).setdefault(scope, {"n": len(qs)})
            allarms = arms_new + sorted({c for a in arms_new for c in comp[a]})
            for a in allarms:
                ho = np.array([hits[a][sq[(q, "ORIG")]] for q in qs], float); hv = np.array([hits[a][sq[(q, v)]] for q in qs], float)
                d = pair(ho, hv)
                lab = "BRITTLE" if (d["delta"] >= 0.02 and d["ci95"][0] > 0) else ("ROBUST" if d["ci95"][1] < 0.02 else "neither")
                cell[a] = dict(orig=float(ho.mean()), rewrite=float(hv.mean()), drop=d["delta"], drop_ci95=d["ci95"], drop_label=lab,
                               status=("registered descriptive" if scope == "all" else "descriptive only (audit rows selected on G/P contrasts)"))
            for a in arms_new:
                for b in comp[a]:
                    dd_a = np.array([hits[a][sq[(q, "ORIG")]] - hits[a][sq[(q, v)]] for q in qs], float)
                    dd_b = np.array([hits[b][sq[(q, "ORIG")]] - hits[b][sq[(q, v)]] for q in qs], float)
                    cell[f"drop({b}) - drop({a})"] = pair(dd_b, dd_a)
                    hv_a = np.array([hits[a][sq[(q, v)]] for q in qs], float); hv_b = np.array([hits[b][sq[(q, v)]] for q in qs], float)
                    cell[f"rewrite: {a} - {b}"] = pair(hv_a, hv_b)
    for a in arms_new:
        out.setdefault("ORIG_all", {})[a] = boot1(np.array([hits[a][sq[(q, "ORIG")]] for q in origq], float))
PV = pd.DataFrame(PV_rows)
