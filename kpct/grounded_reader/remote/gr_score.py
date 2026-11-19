#!/usr/bin/env python3
"""gr_score.py -- grounded_reader track: the SINGLE registered scoring pass (official test split reused; R3 primary
secondary analysis; R1 P3 / R2 rewrites secondary). Runs only after the freeze file exists; every frozen prediction file
is sha256-verified against the freeze first (exit 3 on mismatch). Official multitq_eval.hit_at_k, test.json answers.

Method m = GR_k (the source selected on dev at level k); baseline b = frozen G_k (Reader on the structural bundle).
  (a) full test (n = 54,584): Hits@1 / Hits@10 of m and b, Poisson(1) bootstrap 95% CIs and the paired difference m - b
      (r2_common.summarise, B = 10,000, seed 20261118; two-sided CI; descriptive original-wording cost)
  (b) R3 PRIMARY (578 judge-accepted rewrites, synthetic quid 10q+4; ORIG = the same questions' full-test predictions):
      r3_common.method_tests over the selected levels (c_drop = drop_b - drop_m, c_rw = RW_m - RW_b; one-sided p;
      Holm within the method over levels x {c_drop, c_rw}; SUPPORTED iff Holm p < 0.05)
  (c) R1 P3 (569, 10q+3) and R2 RW (570, 10q+3): the same method_tests, Holm within the method per probe; SECONDARY
  (d) descriptive: m vs P_k (Parser, mean over its 3 S seeds) on R3: RW and drop contrasts (summarise)
Writes results json, per-question hits csv.gz, per-question R3 contrasts json (for the later R3 audit).
"""
import argparse, csv, gzip, hashlib, json, os, sys
import numpy as np
sys.path.insert(0, "/kpct/guided/code"); sys.path.insert(0, "/kpct/robustness_r3/code")
import multitq_eval as ME
import r2_common as R2C
import r3_common as R3

B, SEED = 10000, 20261118
GT = {"k01_s0": "ordS_ep2", "k05_s0": "main_ep2", "k100": "main_ep1"}
CK = {"k01_s0": "ep3", "k05_s0": "ep1", "k100": "ep0.5"}
LEV = {"k01_s0": "1", "k05_s0": "5", "k100": "100"}
GR = "/kpct/grounded_reader"
ROOT = {"r3": "/kpct/robustness_r3/llm", "r1": "/kpct/robustness", "r2": "/kpct/robustness_r2"}
VID = {"r3": 4, "r1": 3, "r2": 3}
FNAME = {"r3": "r3rw", "r1": "r1p3", "r2": "r2rw"}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_ranked(p):
    d = {}
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                x = json.loads(l); rk = x["ranked"]
                d[int(x["quid"])] = json.loads(rk.replace("'", '"')) if isinstance(rk, str) else rk
    return d


def hits(gold, P, quids, squid=lambda q: q, k=1):
    return np.array([ME.hit_at_k(gold[q], P.get(squid(q), []), k) for q in quids], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    FZ = json.load(open(a.freeze))
    bad = [p for p, s in FZ["files"].items() if sha256_file(p) != s]
    if bad:
        print("STOP_TEST_INTEGRITY", bad[:5]); sys.exit(3)
    sel = FZ["selection"]
    gold = {int(q["quid"]): q for q in json.load(open("/data/questions/test.json"))}
    tq = sorted(gold)
    assert len(tq) == 54584
    levels = [k for k in ("k01_s0", "k05_s0", "k100") if sel.get(k) in ("PALONE", "PK")]
    res = dict(freeze=a.freeze, freeze_sha256=sha256_file(a.freeze), selection=sel, levels=levels, test={}, probes={}, vs_P={})
    rows = []
    per_probe = {p: {} for p in ROOT}
    contr = {}
    for k in levels:
        m = sel[k]
        Mt = read_ranked(f"{GR}/final/{k}/{m}/test.jsonl.gz")
        Bt = read_ranked(f"/kpct/guided/preds/G_{k}/pred_G_test_{GT[k]}.jsonl")
        Pt = [read_ranked(f"/kpct/parser/test/{k}/pred_P_{k}_test_{CK[k]}_tauoff_seed{s}.jsonl") for s in range(3)]
        src = json.load(open(f"{GR}/final/{k}/{m}/test.src.json"))
        full = src["counts"]["missing"] == 0
        hm, hb = hits(gold, Mt, tq), hits(gold, Bt, tq)
        hm10, hb10 = hits(gold, Mt, tq, k=10), hits(gold, Bt, tq, k=10)
        res["test"][k] = dict(method=f"GR_{LEV[k]} ({m})", baseline=f"G_{LEV[k]}", n=len(tq), complete=full, assembly=src["counts"],
                              method_hits1=R2C.summarise(hm, B, SEED), baseline_hits1=R2C.summarise(hb, B, SEED),
                              method_hits10=float(hm10.mean()), baseline_hits10=float(hb10.mean()),
                              paired_hits1_m_minus_b=R2C.summarise(hm - hb, B, SEED),
                              label="reuse of the official MultiTQ test split" + ("" if full else "; INCOMPLETE assembly (missing rows count as misses) -> provisional"))
        for q, x, y, x10, y10 in zip(tq, hm, hb, hm10, hb10):
            rows.append(("test", q, q, f"GR_{LEV[k]}", int(x), int(x10))); rows.append(("test", q, q, f"G_{LEV[k]}", int(y), int(y10)))
        for pr in ROOT:
            Mr = read_ranked(f"{GR}/final/{k}/{m}/{FNAME[pr]}.jsonl.gz")
            Br = read_ranked(f"{ROOT[pr]}/G/{k}/pred_G_variants_G_{k}.jsonl")
            Pr = [read_ranked(f"{ROOT[pr]}/P/{k}/pred_P_{k}_variants_P_{k}_tauoff_seed{s}.jsonl") for s in range(3)]
            v = VID[pr]
            qs = sorted(q // 10 for q in Mr if q % 10 == v)
            sq = lambda q, v=v: 10 * q + v
            d = dict(o_m=hits(gold, Mt, qs), r_m=hits(gold, Mr, qs, sq), o_b=hits(gold, Bt, qs), r_b=hits(gold, Br, qs, sq))
            per_probe[pr][LEV[k]] = d
            oP = np.mean([hits(gold, P_, qs) for P_ in Pt], axis=0); rP = np.mean([hits(gold, P_, qs, sq) for P_ in Pr], axis=0)
            res["vs_P"].setdefault(pr, {})[LEV[k]] = dict(n=len(qs), P_orig=float(oP.mean()), P_rw=float(rP.mean()), P_drop=float((oP - rP).mean()),
                                                         rw_m_minus_P=R2C.summarise(d["r_m"] - rP, B, SEED),
                                                         dropP_minus_dropm=R2C.summarise((oP - rP) - (d["o_m"] - d["r_m"]), B, SEED),
                                                         label="descriptive (not registered as a test)")
            for q, om, rm, ob, rb in zip(qs, d["o_m"], d["r_m"], d["o_b"], d["r_b"]):
                rows.append((f"{pr}_orig", q, q, f"GR_{LEV[k]}", int(om), "")); rows.append((f"{pr}_rw", q, 10 * q + v, f"GR_{LEV[k]}", int(rm), ""))
                rows.append((f"{pr}_orig", q, q, f"G_{LEV[k]}", int(ob), "")); rows.append((f"{pr}_rw", q, 10 * q + v, f"G_{LEV[k]}", int(rb), ""))
            if pr == "r3":
                C = R3.contrasts(d["o_m"], d["r_m"], d["o_b"], d["r_b"])
                contr[LEV[k]] = dict(quids=qs, c_drop=C["c_drop"].tolist(), c_rw=C["c_rw"].tolist())
    for pr in ROOT:
        if per_probe[pr]:
            t = R3.method_tests(per_probe[pr], B, SEED)
            res["probes"][pr] = dict(role="PRIMARY secondary analysis (R3 rule)" if pr == "r3" else "SECONDARY (R1/R2; descriptive support)", tests=t)
    json.dump(res, open(a.out + "_results.json", "w"), indent=1)
    json.dump(dict(method="GR (Parser-grounded Reader)", baseline="G_k", per_level=contr), open(a.out + "_r3_contrasts.json", "w"))
    with gzip.open(a.out + "_per_question_hits.csv.gz", "wt", newline="") as f:
        w = csv.writer(f); w.writerow(["set", "quid", "squid", "arm", "hit1", "hit10"]); w.writerows(rows)
    print(json.dumps({k: (round(v["method_hits1"]["delta"], 4), round(v["baseline_hits1"]["delta"], 4), round(v["paired_hits1_m_minus_b"]["delta"], 4)) for k, v in res["test"].items()}))
    for pr, v in res["probes"].items():
        print(pr, json.dumps({lv: {cn: (round(t[cn]["delta"], 4), round(t[cn]["p_holm_one_sided"], 4), t[cn]["label"]) for cn in ("c_drop", "c_rw")} for lv, t in v["tests"].items()}))


if __name__ == "__main__":
    main()
