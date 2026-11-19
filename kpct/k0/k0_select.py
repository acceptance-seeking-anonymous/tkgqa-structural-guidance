#!/usr/bin/env python3
"""k0_select.py -- prompt-design selection of P_0 / G_0 on dev_sel2000 ONLY (rule arms.P_0.selection, arms.G_0.selection).

Every design's dev_sel2000 prediction file is scored with the official scorer (multitq_eval.py, sha256-checked), with a
Poisson(1) bootstrap 95% CI (B = 1000, seed 0, as the guided / parser tracks); each record is appended to the selection
log as it is computed. Choice: best Hits@1 (hit count); tie -> shortest prompt (P_0: registered head chars d1 < d2 < d3;
G_0: fixed-text chars g1 < g2 < g3). STOP_DEV_FAILURE flag: best Hits@1 < 0.30 or its invalid rate > 0.20 (the driver
stops before any test pass when it fires). Only dev.json answers of the 2,000 registered quids are read; no test file is
opened. --mock_questions: plumbing test with placeholder answers (no real dev answer read; scores 0 by construction).
"""
import argparse, fcntl, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gk_common as GC

EVAL_SHA = "1f85bbdcd623f1810e9df2910e042c24d6f5c269d278e82b30ef05387a1e2d69"


def log_append(rec, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")


def boot_ci(h, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.poisson(1.0, size=(B, len(h)))
    m = (w * h).sum(1) / np.maximum(w.sum(1), 1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["P0", "G0"])
    ap.add_argument("--cands", required=True, help="design=pred_file[,...]")
    ap.add_argument("--stats", required=True, help="P0: exec_stats_P0_dev_sel2000.json ; G0: post_stats_G0_dev_sel2000.json")
    ap.add_argument("--prompt_manifest", required=True, help="p0_prompts_manifest.json (chars) | g0_prompts_manifest.json (fixed_chars)")
    ap.add_argument("--dev", default="/data/questions/dev.json")
    ap.add_argument("--manifest", required=True); ap.add_argument("--eval", required=True)
    ap.add_argument("--log", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--mock_questions", default=None, help="plumbing test only: question rows with placeholder answers")
    a = ap.parse_args()
    assert GC.sha256_file(a.eval) == EVAL_SHA, "multitq_eval.py sha256 mismatch"
    sys.path.insert(0, os.path.dirname(os.path.abspath(a.eval)))
    import multitq_eval as ME
    if a.mock_questions:
        Q = sorted(json.load(open(a.mock_questions)), key=lambda q: q["quid"])
        assert all(q["answers"] == ["__PLACEHOLDER__"] for q in Q), "mock file must carry placeholder answers only"
        keep = set(q["quid"] for q in Q); split = "mock"
    else:
        keep = set(GC.load_list(a.manifest, "dev_sel2000"))
        Q = sorted([q for q in json.load(open(a.dev)) if q["quid"] in keep], key=lambda q: q["quid"])
        assert len(Q) == 2000; split = "dev_sel2000"
    st = json.load(open(a.stats))
    pm = json.load(open(a.prompt_manifest))
    chars = {d: v["chars" if a.arm == "P0" else "fixed_chars"] for d, v in pm["designs"].items()}
    res = []
    for item in a.cands.split(","):
        d, pred = item.split("=", 1)
        P = ME.load_predictions(pred)
        assert set(P) == keep, f"{d}: prediction quids != {split}"
        h1 = np.array([ME.hit_at_k(q, P[q["quid"]], 1) for q in Q]); h10 = np.array([ME.hit_at_k(q, P[q["quid"]], 10) for q in Q])
        sd = st["out"][d]
        rec = dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), track="k0", arm=a.arm.replace("0", "_0"), design=d,
                   pred_file=os.path.basename(pred), pred_sha256=GC.sha256_file(pred), hits1=float(h1.mean()), hits1_ci95=boot_ci(h1),
                   hits10=float(h10.mean()), n_hit1=int(h1.sum()), invalid_rate=float(sd["invalid_rate"]),
                   prompt_chars=int(chars[d]), n=len(Q), split=split,
                   per_qtype_hits1={qt: float(np.mean([h for h, q in zip(h1, Q) if q.get("qtype") == qt]))
                                    for qt in sorted({q.get("qtype") for q in Q}) if qt is not None})
        if a.arm == "P0":
            rec.update(reasons=sd.get("reasons"), empty_pool=sd.get("empty_pool"))
        else:
            rec.update(override_rate_vs_p0_top1=sd.get("override_rate_vs_p0_top1"))
        log_append(rec, a.log); res.append(rec)
        print(json.dumps({k: rec[k] for k in ("design", "hits1", "hits1_ci95", "hits10", "invalid_rate")}), flush=True)
    best = max(range(len(res)), key=lambda i: (res[i]["n_hit1"], -res[i]["prompt_chars"]))   # tie -> shortest prompt
    b = res[best]
    out = dict(arm=a.arm, split=split, selected=b["design"], selected_hits1=b["hits1"], selected_hits1_ci95=b["hits1_ci95"],
               selected_hits10=b["hits10"], selected_invalid_rate=b["invalid_rate"], candidates=res,
               rule="best dev_sel2000 Hits@1 (hit count); tie -> shortest prompt",
               DEV_FAILURE_trigger=bool(b["hits1"] < 0.30 or b["invalid_rate"] > 0.20),
               eval_sha256=EVAL_SHA, prompt_manifest_sha256=pm["sha256"], selected_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    json.dump(out, open(a.out, "w"), indent=1)
    log_append(dict(track="k0", decision=f"{a.arm}_design", selected=b["design"], selected_hits1=b["hits1"],
                    DEV_FAILURE_trigger=out["DEV_FAILURE_trigger"], split=split), a.log)
    print(json.dumps({k: out[k] for k in ("selected", "selected_hits1", "DEV_FAILURE_trigger")}))


if __name__ == "__main__":
    main()
