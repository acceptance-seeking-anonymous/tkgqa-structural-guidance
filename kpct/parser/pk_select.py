#!/usr/bin/env python3
"""pk_select.py -- STAGE B ONLY. Checkpoint x tau selection of P_k on dev_sel2000 (rule arms.P_k.selection).

Score = dev_sel2000 Hits@1 (official multitq_eval.hit_at_k, sha256 checked) averaged over the 3 S_k seeds
(P_k hit = per-question hit averaged over the 3 S seeds). Grid = every saved checkpoint (epoch ends; 0.5 epoch when
E_max = 1) x tau in {off, -0.30, -0.10, -0.03}. Choice = highest score; ties -> earlier checkpoint, then tau in the
order off, -0.30, -0.10, -0.03 (tie -> off). Every configuration's score is appended to the P section of the selection
log as it is computed. Reports for the chosen configuration: Hits@1 per seed and mean with a Poisson(1) bootstrap 95%
CI (B = 1000, seed 0), Hits@10, invalid / fallback rates, Hits@1 on dev_sel2000 questions whose template is unseen in
the k% list (bundle template_unseen flag) and on the seen complement, and the STOP_DEV_FAILURE check (best Hits@1 <
0.30 or invalid rate > 0.20). Reads answers ONLY for the 2,000 dev_sel2000 quids (asserted against the registered list).
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pk_common as PC

EXEC_DIR = os.environ.get("KPCT_EXECUTOR", "/kpct/executor")
SCORER_SHA = "1f85bbdcd623f1810e9df2910e042c24d6f5c269d278e82b30ef05387a1e2d69"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", required=True); ap.add_argument("--list", required=True, help="LLM train list of the arm")
    ap.add_argument("--exec_dir", required=True, help="pk_execute.py output dir for dev_sel2000")
    ap.add_argument("--exec_stats", required=True)
    ap.add_argument("--ckpts", required=True, help="adapter names in training order, e.g. ep1,ep2,ep3")
    ap.add_argument("--dev_json", default="/data/questions/dev.json")
    ap.add_argument("--manifest", default="/kpct/parser/code/multitq_kpct_subsets.json")
    ap.add_argument("--bundles", default=None, help="dev_sel2000 bundles (template_unseen flag)")
    ap.add_argument("--log", required=True, help="selection log JSON (appended)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(EXEC_DIR, "struct"))
    scorer = os.path.join(EXEC_DIR, "struct", "multitq_eval.py")
    assert PC.sha256_file(scorer) == SCORER_SHA, "scorer sha256 mismatch"
    import multitq_eval as ME
    sel = PC.load_list(a.manifest, "dev_sel2000")
    S = set(sel)
    Q = {int(q["quid"]): q for q in json.load(open(a.dev_json)) if int(q["quid"]) in S}
    assert set(Q) == S, "dev_sel2000 quids missing from dev.json"
    unseen = {int(r["quid"]): bool(r.get("template_unseen")) for r in PC.iter_jsonl(a.bundles or f"/kpct/bundles/{a.k}/dev_sel2000.jsonl")}
    assert set(unseen) == S
    st = json.load(open(a.exec_stats))
    ckpts = a.ckpts.split(",")
    log = json.load(open(a.log)) if os.path.exists(a.log) else {"P": []}
    log.setdefault("P", [])
    rows = []
    for ci, ck in enumerate(ckpts):
        for ti, t in enumerate(PC.TAUS):
            tn = "off" if t is None else f"{t:+.2f}".replace("+", "p").replace("-", "m").replace(".", "")
            H1, H10 = [], []
            for s in range(3):
                f = os.path.join(a.exec_dir, f"pred_P_{a.k}_dev_sel2000_{ck}_tau{tn}_seed{s}.jsonl")
                P = {int(r["quid"]): r["ranked"] for r in PC.iter_jsonl(f)}
                assert set(P) == S, f"{f}: {len(P)} rows"
                H1.append(np.array([ME.hit_at_k(Q[q], P[q], 1) for q in sel], float))
                H10.append(np.array([ME.hit_at_k(Q[q], P[q], 10) for q in sel], float))
            h1 = np.mean(H1, 0); h10 = np.mean(H10, 0)
            e = dict(arm=f"P_{a.list}", k=a.k, checkpoint=ck, tau=tn, hits1=float(h1.mean()), hits1_per_seed=[float(x.mean()) for x in H1],
                     hits10=float(h10.mean()), fallback_rate=st["adapters"][ck]["tau"][tn]["fallback_rate"],
                     invalid_rate=st["adapters"][ck]["invalid_rate"], utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            log["P"].append(e)
            json.dump(log, open(a.log + ".tmp", "w"), indent=1); os.replace(a.log + ".tmp", a.log)
            rows.append((-round(e["hits1"], 12), ci, ti, e, h1))
    rows.sort(key=lambda x: x[:3])
    best, h1 = rows[0][3], rows[0][4]
    rng = np.random.default_rng(0)
    W = rng.poisson(1.0, size=(1000, len(sel)))
    bs = (W * h1).sum(1) / np.maximum(W.sum(1), 1)
    un = np.array([unseen[q] for q in sel])
    res = dict(arm=f"P_{a.list}", k=a.k, chosen=best, dev_sel2000_hits1=best["hits1"],
               dev_sel2000_hits1_ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
               unseen_n=int(un.sum()), unseen_hits1=float(h1[un].mean()) if un.any() else None,
               seen_n=int((~un).sum()), seen_hits1=float(h1[~un].mean()) if (~un).any() else None,
               STOP_DEV_FAILURE=bool(best["hits1"] < 0.30 or best["invalid_rate"] > 0.20),
               grid=[r[3] for r in rows], scorer_sha256=SCORER_SHA)
    json.dump(res, open(os.path.join(a.out, f"select_P_{a.list}.json"), "w"), indent=1)
    print(json.dumps({k_: res[k_] for k_ in ("chosen", "dev_sel2000_hits1", "dev_sel2000_hits1_ci95", "unseen_hits1", "seen_hits1", "STOP_DEV_FAILURE")}))


if __name__ == "__main__":
    main()
