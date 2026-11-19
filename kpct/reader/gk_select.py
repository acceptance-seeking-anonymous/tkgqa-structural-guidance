#!/usr/bin/env python3
"""gk_select.py -- dev_sel2000 selection and arm freeze for G_k / GU_k (rule selection_discipline, test_discipline).

  select  score every checkpoint's dev_sel2000 prediction file with the official scorer (multitq_eval.py, sha256
          checked), Poisson(1) bootstrap CI (B = 1000, seed 0), invalid rate; best Hits@1 checkpoint, tie -> earlier;
          STOP_DEV_FAILURE flag (best < 0.30 or invalid > 0.20). Every candidate is appended to the selection log
          ($WORK_DIR/multitq_kpct_selection_log.jsonl, flock) as it is computed.
  order   G_1 candidate-order choice at k = 1 subset seed 0: the better of the two selected checkpoints
          (ordS vs ordrandom); |difference| < 0.0005 -> S order (amendment B1, fixed before any dev score).
  freeze  before test inference: sha256 of the selected adapter files, train config/stats, prompt hashes, bundle files
          -> <run>/freeze_<arm>.json. After test inference, `freeze --pred` adds the test prediction sha256 (never scored).
Only dev_sel2000 answers are read (dev.json restricted to the 2,000 registered quids). No test answer is ever read.
"""
import argparse, fcntl, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gk_common as GC

EVAL_SHA = "1f85bbdcd623f1810e9df2910e042c24d6f5c269d278e82b30ef05387a1e2d69"
LOG = os.path.expanduser("$WORK_DIR/multitq_kpct_selection_log.jsonl")


def log_append(rec, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")


def boot_ci(h, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.poisson(1.0, size=(B, len(h)))
    m = (w * h).sum(1) / np.maximum(w.sum(1), 1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def cmd_select(a):
    assert GC.sha256_file(a.eval) == EVAL_SHA, "multitq_eval.py sha256 mismatch"
    sys.path.insert(0, os.path.dirname(os.path.abspath(a.eval)))
    import multitq_eval as ME
    keep = set(GC.load_list(a.manifest, "dev_sel2000"))
    Q = sorted([q for q in json.load(open(a.dev)) if q["quid"] in keep], key=lambda q: q["quid"])
    assert len(Q) == 2000
    res = []
    for item in a.cands.split(","):
        name, pred = item.split("=", 1)
        P = ME.load_predictions(pred)
        assert set(P) == keep, f"{name}: prediction quids != dev_sel2000"
        h1 = np.array([ME.hit_at_k(q, P[q["quid"]], 1) for q in Q]); h10 = np.array([ME.hit_at_k(q, P[q["quid"]], 10) for q in Q])
        raw = pred.replace("/pred_", "/raw_").replace(".jsonl", ".jsonl.gz")
        inv = float(np.mean([not r["valid"] for r in GC.read_jsonl(raw)])) if os.path.exists(raw) else None
        rec = dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), track="guided", arm=a.arm, list=a.list, run=a.run,
                   candidate=name, pred_file=os.path.basename(pred), pred_sha256=GC.sha256_file(pred), hits1=float(h1.mean()),
                   hits1_ci95=boot_ci(h1), hits10=float(h10.mean()), invalid_rate=inv, n=len(Q), split="dev_sel2000")
        log_append(rec, a.log); res.append(rec)
        print(json.dumps(rec), flush=True)
    best = max(range(len(res)), key=lambda i: (round(res[i]["hits1"] * 2000), -i))      # tie -> earlier
    b = res[best]
    out = dict(arm=a.arm, list=a.list, run=a.run, selected=b["candidate"], selected_hits1=b["hits1"], candidates=res,
               DEV_FAILURE_trigger=bool(b["hits1"] < 0.30 or (b["invalid_rate"] or 0) > 0.20))
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("selected", "selected_hits1", "DEV_FAILURE_trigger")}))


def cmd_order(a):
    S, R = json.load(open(a.sel_S)), json.load(open(a.sel_random))
    d = R["selected_hits1"] - S["selected_hits1"]
    choice = "random" if d >= 0.0005 else "S"
    out = dict(rule="G candidate order chosen once at k = 1 subset seed 0 on dev_sel2000; |diff| < 0.0005 -> S order",
               S=S["selected_hits1"], random=R["selected_hits1"], diff=d, chosen=choice)
    json.dump(out, open(a.out, "w"), indent=1); log_append(dict(track="guided", decision="G_order", **out), a.log); print(json.dumps(out))


def cmd_freeze(a):
    fz = json.load(open(a.out)) if os.path.exists(a.out) else {}
    if a.pred:
        fz.setdefault("test_predictions", {})[os.path.basename(a.pred)] = GC.sha256_file(a.pred)
        if a.raw:
            fz["test_predictions"][os.path.basename(a.raw)] = GC.sha256_file(a.raw)
        fz["test_written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    else:
        ad = a.adapter
        fz.update(arm=a.arm, list=a.list, run=a.run, frozen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  adapter_dir=ad, adapter_sha256={f: GC.sha256_file(os.path.join(ad, f)) for f in sorted(os.listdir(ad)) if os.path.isfile(os.path.join(ad, f))},
                  selection=json.load(open(a.selection)) if a.selection else None, prompt_sha256=GC.prompt_texts_sha256(),
                  code_sha256={f: GC.sha256_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), f)) for f in
                               ("gk_common.py", "gk_train.py", "gk_infer.py", "gk_select.py", "gk_run_arm.py")},
                  inputs_sha256={os.path.basename(p): GC.sha256_file(p) for p in (a.inputs.split(",") if a.inputs else [])},
                  order=a.order, decoding=dict(greedy=True, max_new_tokens=GC.MAX_NEW_G))
    json.dump(fz, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in fz.items() if k not in ("selection",)})[:2000])


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("select"); s.add_argument("--cands", required=True, help="name=pred_file,...  (dev_sel2000)")
    s.add_argument("--dev", required=True); s.add_argument("--manifest", required=True); s.add_argument("--eval", required=True)
    s.add_argument("--arm", required=True); s.add_argument("--list", required=True); s.add_argument("--run", required=True)
    s.add_argument("--out", required=True); s.add_argument("--log", default=LOG)
    o = sp.add_parser("order"); o.add_argument("--sel_S", required=True); o.add_argument("--sel_random", required=True)
    o.add_argument("--out", required=True); o.add_argument("--log", default=LOG)
    f = sp.add_parser("freeze"); f.add_argument("--out", required=True); f.add_argument("--arm"); f.add_argument("--list")
    f.add_argument("--run"); f.add_argument("--adapter"); f.add_argument("--selection"); f.add_argument("--inputs", default="")
    f.add_argument("--order", default=None); f.add_argument("--pred", default=None); f.add_argument("--raw", default=None)
    a = ap.parse_args()
    {"select": cmd_select, "order": cmd_order, "freeze": cmd_freeze}[a.cmd](a)


if __name__ == "__main__":
    main()
