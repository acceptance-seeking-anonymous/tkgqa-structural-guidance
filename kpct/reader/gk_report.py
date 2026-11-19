#!/usr/bin/env python3
"""gk_report.py -- build multitq_kpct_guided_freeze.json, multitq_kpct_guided_dev_results.json and the markdown body
of multitq_kpct_guided_report.md from the harvested guided-track outputs (freeze_*, selection_*, session_*,
control_*, train_stats_*, checks, selection log, GPU usage log). Reads no test answer; dev numbers are dev_sel2000 only."""
import argparse, glob, json, os, time, hashlib

ARMS = [("G", "k01_s0"), ("GU", "k01_s0"), ("G", "k100"), ("GU", "k100"), ("G", "k10_s0"), ("GU", "k10_s0"), ("G", "k05_s0"),
        ("G", "k01_s1"), ("G", "k01_s2")]


def J(p):
    return json.load(open(p))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--meta", required=True, help="json with rule / amendment / code / bundle identifiers"); a = ap.parse_args()
    D = a.dir; meta = J(a.meta)
    led = [l.rstrip("\n").split("\t") for l in open(os.path.join(D, "gpu_usage_snapshot.tsv")) if l.strip()]
    hdr = led[0]; led = [dict(zip(hdr, r)) for r in led[1:]]
    g = [r for r in led if r["track"] == "guided"]
    sel_log = [json.loads(l) for l in open(os.path.join(D, "multitq_kpct_guided_selection_log.jsonl"))] if os.path.exists(os.path.join(D, "multitq_kpct_guided_selection_log.jsonl")) else []
    freeze, dev = dict(arms={}, controls={}), dict(arms={}, controls={})
    for arm, lst in ARMS:
        fz = os.path.join(D, f"freeze_{arm}_{lst}.json")
        if not os.path.exists(fz):
            freeze["arms"][f"{arm}_{lst}"] = dict(status="NOT_AVAILABLE"); continue
        F = J(fz)
        S = J(os.path.join(D, f"session_{arm}_{lst}.json")) if os.path.exists(os.path.join(D, f"session_{arm}_{lst}.json")) else {}
        runs = sorted(glob.glob(os.path.join(D, f"selection_{arm}_{lst}_*.json")))
        sels = {os.path.basename(p)[len(f"selection_{arm}_{lst}_"):-5]: J(p) for p in runs}
        ts = {}
        for p in glob.glob(os.path.join(D, f"train_stats_{arm}_{lst}_*.json")):
            t = J(p); ts[os.path.basename(p)[12:-5]] = {k: t.get(k) for k in ("n", "epochs", "steps_run", "len_mean", "target_in_source", "train_s",
                                                                                 "real_tok_per_s", "loss_last10", "list_sha256", "order", "prompt_sha256")}
        tags = [r for r in g if r["tag"] in (f"train_{arm}_{lst}", f"train_{arm}_{lst}_ordS", f"train_{arm}_{lst}_ordrandom", f"sess_{arm}_{lst}")]
        gen = S.get("generations", [])
        test_gen = [x for x in gen if x["split"] == "test" and x["style"] == arm]
        if not test_gen:                                   # session stopped after generating; completed from shards on CPU
            fin = sorted(glob.glob(os.path.join(D, f"finish_{arm}_test_*.json")))
            fin = [J(p) for p in fin if os.path.exists(os.path.join(D, f"freeze_{arm}_{lst}.json")) and
                   os.path.basename(p)[len(f"finish_{arm}_test_"):-5] in os.path.basename(F.get("adapter_dir", "")) + "_" or True]
            fin = [x for x in fin if x.get("pred_file") in (F.get("test_predictions") or {})]
            test_gen = [dict(invalid_rate=x["invalid_rate"], n=x["n"], finished_from_shards=True) for x in fin]
        freeze["arms"][f"{arm}_{lst}"] = dict(
            arm=arm, list=lst, run=F.get("run"), order=F.get("order"), selected_checkpoint=os.path.basename(F.get("adapter_dir", "")),
            adapter_dir_on_host=F.get("adapter_dir", "").replace("/kpct/", "$WORK_DIR/"), adapter_sha256=F.get("adapter_sha256"),
            frozen_utc=F.get("frozen_utc"), test_written_utc=F.get("test_written_utc"),
            test_predictions={k: v for k, v in (F.get("test_predictions") or {}).items() if k.startswith(f"pred_{arm}_") or k.startswith(f"raw_{arm}_")},
            prompt_sha256=F.get("prompt_sha256"), code_sha256=F.get("code_sha256"), bundle_inputs_sha256=F.get("inputs_sha256"),
            decoding=F.get("decoding"), DEV_FAILURE_trigger=F.get("DEV_FAILURE_trigger"),
            test_invalid_rate=test_gen[0]["invalid_rate"] if test_gen else None, test_n=test_gen[0]["n"] if test_gen else None,
            gpu_h=round(sum(float(r["gpu_h"]) for r in tags), 4), usage_log_rows=[(r["tag"], r["cards"], r["seconds"], r["gpu_h"], r["rc"]) for r in tags])
        dev["arms"][f"{arm}_{lst}"] = dict(selected=F.get("run") + "/" + os.path.basename(F.get("adapter_dir", "")),
                                           selections={k: dict(selected=v["selected"], selected_hits1=v["selected_hits1"], DEV_FAILURE_trigger=v["DEV_FAILURE_trigger"],
                                                               checkpoints=[dict(ckpt=c["candidate"], hits1=c["hits1"], hits1_ci95=c["hits1_ci95"], hits10=c["hits10"],
                                                                                 invalid_rate=c["invalid_rate"], pred_sha256=c["pred_sha256"]) for c in v["candidates"]])
                                                       for k, v in sels.items()}, training=ts)
        sel_main = sels.get(F.get("run"))
        dev["arms"][f"{arm}_{lst}"]["dev_sel2000_hits1"] = sel_main["selected_hits1"] if sel_main else None
        if sel_main:
            c = [x for x in sel_main["candidates"] if x["candidate"] == sel_main["selected"]][0]
            dev["arms"][f"{arm}_{lst}"].update(dev_sel2000_hits1_ci95=c["hits1_ci95"], dev_sel2000_hits10=c["hits10"], dev_sel2000_invalid=c["invalid_rate"])
        if arm == "G" and lst == "k01_s0":
            od = os.path.join(D, "order_decision_k01_s0.json")
            if os.path.exists(od):
                dev["order_decision"] = J(od)
            for ctl in ("SHUF", "QONLY"):
                cp = os.path.join(D, f"control_{ctl}_k01_s0_dev.json")
                tp = {k: v for k, v in (F.get("test_predictions") or {}).items() if k.startswith(f"pred_{ctl}_") or k.startswith(f"raw_{ctl}_")}
                cg = [x for x in gen if x["style"] == ctl]
                freeze["controls"][f"{ctl}_G1"] = dict(adapter="G_1 frozen adapter", test_predictions=tp,
                                                        invalid_rate={x["split"]: x["invalid_rate"] for x in cg})
                if os.path.exists(cp):
                    C = J(cp); c0 = C["candidates"][0]
                    dev["controls"][f"{ctl}_G1"] = dict(dev_sel2000_hits1=c0["hits1"], hits1_ci95=c0["hits1_ci95"], hits10=c0["hits10"], invalid_rate=c0["invalid_rate"],
                                                        note="qualifier only, never decision-relevant; not used for any selection")
    checks = {}
    for p in glob.glob(os.path.join(D, "check_*.json")):
        c = J(p); checks[c["subset"]] = dict(status=c["status"], oof=c.get("oof", {}).get("status"), files={k: (v["n"], v["sha256"]) for k, v in c["files"].items()})
    xc = os.path.join(D, "crosscheck_k100.json")
    freeze.update(track="multitq_kpct / guided reasoner", built_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), meta=meta,
                  bundle_checks=checks, k100_crosscheck=J(xc) if os.path.exists(xc) else None,
                  gpu_h_guided_total=round(sum(float(r["gpu_h"]) for r in g), 4), gpu_usage_rows=len(g),
                  test_discipline="test predictions written once per frozen arm, never scored by this track; freeze record written before the first test prompt")
    dev.update(split="dev_sel2000 (n = 2,000)", selection_log_rows=len(sel_log))
    for o, name in ((freeze, "multitq_kpct_guided_freeze.json"), (dev, "multitq_kpct_guided_dev_results.json")):
        body = json.dumps(o, indent=1, sort_keys=True, default=str)
        o["sha256"] = hashlib.sha256(body.encode()).hexdigest()
        json.dump(o, open(os.path.join(a.out, name), "w"), indent=1, default=str)
    print(json.dumps({k: (v.get("dev_sel2000_hits1"), v.get("selected")) for k, v in dev["arms"].items()}, indent=1))


if __name__ == "__main__":
    main()
