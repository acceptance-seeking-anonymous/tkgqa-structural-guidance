"""build_parser_deliverables.py -- assemble multitq_kpct_parser_freeze.json, multitq_kpct_parser_dev_results.json and
multitq_kpct_parser_report.md from the harvested per-arm outputs (out/ dirs of the stage-B jobs). Local, no answers read."""
import json, os, glob, hashlib, time, csv, io

RULE_SHA = "0f5f760450356349a7bf353301520d76de3617f1dfec0b4e4b1430c97b1d6cd7"
ARMS = [("k01_s0", "k01_s0", "P_1"), ("k100", "k100", "P_100"), ("k10_s0", "k10_s0", "P_10"), ("k05_s0", "k05_s0", "P_5")]


def sha_recipe(o):
    d = {k: v for k, v in o.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(d, indent=1, sort_keys=True, default=str).encode()).hexdigest()


def find(name, dirs):
    for d in dirs:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def build(out_dirs, silver_manifest, sref, usage_log_rows, extra_notes):
    arms_f, arms_d = {}, {}
    for lst, sub, arm in ARMS:
        fz_p = find(f"freeze_P_{lst}.json", out_dirs)
        sel_p = find(f"select_P_{lst}.json", out_dirs)
        if fz_p is None or sel_p is None:
            arms_f[arm] = dict(status="NOT_RUN"); arms_d[arm] = dict(status="NOT_RUN"); continue
        fz = json.load(open(fz_p)); sel = json.load(open(sel_p))
        assert fz["sha256"] == sha_recipe(fz), f"{fz_p} sha mismatch"
        tr = json.load(open(find(f"P_{lst}_s0__train_stats.json", out_dirs)))
        ev = json.load(open(find(f"{lst}__exec_stats_P_{sub}_dev_sel2000.json", out_dirs)))
        ck, tn = fz["chosen"]["checkpoint"], fz["chosen"]["tau"]
        gh = sum(float(r["gpu_h"]) for r in usage_log_rows if r["tag"].endswith("_" + lst))
        sm = silver_manifest["lists"][lst]
        arms_f[arm] = dict(status="FROZEN" if fz.get("test") else "FROZEN_PRE_TEST", llm_list=lst, executor_subset=sub,
                           freeze_file=f"parser/freeze/freeze_P_{lst}.json", freeze_sha256=fz["sha256"], frozen_utc=fz["frozen_utc"],
                           chosen=fz["chosen"], adapter=fz["hashes_pre_test"]["adapter"], prompt_sha256=fz["prompt_sha256"],
                           silver=dict(n=sm["n_rows"], kept=sm["n_kept"], kept_fraction=sm["kept_fraction"], LOW_SILVER=sm["LOW_SILVER"]),
                           test=fz.get("test"), gpu_h=round(gh, 4),
                           training=dict(n_examples=tr["n_examples"], epochs=tr["epochs"], steps=tr["total_steps"], s_per_step=tr["s_per_step"],
                                         loss_last10=tr["loss_last10"], data_order_sha256=tr["data_order_sha256"], recipe_check_vs_G=tr["recipe_check_vs_G"]))
        c = sel["chosen"]
        ref = sref.get(lst, {})
        arms_d[arm] = dict(
            llm_list=lst, executor_subset=sub, chosen=dict(checkpoint=ck, tau=tn),
            hits1=c["hits1"], hits1_ci95=sel["dev_sel2000_hits1_ci95"], hits1_per_seed=c["hits1_per_seed"], hits10=c["hits10"],
            unseen=dict(n=sel["unseen_n"], hits1=sel["unseen_hits1"], definition="bundle template_unseen flag: masked template absent from the k% structural train list (k100: full train)"),
            seen=dict(n=sel["seen_n"], hits1=sel["seen_hits1"]),
            fallback_rate=c["fallback_rate"], invalid_rate=c["invalid_rate"], fallback_reasons=ev["adapters"][ck]["reasons"],
            STOP_DEV_FAILURE=sel["STOP_DEV_FAILURE"],
            grid=[dict(checkpoint=g["checkpoint"], tau=g["tau"], hits1=g["hits1"], hits10=g["hits10"], fallback_rate=g["fallback_rate"]) for g in sel["grid"]],
            S_k_reference=dict(source="multitq_kpct_structural_dev_results.json (structural track; not recomputed here)", **ref),
            delta_vs_S_k_dev=dict(hits1=(c["hits1"] - ref["S_hits1_mean"]) if ref.get("S_hits1_mean") is not None else None,
                                  unseen_hits1=(sel["unseen_hits1"] - ref["S_unseen_hits1"]) if ref.get("S_unseen_hits1") is not None else None,
                                  note="descriptive dev_sel2000 difference (DEV_NO_GAIN qualifier input); no test number"))
    not_run = {"P_1 subset seeds 1-2 (k01_s1, k01_s2)": "NOT_RUN (fallback 1 in force; amendment B1/B2)"}
    total_gh = round(sum(float(r["gpu_h"]) for r in usage_log_rows), 4)
    freeze = dict(name="multitq_kpct_parser_freeze", track="multitq_kpct / LLM parser (P_k)", rule_sha256=RULE_SHA,
                  subsets_manifest_sha256="de8a9c22d51f4182f88063c57d408530c351660ec23caea29b8559878ee88e5a",
                  amendments=["B1", "B2 (fallbacks 1, 2 in force)"],
                  arms=arms_f, not_run=not_run, P_0="not part of this run (training-free phase)",
                  gpu=dict(card="GPU 1", total_gpu_h=total_gh, usage_log_rows=usage_log_rows),
                  test_scored=False, notes=extra_notes, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    freeze["sha256"] = sha_recipe(freeze)
    dev = dict(name="multitq_kpct_parser_dev_results", split="dev_sel2000 (n = 2000; selection split)", rule_sha256=RULE_SHA,
               statistic="official multitq_eval.hit_at_k (sha256 1f85bbdc...); P_k hit = per-question hit averaged over the 3 S_k seeds; 95% CI Poisson(1) bootstrap B = 1000 seed 0",
               selection="checkpoint x tau in {off, -0.30, -0.10, -0.03}; best mean Hits@1; ties -> earlier checkpoint, then off",
               arms=arms_d, not_run=not_run, created_utc=freeze["created_utc"])
    dev["sha256"] = sha_recipe(dev)
    return freeze, dev
