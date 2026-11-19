#!/usr/bin/env python3
"""cs_freeze.py -- freeze record of the clean-silver P_1 follow-up (multitq_p1_clean_silver_rule.json), written BEFORE the
comparison is scored: sha256 of the rule, the clean silver set, every adapter checkpoint of P_1clean / P_1rerun / P_1order,
their training stats, the dev raw generations, every dev prediction file (3 S seeds x 4 taus x 9 checkpoints), the
selection outputs, the code, and the referenced frozen P_1 artefacts (re-verified against parser/freeze/freeze_P_k01_s0.json).
Dev only: no test file exists for any variant (asserted)."""
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pk_common as PC

K = "/kpct"; Q = f"{K}/parser_clean"; P = f"{K}/parser"
VARIANTS = ["clean", "rerun", "order"]


def h(paths):
    return {os.path.relpath(p, K): PC.sha256_file(p) for p in sorted(paths) if os.path.isfile(p)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    rule = json.load(open(f"{Q}/multitq_p1_clean_silver_rule.json"))
    assert rule["sha256"] == PC.sha_recipe(rule), "rule file modified"
    fz1 = json.load(open(f"{P}/freeze/freeze_P_k01_s0.json"))
    assert fz1["sha256"] == PC.sha_recipe(fz1)
    frozen_ok = {k: PC.sha256_file(f"{K}/{k}") == v for k, v in fz1["hashes_pre_test"]["adapter"].items()}
    dev_ok = {fn: PC.sha256_file(f"{P}/exec/k01_s0/{fn}") == v for fn, v in fz1["dev_sel2000"]["chosen_pred_files"].items()}
    assert all(frozen_ok.values()) and all(dev_ok.values()), f"STOP_INTEGRITY: frozen P_1 artefacts changed {frozen_ok} {dev_ok}"
    assert not glob.glob(f"{Q}/**/*test*.jsonl*", recursive=True), "a test file exists in parser_clean (STOP_NO_TEST)"
    est = json.load(open(f"{Q}/exec/exec_stats_P_k01_s0_dev_sel2000.json"))
    for ad in est["adapters"].values():
        for t in ad["tau"].values():
            for fn, hv in t["files"].items():
                assert PC.sha256_file(f"{Q}/exec/{fn}") == hv, fn
    ist = json.load(open(f"{Q}/raw/infer_stats_P_dev_sel2000.json"))
    variants = {}
    for v in VARIANTS:
        ad = f"{Q}/adapters/P_k01_s0{v}_s0"
        ts = json.load(open(f"{ad}/train_stats.json"))
        variants[v] = dict(
            adapter_dir=os.path.relpath(ad, K), adapters=h(glob.glob(f"{ad}/ep*/adapter_model.safetensors") + glob.glob(f"{ad}/ep*/adapter_config.json")),
            train_stats=h([f"{ad}/train_stats.json", f"{ad}/train_log.txt"]),
            training=dict(n_examples=ts["n_examples"], silver_file=ts["silver_file"], silver_sha256=ts["silver_sha256"], seed=ts["seed"],
                          order_seed=ts.get("order_seed"), epochs=ts["epochs"], total_steps=ts["total_steps"], data_order_sha256=ts["data_order_sha256"],
                          s_per_step=ts["s_per_step"], recipe_check_vs_G=ts["recipe_check_vs_G"], loss_last10=ts["loss_last10"]),
            raw=h(glob.glob(f"{Q}/raw/raw_P_dev_sel2000_{v}_ep*.jsonl.gz")),
            raw_adapter_sha256={n: x["adapter_sha256"] for n, x in ist["adapters"].items() if n.startswith(v + "_")},
            predictions={f"parser_clean/exec/{fn}": hv for n, x in est["adapters"].items() if n.startswith(v + "_")
                         for t in x["tau"].values() for fn, hv in t["files"].items()},
            selection=h([f"{Q}/select/{v}/select_P_k01_s0.json", f"{Q}/select/selection_log_{v}.json"]))
        for n, sha in variants[v]["raw_adapter_sha256"].items():
            assert sha == PC.sha256_file(f"{ad}/{n.split('_', 1)[1]}/adapter_model.safetensors"), f"decoded adapter {n} differs from the trained one"
    fz = dict(name="multitq_p1_clean_silver_freeze", rule_sha256=rule["sha256"], split="dev_sel2000 only (no test file exists)",
              clean_silver=h(glob.glob(f"{Q}/silver/silver_*_clean_*")), variants=variants,
              exec_stats=h([f"{Q}/exec/exec_stats_P_k01_s0_dev_sel2000.json"]), infer_stats=h([f"{Q}/raw/infer_stats_P_dev_sel2000.json"]),
              frozen_P1_reference=dict(freeze_file="parser/freeze/freeze_P_k01_s0.json", freeze_sha256=fz1["sha256"],
                                       adapter=fz1["hashes_pre_test"]["adapter"], dev_pred_files=fz1["dev_sel2000"]["chosen_pred_files"],
                                       reverified=True),
              code=h(glob.glob(f"{Q}/code/*.py") + glob.glob(f"{Q}/code/*.sh")),
              frozen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    fz["sha256"] = PC.sha_recipe(fz)
    json.dump(fz, open(a.out, "w"), indent=1)
    print(json.dumps({"sha256": fz["sha256"], "n_pred_files": sum(len(v["predictions"]) for v in variants.values())}))


if __name__ == "__main__":
    main()
