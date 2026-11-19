#!/usr/bin/env python3
"""ps_freeze.py -- parser LoRA-seed amendment (multitq_kpct_seed_amendment_P): pre-test and test freezes per arm.
pre_test: requires STOP_RECIPE false and STOP_DEV_FAILURE false; records the chosen checkpoint x tau, the adapter sha256
(train_stats on the 4090 == pk_infer re-hash at decode time) and sha256 of every input of the test pass.
test: re-verifies every pre-test hash, checks the test decode used the frozen adapter (sha256), and records the sha256
of the unscored test prediction files; also writes the combined multitq_kpct_seed_freeze_P.json. Stdlib only.
"""
import argparse, glob, hashlib, json, os, sys, time
K = os.path.expanduser("$WORK_DIR"); PS = os.path.join(K, "parser_seeds")
def sh(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def recipe_sha(d):
    return hashlib.sha256(json.dumps({k: v for k, v in d.items() if k != "sha256"}, indent=1, sort_keys=True, default=str).encode()).hexdigest()
def rel(p): return os.path.relpath(p, K)
TN2T = {"off": "off", "m030": "-0.30", "m010": "-0.10", "m003": "-0.03"}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["pre_test", "test"])
    ap.add_argument("--arm", required=True); ap.add_argument("--list", required=True); ap.add_argument("--sub", required=True)
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--map", required=True, help="ckpt=dev adapter name in the decode session, e.g. ep1=s1ep1,ep2=s1ep2")
    a = ap.parse_args()
    fz_path = os.path.join(PS, "freeze", f"freeze_{a.arm}.json"); os.makedirs(os.path.dirname(fz_path), exist_ok=True)
    M = dict(x.split("=") for x in a.map.split(","))
    if a.mode == "pre_test":
        assert not os.path.exists(fz_path), "pre-test freeze exists"
        T = os.path.join(PS, "train", a.arm)
        ts = json.load(open(os.path.join(T, "train_stats.json"))); rc = json.load(open(os.path.join(T, "recipe_check.json")))
        assert ts["seed"] == a.seed and not rc["STOP_RECIPE"], "STOP_RECIPE"
        sel_f = os.path.join(PS, "select", a.arm, f"select_P_{a.list}.json"); sel = json.load(open(sel_f))
        assert not sel["STOP_DEV_FAILURE"], "STOP_DEV_FAILURE"
        ck, tn = sel["chosen"]["checkpoint"], sel["chosen"]["tau"]
        inf_f = os.path.join(PS, "raw", a.arm, "infer_stats_P_dev_sel2000.json"); inf = json.load(open(inf_f))
        ad_sha = ts["adapters"][ck]; assert inf["adapters"][M[ck]]["adapter_sha256"] == ad_sha, "adapter sha mismatch train vs decode"
        H = {}
        for p in [os.path.join(PS, "amendment", "multitq_kpct_seed_amendment_P.json"), os.path.join(K, "parser", "silver", f"silver_{a.list}_train.jsonl"),
                  os.path.join(K, "parser", "code", "multitq_kpct_subsets.json"), os.path.join(T, "train_stats.json"), os.path.join(T, "recipe_check.json"), inf_f, sel_f,
                  os.path.join(PS, "select", a.arm, "selection_log.json"), os.path.join(PS, "exec", a.arm, f"exec_stats_P_{a.sub}_dev_sel2000.json")]:
            H[rel(p)] = sh(p)
        for c, n in M.items():
            p = os.path.join(PS, "raw", a.arm, f"raw_P_dev_sel2000_{n}.jsonl.gz"); H[rel(p)] = sh(p)
        for s in range(3):
            p = os.path.join(PS, "exec", a.arm, f"pred_P_{a.sub}_dev_sel2000_{ck}_tau{tn}_seed{s}.jsonl"); H[rel(p)] = sh(p)
            p = os.path.join(K, "preds", a.sub, f"test_S_seed{s}.jsonl"); H[rel(p)] = sh(p)
        for p in sorted(glob.glob(os.path.join(K, "executor", "models", a.sub, "*"))) + sorted(glob.glob(os.path.join(K, "executor", "*.py"))) \
                 + sorted(glob.glob(os.path.join(K, "executor", "struct", "*.py"))) + [os.path.join(K, "parser", "code", f) for f in ("pk_execute.py", "pk_select.py", "pk_common.py", "pk_infer.py", "run_cpu_p.sh")] \
                 + [os.path.join(PS, "code", f) for f in ("ps_freeze.py", "ps_dev.sh", "ps_test.sh", "ps_score.py") if os.path.exists(os.path.join(PS, "code", f))]:
            if os.path.isfile(p): H[rel(p)] = sh(p)
        H[rel(os.path.join(K, "bundles", a.sub, "test.jsonl"))] = sh(os.path.join(K, "bundles", a.sub, "test.jsonl"))
        fz = dict(name=f"freeze_{a.arm}", amendment="multitq_kpct_seed_amendment_P v1.0", arm=a.arm, llm_list=a.list, executor_subset=a.sub, lora_seed=a.seed,
                  chosen=dict(checkpoint=ck, tau=tn, tau_value=TN2T[tn]), adapter=dict(file=f"second-host scratch:/adapters/P_{a.list}_s{a.seed}/{ck}/adapter_model.safetensors", sha256=ad_sha,
                  all_checkpoints=ts["adapters"]), training=dict(steps=ts["total_steps"], s_per_step=ts["s_per_step"], train_s=ts["train_s"], peak_mem_gb=ts["peak_mem_gb"],
                  train_tok=ts["train_tok"], grad_ckpt=ts.get("grad_ckpt"), versions=ts["versions"], data_order_sha256=ts["data_order_sha256"], loss_last10=ts["loss_last10"]),
                  recipe_check=rc, dev_sel2000={k: sel[k] for k in ("dev_sel2000_hits1", "dev_sel2000_hits1_ci95", "unseen_n", "unseen_hits1", "seen_n", "seen_hits1", "STOP_DEV_FAILURE")} | dict(chosen=sel["chosen"]),
                  hashes_pre_test=H, frozen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        fz["sha256"] = recipe_sha(fz)
        json.dump(fz, open(fz_path, "w"), indent=1)
        print(json.dumps(dict(arm=a.arm, chosen=fz["chosen"], adapter_sha256=ad_sha, n_hashes=len(H), dev=fz["dev_sel2000"]["dev_sel2000_hits1"], sha256=fz["sha256"])))
    else:
        fz = json.load(open(fz_path)); assert "test" not in fz, "test freeze exists"
        bad = [p for p, h in fz["hashes_pre_test"].items() if sh(os.path.join(K, p)) != h]
        assert not bad, f"pre-test hash mismatch: {bad}"
        ck, tn = fz["chosen"]["checkpoint"], fz["chosen"]["tau"]
        R = os.path.join(PS, "raw", a.arm); inf = json.load(open(os.path.join(R, "infer_stats_P_test.json")))
        tname = [f"s{a.seed}{ck}"]; assert tname[0] in inf["adapters"], f"{tname[0]} not in test session"
        assert inf["adapters"][tname[0]]["adapter_sha256"] == fz["adapter"]["sha256"], "test decode adapter != frozen"
        rawf = os.path.join(R, inf["adapters"][tname[0]]["raw_file"])
        Tdir = os.path.join(PS, "test", a.arm); est = json.load(open(os.path.join(Tdir, f"exec_stats_P_{a.sub}_test.json")))
        preds = {rel(os.path.join(Tdir, f"pred_P_{a.sub}_test_{tname[0]}_tau{tn}_seed{s}.jsonl")): sh(os.path.join(Tdir, f"pred_P_{a.sub}_test_{tname[0]}_tau{tn}_seed{s}.jsonl")) for s in range(3)}
        for p in preds:
            n = sum(1 for _ in open(os.path.join(K, p))); assert n == 54584, f"{p}: {n} rows"
        ad = est["adapters"][tname[0]]
        fz["test"] = dict(n=54584, decode_adapter_name=tname[0], raw_file=rel(rawf), raw_sha256=sh(rawf), infer_stats_sha256=sh(os.path.join(R, "infer_stats_P_test.json")),
                          decode_seconds=inf["adapters"][tname[0]]["seconds"], prompts_per_s=inf["adapters"][tname[0]]["prompts_per_s"],
                          prompt_ids_sha256_first100=inf.get("prompt_ids_sha256_first100"), pred_files=preds,
                          exec_stats_sha256=sh(os.path.join(Tdir, f"exec_stats_P_{a.sub}_test.json")), invalid_rate=ad.get("invalid_rate"),
                          fallback_rate=ad["tau"][tn]["fallback_rate"], fallback_reasons=ad["tau"][tn].get("fallback_reasons"), scored=False,
                          written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), pre_test_hashes_reverified=len(fz["hashes_pre_test"]))
        fz["sha256"] = recipe_sha(fz)
        json.dump(fz, open(fz_path, "w"), indent=1)
        print(json.dumps(dict(arm=a.arm, test_pred_files=preds, invalid=fz["test"]["invalid_rate"], fallback=fz["test"]["fallback_rate"], sha256=fz["sha256"])))
    # combined freeze (every arm with a test block)
    allf = {}
    for f in sorted(glob.glob(os.path.join(PS, "freeze", "freeze_*.json"))):
        d = json.load(open(f)); allf[d["arm"]] = dict(freeze_file=rel(f), freeze_sha256=d["sha256"], frozen_utc=d["frozen_utc"], chosen=d["chosen"],
                                                       adapter_sha256=d["adapter"]["sha256"], test_pred_files=(d.get("test") or {}).get("pred_files"),
                                                       test_written_utc=(d.get("test") or {}).get("written_utc"))
    comb = dict(name="multitq_kpct_seed_freeze_P", amendment="multitq_kpct_seed_amendment_P v1.0", arms=allf, test_scored=False,
                updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    comb["sha256"] = recipe_sha(comb)
    json.dump(comb, open(os.path.join(PS, "multitq_kpct_seed_freeze_P.json"), "w"), indent=1)
if __name__ == "__main__":
    main()
