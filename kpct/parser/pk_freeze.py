#!/usr/bin/env python3
"""pk_freeze.py -- freeze record of one P_k arm (rule test_discipline).

--mode pre_test : written AFTER dev_sel2000 selection and BEFORE any test generation: sha256 of the chosen adapter,
                  its training stats, the silver set, the S_k / classifier / linker / padding models used at execution,
                  the prompt, every P-track and executor code file, the rule and the subset manifest, the chosen
                  checkpoint and tau and the dev_sel2000 selection summary.
--mode test     : re-verifies every pre-test hash (STOP_TEST_INTEGRITY) and adds the sha256 of the test raw generations
                  and of the 3 test prediction files (one per S_k seed). Test predictions are NOT scored here.
"""
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pk_common as PC

K = "/kpct"


def tau_name(t):
    return "off" if t is None else f"{t:+.2f}".replace("+", "p").replace("-", "m").replace(".", "")


def hashes(paths):
    return {os.path.relpath(p, K): PC.sha256_file(p) for p in sorted(paths) if os.path.isfile(p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["pre_test", "test"])
    ap.add_argument("--list", required=True); ap.add_argument("--subset", required=True)
    ap.add_argument("--silver_manifest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = f"{K}/parser"
    sel = json.load(open(f"{P}/select/select_P_{a.list}.json"))
    ck, tn = sel["chosen"]["checkpoint"], sel["chosen"]["tau"]
    adir = f"{P}/adapters/P_{a.list}_s0"
    pre = dict(
        adapter=hashes([f"{adir}/{ck}/adapter_model.safetensors", f"{adir}/{ck}/adapter_config.json", f"{adir}/train_stats.json"]),
        silver=hashes([a.silver_manifest, a.silver_manifest.replace("_manifest.json", "_train.jsonl")]),
        execution_models=hashes(glob.glob(f"{K}/executor/models/{a.subset}/*")),
        code=hashes(glob.glob(f"{P}/code/*.py") + glob.glob(f"{P}/code/*.sh") + [f"{K}/executor/kpct_executor.py", f"{K}/executor/kpct_core.py"]
                    + glob.glob(f"{K}/executor/struct/*.py")),
        dev_bundles=hashes([f"{K}/bundles/{a.subset}/dev_sel2000.jsonl"]),
        subsets_manifest=hashes([f"{P}/code/multitq_kpct_subsets.json"]))
    if a.mode == "pre_test":
        dev_st = json.load(open(f"{P}/exec/{a.list}/exec_stats_P_{a.subset}_dev_sel2000.json"))
        fz = dict(
            name=f"freeze_P_{a.list}", arm=f"P_{PC.k_of(a.list)}", llm_list=a.list, executor_subset=a.subset,
            rule_sha256="0f5f760450356349a7bf353301520d76de3617f1dfec0b4e4b1430c97b1d6cd7",
            subsets_manifest_sha256="de8a9c22d51f4182f88063c57d408530c351660ec23caea29b8559878ee88e5a",
            chosen=dict(checkpoint=ck, tau=tn), model=PC.MODEL_8B, prompt_sha256=PC.prompt_sha256(),
            decoding=dict(greedy=True, max_new_tokens=PC.MAX_NEW_P, engine="vLLM (tkgqa-vllm:latest), multi-LoRA, logprobs=0"),
            execution=dict(probs="classifier", s_seeds=[0, 1, 2], fallback="S_k own output (preds/<k>/<split>_S_seed<b>.jsonl)"),
            dev_sel2000=dict(hits1=sel["dev_sel2000_hits1"], hits1_ci95=sel["dev_sel2000_hits1_ci95"], hits1_per_seed=sel["chosen"]["hits1_per_seed"],
                             hits10=sel["chosen"]["hits10"], fallback_rate=sel["chosen"]["fallback_rate"], invalid_rate=sel["chosen"]["invalid_rate"],
                             unseen_n=sel["unseen_n"], unseen_hits1=sel["unseen_hits1"], STOP_DEV_FAILURE=sel["STOP_DEV_FAILURE"],
                             chosen_pred_files=dev_st["adapters"][ck]["tau"][tn]["files"]),
            hashes_pre_test=pre, frozen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), test=None)
    else:
        fz = json.load(open(a.out))
        assert fz["sha256"] == PC.sha_recipe(fz), "freeze file modified"
        bad = {g: [k for k in fz["hashes_pre_test"][g] if pre[g].get(k) != fz["hashes_pre_test"][g][k]] for g in fz["hashes_pre_test"]}
        bad = {g: v for g, v in bad.items() if v}
        assert not bad, f"STOP_TEST_INTEGRITY: pre-test hashes changed: {bad}"
        st = json.load(open(f"{P}/test/{a.list}/exec_stats_P_{a.subset}_test.json"))
        raw = f"{P}/raw/{a.list}/raw_P_test_{ck}.jsonl.gz"
        ist = json.load(open(f"{P}/raw/{a.list}/infer_stats_P_test.json"))
        assert ist["adapters"][ck]["adapter_sha256"] == fz["hashes_pre_test"]["adapter"][os.path.relpath(f"{adir}/{ck}/adapter_model.safetensors", K)], \
            "test generations were not made with the frozen adapter"
        files = st["adapters"][ck]["tau"][tn]["files"]
        for fn, h in files.items():
            assert PC.sha256_file(f"{P}/test/{a.list}/{fn}") == h
        fz["test"] = dict(n=st["n"], raw_file=os.path.relpath(raw, K), raw_sha256=PC.sha256_file(raw),
                          pred_files={f"parser/test/{a.list}/{fn}": h for fn, h in files.items()},
                          invalid_rate=st["adapters"][ck]["invalid_rate"], fallback_rate=st["adapters"][ck]["tau"][tn]["fallback_rate"],
                          fallback_reasons=st["adapters"][ck]["reasons"], scored=False,
                          written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    fz["sha256"] = PC.sha_recipe(fz)
    json.dump(fz, open(a.out + ".tmp", "w"), indent=1); os.replace(a.out + ".tmp", a.out)
    print(json.dumps({"mode": a.mode, "chosen": fz["chosen"], "sha256": fz["sha256"]}))


if __name__ == "__main__":
    main()
