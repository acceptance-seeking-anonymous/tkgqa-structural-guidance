#!/usr/bin/env python3
"""k0_freeze.py -- freeze records of the k = 0 arms P_0 and G_0 (rule test_discipline).

--mode pre_test : written AFTER the dev_sel2000 design selection and BEFORE any test prompt of the arm: rule / subset
                  manifest / amendment references, model and decoding, the selected design and its registered sha256,
                  the dev selection summary, sha256 of every k0 code file, the P_0 parser-track code copies, the k0
                  executor (code + KG-statistics padding), the prompt manifests, the dev generations / executions /
                  predictions, and (G_0) the P_0 freeze and the dev + test k0 bundles.
--mode test     : re-verifies every pre-test hash (STOP_TEST_INTEGRITY: any mismatch -> exit 1, nothing added) and adds
                  the sha256 of the test generations and prediction files. Test predictions are NOT scored here.
"""
import argparse, glob, hashlib, json, os, sys, time

K = "/kpct"                                                  # relpath base of the recorded hashes
K0 = os.environ.get("KPCT_K0", f"{K}/k0")                    # arm outputs (the dry run uses a separate tree)
CODE = os.environ.get("KPCT_K0_CODE", f"{K}/k0/code")
EXE = os.environ.get("KPCT_EXECUTOR", f"{K}/executor")
RULE = dict(file="multitq_kpct_decision_rule.json",
            sha256="0f5f760450356349a7bf353301520d76de3617f1dfec0b4e4b1430c97b1d6cd7")
SUBSETS_SHA = "de8a9c22d51f4182f88063c57d408530c351660ec23caea29b8559878ee88e5a"
AMENDMENTS = {"B1": "multitq_kpct_amendment_B1.json", "B2": "multitq_kpct_amendment_B2.json"}
MODEL = "meta-llama/Llama-3.1-70B-Instruct"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hashes(paths):
    return {os.path.relpath(p, K): sha256_file(p) for p in sorted(set(paths)) if os.path.isfile(p)}


def code_hashes():
    c = CODE
    return hashes(glob.glob(f"{c}/*.py") + glob.glob(f"{c}/*.sh") + glob.glob(f"{c}/*.json") + glob.glob(f"{c}/parser/*.py")
                  + glob.glob(f"{c}/parser/*.json") + glob.glob(f"{c}/parser/p0_prompts/*") + glob.glob(f"{c}/g0_prompts/*")
                  + [f"{EXE}/kpct_executor.py", f"{EXE}/kpct_core.py", f"{EXE}/models/k0/FB.json"]
                  + glob.glob(f"{EXE}/struct/*.py"))


def pre_hashes(arm, sel):
    d = sel["selected"]
    h = dict(code=code_hashes())
    if arm == "P0":
        h["dev"] = hashes(glob.glob(f"{K0}/p0/raw/raw_P0_dev_sel2000_*.jsonl.gz") + glob.glob(f"{K0}/p0/exec/*_P0_dev_sel2000_*")
                          + [f"{K0}/p0/exec/exec_stats_P0_dev_sel2000.json", f"{K0}/select/select_P0.json"])
    else:
        h["p0_freeze"] = hashes([f"{K0}/freeze/freeze_P0.json"])
        h["bundles"] = hashes([f"{K0}/bundles/dev_sel2000.jsonl", f"{K0}/bundles/test.jsonl",
                               f"{K0}/bundles/bundle_stats_dev_sel2000.json", f"{K0}/bundles/bundle_stats_test.json"])
        h["dev"] = hashes(glob.glob(f"{K0}/g0/raw/raw_G0_dev_sel2000_*.jsonl.gz") + glob.glob(f"{K0}/g0/pred/*_G0_dev_sel2000_*")
                          + [f"{K0}/g0/pred/post_stats_G0_dev_sel2000.json", f"{K0}/select/select_G0.json"])
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["P0", "G0"])
    ap.add_argument("--mode", required=True, choices=["pre_test", "test"])
    ap.add_argument("--dry", action="store_true", help="plumbing dry run (mock answers): STOP_DEV_FAILURE not enforced, record marked dry")
    a = ap.parse_args()
    out = f"{K0}/freeze/freeze_{a.arm}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sel = json.load(open(f"{K0}/select/select_{a.arm}.json"))
    d = sel["selected"]
    if a.mode == "pre_test":
        assert a.dry or not sel["DEV_FAILURE_trigger"], "STOP_DEV_FAILURE fired: one logged bug-fix iteration (amendment) is required first"
        if a.arm == "P0":
            pm = json.load(open(f"{CODE}/parser/p0_prompts/p0_prompts_manifest.json"))
            ex = json.load(open(f"{CODE}/parser/p0_exemplars.json"))
            prompt = dict(manifest_sha256=pm["sha256"], design=d, design_sha256=pm["designs"][d]["sha256"],
                          design_file=pm["designs"][d]["file"], exemplars_sha256=ex["sha256"], exemplar_quids=ex["quids"],
                          decoding=dict(greedy=True, max_new_tokens=160, logprobs="sampled tokens (diagnostic)"))
            execution = dict(executor="k0 (operators + retrieval + RULE ordering; no S, classifier, sector aliases or block list)",
                             entities="exact KG name, else base linker (stemming, demonyms)", invalid="no prediction (miss)",
                             padding="KG statistics only (executor models/k0/FB.json)")
        else:
            gm = json.load(open(f"{CODE}/g0_prompts/g0_prompts_manifest.json"))
            p0f = json.load(open(f"{K0}/freeze/freeze_P0.json"))
            assert p0f.get("test_predictions"), "P_0 must be frozen with its test predictions before G_0 is frozen"
            prompt = dict(manifest_sha256=gm["sha256"], design=d, design_fixed_text_sha256=gm["designs"][d]["fixed_text_sha256"],
                          design_example_render_sha256=gm["designs"][d]["example_render_sha256"],
                          decoding=dict(greedy=True, max_new_tokens=48), zero_shot=True)
            execution = dict(bundle="k0 bundles from P_0 (design %s) parses: parse, top-20 RULE-ordered candidates with RULE ranks, "
                             "no S scores (s_rank slot = RULE rank), 20 provenance facts, operator trace" % p0f["chosen"]["design"],
                             ranked_list="[answer] + P_0 list (deduplicated); invalid answer stays at rank 1",
                             p0_freeze_sha256=sha256_file(f"{K0}/freeze/freeze_P0.json"))
        fz = dict(name=f"freeze_{a.arm}", arm=a.arm.replace("0", "_0"), k=0, rule=RULE, subsets_manifest_sha256=SUBSETS_SHA,
                  amendments=AMENDMENTS, model=dict(name=MODEL, dtype="bfloat16", engine="vLLM 0.10.2 (tkgqa-vllm:latest)",
                  tensor_parallel=2, disable_custom_all_reduce=True, enforce_eager=True, max_model_len=4096, prefix_caching=True, seed=0),
                  chosen=dict(design=d), prompt=prompt, execution=execution,
                  dev_sel2000=dict(hits1=sel["selected_hits1"], hits1_ci95=sel["selected_hits1_ci95"], hits10=sel["selected_hits10"],
                                   invalid_rate=sel["selected_invalid_rate"], STOP_DEV_FAILURE=sel["DEV_FAILURE_trigger"],
                                   all_designs={c["design"]: dict(hits1=c["hits1"], hits1_ci95=c["hits1_ci95"], hits10=c["hits10"],
                                                                  invalid_rate=c["invalid_rate"]) for c in sel["candidates"]}),
                  dry_run=bool(a.dry), hashes_pre_test=pre_hashes(a.arm, sel), frozen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), test_predictions=None)
        json.dump(fz, open(out, "w"), indent=1)
        print(json.dumps(dict(freeze=out, design=d, sha256=sha256_file(out))))
        return
    fz = json.load(open(out))
    now = pre_hashes(a.arm, sel)
    bad = [(grp, f) for grp, hs in fz["hashes_pre_test"].items() for f, h in hs.items() if now.get(grp, {}).get(f) != h]
    if bad:
        print(json.dumps(dict(STOP_TEST_INTEGRITY=True, mismatches=bad[:20])))
        sys.exit(1)
    if a.arm == "P0":
        files = [f"{K0}/p0/raw/raw_P0_test_{d}.jsonl.gz", f"{K0}/p0/exec/exec_P0_test_{d}.jsonl.gz", f"{K0}/p0/exec/pred_P0_test_{d}.jsonl"]
        st = json.load(open(f"{K0}/p0/exec/exec_stats_P0_test.json"))["out"][d]
        rates = dict(invalid_rate=st["invalid_rate"], reasons=st["reasons"], empty_pool=st["empty_pool"])
    else:
        files = [f"{K0}/g0/raw/raw_G0_test_{d}.jsonl.gz", f"{K0}/g0/pred/post_G0_test_{d}.jsonl.gz", f"{K0}/g0/pred/pred_G0_test_{d}.jsonl"]
        st = json.load(open(f"{K0}/g0/pred/post_stats_G0_test.json"))["out"][d]
        rates = dict(invalid_rate=st["invalid_rate"], override_rate_vs_p0_top1=st["override_rate_vs_p0_top1"])
    for f in files:
        assert os.path.isfile(f), f
    fz["test_predictions"] = {os.path.basename(f): sha256_file(f) for f in files}
    fz["test_prediction_file"] = os.path.basename(files[-1])
    fz["test_output_rates"] = rates
    fz["test_frozen_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    fz["test_scored"] = False
    json.dump(fz, open(out, "w"), indent=1)
    print(json.dumps(dict(freeze=out, test_predictions=fz["test_predictions"])))


if __name__ == "__main__":
    main()
