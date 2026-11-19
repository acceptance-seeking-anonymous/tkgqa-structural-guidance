#!/usr/bin/env python3
"""gr_infer.py -- grounded_reader track: the FROZEN Reader adapters G_k on Parser-grounded bundles, one vLLM session,
several items in the registered cut order, each guarded by the GPU-hour cap.

Decoding, prompt construction, validity and output format are the frozen guided-track code, imported unchanged:
  engine   = gk_infer.main's LLM(...) arguments (Llama-3.1-8B-Instruct, bf16, enable_lora, max_lora_rank 16, max_loras 1,
             max_model_len 2304, seed 0, CUDA graphs) with gpu_memory_utilization from the plan
  prompts  = gk_common.normalize_bundle + gk_common.fit_g_prompt(order 'S')   (2,048-token cap, trailing fact groups dropped)
  decoding = SamplingParams(temperature 0, max_tokens gk_common.MAX_NEW_G = 48, seed 0)
  answers  = gk_common.canon_answer (KG name / valid time, else invalid at rank 1)
  outputs  = gk_infer.write_outputs (raw_G_<split>_<adapter>.jsonl.gz, pred_G_<split>_<adapter>.jsonl; tail = the item's
             s_list = S seed-0 list of the bundle's pool)
Plan JSON: {adapters: {name: path}, gpu_mem, budget_s, sec_per_prompt, items: [{name, adapter, bundles, quids|null,
s_list, out, split}]}. Before each item: elapsed + n_prompts * sec_per_prompt + 20 s > budget_s -> NOT_RUN (recorded).
Items whose prediction file already exists are skipped (resume). Progress is written after every item.
Never reads an answer.
"""
import argparse, json, os, sys, time, types
sys.path.insert(0, os.environ.get("GK_CODE", "/kpct/guided/code"))
import numpy as np
import gk_common as GC          # frozen
import gk_infer as GI           # frozen (write_outputs)


def log(m):
    print(time.strftime("%H:%M:%S") + " " + m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--kg", default="/data/kg/full.txt")
    ap.add_argument("--stats", required=True)
    a = ap.parse_args()
    t_start = time.time()
    P = json.load(open(a.plan))
    time_cap = float(P["budget_s"]); spp = float(P["sec_per_prompt"])
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.inputs import TokensPrompt
    import vllm, transformers
    kg = GC.load_kg(a.kg)
    enorm = GC.entity_norm_map(sorted({f[0] for f in kg} | {f[2] for f in kg}))
    t0 = time.time()
    llm = LLM(model=GC.MODEL_8B, dtype="bfloat16", enable_lora=True, max_lora_rank=16, max_loras=1,
              max_model_len=int(P.get("max_model_len", 2304)), gpu_memory_utilization=float(P["gpu_mem"]), seed=0,
              enforce_eager=bool(P.get("eager", False)))
    t_init = time.time() - t0
    tok = llm.get_tokenizer()
    log(f"vllm {vllm.__version__} transformers {transformers.__version__} init {t_init:.0f}s time_cap {time_cap:.0f}s")
    lids = {n: i + 1 for i, n in enumerate(sorted(P["adapters"]))}
    stats = dict(plan=a.plan, init_s=round(t_init, 1), vllm=vllm.__version__, prompt_sha256=GC.prompt_texts_sha256(),
                 adapters={n: dict(path=p, adapter_sha256=GC.sha256_file(os.path.join(p, "adapter_model.safetensors")))
                           for n, p in P["adapters"].items()}, items=[])
    sp = SamplingParams(temperature=0.0, max_tokens=GC.MAX_NEW_G, seed=0)
    for it in P["items"]:
        aname = it["adapter"]
        out_pred = os.path.join(it["out"], f"pred_G_{it['split']}_{aname}.jsonl")
        rec = dict(name=it["name"], adapter=aname, split=it["split"], bundles=it["bundles"], quids=it.get("quids"))
        if os.path.exists(out_pred):
            rec.update(status="SKIPPED_EXISTS"); stats["items"].append(rec); continue
        keep = set(int(x) for x in json.load(open(it["quids"]))) if it.get("quids") else None
        bl = [GC.normalize_bundle(r) for r in GC.read_jsonl(it["bundles"])]
        bl = sorted([b for b in bl if keep is None or b["quid"] in keep], key=lambda b: b["quid"])
        if keep is not None:
            assert len(bl) == len(keep), f"{it['name']}: {len(bl)} bundles for {len(keep)} quids"
        assert not any(b["gold"] for b in bl), "dev/test bundles must not carry gold"
        el = time.time() - t_start
        proj = len(bl) * spp
        rec.update(n=len(bl), elapsed_before_s=round(el, 1), projected_s=round(proj, 1))
        if el + proj + 20 > time_cap:
            rec.update(status="NOT_RUN_BUDGET"); stats["items"].append(rec)
            log(f"NOT_RUN {it['name']} n={len(bl)} elapsed {el:.0f}s projected {proj:.0f}s time_cap {time_cap:.0f}s")
            json.dump(stats, open(a.stats, "w"), indent=1); continue
        os.makedirs(it["out"], exist_ok=True)
        if not bl:
            rec.update(status="EMPTY"); stats["items"].append(rec); continue
        Pp = [GC.fit_g_prompt(tok, b, order="S") for b in bl]
        prompts = [p[0] for p in Pp]
        lr = LoRARequest(aname, lids[aname], P["adapters"][aname])
        t1 = time.time()
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp, lora_request=lr, use_tqdm=False)
        dt = time.time() - t1
        st = dict(n=len(prompts), seconds=round(dt, 2), prompts_per_s=len(prompts) / dt, prompt_tok_mean=float(np.mean([len(p) for p in prompts])),
                  out_tok_mean=float(np.mean([len(o.outputs[0].token_ids) for o in outs])), n_truncated=int(sum(p[2] > 0 for p in Pp)))
        raw, preds, n_valid = [], [], 0
        for b, o in zip(bl, outs):
            txt = o.outputs[0].text
            ans, ok = GC.canon_answer(txt, enorm); n_valid += ok
            raw.append(dict(quid=b["quid"], text=txt, answer=ans, valid=bool(ok), n_out=len(o.outputs[0].token_ids)))
            preds.append((b["quid"], ans, None))
        st["invalid_rate"] = 1 - n_valid / max(1, len(bl))
        s_lists = GI.load_s_lists(it["s_list"], set(b["quid"] for b in bl))
        ns = types.SimpleNamespace(out=it["out"], tag="", split=it["split"])
        GI.write_outputs(ns, "G", aname, raw, preds, bl, s_lists, [], st, split=it["split"])
        rec.update(status="DONE", **st)
        stats["items"].append(rec)
        stats["wall_s"] = round(time.time() - t_start, 1)
        json.dump(stats, open(a.stats, "w"), indent=1)
        log(f"{it['name']} {aname} n={len(prompts)} {dt:.1f}s {st['prompts_per_s']:.2f}/s tok {st['prompt_tok_mean']:.0f} invalid {st['invalid_rate']:.4f}")
    stats["wall_s"] = round(time.time() - t_start, 1)
    json.dump(stats, open(a.stats, "w"), indent=1)


if __name__ == "__main__":
    main()
