#!/usr/bin/env python3
"""nm_infer.py -- greedy decoding of the normalizer N (rule multitq_normalizer_rule.json, section inference).
vLLM 0.10.2 (image vllm/vllm-openai:v0.10.2 on the second GPU host RTX 4090), Llama-3.1-8B-Instruct bf16 + LoRA (LoRARequest), one
engine session, the listed adapters decoded one after another over the same inputs; temperature 0, seed 0,
max 128 new tokens; prompt = nm_common.n_user(text) in the training chat template (fixed date_string).
Before decoding, the prompt token ids of the 50 probe inputs written by nm_train.py are recomputed and must be identical
(tokenizer check; exit 4 otherwise). Reads only item ids and texts. Output per adapter:
  raw_<name>.jsonl.gz rows {iid, out, finish_reason, n_out}; infer_stats.json (sha256, timings).
"""
import argparse, gzip, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nm_common as NC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True, help="jsonl {iid, text}")
    ap.add_argument("--adapters", required=True, help="name=path[,name=path...]")
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=NC.MAX_LEN)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.inputs import TokensPrompt
    import vllm, transformers
    os.makedirs(a.out, exist_ok=True)
    rows = [json.loads(l) for l in open(a.items) if l.strip()]
    assert len({r["iid"] for r in rows}) == len(rows)
    adapters = [x.split("=", 1) for x in a.adapters.split(",")]
    stats = dict(items=os.path.basename(a.items), items_sha256=NC.sha256_file(a.items), n=len(rows),
                 prompt_sha256=NC.prompt_sha256(), adapters={}, gpu_mem=a.gpu_mem, max_model_len=a.max_model_len)
    t0 = time.time()
    llm = LLM(model=NC.MODEL_8B, dtype="bfloat16", enable_lora=True, max_lora_rank=16, max_loras=1,
              max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem, seed=0)
    stats.update(init_s=round(time.time() - t0, 1), vllm=vllm.__version__, transformers=transformers.__version__)
    tok = llm.get_tokenizer()
    probe = json.load(open(a.probe))
    bad = [p["input"] for p in probe if NC.encode_prompt(tok, NC.n_user(p["input"])) != p["ids"]]
    stats["probe_check"] = dict(n=len(probe), mismatches=len(bad))
    if bad:
        json.dump(stats, open(os.path.join(a.out, "infer_stats.json"), "w"), indent=1)
        print("TOKENIZER_MISMATCH", len(bad), flush=True)
        sys.exit(4)
    stop = sorted({i for i in (tok.convert_tokens_to_ids(t) for t in ("<|eot_id|>", "<|end_of_text|>", "<|eom_id|>")) if i is not None and i >= 0})
    prompts = [NC.encode_prompt(tok, NC.n_user(r["text"])) for r in rows]
    stats["prompt_tok_mean"] = sum(len(p) for p in prompts) / max(1, len(prompts))
    sp = SamplingParams(temperature=0.0, max_tokens=NC.MAX_NEW, seed=0, stop_token_ids=stop)
    for li, (name, path) in enumerate(adapters):
        sa = os.path.join(path, "adapter_model.safetensors")
        stats["adapters"][name] = dict(path=path, sha256=NC.sha256_file(sa))
        t1 = time.time()
        res = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp, lora_request=LoRARequest(name, li + 1, path), use_tqdm=False)
        dt = time.time() - t1
        fp = os.path.join(a.out, f"raw_{name}.jsonl.gz")
        with gzip.open(fp + ".tmp", "wt", encoding="utf-8") as f:
            for r, o in zip(rows, res):
                g = o.outputs[0]
                f.write(json.dumps(dict(iid=r["iid"], out=g.text, finish_reason=g.finish_reason, n_out=len(g.token_ids)), ensure_ascii=False) + "\n")
        os.replace(fp + ".tmp", fp)
        stats["adapters"][name].update(gen_s=round(dt, 1), per_s=round(len(rows) / max(dt, 1e-9), 1), raw_sha256=NC.sha256_file(fp))
        print(name, len(rows), round(dt, 1), "s", flush=True)
        json.dump(stats, open(os.path.join(a.out, "infer_stats.json"), "w"), indent=1)
    stats["wall_s"] = round(time.time() - t0, 1)
    json.dump(stats, open(os.path.join(a.out, "infer_stats.json"), "w"), indent=1)
    os.system(f"chmod -R a+rX {a.out}")


if __name__ == "__main__":
    main()
