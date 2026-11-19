#!/usr/bin/env python3
"""pk_infer.py -- parse inference of P_k adapters (rule arms.P_k.decoding): greedy, max 160 new tokens, one user turn
with the training chat template (fixed date_string); token log-probs of the sampled tokens (logprobs=0) for the tau
fallback. Several adapters (the epoch checkpoints of one run) are decoded in ONE vLLM engine session (multi-LoRA, one
adapter at a time). Reads only quid + question text of the split (from the structural bundles, which carry no gold);
never reads a dev/test answer. Output: raw_P_<split>_<adapter><tag>.jsonl.gz rows
  {quid, text, n_out, mean_logprob (stop tokens excluded), min_logprob, finish_reason}
and infer_stats_P_<split><tag>.json (sha256 of every raw file, adapter sha256, throughput).

--backend hf --device cpu : CPU I/O dry run with transformers generate (same prompts, same greedy decoding,
log-probs from compute_transition_scores); the vLLM engine path is only exercised on the GPU.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pk_common as PC


def log(m):
    print(time.strftime("%H:%M:%S") + " " + m, flush=True)


def load_questions(a):
    keep = set(int(x) for x in json.load(open(a.quids))) if a.quids else None
    rows = []
    for r in PC.iter_jsonl(a.bundles):
        q = int(r["quid"])
        if keep is not None and q not in keep:
            continue
        assert not r.get("gold"), "dev/test bundles must not carry gold"
        rows.append((q, r["question"]))
    rows.sort()
    if keep is not None:
        assert len(rows) == len(keep), f"{len(rows)} bundle rows for {len(keep)} quids"
    if a.limit:
        rows = rows[:a.limit]
    return rows


def stop_ids_of(tok):
    ids = set()
    for t in ("<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"):
        i = tok.convert_tokens_to_ids(t)
        if i is not None and i >= 0:
            ids.add(int(i))
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    return ids


def run_vllm(a, rows, adapters, stats):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.inputs import TokensPrompt
    import vllm, transformers
    t0 = time.time()
    llm = LLM(model=a.model, dtype="bfloat16", enable_lora=True, max_lora_rank=16, max_loras=1,
              max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem, seed=0, enforce_eager=a.eager)
    stats.update(init_s=round(time.time() - t0, 1), vllm=vllm.__version__, transformers=transformers.__version__)
    tok = llm.get_tokenizer()
    stop = stop_ids_of(tok)
    prompts = [PC.encode_prompt(tok, PC.p_user(qt)) for _, qt in rows]
    stats["prompt_ids_sha256_first100"] = PC.sha256_text(json.dumps(prompts[:100]))
    stats["prompt_tok_mean"] = float(np.mean([len(p) for p in prompts]))
    sp = SamplingParams(temperature=0.0, max_tokens=PC.MAX_NEW_P, seed=0, logprobs=0)
    out = {}
    for li, (aname, apath) in enumerate(adapters):
        lr = LoRARequest(aname, li + 1, apath) if apath != "base" else None
        t1 = time.time()
        res = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp, lora_request=lr, use_tqdm=False)
        dt = time.time() - t1
        raw = []
        for (q, _), o in zip(rows, res):
            g = o.outputs[0]
            ids = list(g.token_ids)
            lps = [(d.get(t).logprob if d is not None and d.get(t) is not None else None) for t, d in zip(ids, g.logprobs or [None] * len(ids))]
            raw.append(dict(quid=q, text=g.text, n_out=len(ids), mean_logprob=PC.mean_logprob(ids, lps, stop),
                            min_logprob=min([x for t, x in zip(ids, lps) if x is not None and t not in stop] or [float("-inf")]),
                            finish_reason=g.finish_reason))
        out[aname] = (raw, dt)
    return out


def run_hf(a, rows, adapters, stats):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    tok = AutoTokenizer.from_pretrained(a.model)
    stop = stop_ids_of(tok)
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev)
    stats.update(init_s=round(time.time() - t0, 1), backend="hf")
    prompts = [PC.encode_prompt(tok, PC.p_user(qt)) for _, qt in rows]
    stats["prompt_ids_sha256_first100"] = PC.sha256_text(json.dumps(prompts[:100]))
    stats["prompt_tok_mean"] = float(np.mean([len(p) for p in prompts]))
    pad = tok.convert_tokens_to_ids("<|finetune_right_pad_id|>")
    out = {}
    for aname, apath in adapters:
        model = PeftModel.from_pretrained(base, apath).to(dev) if apath != "base" else base
        model.eval()
        t1 = time.time(); raw = []
        for b0 in range(0, len(prompts), a.hf_batch):
            P = prompts[b0:b0 + a.hf_batch]
            ml = max(len(p) for p in P)
            ids = torch.full((len(P), ml), pad, dtype=torch.long); att = torch.zeros((len(P), ml), dtype=torch.long)
            for i, p in enumerate(P):                       # left padding for generation
                ids[i, ml - len(p):] = torch.tensor(p); att[i, ml - len(p):] = 1
            with torch.no_grad():
                g = model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), do_sample=False, max_new_tokens=PC.MAX_NEW_P,
                                   output_scores=True, return_dict_in_generate=True, pad_token_id=pad, eos_token_id=sorted(stop))
            sc = model.compute_transition_scores(g.sequences, g.scores, normalize_logits=True).float().cpu().numpy()
            for i in range(len(P)):
                gen = g.sequences[i, ml:].tolist()
                lps = sc[i].tolist()
                n = len(gen)
                for j, t in enumerate(gen):                 # cut at the first stop token (kept, as vLLM does)
                    if t in stop:
                        n = j + 1; break
                gen, lps = gen[:n], lps[:n]
                gen = [t for t in gen if t != pad]
                txt = tok.decode([t for t in gen if t not in stop], skip_special_tokens=True)
                q = rows[b0 + i][0]
                raw.append(dict(quid=q, text=txt, n_out=len(gen), mean_logprob=PC.mean_logprob(gen, lps, stop),
                                min_logprob=min([x for t, x in zip(gen, lps) if t not in stop] or [float("-inf")]),
                                finish_reason=("stop" if gen and gen[-1] in stop else "length")))
        out[aname] = (raw, time.time() - t1)
        if apath != "base":
            base = model.unload() if hasattr(model, "unload") else base
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="dev_sel2000 | test | train_dry")
    ap.add_argument("--bundles", required=True, help="bundles/<subset>/<split>.jsonl (quid + question read)")
    ap.add_argument("--quids", default=None, help="json list restricting the split")
    ap.add_argument("--adapters", required=True, help="name=path[,name=path...]  (path 'base' = no adapter)")
    ap.add_argument("--out", required=True); ap.add_argument("--tag", default="")
    ap.add_argument("--model", default=PC.MODEL_8B)
    ap.add_argument("--max_model_len", type=int, default=1024); ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--hf_batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--allow_gold", action="store_true", help="dry run on TRAIN bundles only (gold field present, never read)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()
    if a.allow_gold:
        assert a.split.startswith("train"), "--allow_gold only for train dry runs"
        rows = sorted((int(r["quid"]), r["question"]) for r in PC.iter_jsonl(a.bundles))[:a.limit or None]
    else:
        rows = load_questions(a)
    adapters = [x.split("=", 1) for x in a.adapters.split(",")]
    stats = dict(arm="P", split=a.split, n=len(rows), model=a.model, max_new_tokens=PC.MAX_NEW_P, decoding="greedy",
                 prompt_sha256=PC.prompt_sha256(), adapters={}, backend=a.backend)
    res = run_vllm(a, rows, adapters, stats) if a.backend == "vllm" else run_hf(a, rows, adapters, stats)
    for aname, apath in adapters:
        raw, dt = res[aname]
        f = os.path.join(a.out, f"raw_P_{a.split}_{aname}{a.tag}.jsonl.gz")
        PC.write_jsonl(f, raw)
        af = os.path.join(apath, "adapter_model.safetensors")
        stats["adapters"][aname] = dict(path=apath, adapter_sha256=PC.sha256_file(af) if os.path.exists(af) else None,
                                        raw_file=os.path.basename(f), raw_sha256=PC.sha256_file(f), seconds=round(dt, 2),
                                        prompts_per_s=len(raw) / max(dt, 1e-9), out_tok_mean=float(np.mean([r["n_out"] for r in raw])),
                                        finish_length_rate=float(np.mean([r["finish_reason"] == "length" for r in raw])))
        log(f"{aname} n={len(raw)} {dt:.1f}s out_tok {stats['adapters'][aname]['out_tok_mean']:.1f}")
    stats["wall_s"] = round(time.time() - t_start, 1)
    json.dump(stats, open(os.path.join(a.out, f"infer_stats_P_{a.split}{a.tag}.json"), "w"), indent=1)
    os.system(f"chmod -R a+rX {a.out}")


if __name__ == "__main__":
    main()
