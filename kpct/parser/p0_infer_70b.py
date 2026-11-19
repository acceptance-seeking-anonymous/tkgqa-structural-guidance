#!/usr/bin/env python3
"""p0_infer_70b.py -- P_0 parse inference: Llama-3.1-70B-Instruct, bf16, vLLM tensor-parallel 2 on BOTH cards
(rule arms.P_0; training-free phase).

Recipe (verified on the GPU host): LLM(tensor_parallel_size=2, dtype=bfloat16, gpu_memory_utilization=0.90,
max_model_len=4096, enforce_eager=True); docker --ipc=host --shm-size=16g, VLLM_WORKER_MULTIPROC_METHOD=spawn,
NCCL_P2P_DISABLE=1; this file must be a real file (spawn re-exec). Prefix caching on (the head is shared).
Greedy, max 160 new tokens, logprobs of the sampled tokens recorded (diagnostic only; P_0 has no tau).
Every requested design (p0_prompt_<d>.txt, sha256-checked against p0_prompts_manifest.json) is decoded in ONE
engine session. Reads only quid + question of the split; answers are dropped on load and never used.
Output: raw_P0_<split>_<design>.jsonl.gz {quid, text, n_out, mean_logprob, finish_reason} + infer_stats_P0_<split>.json.
--dry_prompts: tokenizer-only prompt build (CPU; no engine), writes prompt token statistics.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pk_common as PC
from p0_prompts import p0_user


def log(m):
    print(time.strftime("%H:%M:%S") + " " + m, flush=True)


def load_questions(path, quids=None, limit=0):
    keep = set(int(x) for x in quids) if quids is not None else None
    Q = [(int(q["quid"]), q["question"]) for q in json.load(open(path)) if keep is None or int(q["quid"]) in keep]
    Q.sort()
    if keep is not None:
        assert len(Q) == len(keep), f"{len(Q)} questions for {len(keep)} quids"
    return Q[:limit] if limit else Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="dev_sel2000 | test | train_dry")
    ap.add_argument("--questions", required=True, help="questions json of the split (quid + question read)")
    ap.add_argument("--quids_list", default=None, help="registered list name restricting the split (dev_sel2000)")
    ap.add_argument("--quids", default=None, help="json file with a quid list (dry runs)")
    ap.add_argument("--manifest", default="/kpct/parser/code/multitq_kpct_subsets.json")
    ap.add_argument("--prompts", required=True, help="dir with p0_prompt_<d>.txt + p0_prompts_manifest.json")
    ap.add_argument("--designs", default="d1,d2,d3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=PC.MODEL_70B)
    ap.add_argument("--tp", type=int, default=2); ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    ap.add_argument("--dry_prompts", action="store_true"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()
    pm = json.load(open(os.path.join(a.prompts, "p0_prompts_manifest.json")))
    assert pm["sha256"] == PC.sha_recipe(pm), "prompt manifest sha256 mismatch"
    heads = {}
    for d in a.designs.split(","):
        h = open(os.path.join(a.prompts, f"p0_prompt_{d}.txt"), encoding="utf-8").read()
        assert PC.sha256_text(h) == pm["designs"][d]["sha256"], f"prompt {d} differs from its registered sha256"
        heads[d] = h
    quids = None
    if a.quids_list:
        quids = PC.load_list(a.manifest, a.quids_list)
    elif a.quids:
        quids = json.load(open(a.quids))
    Q = load_questions(a.questions, quids, a.limit)
    stats = dict(arm="P_0", split=a.split, n=len(Q), model=a.model, tp=a.tp, max_new_tokens=PC.MAX_NEW_P, decoding="greedy",
                 prompt_manifest_sha256=pm["sha256"], designs={d: pm["designs"][d]["sha256"] for d in heads}, out={})
    if a.dry_prompts:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
        for d, h in heads.items():
            P = [PC.encode_prompt(tok, p0_user(h, qt)) for _, qt in Q]
            L = [len(p) for p in P]
            stats["out"][d] = dict(prompt_tok_mean=float(np.mean(L)), prompt_tok_max=int(max(L)),
                                   fits=bool(max(L) + PC.MAX_NEW_P <= a.max_model_len),
                                   first_prompt_text_sha256=PC.sha256_text(tok.decode(P[0])))
            open(os.path.join(a.out, f"dry_prompt_{d}_first.txt"), "w").write(tok.decode(P[0]))
        json.dump(stats, open(os.path.join(a.out, f"dry_prompts_P0_{a.split}.json"), "w"), indent=1)
        print(json.dumps(stats["out"]))
        return
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    import vllm
    t0 = time.time()
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem,
              max_model_len=a.max_model_len, enforce_eager=True, seed=0, enable_prefix_caching=True,
              disable_custom_all_reduce=True)   # amendment B1: custom all-reduce hung the TP2 collective on these PCIe cards
    stats.update(init_s=round(time.time() - t0, 1), vllm=vllm.__version__)
    tok = llm.get_tokenizer()
    stop = set(int(tok.convert_tokens_to_ids(t)) for t in ("<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"))
    sp = SamplingParams(temperature=0.0, max_tokens=PC.MAX_NEW_P, seed=0, logprobs=0)
    for d, h in heads.items():
        P = [PC.encode_prompt(tok, p0_user(h, qt)) for _, qt in Q]
        assert max(len(p) for p in P) + PC.MAX_NEW_P <= a.max_model_len, f"{d}: prompt too long"
        t1 = time.time()
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in P], sp, use_tqdm=False)
        dt = time.time() - t1
        raw = []
        for (q, _), o in zip(Q, outs):
            g = o.outputs[0]; ids = list(g.token_ids)
            lps = [(dd.get(t).logprob if dd is not None and dd.get(t) is not None else None) for t, dd in zip(ids, g.logprobs or [None] * len(ids))]
            raw.append(dict(quid=q, text=g.text, n_out=len(ids), mean_logprob=PC.mean_logprob(ids, lps, stop), finish_reason=g.finish_reason))
        f = os.path.join(a.out, f"raw_P0_{a.split}_{d}.jsonl.gz")
        PC.write_jsonl(f, raw)
        stats["out"][d] = dict(raw_file=os.path.basename(f), raw_sha256=PC.sha256_file(f), seconds=round(dt, 1),
                               prompts_per_s=len(P) / dt, prompt_tok_mean=float(np.mean([len(p) for p in P])),
                               out_tok_mean=float(np.mean([r["n_out"] for r in raw])))
        log(f"{d} n={len(P)} {dt:.0f}s")
        json.dump(stats, open(os.path.join(a.out, f"infer_stats_P0_{a.split}.json"), "w"), indent=1)
    stats["wall_s"] = round(time.time() - t_start, 1)
    json.dump(stats, open(os.path.join(a.out, f"infer_stats_P0_{a.split}.json"), "w"), indent=1)
    os.system(f"chmod -R a+rX {a.out}")


if __name__ == "__main__":
    main()
