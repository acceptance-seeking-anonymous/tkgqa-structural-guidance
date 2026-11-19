#!/usr/bin/env python3
"""r2_gen.py -- R2 rewrite generation with Qwen/Qwen2.5-72B-Instruct-AWQ (open weights, 4-bit AWQ) in vLLM on ONE card.
One user turn = the prompt text given in the input file (R1 GEN_PROMPT, optionally + GEN_RETRY_SUFFIX), the model's
own default chat template, greedy decoding (temperature 0, seed 0), max 300 new tokens. Writes raw generations only;
mechanical checks and the judge run elsewhere. Never reads answers.
v2: the code runs under a __main__ guard (vLLM spawn start-up); prompts, model, decoding and outputs are unchanged.
  python3 r2_gen.py --inp prompts.jsonl --out gen.jsonl [--model Qwen/Qwen2.5-72B-Instruct-AWQ]
"""
import argparse, glob, hashlib, json, os, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inp', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='Qwen/Qwen2.5-72B-Instruct-AWQ')
    ap.add_argument('--max_tokens', type=int, default=300)
    ap.add_argument('--gpu_mem', type=float, default=0.90); ap.add_argument('--max_model_len', type=int, default=2048)
    a = ap.parse_args()
    t0 = time.time()
    from vllm import LLM, SamplingParams
    import vllm
    rows = [json.loads(l) for l in open(a.inp) if l.strip()]
    llm = LLM(model=a.model, dtype='float16', max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem, seed=0,
              enforce_eager=True, trust_remote_code=False)
    tok = llm.get_tokenizer()
    t_init = time.time() - t0
    texts = [tok.apply_chat_template([{'role': 'user', 'content': r['prompt']}], tokenize=False, add_generation_prompt=True) for r in rows]
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, seed=0)
    t1 = time.time()
    outs = llm.generate(texts, sp, use_tqdm=False)
    dt = time.time() - t1
    with open(a.out, 'w') as f:
        for r, o in zip(rows, outs):
            g = o.outputs[0]
            f.write(json.dumps(dict(quid=r['quid'], attempt=r.get('attempt'), text=g.text, finish_reason=g.finish_reason,
                                    n_out=len(g.token_ids), prompt_sha256=hashlib.sha256(r['prompt'].encode()).hexdigest()), ensure_ascii=False) + '\n')
    snap = sorted(glob.glob(f"/hf/hub/models--{a.model.replace('/', '--')}/snapshots/*"))
    cfg = os.path.join(snap[-1], 'config.json') if snap else None
    stats = dict(model=a.model, snapshot=os.path.basename(snap[-1]) if snap else None,
                 config_sha256=hashlib.sha256(open(cfg, 'rb').read()).hexdigest() if cfg and os.path.exists(cfg) else None,
                 vllm=vllm.__version__, n=len(rows), init_s=round(t_init, 1), gen_s=round(dt, 1),
                 chat_template_sha256=hashlib.sha256((tok.chat_template or '').encode()).hexdigest(),
                 example_rendered_prefix=texts[0][:400] if texts else None, decoding=dict(temperature=0.0, seed=0, max_tokens=a.max_tokens),
                 finish_length=sum(o.outputs[0].finish_reason == 'length' for o in outs))
    json.dump(stats, open(a.out.replace('.jsonl', '_stats.json'), 'w'), indent=1)
    print(json.dumps(stats)[:1500], flush=True)


if __name__ == '__main__':      # vLLM spawns its engine core with multiprocessing 'spawn', which re-imports this file
    main()
