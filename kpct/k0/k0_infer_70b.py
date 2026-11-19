#!/usr/bin/env python3
"""k0_infer_70b.py -- k = 0 arms P_0 and G_0: Llama-3.1-70B-Instruct, bf16, vLLM tensor-parallel 2 on BOTH cards.

Recipe (verified on the GPU host; amendment B1): LLM(tensor_parallel_size=2, dtype=bfloat16,
gpu_memory_utilization=0.90, max_model_len=4096, enforce_eager=True, disable_custom_all_reduce=True); docker --ipc=host
--shm-size=16g, VLLM_WORKER_MULTIPROC_METHOD=spawn, NCCL_P2P_DISABLE=1. This file must be a real file (spawn re-exec).
Prefix caching on (each design's head is shared by all prompts). Greedy decoding, seed 0.

  --arm P0: prompts byte-identical to the LLM-parser track's p0_infer_70b.py (p0_prompts.p0_user(head, question),
            pk_common.encode_prompt); heads sha256-checked against p0_prompts_manifest.json; max 160 new tokens, logprobs
            of the sampled tokens recorded (diagnostic; P_0 has no tau). Reads only quid + question of the split.
            Output raw_P0_<split>_<d>.jsonl.gz {quid, text, n_out, mean_logprob, finish_reason} (= p0_infer_70b format,
            consumed unchanged by p0_execute.py).
  --arm G0: prompts from the k0 guidance bundles (k0_bundles.py; no gold) with g0_prompts.fit_g0_prompt (designs
            verified against g0_prompts_manifest.json); max 48 new tokens.
            Output raw_G0_<split>_<g>.jsonl.gz {quid, text, n_out, finish_reason, prompt_tok, n_groups_dropped}.
Generation runs in shards of --shard prompts cached on disk (rule STOP_HOST: a restarted job resumes from the shards;
a shard is reused only if its quids and design sha256 match). --dry_prompts: tokenizer only (CPU), prompt statistics.
"""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PARSER_CODE = os.environ.get("KPCT_PARSER_CODE", "/kpct/k0/code/parser")      # copy of the parser track's P_0 code
sys.path.insert(1, PARSER_CODE)
import pk_common as PC                                   # noqa: E402  (parser track, sha-pinned in k0_hashes.json)
from p0_prompts import p0_user                          # noqa: E402
import gk_common as GC                                   # noqa: E402  (final guided-track version, sha-pinned)
import g0_prompts as G0                                  # noqa: E402

MAX_NEW = {"P0": PC.MAX_NEW_P, "G0": G0.MAX_NEW}


def log(m):
    print(time.strftime("%H:%M:%S") + " " + m, flush=True)


def load_questions(path, keep):
    Q = [(int(q["quid"]), q["question"]) for q in json.load(open(path)) if keep is None or int(q["quid"]) in keep]
    Q.sort()
    if keep is not None:
        assert len(Q) == len(keep), f"{len(Q)} questions for {len(keep)} quids"
    return Q


def load_bundles(path, keep):
    bl = [GC.normalize_bundle(r) for r in GC.read_jsonl(path)]
    bl = sorted([b for b in bl if keep is None or b["quid"] in keep], key=lambda b: b["quid"])
    if keep is not None:
        assert len(bl) == len(keep), f"{len(bl)} bundles for {len(keep)} quids"
    assert not any(b["gold"] for b in bl), "dev/test bundles must not carry gold"
    return bl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["P0", "G0"])
    ap.add_argument("--split", required=True, help="dev_sel2000 | test | dry")
    ap.add_argument("--questions", default=None, help="P0: questions json of the split (quid + question read)")
    ap.add_argument("--bundles", default=None, help="G0: k0 bundle file of the split")
    ap.add_argument("--quids_list", default=None, help="registered list restricting the split (dev_sel2000)")
    ap.add_argument("--quids", default=None, help="json quid list (dry runs)")
    ap.add_argument("--manifest", default="/kpct/k0/code/multitq_kpct_subsets.json")
    ap.add_argument("--p0_prompts", default=os.path.join(PARSER_CODE, "p0_prompts"))
    ap.add_argument("--g0_manifest", default="/kpct/k0/code/g0_prompts/g0_prompts_manifest.json")
    ap.add_argument("--designs", required=True, help="P0: d1,d2,d3 ; G0: g1,g2,g3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=4000)
    ap.add_argument("--model", default=GC.MODEL_70B)
    ap.add_argument("--tp", type=int, default=2); ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    ap.add_argument("--dry_prompts", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    shard_dir = os.path.join(a.out, "shards"); os.makedirs(shard_dir, exist_ok=True)
    t_start = time.time()
    designs = a.designs.split(",")
    keep = None
    if a.quids_list:
        keep = set(PC.load_list(a.manifest, a.quids_list))
    elif a.quids:
        keep = set(int(x) for x in json.load(open(a.quids)))
    # ------------------------------------------------------------------ registered prompt texts
    dsha = {}
    if a.arm == "P0":
        pm = json.load(open(os.path.join(a.p0_prompts, "p0_prompts_manifest.json")))
        assert pm["sha256"] == PC.sha_recipe(pm), "P_0 prompt manifest sha256 mismatch"
        heads = {}
        for d in designs:
            h = open(os.path.join(a.p0_prompts, f"p0_prompt_{d}.txt"), encoding="utf-8").read()
            assert PC.sha256_text(h) == pm["designs"][d]["sha256"], f"P_0 prompt {d} differs from its registered sha256"
            heads[d] = h; dsha[d] = pm["designs"][d]["sha256"]
        man_sha = pm["sha256"]
        items = load_questions(a.questions, keep)
        quids = [q for q, _ in items]
    else:
        gm = G0.verify(a.g0_manifest)
        for d in designs:
            assert d in gm["designs"], d
            dsha[d] = gm["designs"][d]["fixed_text_sha256"]
        man_sha = gm["sha256"]
        bl = load_bundles(a.bundles, keep)
        quids = [b["quid"] for b in bl]
        stats_b = dict(bundle_file=os.path.basename(a.bundles), bundle_sha256=GC.sha256_file(a.bundles))
    stats = dict(arm=a.arm, split=a.split, n=len(quids), model=a.model, tp=a.tp, max_new_tokens=MAX_NEW[a.arm],
                 decoding="greedy", prompt_manifest_sha256=man_sha, designs=dsha, out={}, shard=a.shard)
    if a.arm == "G0":
        stats.update(stats_b)
    sf = os.path.join(a.out, f"infer_stats_{a.arm}_{a.split}.json")
    if os.path.exists(sf) and not a.dry_prompts:          # resumed job: keep the per-design records of completed designs
        old = json.load(open(sf))
        if old.get("prompt_manifest_sha256") == man_sha and old.get("n") == len(quids):
            stats["out"].update(old.get("out", {}))
            stats["previous_sessions"] = old.get("previous_sessions", []) + [{k: old.get(k) for k in ("init_s", "wall_s")}]

    def build(tok, d):
        """-> (prompt_ids list, per-prompt extra dict list)"""
        if a.arm == "P0":
            P = [PC.encode_prompt(tok, p0_user(heads[d], qt)) for _, qt in items]
            return P, [{} for _ in P]
        P, X = [], []
        for b in bl:
            p, nd = G0.fit_g0_prompt(tok, b, d)
            P.append(p); X.append(dict(prompt_tok=len(p), n_groups_dropped=nd))
        return P, X

    if a.dry_prompts:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
        for d in designs:
            P, X = build(tok, d)
            L = [len(p) for p in P]
            stats["out"][d] = dict(prompt_tok_mean=float(np.mean(L)), prompt_tok_max=int(max(L)),
                                   fits=bool(max(L) + MAX_NEW[a.arm] <= (a.max_model_len if a.arm == "P0" else G0.MAX_LEN)),
                                   n_truncated=int(sum(x.get("n_groups_dropped", 0) > 0 for x in X)),
                                   first_prompt_text_sha256=PC.sha256_text(tok.decode(P[0])))
            for i in range(min(3, len(P))):
                open(os.path.join(a.out, f"dry_prompt_{a.arm}_{d}_{i}.txt"), "w").write(tok.decode(P[i]))
        json.dump(stats, open(os.path.join(a.out, f"dry_prompts_{a.arm}_{a.split}.json"), "w"), indent=1)
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
    log(f"engine up {stats['init_s']}s vllm {vllm.__version__}")
    tok = llm.get_tokenizer()
    stop = set(int(tok.convert_tokens_to_ids(t)) for t in ("<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"))
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW[a.arm], seed=0, logprobs=(0 if a.arm == "P0" else None))
    for d in designs:
        f = os.path.join(a.out, f"raw_{a.arm}_{a.split}_{d}.jsonl.gz")
        if os.path.exists(f) and os.path.exists(f + ".done"):
            log(f"{d}: {os.path.basename(f)} complete, skipped"); continue
        P, X = build(tok, d)
        lim = a.max_model_len if a.arm == "P0" else G0.MAX_LEN
        assert max(len(p) for p in P) + MAX_NEW[a.arm] <= lim, f"{d}: prompt too long"
        raw, secs, n_gen = [], 0.0, 0
        for si, s0 in enumerate(range(0, len(P), a.shard)):
            spath = os.path.join(shard_dir, f"{a.arm}_{a.split}_{d}_{si:03d}.json")
            qs = quids[s0:s0 + a.shard]
            if os.path.exists(spath):
                sh = json.load(open(spath))
                assert sh["quids"] == qs and sh["design_sha256"] == dsha[d] and sh["manifest_sha256"] == man_sha, f"stale shard {spath}"
            else:
                t1 = time.time()
                outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in P[s0:s0 + a.shard]], sp, use_tqdm=False)
                dt = time.time() - t1
                rows = []
                for q, o, x in zip(qs, outs, X[s0:s0 + a.shard]):
                    g = o.outputs[0]; ids = list(g.token_ids)
                    r = dict(quid=q, text=g.text, n_out=len(ids))
                    if a.arm == "P0":
                        lps = [(dd.get(t).logprob if dd is not None and dd.get(t) is not None else None)
                               for t, dd in zip(ids, g.logprobs or [None] * len(ids))]
                        r["mean_logprob"] = PC.mean_logprob(ids, lps, stop)
                    r["finish_reason"] = g.finish_reason
                    r.update(x)
                    rows.append(r)
                sh = dict(quids=qs, design_sha256=dsha[d], manifest_sha256=man_sha, seconds=round(dt, 2), rows=rows)
                json.dump(sh, open(spath + ".tmp", "w")); os.replace(spath + ".tmp", spath); n_gen += len(qs)
                log(f"{a.arm} {a.split} {d} shard {si} n={len(qs)} {dt:.0f}s ({len(qs) / max(dt, 1e-9):.2f}/s)")
            raw += sh["rows"]; secs += sh["seconds"]
        assert [r["quid"] for r in raw] == quids
        PC.write_jsonl(f, raw)
        open(f + ".done", "w").write(PC.sha256_file(f))
        stats["out"][d] = dict(raw_file=os.path.basename(f), raw_sha256=PC.sha256_file(f), seconds=round(secs, 1),
                               prompts_per_s=len(P) / max(secs, 1e-9), prompt_tok_mean=float(np.mean([len(p) for p in P])),
                               out_tok_mean=float(np.mean([r["n_out"] for r in raw])),
                               finish_length_rate=float(np.mean([r["finish_reason"] == "length" for r in raw])),
                               n_generated_this_session=n_gen)
        if a.arm == "G0":
            stats["out"][d]["n_truncated"] = int(sum(r["n_groups_dropped"] > 0 for r in raw))
        log(f"{d} n={len(P)} {secs:.0f}s")
        json.dump(stats, open(sf, "w"), indent=1)
    stats["wall_s"] = round(time.time() - t_start, 1)
    json.dump(stats, open(sf, "w"), indent=1)
    os.system(f"chmod -R a+rX {a.out}")


if __name__ == "__main__":
    main()
