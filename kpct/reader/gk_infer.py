#!/usr/bin/env python3
"""gk_infer.py -- vLLM inference for G_k, GU_k, SHUF_G1 and QONLY_G1 (rule arms.G_k.output / arms.GU_k.output / controls_k1).

Greedy decoding, max 48 new tokens, one user turn with the same chat template (fixed date_string) as training.
Several adapters (e.g. the epoch checkpoints of one run) are scored in ONE engine session (multi-LoRA, one at a time).
The generated string is valid iff, after stripping, whitespace collapsing, case folding and _ -> space, it equals a KG
entity name (it is then written as that KG name) or is a time string YYYY / YYYY-MM / YYYY-MM-DD in 2005-01-01..2015-12-31.
Ranked list: G / SHUF / QONLY = [answer] + S_k seed-0 list (deduplicated); GU = [answer] + entities and dates of the
retrieved facts in retrieval order, deduplicated, padded with the most frequent k% train answers. An invalid answer
stays at rank 1 (a miss); no fallback. This script never reads answers of dev or test (scoring is gk_select.py, dev only).

  --split bench   stage-0 throughput: 500 benchmark prompts per style (G, GU, PB), outputs discarded
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gk_common as GC
from gk_train import P_HEAD


def log(m):
    print(time.strftime("%H:%M:%S") + " " + m, flush=True)


def load_s_lists(path, quids):
    """S_k seed-0 ranked lists {quid: [..]} from a prediction JSONL {"quid", "ranked"}"""
    out = {}
    for r in GC.read_jsonl(path):
        q = int(r["quid"])
        if q in quids:
            rk = r["ranked"]
            out[q] = json.loads(rk.replace("'", '"')) if isinstance(rk, str) else rk
    return out


# ---------------------------------------------------------------------------------------------- reusable pieces
def make_prompts(tok, arm, split_rows=None, questions=None, gu=None, order="S", keep=None):
    """arm in G | SHUF | QONLY (split_rows = bundle rows) or GU (questions [{quid, question}], gu {quid: facts}).
    -> (quids, prompt_ids, extra, info); extra = normalised bundles (G family) or gu facts (GU)"""
    info = {}
    if arm in ("G", "SHUF", "QONLY"):
        bl = [GC.normalize_bundle(r) for r in split_rows]
        bl = sorted([b for b in bl if keep is None or b["quid"] in keep], key=lambda b: b["quid"])
        if keep is not None:
            assert len(bl) == len(keep), f"{len(bl)} bundles for {len(keep)} quids"
        assert not any(b["gold"] for b in bl), "dev/test bundles must not carry gold"
        if arm == "G":
            P = [GC.fit_g_prompt(tok, b, order=order) for b in bl]
        elif arm == "SHUF":
            dmap = GC.derangement_map([b["quid"] for b in bl], seed=0); byq = {b["quid"]: b for b in bl}
            P = [GC.fit_g_prompt(tok, b, order=order, bundle_from=byq[dmap[b["quid"]]]) for b in bl]
            info["shuf_derangement_sha256"] = GC.sha256_text(json.dumps(sorted(dmap.items())))
        else:
            P = [(GC.encode_prompt(tok, GC.qonly_user(b)), [], 0) for b in bl]
        info["n_truncated"] = int(sum(p[2] > 0 for p in P))
        return [b["quid"] for b in bl], [p[0] for p in P], bl, info
    Q = sorted([dict(quid=int(q["quid"]), question=q["question"]) for q in questions if keep is None or int(q["quid"]) in keep], key=lambda q: q["quid"])
    if keep is not None:
        assert len(Q) == len(keep)
    P = [GC.fit_gu_prompt(tok, q["question"], gu[q["quid"]]) for q in Q]
    info["n_truncated"] = int(sum(p[2] > 0 for p in P))
    return [q["quid"] for q in Q], [p[0] for p in P], gu, info


def generate(llm, prompts, max_tokens, lora_request, logprobs=False):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=0, logprobs=0 if logprobs else None)
    t1 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp, lora_request=lora_request, use_tqdm=False)
    dt = time.time() - t1
    st = dict(n=len(prompts), seconds=round(dt, 2), prompts_per_s=len(prompts) / dt, prompt_tok_mean=float(np.mean([len(p) for p in prompts])),
              out_tok_mean=float(np.mean([len(o.outputs[0].token_ids) for o in outs])), prompt_tok_per_s=sum(len(p) for p in prompts) / dt)
    return outs, st


def postprocess(quids, outs, enorm):
    raw, preds, n_valid = [], [], 0
    for q, o in zip(quids, outs):
        txt = o.outputs[0].text
        ans, ok = GC.canon_answer(txt, enorm); n_valid += ok
        raw.append(dict(quid=q, text=txt, answer=ans, valid=bool(ok), n_out=len(o.outputs[0].token_ids)))
        preds.append((q, ans, None))
    return raw, preds, 1 - n_valid / max(1, len(quids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["G", "GU", "SHUF", "QONLY", "BENCH"])
    ap.add_argument("--split", required=True, help="dev_sel2000 | test | bench")
    ap.add_argument("--bundles", default=None, help="G/SHUF/QONLY: bundles/<subset>/<split>.jsonl(.gz)")
    ap.add_argument("--gu_facts", default=None, help="GU: retrieval jsonl.gz for the split")
    ap.add_argument("--questions", default=None, help="GU: questions json of the split (question text only)")
    ap.add_argument("--quids", default=None, help="json list restricting the split (dev_sel2000 quids)")
    ap.add_argument("--adapters", required=True, help="name=path[,name=path...]")
    ap.add_argument("--order", default="S", choices=["S", "random"])
    ap.add_argument("--s_list", default=None, help="G: S_k seed-0 prediction JSONL for the ranked-list tail")
    ap.add_argument("--gu_pad", default=None, help="GU: gu_pad.json written by gk_train.py (k%% list answers)")
    ap.add_argument("--kg", required=True, help="kg/full.txt (entity vocabulary for validity)")
    ap.add_argument("--out", required=True); ap.add_argument("--tag", default="")
    ap.add_argument("--max_model_len", type=int, default=2304); ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--bench_file", default=None); ap.add_argument("--bench_n", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=0, help="bench: untimed-for-throughput warm-up prompts (G style, lora rows) first")
    ap.add_argument("--bench_order", default="G,GU,PB", help="bench: style order; a repeated style gets suffix 2")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.inputs import TokensPrompt
    import vllm, transformers
    kg = GC.load_kg(a.kg)
    enorm = GC.entity_norm_map(sorted({f[0] for f in kg} | {f[2] for f in kg}))
    t0 = time.time()
    llm = LLM(model=GC.MODEL_8B, dtype="bfloat16", enable_lora=True, max_lora_rank=16, max_loras=1,
              max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem, seed=0, enforce_eager=a.eager)
    t_init = time.time() - t0
    tok = llm.get_tokenizer()
    log(f"vllm {vllm.__version__} transformers {transformers.__version__} init {t_init:.0f}s")
    adapters = [x.split("=", 1) for x in a.adapters.split(",")]
    stats = dict(arm=a.arm, split=a.split, init_s=round(t_init, 1), vllm=vllm.__version__, eager=a.eager,
                 prompt_sha256=GC.prompt_texts_sha256(), adapters={})
    # ------------------------------------------------------------------ build prompts
    jobs = []     # (style, quids, prompt_ids, max_tokens, extra)
    if a.split == "bench":
        rows = [r for r in GC.read_jsonl(a.bench_file) if r["use"] == "inf"][:a.bench_n]
        gu = {int(r["quid"]): [tuple(f) for f in r["facts"]] for r in GC.read_jsonl(a.gu_facts)} if a.gu_facts else {}
        bl = [GC.normalize_bundle(r) for r in rows]
        jobs.append(("G", [b["quid"] for b in bl], [GC.fit_g_prompt(tok, b)[0] for b in bl], GC.MAX_NEW_G, None))
        if gu:
            jobs.append(("GU", [r["quid"] for r in rows], [GC.fit_gu_prompt(tok, r["question"], gu[int(r["quid"])])[0] for r in rows], GC.MAX_NEW_G, None))
        jobs.append(("PB", [r["quid"] for r in rows], [GC.encode_prompt(tok, P_HEAD + f"\nQuestion: {r['question']}\n") for r in rows], 160, "logprobs"))
        byst = {j[0]: j for j in jobs}; jobs = []
        for st_ in a.bench_order.split(","):
            j = byst[st_.rstrip("2")]
            jobs.append((st_,) + j[1:])
        if a.warmup:
            wr = [GC.normalize_bundle(r) for r in GC.read_jsonl(a.bench_file) if r["use"] == "lora"][:a.warmup]
            jobs.insert(0, ("WARMUP_G", [b["quid"] for b in wr], [GC.fit_g_prompt(tok, b)[0] for b in wr], GC.MAX_NEW_G, None))
        tr_rows = [r for r in GC.read_jsonl(a.bench_file) if r["use"] == "lora"][:100]
        stats["prompt_ids_sha256_first100_G_lora"] = GC.sha256_text(json.dumps([GC.fit_g_prompt(tok, GC.normalize_bundle(r), GC.g_target(GC.normalize_bundle(r, True))[0])[0] for r in tr_rows]))
    else:
        keep = set(int(x) for x in json.load(open(a.quids))) if a.quids else None
        if a.arm in ("G", "SHUF", "QONLY"):
            bl = [GC.normalize_bundle(r) for r in GC.read_jsonl(a.bundles)]
            bl = sorted([b for b in bl if keep is None or b["quid"] in keep], key=lambda b: b["quid"])
            if keep is not None:
                assert len(bl) == len(keep), f"{len(bl)} bundles for {len(keep)} quids"
            assert not any(b["gold"] for b in bl), "dev/test bundles must not carry gold"
            if a.arm == "G":
                P = [GC.fit_g_prompt(tok, b, order=a.order) for b in bl]
            elif a.arm == "SHUF":
                dmap = GC.derangement_map([b["quid"] for b in bl], seed=0); byq = {b["quid"]: b for b in bl}
                P = [GC.fit_g_prompt(tok, b, order=a.order, bundle_from=byq[dmap[b["quid"]]]) for b in bl]
                stats["shuf_derangement_sha256"] = GC.sha256_text(json.dumps(sorted(dmap.items())))
            else:
                P = [(GC.encode_prompt(tok, GC.qonly_user(b)), [], 0) for b in bl]
            stats["n_truncated"] = int(sum(p[2] > 0 for p in P))
            jobs.append((a.arm, [b["quid"] for b in bl], [p[0] for p in P], GC.MAX_NEW_G, bl))
        else:
            Q = json.load(open(a.questions))
            Q = sorted([dict(quid=q["quid"], question=q["question"]) for q in Q if keep is None or q["quid"] in keep], key=lambda q: q["quid"])
            gu = {int(r["quid"]): [tuple(f) for f in r["facts"]] for r in GC.read_jsonl(a.gu_facts)}
            P = [GC.fit_gu_prompt(tok, q["question"], gu[q["quid"]]) for q in Q]
            stats["n_truncated"] = int(sum(p[2] > 0 for p in P))
            jobs.append(("GU", [q["quid"] for q in Q], [p[0] for p in P], GC.MAX_NEW_G, gu))
    s_lists = load_s_lists(a.s_list, set(jobs[0][1])) if a.s_list else None
    pad = json.load(open(a.gu_pad)) if a.gu_pad else []
    # ------------------------------------------------------------------ generate
    for li, (aname, apath) in enumerate(adapters):
        lr = LoRARequest(aname, li + 1, apath)
        st_a = {}
        for style, quids, prompts, mx, extra in jobs:
            sp = SamplingParams(temperature=0.0, max_tokens=mx, seed=0, logprobs=0 if extra == "logprobs" else None)
            t1 = time.time()
            outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp, lora_request=lr, use_tqdm=False)
            dt = time.time() - t1
            n_out = [len(o.outputs[0].token_ids) for o in outs]
            st = dict(n=len(prompts), seconds=round(dt, 2), prompts_per_s=len(prompts) / dt, prompt_tok_mean=float(np.mean([len(p) for p in prompts])),
                      out_tok_mean=float(np.mean(n_out)), prompt_tok_per_s=sum(len(p) for p in prompts) / dt)
            if a.split != "bench":
                raw, preds, n_valid = [], [], 0
                for q, o in zip(quids, outs):
                    txt = o.outputs[0].text
                    ans, ok = GC.canon_answer(txt, enorm); n_valid += ok
                    raw.append(dict(quid=q, text=txt, answer=ans, valid=bool(ok), n_out=len(o.outputs[0].token_ids)))
                    preds.append((q, ans, None))
                st["invalid_rate"] = 1 - n_valid / max(1, len(quids))
                st_a[style] = st
                write_outputs(a, style, aname, raw, preds, extra, s_lists, pad, st)
            else:
                st_a[style] = st
            log(f"{aname} {style} n={len(prompts)} {dt:.1f}s {st['prompts_per_s']:.2f} prompts/s prompt_tok {st['prompt_tok_mean']:.0f} out_tok {st['out_tok_mean']:.1f}")
        stats["adapters"][aname] = dict(path=apath, adapter_sha256=GC.sha256_file(os.path.join(apath, "adapter_model.safetensors"))
                                        if os.path.exists(os.path.join(apath, "adapter_model.safetensors")) else None, styles=st_a)
    stats["wall_s"] = round(time.time() - t_start, 1)
    json.dump(stats, open(os.path.join(a.out, f"infer_stats_{a.arm}_{a.split}{a.tag}.json"), "w"), indent=1)


def write_outputs(a, style, aname, raw, preds, extra, s_lists, pad, st, split=None):
    """raw generations + ranked prediction JSONL {"quid", "ranked"} (official scorer format) + sha256"""
    base = f"{style}_{split or a.split}_{aname}{a.tag}"
    by_q = {x["quid"]: x for x in extra} if style != "GU" else None
    rows = []
    for q, ans, _ in preds:
        if style == "GU":
            rk = GC.gu_ranked(ans, extra[q], pad)
        else:
            tail = s_lists.get(q) if s_lists is not None else None
            if tail is None:
                tail = [c["candidate"] for c in by_q[q]["candidates"]]
            rk = GC.g_ranked(ans, tail)
        rows.append(dict(quid=q, ranked=rk))
    rp = os.path.join(a.out, f"raw_{base}.jsonl.gz"); pp = os.path.join(a.out, f"pred_{base}.jsonl")
    GC.write_jsonl(rp, raw); GC.write_jsonl(pp, rows)
    st["raw_file"], st["pred_file"] = os.path.basename(rp), os.path.basename(pp)
    st["raw_sha256"], st["pred_sha256"] = GC.sha256_file(rp), GC.sha256_file(pp)
    st["tail_source"] = "S_k seed-0 list" if (style != "GU" and s_lists is not None) else ("bundle candidates" if style != "GU" else "retrieved facts + k% pad")


if __name__ == "__main__":
    main()
