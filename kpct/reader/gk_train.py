#!/usr/bin/env python3
"""gk_train.py -- LoRA training of the guided reasoner G_k and the unguided control GU_k (rule arms.G_k / arms.GU_k).

Recipe (fixed a priori by the rule): Llama-3.1-8B-Instruct; LoRA r 16, alpha 32, dropout 0.05 on q,k,v,o,gate,up,down;
AdamW lr 2e-4, weight decay 0, cosine schedule, 3% warm-up, grad clip 1.0, effective batch 16 questions, bf16; loss on
the answer tokens only (incl. <|eot_id|>); tokenizer chat template, one user turn (date_string fixed); max length 2,048
tokens; seed 0 sets the LoRA init and the data order. E_max = 3 (k = 1), 2 (k = 5), 1 (k = 10, 100); an adapter is saved
at every epoch end, and at 0.5 epoch when E_max = 1 (checkpoint selection on dev_sel2000 happens in gk_select.py).
Resume checkpoint every 500 optimizer steps (STOP_HOST). One card per process, model.to('cuda:0'), no device_map.

Targets: G -- the gold answer ranked highest in the bundle candidate list, answers[0] if none is in the list;
GU -- the gold answer occurring earliest in the 20 retrieved facts, answers[0] if none occurs. Every question of the
k% list is trained (no skipping).
--bench_steps N: stage-0 throughput benchmark on the benchmark questions (outside every training list); the adapter
is written only for the vLLM load test and deleted by the benchmark driver.
"""
import argparse, collections, json, math, os, random, sys, time, shutil
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gk_common as GC

P_HEAD = ("Parse the temporal question about the ICEWS knowledge graph into JSON with the keys mentions (list of "
          "{text, entity}), relation, answer_role (subject|object), qtype, answer_type (entity|time), granularity "
          "(day|month|year), time, anchor, order. Use exact KG entity and relation names.\n")


def log(msg, fh=None):
    s = time.strftime("%H:%M:%S") + " " + msg
    print(s, flush=True)
    if fh:
        fh.write(s + "\n"); fh.flush()


def build_examples(a, tok, lf):
    """-> list of dict(quid, prompt_ids, answer_ids, target, in_list, dropped) and info"""
    info = {}
    if a.bench_steps:
        rows = [r for r in GC.read_jsonl(a.bench_file) if r["use"] == "lora"]
    elif a.arm == "G":
        rows = GC.read_jsonl(a.bundles)
    else:
        Qtr = {q["quid"]: q for q in json.load(open(a.questions))}
        keep = set(GC.load_list(a.manifest, a.list))
        rows = [dict(quid=q, question=Qtr[q]["question"], gold=Qtr[q]["answers"]) for q in sorted(keep)]
    if not a.bench_steps:
        keep = set(GC.load_list(a.manifest, a.list))
        got = set(int(r["quid"]) for r in rows)
        assert got == keep, f"train rows {len(got)} != list {a.list} ({len(keep)}); missing {len(keep - got)}, extra {len(got - keep)}"
        info["list"] = a.list; info["list_sha256"] = GC.list_sha256(keep)
        if a.arm == "G":                       # gold of the bundle must equal train.json answers
            Qtr = {q["quid"]: q for q in json.load(open(a.questions))}
            bad = [r["quid"] for r in rows if list(r.get("gold") or []) != Qtr[int(r["quid"])]["answers"]]
            assert not bad, f"{len(bad)} bundle gold fields differ from train.json answers (e.g. {bad[:3]})"
    gu = None
    if a.arm == "GU":
        gu = {int(r["quid"]): [tuple(f) for f in r["facts"]] for r in GC.read_jsonl(a.gu_facts)}
        miss = [r["quid"] for r in rows if int(r["quid"]) not in gu]
        assert not miss, f"{len(miss)} train quids without GU retrieval"
    ex, n_in, n_drop, t0 = [], 0, 0, time.time()
    for r in rows:
        if a.arm == "G":
            b = GC.normalize_bundle(r, require_gold=True)
            tgt, inl = GC.g_target(b)
            p, ans, dr = GC.fit_g_prompt(tok, b, tgt, order=a.order)
        elif a.arm == "GU":
            facts = gu[int(r["quid"])]
            tgt, inl = GC.gu_target(r["gold"], facts)
            p, ans, dr = GC.fit_gu_prompt(tok, r["question"], facts, tgt)
        else:   # PB: P-style prompt (benchmark only)
            u = P_HEAD + f"\nQuestion: {r['question']}\n"
            p = GC.encode_prompt(tok, u); tgt = r["p_target"]; ans = GC.encode_answer(tok, u, tgt, p); inl, dr = True, 0
        n_in += inl; n_drop += dr > 0
        ex.append(dict(quid=int(r["quid"]), p=p, a=ans, target=tgt, in_list=bool(inl)))
    info.update(n=len(ex), target_in_source=n_in / max(1, len(ex)), n_truncated=n_drop, tokenize_s=round(time.time() - t0, 1))
    # GU ranked-list padding: most frequent answers of the k% train list (k% list only; rule GU_k.output)
    if a.arm == "GU" and not a.bench_steps:
        cnt = collections.Counter(x for r in rows for x in r["gold"])
        info["gu_pad"] = [x for x, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:50]]
    return ex, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["G", "GU", "PB"])
    ap.add_argument("--list", default=None, help="registered LLM train list (k01_s0, k05_s0, k10_s0, k01_s1, k01_s2, k100, cap20k_*)")
    ap.add_argument("--bundles", default=None, help="G: bundles/<subset>/train.jsonl(.gz)")
    ap.add_argument("--gu_facts", default=None, help="GU: retrieval jsonl.gz covering the train quids")
    ap.add_argument("--questions", default=None, help="train.json (GU question text + gold; G gold check)")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--order", default="S", choices=["S", "random"])
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True); ap.add_argument("--name", default=None)
    ap.add_argument("--train_tok", type=int, default=8192, help="micro-batch padded-token budget")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--bench_steps", type=int, default=0); ap.add_argument("--bench_file", default=None)
    ap.add_argument("--ckpt_every", type=int, default=500)
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
    import transformers, peft
    from peft import LoraConfig, get_peft_model
    k = GC.k_of(a.list) if a.list else "bench"
    E = a.epochs if a.epochs is not None else (1 if a.bench_steps else GC.E_MAX[k])
    name = a.name or (f"{a.arm}_{a.list}_{'ord' + a.order + '_' if a.arm == 'G' else ''}s{a.seed}" + ("_bench" if a.bench_steps else ""))
    adir = os.path.join(a.out, name); os.makedirs(adir, exist_ok=True)
    lf = open(os.path.join(adir, "train_log.txt"), "a")
    t_start = time.time()
    log(f"{name} torch {torch.__version__} transformers {transformers.__version__} peft {peft.__version__} "
        f"gpu {torch.cuda.get_device_name(0)} E={E}", lf)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    tok = AutoTokenizer.from_pretrained(GC.MODEL_8B)
    pad_id = tok.convert_tokens_to_ids("<|finetune_right_pad_id|>")
    ex, info = build_examples(a, tok, lf)
    Ls = np.array([len(e["p"]) + len(e["a"]) for e in ex])
    log(f"examples {len(ex)} len mean {Ls.mean():.0f} p95 {np.percentile(Ls, 95):.0f} max {Ls.max()} target_in_source "
        f"{info['target_in_source']:.4f} truncated {info['n_truncated']} tokenize {info['tokenize_s']}s", lf)
    stats = dict(name=name, arm=a.arm, list=a.list, order=a.order if a.arm == "G" else None, seed=a.seed, epochs=E,
                 lora=GC.LORA, opt=GC.OPT, max_len=GC.MAX_LEN, chat_date=GC.CHAT_DATE, train_tok=a.train_tok,
                 prompt_sha256=GC.prompt_texts_sha256(), model=GC.MODEL_8B, versions=dict(torch=torch.__version__,
                 transformers=transformers.__version__, peft=peft.__version__), len_mean=float(Ls.mean()),
                 len_p95=float(np.percentile(Ls, 95)), len_max=int(Ls.max()), answer_tokens_mean=float(np.mean([len(e["a"]) for e in ex])),
                 **{k_: v for k_, v in info.items() if k_ not in ("gu_pad", "list")})
    if "gu_pad" in info:
        json.dump(info["gu_pad"], open(os.path.join(adir, "gu_pad.json"), "w"))
    # token-id fingerprint of the first 100 prompts (cross-check against the vLLM tokenizer)
    stats["prompt_ids_sha256_first100"] = GC.sha256_text(json.dumps([e["p"] for e in ex[:100]]))
    t_l0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(GC.MODEL_8B, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0")
    torch.manual_seed(a.seed)
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", bias="none", **GC.LORA))
    if a.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads()
    stats["load_s"] = round(time.time() - t_l0, 1)
    inner, head = model.get_base_model().model, model.get_base_model().lm_head
    params = [p for p in model.parameters() if p.requires_grad]
    B = GC.OPT["eff_batch"]
    spe = math.ceil(len(ex) / B)                          # optimizer steps per epoch
    total = int(math.ceil(E * spe))
    if a.bench_steps:
        total_run = min(total, a.bench_steps)
    else:
        total_run = total
    opt = torch.optim.AdamW(params, lr=GC.OPT["lr"], weight_decay=GC.OPT["weight_decay"])
    sch = get_cosine_schedule_with_warmup(opt, int(math.ceil(GC.OPT["warmup"] * total)), total)
    # data order: per epoch, permutation default_rng(1000*seed + 7 + epoch); sort by length inside chunks of 64 batches;
    # batch order shuffled within the chunk (length bucketing; deterministic per seed)
    plan = []
    for ep in range(int(math.ceil(E))):
        rng = np.random.default_rng(1000 * a.seed + 7 + ep); perm = rng.permutation(len(ex)); CH = B * 64; ep_plan = []
        for c0 in range(0, len(perm), CH):
            ch = sorted(perm[c0:c0 + CH], key=lambda i: Ls[i])
            bs = [ch[j:j + B] for j in range(0, len(ch), B)]
            for j in rng.permutation(len(bs)):
                ep_plan.append(bs[j])
        plan += ep_plan
    plan = plan[:total]
    save_at = {int(round(e_ * spe)): f"ep{e_:g}" for e_ in ([0.5, 1] if E == 1 else list(range(1, int(E) + 1)))}
    save_at[total] = f"ep{E:g}"
    done_dir = os.path.join(adir, save_at[total], "adapter_model.safetensors")
    if not a.bench_steps and os.path.exists(done_dir) and os.path.exists(os.path.join(adir, "train_stats.json")):
        log(f"{name}: final checkpoint {save_at[total]} already exists -- nothing to do", lf)
        return
    ck = os.path.join(adir, "ckpt.pt"); step0 = 0
    if os.path.exists(ck) and not a.bench_steps:
        st = torch.load(ck, map_location="cuda:0", weights_only=False)
        for p_, v in zip(params, st["params"]):
            p_.data.copy_(v)
        opt.load_state_dict(st["opt"]); sch.load_state_dict(st["sch"]); step0 = st["step"]
        torch.set_rng_state(st["rng_cpu"]); torch.cuda.set_rng_state(st["rng_cuda"])
        log(f"resumed at step {step0}", lf)
    model.train(); t0 = time.time(); losses = []; ntok_pad = ntok_real = 0; step_times = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(step0, total_run):
        ts = time.time()
        grp = plan[step]
        n_ans = sum(len(ex[i]["a"]) for i in grp)
        mbs, cur = [], []
        for i in grp:
            if cur and (len(cur) + 1) * max(Ls[j] for j in cur + [i]) > a.train_tok:
                mbs.append(cur); cur = []
            cur.append(i)
        mbs.append(cur)
        tot = 0.0
        for mb in mbs:
            ml = int(max(Ls[i] for i in mb))
            ids = torch.full((len(mb), ml), pad_id, dtype=torch.long); att = torch.zeros((len(mb), ml), dtype=torch.long)
            pos_b, pos_t, tgt = [], [], []
            for r_, i in enumerate(mb):
                seq = ex[i]["p"] + ex[i]["a"]; ids[r_, :len(seq)] = torch.tensor(seq); att[r_, :len(seq)] = 1
                lp = len(ex[i]["p"])
                for j, t_ in enumerate(ex[i]["a"]):
                    pos_b.append(r_); pos_t.append(lp - 1 + j); tgt.append(t_)
            ntok_pad += len(mb) * ml; ntok_real += int(att.sum())
            ids, att = ids.to("cuda:0"), att.to("cuda:0")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hs = inner(input_ids=ids, attention_mask=att).last_hidden_state
                h_sel = hs[torch.tensor(pos_b, device="cuda:0"), torch.tensor(pos_t, device="cuda:0")]
                logits = head(h_sel).float()
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor(tgt, device="cuda:0"), reduction="sum") / n_ans
            loss.backward(); tot += float(loss)
        torch.nn.utils.clip_grad_norm_(params, GC.OPT["grad_clip"])
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        losses.append(tot); step_times.append(time.time() - ts)
        s1 = step + 1
        if s1 % 50 == 0 or s1 == total_run:
            el = time.time() - t0
            log(f"step {s1}/{total} loss {np.mean(losses[-50:]):.4f} {el / max(1, s1 - step0):.2f}s/it "
                f"{ntok_pad / el:.0f} padded tok/s lr {sch.get_last_lr()[0]:.2e}", lf)
        if not a.bench_steps and s1 in save_at:
            d = os.path.join(adir, save_at[s1]); model.save_pretrained(d)
            json.dump(dict(step=s1, epoch=s1 / spe, loss_last50=float(np.mean(losses[-50:]))), open(os.path.join(d, "ckpt_info.json"), "w"))
            log(f"saved {save_at[s1]} at step {s1}", lf)
        if not a.bench_steps and (s1 % a.ckpt_every == 0) and s1 < total:
            torch.save(dict(params=[p_.detach().clone() for p_ in params], opt=opt.state_dict(), sch=sch.state_dict(), step=s1,
                            rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state()), ck + ".tmp")
            os.replace(ck + ".tmp", ck)
    el = time.time() - t0
    n_steps = total_run - step0
    stats.update(steps_per_epoch=spe, total_steps=total, steps_run=n_steps, train_s=round(el, 1),
                 s_per_step=el / max(1, n_steps), s_per_step_after3=float(np.mean(step_times[3:])) if len(step_times) > 3 else None,
                 padded_tok_per_s=ntok_pad / max(el, 1e-9), real_tok_per_s=ntok_real / max(el, 1e-9),
                 padded_tokens=ntok_pad, real_tokens=ntok_real, peak_mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
                 loss_first10=float(np.mean(losses[:10])), loss_last10=float(np.mean(losses[-10:])), wall_s=round(time.time() - t_start, 1))
    if a.bench_steps:
        model.save_pretrained(os.path.join(adir, "bench_adapter"))      # vLLM load test only; deleted by the driver
    elif os.path.exists(ck):
        os.remove(ck)
    json.dump(stats, open(os.path.join(adir, "train_stats.json"), "w"), indent=1)
    log(json.dumps({k_: stats[k_] for k_ in ("s_per_step", "padded_tok_per_s", "peak_mem_gb", "len_mean", "wall_s")}), lf)


if __name__ == "__main__":
    main()
