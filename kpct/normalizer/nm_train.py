#!/usr/bin/env python3
"""nm_train.py -- LoRA training of the question normalizer N (rule multitq_normalizer_rule.json, section training).

Loop = frozen pk_train.py / gk_train.py (rule arms.G_k.lora_recipe): Llama-3.1-8B-Instruct; LoRA r 16, alpha 32,
dropout 0.05 on q,k,v,o,gate,up,down; AdamW lr 2e-4, wd 0, cosine, 3% warm-up, grad clip 1.0, effective batch 16, bf16;
loss on the answer tokens only (incl. <|eot_id|>); chat template, one user turn, fixed date_string; seed 0 = LoRA init
and data order (per-epoch permutation default_rng(1000*seed + 7 + epoch), length-sorted chunks of 64 batches, batch
order shuffled within a chunk). Micro-batches by padded-token budget --train_tok. Epochs E (default 2); adapters saved
at the registered epochs (0.5, 1, 2). Resume checkpoint every 500 optimizer steps.
--max_steps M --dry : throughput dry run (adapter discarded), writes dry_stats.json.
Also writes probe_prompt_ids.json (prompt token ids of the first 50 training inputs) for the inference-side check.
"""
import argparse, json, math, os, random, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nm_common as NC


def log(msg, fh=None):
    s = time.strftime("%H:%M:%S") + " " + msg
    print(s, flush=True)
    if fh:
        fh.write(s + "\n"); fh.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="N_k05_s0_s0")
    ap.add_argument("--epochs", type=float, default=NC.E_MAX)
    ap.add_argument("--train_tok", type=int, default=NC.TRAIN_TOK)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--model", default=NC.MODEL_8B)
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
    import transformers, peft
    from peft import LoraConfig, get_peft_model
    seed = NC.SEED
    E = a.epochs
    adir = os.path.join(a.out, a.name + ("_dry" if a.dry else "")); os.makedirs(adir, exist_ok=True)
    lf = open(os.path.join(adir, "train_log.txt"), "a")
    t_start = time.time()
    dev = torch.device("cuda:0")
    log(f"{a.name} torch {torch.__version__} transformers {transformers.__version__} peft {peft.__version__} "
        f"({torch.cuda.get_device_name(0)}) E={E} dry={a.dry}", lf)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    rows = [json.loads(l) for l in open(a.pairs) if l.strip()]
    assert len({r["ex_id"] for r in rows}) == len(rows)
    rows.sort(key=lambda r: r["ex_id"])
    tok = AutoTokenizer.from_pretrained(a.model)
    pad_id = tok.convert_tokens_to_ids("<|finetune_right_pad_id|>")
    ex, n_long = [], 0
    for r in rows:
        u = NC.n_user(r["input"])
        p = NC.encode_prompt(tok, u)
        ans = NC.encode_answer(tok, u, r["target"], p)
        if len(p) + len(ans) > NC.MAX_LEN:
            n_long += 1; continue
        ex.append(dict(ex_id=int(r["ex_id"]), p=p, a=ans))
    probe = [dict(input=r["input"], ids=NC.encode_prompt(tok, NC.n_user(r["input"]))) for r in rows[:50]]
    json.dump(probe, open(os.path.join(adir, "probe_prompt_ids.json"), "w"))
    Ls = np.array([len(e["p"]) + len(e["a"]) for e in ex])
    log(f"examples {len(ex)} (rows {len(rows)}, over max_len {n_long}) len mean {Ls.mean():.1f} p95 "
        f"{np.percentile(Ls, 95):.0f} max {Ls.max()} answer tok mean {np.mean([len(e['a']) for e in ex]):.1f}", lf)
    stats = dict(name=a.name, n_rows=len(rows), n_examples=len(ex), n_over_max_len=n_long, epochs=E, lora=NC.LORA,
                 opt=NC.OPT, max_len=NC.MAX_LEN, chat_date=NC.CHAT_DATE, train_tok=a.train_tok, seed=seed,
                 prompt_sha256=NC.prompt_sha256(), model=a.model, pairs_sha256=NC.sha256_file(a.pairs),
                 versions=dict(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__),
                 len_mean=float(Ls.mean()), len_p95=float(np.percentile(Ls, 95)), len_max=int(Ls.max()),
                 answer_tokens_mean=float(np.mean([len(e["a"]) for e in ex])), dry_run=a.dry)
    t_l0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev)
    torch.manual_seed(seed)
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", bias="none", **NC.LORA))
    stats["load_s"] = round(time.time() - t_l0, 1)
    inner, head = model.get_base_model().model, model.get_base_model().lm_head
    params = [p for p in model.parameters() if p.requires_grad]
    B = NC.OPT["eff_batch"]
    spe = math.ceil(len(ex) / B)
    total = int(math.ceil(E * spe))
    total_run = min(total, a.max_steps) if a.max_steps else total
    opt = torch.optim.AdamW(params, lr=NC.OPT["lr"], weight_decay=NC.OPT["weight_decay"])
    sch = get_cosine_schedule_with_warmup(opt, int(math.ceil(NC.OPT["warmup"] * total)), total)
    plan = []
    for ep in range(int(math.ceil(E))):
        rng = np.random.default_rng(1000 * seed + 7 + ep); perm = rng.permutation(len(ex)); CH = B * 64; ep_plan = []
        for c0 in range(0, len(perm), CH):
            ch = sorted(perm[c0:c0 + CH], key=lambda i: Ls[i])
            bs = [ch[j:j + B] for j in range(0, len(ch), B)]
            for j in rng.permutation(len(bs)):
                ep_plan.append(bs[j])
        plan += ep_plan
    plan = plan[:total]
    stats["data_order_sha256"] = NC.sha256_text(json.dumps([[int(i) for i in g] for g in plan]))
    save_at = {int(round(e_ * spe)): f"ep{e_:g}" for e_ in NC.SAVE_AT_EPOCHS if e_ <= E}
    save_at[total] = f"ep{E:g}"
    ck = os.path.join(adir, "ckpt.pt"); step0 = 0
    if os.path.exists(ck) and not a.dry:
        st = torch.load(ck, map_location=dev, weights_only=False)
        for p_, v in zip(params, st["params"]):
            p_.data.copy_(v)
        opt.load_state_dict(st["opt"]); sch.load_state_dict(st["sch"]); step0 = st["step"]
        torch.set_rng_state(st["rng_cpu"]); torch.cuda.set_rng_state(st["rng_cuda"])
        log(f"resumed at step {step0}", lf)
    model.train(); t0 = time.time(); losses = []; ntok_pad = ntok_real = 0
    torch.cuda.reset_peak_memory_stats()
    for step in range(step0, total_run):
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
            ids, att = ids.to(dev), att.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hs = inner(input_ids=ids, attention_mask=att).last_hidden_state
                h_sel = hs[torch.tensor(pos_b, device=dev), torch.tensor(pos_t, device=dev)]
                logits = head(h_sel).float()
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor(tgt, device=dev), reduction="sum") / n_ans
            loss.backward(); tot += float(loss)
        torch.nn.utils.clip_grad_norm_(params, NC.OPT["grad_clip"])
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        losses.append(tot)
        s1 = step + 1
        if s1 % 50 == 0 or s1 == total_run or (a.dry and s1 % 10 == 0):
            el = time.time() - t0
            log(f"step {s1}/{total} loss {np.mean(losses[-50:]):.4f} {el / max(1, s1 - step0):.3f}s/it "
                f"{ntok_pad / el:.0f} padded tok/s lr {sch.get_last_lr()[0]:.2e} peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GB", lf)
        if not a.dry and s1 in save_at:
            d = os.path.join(adir, save_at[s1]); model.save_pretrained(d)
            json.dump(dict(step=s1, epoch=s1 / spe, loss_last50=float(np.mean(losses[-50:]))), open(os.path.join(d, "ckpt_info.json"), "w"))
            log(f"saved {save_at[s1]} at step {s1}", lf)
        if not a.dry and (s1 % a.ckpt_every == 0) and s1 < total:
            torch.save(dict(params=[p_.detach().clone() for p_ in params], opt=opt.state_dict(), sch=sch.state_dict(), step=s1,
                            rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state()), ck + ".tmp")
            os.replace(ck + ".tmp", ck)
    el = time.time() - t0
    n_steps = total_run - step0
    stats.update(steps_per_epoch=spe, total_steps=total, steps_run=n_steps, save_at=save_at, train_s=round(el, 1),
                 s_per_step=el / max(1, n_steps), padded_tok_per_s=ntok_pad / max(el, 1e-9),
                 peak_mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
                 loss_first10=float(np.mean(losses[:10])) if losses else None,
                 loss_last10=float(np.mean(losses[-10:])) if losses else None, wall_s=round(time.time() - t_start, 1))
    if a.dry:
        stats["projected_full_train_s"] = stats["s_per_step"] * total
        json.dump(stats, open(os.path.join(adir, "dry_stats.json"), "w"), indent=1)
    else:
        if os.path.exists(ck):
            os.remove(ck)
        stats["adapters"] = {}
        for dname in sorted(os.listdir(adir)):
            f = os.path.join(adir, dname, "adapter_model.safetensors")
            if os.path.exists(f):
                stats["adapters"][dname] = NC.sha256_file(f)
        json.dump(stats, open(os.path.join(adir, "train_stats.json"), "w"), indent=1)
    os.system(f"chmod -R a+rX {adir}")
    log(json.dumps({k_: stats.get(k_) for k_ in ("n_examples", "total_steps", "steps_run", "s_per_step", "peak_mem_gb", "wall_s", "projected_full_train_s")}), lf)


if __name__ == "__main__":
    main()
