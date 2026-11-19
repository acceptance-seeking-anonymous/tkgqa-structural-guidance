#!/usr/bin/env python3
"""gk_run_arm.py -- one vLLM session per arm: dev_sel2000 checkpoint scoring -> selection -> freeze -> test predictions.

Order of operations inside the session (rule selection_discipline / test_discipline):
  1. dev_sel2000 generations for every checkpoint of every run (e.g. ordS and ordrandom at k = 1 subset seed 0)
  2. official-scorer Hits@1 per checkpoint, appended to the selection log as computed; best checkpoint per run
     (tie -> earlier); for G_1 the candidate order is decided once here (|diff| < 0.0005 -> S order)
  3. freeze_<arm>.json written (adapter sha256, selection, prompt and code sha256, bundle sha256) BEFORE any test prompt
  4. test generations with the frozen adapter -> pred_*.jsonl (+ raw), sha256 added to the freeze; test is NOT scored
  5. optional controls with the same frozen adapter (G_1 only): SHUF and QONLY on dev_sel2000 and test
     (control dev numbers are logged as qualifiers, never used for selection)
--no_test: dev-only session (e.g. if the order grid is run separately). STOP_DEV_FAILURE is flagged in the freeze;
the one logged bug-fix iteration is a separate, manual amendment.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gk_common as GC
import gk_infer as GI
import gk_select as GS


def ckpts(run_dir):
    eps = sorted([d for d in os.listdir(run_dir) if d.startswith("ep") and os.path.exists(os.path.join(run_dir, d, "adapter_model.safetensors"))],
                 key=lambda d: float(d[2:]))
    assert eps, f"no checkpoints in {run_dir}"
    return [(d, os.path.join(run_dir, d)) for d in eps]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["G", "GU"]); ap.add_argument("--list", required=True)
    ap.add_argument("--runs", required=True, help="name:order:run_dir[,name:order:run_dir]  (order S|random; GU: S)")
    ap.add_argument("--bundles_dir", default=None, help="G: dir with dev_sel2000/test bundle files")
    ap.add_argument("--gu_dev", default=None); ap.add_argument("--gu_test", default=None)
    ap.add_argument("--dev_q", required=True); ap.add_argument("--test_q", required=True)
    ap.add_argument("--s_list_dev", default=None); ap.add_argument("--s_list_test", default=None)
    ap.add_argument("--controls", default="", help="SHUF,QONLY (G_1 only)"); ap.add_argument("--controls_test", type=int, default=1)
    ap.add_argument("--no_test", action="store_true")
    ap.add_argument("--manifest", required=True); ap.add_argument("--eval", required=True); ap.add_argument("--kg", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--log", default=GS.LOG); ap.add_argument("--tag", default="")
    ap.add_argument("--max_model_len", type=int, default=2304); ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run even if this arm's session is complete")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fz0 = os.path.join(a.out, f"freeze_{a.arm}_{a.list}.json"); ss0 = os.path.join(a.out, f"session_{a.arm}_{a.list}{a.tag}.json")
    if not a.force and os.path.exists(fz0) and os.path.exists(ss0) and (a.no_test or json.load(open(fz0)).get("test_predictions")):
        print(f"session {a.arm} {a.list} already complete ({ss0}); nothing to do", flush=True)
        return
    t_start = time.time()
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    kg = GC.load_kg(a.kg)
    enorm = GC.entity_norm_map(sorted({f[0] for f in kg} | {f[2] for f in kg}))
    dsel = set(GC.load_list(a.manifest, "dev_sel2000"))
    llm = LLM(model=GC.MODEL_8B, dtype="bfloat16", enable_lora=True, max_lora_rank=16, max_loras=1,
              max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem, seed=0, enforce_eager=a.eager)
    tok = llm.get_tokenizer()
    t_init = time.time() - t_start
    runs = [r.split(":", 2) for r in a.runs.split(",")]
    sess = dict(arm=a.arm, list=a.list, init_s=round(t_init, 1), runs={}, generations=[])

    def bfile(split):
        for ext in (".jsonl", ".jsonl.gz"):
            p = os.path.join(a.bundles_dir, split + ext)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"{a.bundles_dir}/{split}.jsonl")

    def prompts(arm, split, order):
        keep = dsel if split == "dev_sel2000" else None
        if a.arm == "G":
            return GI.make_prompts(tok, arm, split_rows=GC.read_jsonl(bfile(split)), order=order, keep=keep)
        qf = a.dev_q if split == "dev_sel2000" else a.test_q
        gu = {int(r["quid"]): [tuple(f) for f in r["facts"]] for r in GC.read_jsonl(a.gu_dev if split == "dev_sel2000" else a.gu_test)}
        Q = [dict(quid=q["quid"], question=q["question"]) for q in json.load(open(qf))]      # question text only
        return GI.make_prompts(tok, "GU", questions=Q, gu=gu, keep=keep)

    ns = argparse.Namespace(out=a.out, split=None, tag=a.tag)
    lid = [0]

    shard_dir = os.path.join(a.out, "shards"); os.makedirs(shard_dir, exist_ok=True)
    SHARD = 5000

    def gen_write(style, split, aname, apath, P, s_list_path, pad):
        """generation in shards of 5,000 prompts cached on disk (STOP_HOST: a restarted session resumes from the shards)"""
        quids, pids, extra, info = P
        lid[0] += 1
        lr = LoRARequest(aname, lid[0], apath)
        texts, n_out, secs, n_gen = [], [], 0.0, 0
        for si, s0 in enumerate(range(0, len(pids), SHARD)):
            sp = os.path.join(shard_dir, f"{style}_{split}_{aname}{a.tag}_{si:03d}.json")
            if os.path.exists(sp):
                d = json.load(open(sp))
                assert d["quids"] == quids[s0:s0 + SHARD] and d["adapter_sha256"] == GC.sha256_file(os.path.join(apath, "adapter_model.safetensors"))
            else:
                outs, st0 = GI.generate(llm, pids[s0:s0 + SHARD], GC.MAX_NEW_G, lr)
                d = dict(quids=quids[s0:s0 + SHARD], text=[o.outputs[0].text for o in outs], n_out=[len(o.outputs[0].token_ids) for o in outs],
                         seconds=st0["seconds"], adapter_sha256=GC.sha256_file(os.path.join(apath, "adapter_model.safetensors")))
                json.dump(d, open(sp + ".tmp", "w")); os.replace(sp + ".tmp", sp); n_gen += len(d["quids"])
            texts += d["text"]; n_out += d["n_out"]; secs += d["seconds"]
        import numpy as np
        st = dict(n=len(pids), seconds=round(secs, 2), prompts_per_s=len(pids) / max(secs, 1e-9), prompt_tok_mean=float(np.mean([len(p) for p in pids])),
                  out_tok_mean=float(np.mean(n_out)), n_generated_this_session=n_gen)

        class _O:
            def __init__(self, t, n):
                self.text, self.token_ids = t, [0] * n

        class _R:
            def __init__(self, t, n):
                self.outputs = [_O(t, n)]
        outs = [_R(t, n) for t, n in zip(texts, n_out)]
        raw, preds, inv = GI.postprocess(quids, outs, enorm)
        st["invalid_rate"] = inv; st.update(info)
        s_lists = GI.load_s_lists(s_list_path, set(quids)) if s_list_path else None
        GI.write_outputs(ns, style, aname, raw, preds, extra, s_lists, pad, st, split=split)
        sess["generations"].append(dict(style=style, split=split, adapter=aname, **st))
        print(time.strftime("%H:%M:%S"), style, split, aname, f"{st['prompts_per_s']:.1f}/s invalid {inv:.4f}", flush=True)
        return os.path.join(a.out, st["pred_file"])

    # ---------------------------------------------------------------- 1-2 dev_sel2000 + selection
    sel = {}
    for name, order, rdir in runs:
        pad_p = os.path.join(rdir, "gu_pad.json")
        pad = json.load(open(pad_p)) if os.path.exists(pad_p) else []
        P = prompts(a.arm, "dev_sel2000", order)
        cands = []
        for ep, path in ckpts(rdir):
            cands.append(f"{ep}=" + gen_write(a.arm, "dev_sel2000", f"{name}_{ep}", path, P, a.s_list_dev, pad))
        so = os.path.join(a.out, f"selection_{a.arm}_{a.list}_{name}.json")
        GS.cmd_select(argparse.Namespace(cands=",".join(cands), dev=a.dev_q, manifest=a.manifest, eval=a.eval, arm=a.arm,
                                         list=a.list, run=name, out=so, log=a.log))
        sel[name] = (so, order, rdir, pad)
    if len(runs) > 1:
        (n1, o1, *_), (n2, o2, *_) = runs[0], runs[1]
        S_name, R_name = (n1, n2) if o1 == "S" else (n2, n1)
        od = os.path.join(a.out, f"order_decision_{a.list}.json")
        GS.cmd_order(argparse.Namespace(sel_S=sel[S_name][0], sel_random=sel[R_name][0], out=od, log=a.log))
        chosen = S_name if json.load(open(od))["chosen"] == "S" else R_name
    else:
        chosen = runs[0][0]
    so, order, rdir, pad = sel[chosen]
    S = json.load(open(so))
    ad = os.path.join(rdir, S["selected"])
    # ---------------------------------------------------------------- 3 freeze before any test prompt
    fz = os.path.join(a.out, f"freeze_{a.arm}_{a.list}.json")
    inputs = [bfile("dev_sel2000"), bfile("test")] if a.arm == "G" else [a.gu_dev, a.gu_test]
    GS.cmd_freeze(argparse.Namespace(out=fz, arm=a.arm, list=a.list, run=chosen, adapter=ad, selection=so, inputs=",".join(inputs),
                                     order=order, pred=None, raw=None))
    fzd = json.load(open(fz)); fzd["DEV_FAILURE_trigger"] = S["DEV_FAILURE_trigger"]; json.dump(fzd, open(fz, "w"), indent=1)
    if a.no_test:
        sess["note"] = "dev-only session"
    else:
        # ------------------------------------------------------------ 4 test predictions (not scored)
        aname = f"{chosen}_{S['selected']}"
        pt = gen_write(a.arm, "test", aname, ad, prompts(a.arm, "test", order), a.s_list_test, pad)
        GS.cmd_freeze(argparse.Namespace(out=fz, pred=pt, raw=pt.replace("/pred_", "/raw_").replace(".jsonl", ".jsonl.gz"), arm=None, list=None,
                                         run=None, adapter=None, selection=None, inputs="", order=None))
        # ------------------------------------------------------------ 5 controls (G_1)
        for ctl in [c for c in a.controls.split(",") if c]:
            pdv = gen_write(ctl, "dev_sel2000", aname, ad, prompts(ctl, "dev_sel2000", order), a.s_list_dev, pad)
            GS.cmd_select(argparse.Namespace(cands=f"{ctl}={pdv}", dev=a.dev_q, manifest=a.manifest, eval=a.eval, arm=f"{ctl}_control",
                                             list=a.list, run=aname, out=os.path.join(a.out, f"control_{ctl}_{a.list}_dev.json"), log=a.log))
            if a.controls_test:
                ptc = gen_write(ctl, "test", aname, ad, prompts(ctl, "test", order), a.s_list_test, pad)
                GS.cmd_freeze(argparse.Namespace(out=fz, pred=ptc, raw=ptc.replace("/pred_", "/raw_").replace(".jsonl", ".jsonl.gz"), arm=None,
                                                 list=None, run=None, adapter=None, selection=None, inputs="", order=None))
    sess["wall_s"] = round(time.time() - t_start, 1)
    json.dump(sess, open(os.path.join(a.out, f"session_{a.arm}_{a.list}{a.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
