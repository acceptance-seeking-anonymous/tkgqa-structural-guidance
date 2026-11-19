#!/usr/bin/env python3
"""abl_run.py -- G_1 component ablation, one vLLM session on one card (rule multitq_g1_ablation_rule.json).

Phase SEL : dev_sel2000, FROZEN bundles + frozen GU facts, all variants; gate S4 (A_full reproduces the frozen G_1).
Phase MAIN: D_main blocks of 2,000 quids in the registered order, rebuilt bundles, all variants (A_GUfacts only if S3);
            gates S2 (bundle identity, from abl_build_identity.json), S3 (GU identity), S5, GPU-hour cap S6, yield S7.
Raw generations are written per block (resumable). Never reads an answer except dev_sel2000 answers for the S4 gate.
"""
import argparse, gzip, json, os, sys, time
T_START = time.time()
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gk_common as GC
import gk_ablation as AB

CAP, RESERVE, BLOCK = 4.0, 0.10, 2000
ADAPTER_SHA = "4d81d7a017e391af0c8118e55c9e707b53ca9b607cc397572832c7da54a3440a"


def log(*m):
    print(time.strftime("%H:%M:%S"), *m, flush=True)


def usage_log_gpu_h(path, track="ablation_g1"):
    s = 0.0
    if os.path.exists(path):
        for l in open(path):
            p = l.rstrip("\n").split("\t")
            if len(p) >= 7 and p[0] == track:
                s += float(p[6])
    return s


def load_gu(path):
    out = {}
    for x in GC.read_jsonl(path):
        out[int(x["quid"])] = [tuple(f) for f in x["facts"]], x["fact_ids"]
    return out


def s_lists(path, keep):
    out = {}
    for r in GC.read_jsonl(path):
        q = int(r["quid"])
        if q in keep:
            rk = r["ranked"]
            out[q] = json.loads(rk.replace("'", '"')) if isinstance(rk, str) else rk
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--root", required=True, help="persistent ablation dir (/kpct/ablation_g1)")
    ap.add_argument("--sel_bundles", required=True); ap.add_argument("--sel_gu", required=True)
    ap.add_argument("--sel_s0", required=True); ap.add_argument("--sel_frozen_pred", required=True)
    ap.add_argument("--sel_quids", required=True); ap.add_argument("--order", required=True)
    ap.add_argument("--main_bundles", required=True); ap.add_argument("--main_s0", required=True)
    ap.add_argument("--main_gu", required=True); ap.add_argument("--build_identity", required=True)
    ap.add_argument("--dev_q", required=True); ap.add_argument("--kg", required=True); ap.add_argument("--eval", required=True)
    ap.add_argument("--usage_log", required=True); ap.add_argument("--out", default=".")
    ap.add_argument("--max_model_len", type=int, default=2304); ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--dry", action="store_true", help="CPU dry run: HF tokenizer, stub generation, first main block only")
    a = ap.parse_args()
    R = a.root if not a.dry else os.path.join(a.root, "dry"); os.makedirs(f"{R}/gen", exist_ok=True)
    prog_p = os.path.join(a.out, "abl_run_progress.json")
    prog = dict(started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(T_START)), gates={}, blocks=[], stop=None)

    def save():
        prog["elapsed_s"] = round(time.time() - T_START, 1)
        json.dump(prog, open(prog_p + ".tmp", "w"), indent=1); os.replace(prog_p + ".tmp", prog_p)
        json.dump(prog, open(f"{R}/abl_run_progress.json", "w"), indent=1)

    # ---------------------------------------------------------------- S1
    sha = GC.sha256_file(os.path.join(a.adapter, "adapter_model.safetensors"))
    prog["gates"]["S1_adapter_sha256"] = dict(got=sha, ok=sha == ADAPTER_SHA)
    if sha != ADAPTER_SHA:
        prog["stop"] = "S1"; save(); sys.exit(2)
    # ---------------------------------------------------------------- S3 (GU identity on dev_sel2000)
    sel = sorted(int(q) for q in json.load(open(a.sel_quids)))
    gu_sel = load_gu(a.sel_gu); gu_main = load_gu(a.main_gu)
    same = sum(gu_main[q][1] == gu_sel[q][1] for q in sel)
    s3 = same / len(sel)
    prog["gates"]["S3_gu_identity_dev_sel2000"] = dict(identical_ordered_top20=s3, ok=s3 >= 0.99)
    bid = json.load(open(a.build_identity))
    prog["gates"]["S2_bundle_identity"] = dict(rendered=bid["rendered_user_turn_identity"], s0=bid["field_identity"]["s0_list"], ok=bool(bid["S2_pass"]))
    save()
    # ---------------------------------------------------------------- engine
    import multitq_eval as ME
    kg = GC.load_kg(a.kg)
    enorm = GC.entity_norm_map(sorted({f[0] for f in kg} | {f[2] for f in kg}))
    if a.dry:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(GC.MODEL_8B); lr = None

        class _O:
            def __init__(self):
                self.text, self.token_ids = "", []

        class _R:
            outputs = [_O()]

        def generate(llm_, pids, mx, lr_):
            return [_R() for _ in pids], dict(seconds=1.0, prompts_per_s=len(pids), prompt_tok_mean=float(np.mean([len(x) for x in pids])), prompt_tok_per_s=0.0)
        llm = None
    else:
        from vllm import LLM
        from vllm.lora.request import LoRARequest
        llm = LLM(model=GC.MODEL_8B, dtype="bfloat16", enable_lora=True, max_lora_rank=16, max_loras=1,
                  max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem, seed=0)
        tok = llm.get_tokenizer()
        lr = LoRARequest("G_k01_s0_ordS_ep2", 1, a.adapter)
        generate = __import__("gk_infer").generate
    prog["init_s"] = round(time.time() - T_START, 1); save()

    def run(bundles, gu, variants, tag):
        """-> dict with per-variant lists and timing; one generate call for all variants"""
        meta, pids = [], []
        t0 = time.time()
        for b in bundles:
            for v in variants:
                p, dropped, shown = AB.fit_ab_prompt(tok, b, v, gu.get(b["quid"], (None,))[0] if v == "A_GUfacts" else None)
                meta.append((b["quid"], v, dropped, shown)); pids.append(p)
        t_tok = time.time() - t0
        outs, st = generate(llm, pids, GC.MAX_NEW_G, lr)
        rows = []
        for (q, v, dr, sh), p, o in zip(meta, pids, outs):
            txt = o.outputs[0].text
            ans, ok = GC.canon_answer(txt, enorm)
            rows.append(dict(quid=q, variant=v, text=txt, answer=ans, valid=bool(ok), n_out=len(o.outputs[0].token_ids),
                             prompt_len=len(p), groups_dropped=dr, groups_shown=sh))
        return rows, dict(tag=tag, n_prompts=len(pids), tokenize_s=round(t_tok, 1), gen_s=st["seconds"],
                          prompts_per_s=st["prompts_per_s"], prompt_tok_mean=st["prompt_tok_mean"], prompt_tok_per_s=st["prompt_tok_per_s"])

    def write_block(path, rows, st):
        with gzip.open(path + ".tmp", "wt") as f:
            json.dump(dict(stats=st, adapter_sha256=sha, rows=rows), f)
        os.replace(path + ".tmp", path)

    # ---------------------------------------------------------------- phase SEL
    sp = f"{R}/gen/sel_dev_sel2000.json.gz"
    Bsel = sorted([GC.normalize_bundle(r) for r in GC.read_jsonl(a.sel_bundles)], key=lambda b: b["quid"])
    assert [b["quid"] for b in Bsel] == sel
    # A_full prompt ids == the frozen G prompt construction
    n_same = sum(AB.fit_ab_prompt(tok, b, "A_full")[0] == GC.fit_g_prompt(tok, b, order="S")[0] for b in Bsel)
    prog["gates"]["A_full_prompt_ids_equal_fit_g_prompt"] = n_same / len(Bsel)
    assert n_same == len(Bsel)
    if os.path.exists(sp):
        d = json.load(gzip.open(sp, "rt")); rows, st = d["rows"], d["stats"]; assert d["adapter_sha256"] == sha
    else:
        rows, st = run(Bsel, gu_sel, AB.VARIANTS, "dev_sel2000")
        st["elapsed_at_end_s"] = round(time.time() - T_START, 1); write_block(sp, rows, st)
    prog["sel"] = st
    sel_rate = st["gen_s"] / len(Bsel)          # seconds per question, all variants
    frz = {}
    for x in GC.read_jsonl(a.sel_frozen_pred):
        rk = x["ranked"]; rk = json.loads(rk.replace("'", '"')) if isinstance(rk, str) else rk
        frz[int(x["quid"])] = rk[0]
    s0sel = s_lists(a.sel_s0, set(sel))
    Q = {q["quid"]: q for q in json.load(open(a.dev_q)) if q["quid"] in set(sel)}
    full = {r["quid"]: r for r in rows if r["variant"] == "A_full"}
    agree = float(np.mean([full[q]["answer"] == frz[q] for q in sel]))
    h1 = float(np.mean([ME.hit_at_k(Q[q], GC.g_ranked(full[q]["answer"], s0sel[q]), 1) for q in sel]))
    s4 = agree >= 0.99 and abs(h1 - 0.7990) <= 0.005
    prog["gates"]["S4_A_full_reproduces_frozen"] = dict(rank1_agreement=agree, hits1=h1, frozen_hits1=0.7990, ok=bool(s4))
    save()
    log("SEL", json.dumps(prog["gates"]["S4_A_full_reproduces_frozen"]), f"{sel_rate:.3f} s/question")
    if not s4 and not a.dry:
        prog["stop"] = "S4"; save(); sys.exit(0)
    if not bid["S2_pass"]:
        prog["stop"] = "S2 (D_main NOT_RUN)"; save(); sys.exit(0)
    # ---------------------------------------------------------------- phase MAIN
    order = json.load(open(a.order))
    variants = [v for v in AB.VARIANTS if v != "A_GUfacts" or s3 >= 0.99]
    keep = set(order)
    Bm = {}
    for r in GC.read_jsonl(a.main_bundles):
        if int(r["quid"]) in keep:
            b = GC.normalize_bundle(r); Bm[b["quid"]] = b
    excluded = sorted(keep - set(Bm))            # frozen bundle builder raised (abl_build_identity.json builder_errors)
    assert set(excluded) <= {int(e["quid"]) for e in bid["builder_errors"]["rows"]}, "missing bundles not explained by builder errors"
    prog["excluded_builder_errors"] = excluded
    blocks = [[q for q in order[i:i + BLOCK] if q in Bm] for i in range(0, len(order), BLOCK)]   # registered block membership
    done_s = []
    for bi, bq in enumerate(blocks):
        bp = f"{R}/gen/main_block_{bi:03d}.json.gz"
        if os.path.exists(bp):
            d = json.load(gzip.open(bp, "rt"))
            assert d["adapter_sha256"] == sha and [r["quid"] for r in d["rows"][::len(variants)]] == bq
            prog["blocks"].append(dict(block=bi, n=len(bq), resumed=True, **d["stats"])); save()
            continue
        if os.path.exists(f"{R}/YIELD"):
            prog["stop"] = f"S7 yield before block {bi}"; save(); break
        spent = (usage_log_gpu_h(a.usage_log) if not a.dry else 0.0) + (time.time() - T_START) / 3600
        proj = (np.mean(done_s) if done_s else sel_rate * len(bq) * 1.05) / 3600
        prog["gpu_cap_check"] = dict(block=bi, spent_gpu_h=round(spent, 4), next_block_gpu_h=round(proj, 4), cap=CAP, reserve=RESERVE)
        if spent + proj + RESERVE > CAP:
            prog["stop"] = f"S6 GPU-hour cap before block {bi}"; save(); break
        t0 = time.time()
        rows, st = run([Bm[q] for q in bq], gu_main, variants, f"main_block_{bi:03d}")
        st["block_wall_s"] = round(time.time() - t0, 1); st["elapsed_at_end_s"] = round(time.time() - T_START, 1)
        write_block(bp, rows, st)
        done_s.append(st["block_wall_s"])
        inv = float(np.mean([not r["valid"] for r in rows if r["variant"] == "A_full"]))
        prog["blocks"].append(dict(block=bi, n=len(bq), A_full_invalid=inv, **st)); save()
        log(f"block {bi} {len(bq)} q {st['block_wall_s']} s {st['prompts_per_s']:.1f} prompts/s A_full invalid {inv:.4f}")
        if a.dry:
            prog["stop"] = "dry run: first block only"; save(); break
        if bi == 0 and inv > 0.02:
            prog["stop"] = "S5"; save(); break
    else:
        prog["stop"] = "complete"
    prog["variants_main"] = variants
    save()


if __name__ == "__main__":
    main()
