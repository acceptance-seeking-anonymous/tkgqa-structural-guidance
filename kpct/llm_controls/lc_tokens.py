#!/usr/bin/env python3
"""lc_tokens.py -- GPLAIN_1 vs G_1 prompt-token lengths and input-identity checks (amendment LC1; CPU, gold-free for
dev/test). For every k01_s0 bundle (train 3,868 with targets; dev_sel2000; test 54,584) it renders the frozen G_1 prompt
(gk_common.fit_g_prompt, order S) and the GPLAIN_1 prompt (gk_plain.fit_plain_prompt) with the Llama-3.1-8B-Instruct
chat template, and checks: same candidate multiset, same fact groups, same train target string and answer token ids.
Writes /workspace/tokens_gplain_vs_g.json and /workspace/tokens_gplain_vs_g_per_question.csv.gz."""
import argparse, csv, gzip, json, os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.environ.get("GK_CODE", "/kpct/guided/code"))
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
import gk_common as GC
import gk_plain

FIT_G = GC.fit_g_prompt          # captured before any patch (this script never patches)
TOK = None


def init():
    global TOK
    from transformers import AutoTokenizer
    TOK = AutoTokenizer.from_pretrained(GC.MODEL_8B)


def one(args):
    r, split = args
    b = GC.normalize_bundle(r, require_gold=(split == "train"))
    tgt = GC.g_target(b)[0] if split == "train" else None
    pg, ag, dg = FIT_G(TOK, b, tgt, order="S")
    pp, ap, dp = gk_plain.fit_plain_prompt(TOK, b, tgt)
    same_c = sorted(c["candidate"] for c in b["candidates"]) == sorted(gk_plain.plain_cands(b))
    fg = sorted((h, r_, t, tuple(ds)) for h, r_, t, ds in GC.group_facts(b["facts"])[0])
    fp = sorted((h, r_, t, tuple(ds)) for h, r_, t, ds in gk_plain.plain_groups(b["facts"]))
    return dict(quid=b["quid"], split=split, g_tok=len(pg), plain_tok=len(pp), g_drop=dg, plain_drop=dp,
                n_cands=len(b["candidates"]), n_facts=len(b["facts"]), n_groups=len(fg), same_cands=same_c, same_facts=fg == fp,
                same_answer_ids=(ag == ap) if split == "train" else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles_dir", default="/kpct/bundles/k01_s0"); ap.add_argument("--manifest", required=True)
    ap.add_argument("--workers", type=int, default=12); ap.add_argument("--out", default="/workspace")
    a = ap.parse_args()
    t0 = time.time()
    dsel = set(GC.load_list(a.manifest, "dev_sel2000"))
    rows = []
    for split in ("train", "dev_sel2000", "test"):
        p = next(os.path.join(a.bundles_dir, split + e) for e in (".jsonl", ".jsonl.gz") if os.path.exists(os.path.join(a.bundles_dir, split + e)))
        R = GC.read_jsonl(p)
        if split == "dev_sel2000":
            R = [r for r in R if int(r["quid"]) in dsel]
        print(split, len(R), p, GC.sha256_file(p), flush=True)
        with Pool(a.workers, initializer=init) as pool:
            rows += pool.map(one, [(r, split) for r in R], chunksize=64)
    summ = dict(script="lc_tokens.py", gk_plain_sha256=GC.sha256_file(gk_plain.__file__), gk_common_sha256=GC.sha256_file(GC.__file__),
                tokenizer=GC.MODEL_8B, chat_date=GC.CHAT_DATE, max_len=GC.MAX_LEN, secs=None, splits={})
    for split in ("train", "dev_sel2000", "test"):
        S = [r for r in rows if r["split"] == split]
        g = np.array([r["g_tok"] for r in S]); pl = np.array([r["plain_tok"] for r in S]); d = g - pl
        summ["splits"][split] = dict(
            n=len(S), g_tok_mean=float(g.mean()), g_tok_median=float(np.median(g)), g_tok_p95=float(np.percentile(g, 95)), g_tok_max=int(g.max()),
            plain_tok_mean=float(pl.mean()), plain_tok_median=float(np.median(pl)), plain_tok_p95=float(np.percentile(pl, 95)), plain_tok_max=int(pl.max()),
            diff_mean=float(d.mean()), diff_median=float(np.median(d)), diff_min=int(d.min()), diff_max=int(d.max()), ratio_mean=float((pl / g).mean()),
            g_truncated=int(sum(r["g_drop"] > 0 for r in S)), plain_truncated=int(sum(r["plain_drop"] > 0 for r in S)),
            same_cands_all=bool(all(r["same_cands"] for r in S)), same_facts_all=bool(all(r["same_facts"] for r in S)),
            same_answer_ids_all=(bool(all(r["same_answer_ids"] for r in S)) if split == "train" else None))
    summ["secs"] = round(time.time() - t0, 1)
    json.dump(summ, open(os.path.join(a.out, "tokens_gplain_vs_g.json"), "w"), indent=1)
    with gzip.open(os.path.join(a.out, "tokens_gplain_vs_g_per_question.csv.gz"), "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    main()
