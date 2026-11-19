#!/usr/bin/env python3
"""pk_silver.py -- answer-consistent silver parse targets for P_k (rule arms.P_k.targets).

Target = the k%-trained structural parser's top-1 hypothesis on each question of the arm's LLM train list
(bundles/<subset>/train.jsonl: in_sample_parse.top1_parse = the full-subset, non-OOF parser), serialised to the compact
P JSON (pk_common.compact_from_parse), kept iff EXECUTING THAT SERIALISED JSON (kpct_executor.compact_to_parse with no
structural parse appended -> operators + RULE ordering) returns a gold answer at rank 1 under official matching
(multitq_eval.hit_at_k). k = 100: the frozen v3 parser on reader_train60k (in_sample_parse of the k100 bundles; if a
row carries none, Executor('k100').parse_question recomputes it), same filter.

Also reported per row: the structural track's own flags (executes_to_gold for the top-1 parse in executor schema,
executes_to_gold_full_parse) and whether the serialised JSON reproduces the structural top-1 RULE top-10 exactly
(serialisation fidelity). Train answers are read only for quids of the registered list (asserted).

CPU only. Run inside tkgqa-gpu:latest with /kpct mounted:
  python3 pk_silver.py --subset k01_s0 --list k01_s0 --out /kpct/parser/silver --workers 8
"""
import argparse, collections, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pk_common as PC

EXEC_DIR = os.environ.get("KPCT_EXECUTOR", "/kpct/executor")
sys.path.insert(0, EXEC_DIR)
_W = {}


def top1_of(row, k):
    isp = row.get("in_sample_parse") or {}
    if isp.get("top1_parse"):
        return isp["top1_parse"], "in_sample_parse.top1_parse"
    import kpct_executor as X
    full = isp.get("parse") or X.get(k).parse_question(row["question"], row["quid"])
    p = json.loads(json.dumps(full))
    p["relations"] = p["relations"][:1]
    for m in p["mentions"]:
        m["alternatives"] = m["alternatives"][:1]
    return p, ("in_sample_parse.parse[top-1]" if isp.get("parse") else "Executor.parse_question[top-1]")


def work(rows):
    import kpct_executor as X
    import kpct_core as K
    MC, ME = K.MC, K.ME
    ex = X.get(_W["k"])
    kg = K.kg()
    out = []
    for r in rows:
        q = r["question"]
        gold = {"answers": r["gold"], "answer_type": r["gold_meta"]["answer_type"]}
        o = dict(quid=int(r["quid"]), question=q, qtype=r["gold_meta"]["qtype"], answer_type=r["gold_meta"]["answer_type"],
                 qlabel=r["gold_meta"].get("qlabel"))
        isp = r.get("in_sample_parse") or {}
        o["flag_executes_to_gold"] = isp.get("executes_to_gold")
        o["flag_executes_to_gold_full_parse"] = isp.get("executes_to_gold_full_parse")
        try:
            top1, src = top1_of(r, _W["k"])
            o["top1_source"] = src
            rec, info = K.rec_info_from_parse(top1, q)
            anchored = info["qt"] in ("after_first", "before_last") or (info["qt"] in ("before_after", "equal_multi") and info["tp"] is None)
            ai = MC.anchor_index(rec["ws"], rec["ments"], q) if anchored else None
            anc = kg.ent_names[rec["ments"][ai][2][0][0]] if (ai is not None and rec["ments"][ai][2]) else None
            c = PC.compact_from_parse(top1, q, anc)
            tgt = PC.dumps_compact(c)
            o["target"] = tgt
            c2 = PC.validate_compact(json.loads(tgt))
            cp = X.compact_to_parse(c2, q, None)
            res = ex.execute(cp, question=q, probs="given")
            o["rt_valid"] = bool(res["valid"]); o["rt_error"] = res["error"]; o["rt_pool"] = res["pool_size"]
            o["rt_hit1"] = int(ME.hit_at_k(gold, res["rule"], 1)) if res["valid"] else 0
            ref = isp.get("rule_top10_top1")
            o["rt_top10_identical"] = (res["rule"][:10] == ref) if ref is not None else None
            o["rt_top1_identical"] = (res["rule"][:1] == ref[:1]) if ref is not None else None
            o["n_mentions"] = len(c["mentions"]); o["has_anchor"] = anc is not None; o["order"] = c["order"]
            o["n_alt_top1"] = max([len(m["alternatives"]) for m in top1["mentions"]] or [0])
        except Exception as e_:          # PInvalid / ParseError / anything else: recorded, row dropped
            o.update(target=o.get("target"), rt_valid=False, rt_error=f"{type(e_).__name__}: {e_}", rt_hit1=0)
        o["kept"] = bool(o.get("rt_hit1") == 1)
        out.append(o)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True, help="structural subset whose parser/executor produced the parses (k01_s0 ...)")
    ap.add_argument("--list", default=None, help="registered LLM train list (default = --subset; k100 -> reader_train60k)")
    ap.add_argument("--bundles", default=None, help="default /kpct/bundles/<subset>/train.jsonl")
    ap.add_argument("--manifest", default="/kpct/parser/code/multitq_kpct_subsets.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="dry-run: first N list quids only")
    a = ap.parse_args()
    t0 = time.time()
    lst = a.list or a.subset
    bfile = a.bundles or f"/kpct/bundles/{a.subset}/train.jsonl"
    keep = PC.load_list(a.manifest, lst)
    keep_s = set(keep)
    rows = [r for r in PC.iter_jsonl(bfile) if int(r["quid"]) in keep_s]
    got = {int(r["quid"]) for r in rows}
    assert got == keep_s, f"bundle rows {len(got)} != list {lst} ({len(keep_s)}); missing {len(keep_s - got)}"
    assert all(r.get("gold") for r in rows), "train bundle rows without gold"
    rows.sort(key=lambda r: int(r["quid"]))
    if a.limit:
        rows = rows[:a.limit]
    import kpct_executor as X
    _W["k"] = a.subset
    X.get(a.subset)
    chunks = [rows[i:i + 50] for i in range(0, len(rows), 50)]
    if a.workers <= 1:
        res = [x for ch in chunks for x in work(ch)]
    else:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.workers) as pool:
            res = [x for r in pool.map(work, chunks) for x in r]
    os.makedirs(a.out, exist_ok=True)
    tag = f"{lst}" + (f"_from_{a.subset}" if lst != a.subset else "") + (f"_dry{a.limit}" if a.limit else "")
    f_all = os.path.join(a.out, f"silver_{tag}_all.jsonl")
    f_tr = os.path.join(a.out, f"silver_{tag}_train.jsonl")
    PC.write_jsonl(f_all, res)
    kept = [dict(quid=r["quid"], question=r["question"], target=r["target"]) for r in res if r["kept"]]
    PC.write_jsonl(f_tr, kept)
    n = len(res); nk = len(kept)
    by_qt = collections.defaultdict(lambda: [0, 0])
    by_at = collections.defaultdict(lambda: [0, 0])
    for r in res:
        by_qt[r["qtype"]][0 if r["kept"] else 1] += 1
        by_at[r["answer_type"]][0 if r["kept"] else 1] += 1
    conc = collections.Counter((bool(r["kept"]), bool(r.get("flag_executes_to_gold"))) for r in res)
    errs = collections.Counter((r.get("rt_error") or "").split(":")[0] for r in res if not r.get("rt_valid"))
    man = dict(
        track="multitq_kpct / LLM parser (P_k)", rule_sha256="0f5f760450356349a7bf353301520d76de3617f1dfec0b4e4b1430c97b1d6cd7",
        subset=a.subset, llm_list=lst, list_n=len(keep), list_sha256=PC.list_sha256(keep), n_rows=n, dry_run_limit=a.limit or None,
        keep_rule="serialised compact top-1 parse executed alone (compact_to_parse without structural hypotheses; operators + RULE order) returns a gold answer at rank 1 (official matching)",
        n_kept=nk, kept_fraction=nk / max(1, n), LOW_SILVER=(nk / max(1, n) < 0.5),
        kept_quids_sha256=PC.list_sha256([r["quid"] for r in res if r["kept"]]),
        structural_flag_executes_to_gold_rate=sum(bool(r.get("flag_executes_to_gold")) for r in res) / max(1, n),
        structural_flag_full_parse_rate=sum(bool(r.get("flag_executes_to_gold_full_parse")) for r in res) / max(1, n),
        concordance_kept_vs_flag={f"kept={k_[0]}|flag={k_[1]}": v for k_, v in sorted(conc.items())},
        serialisation_top10_identical_rate=(sum(1 for r in res if r.get("rt_top10_identical")) / max(1, sum(1 for r in res if r.get("rt_top10_identical") is not None))),
        serialisation_top1_identical_rate=(sum(1 for r in res if r.get("rt_top1_identical")) / max(1, sum(1 for r in res if r.get("rt_top1_identical") is not None))),
        roundtrip_invalid=sum(1 for r in res if not r.get("rt_valid")), roundtrip_errors=dict(errs),
        by_qtype={k_: dict(kept=v[0], dropped=v[1], kept_fraction=v[0] / max(1, v[0] + v[1])) for k_, v in sorted(by_qt.items())},
        by_answer_type={k_: dict(kept=v[0], dropped=v[1], kept_fraction=v[0] / max(1, v[0] + v[1])) for k_, v in sorted(by_at.items())},
        qtype_share_kept={k_: v[0] / max(1, nk) for k_, v in sorted(by_qt.items())},
        qtype_share_dropped={k_: v[1] / max(1, n - nk) for k_, v in sorted(by_qt.items())},
        top1_sources=dict(collections.Counter(r.get("top1_source") for r in res)),
        target_chars_mean=(sum(len(r["target"]) for r in res if r.get("target")) / max(1, sum(1 for r in res if r.get("target")))),
        files={os.path.basename(f_all): PC.sha256_file(f_all), os.path.basename(f_tr): PC.sha256_file(f_tr)},
        bundle_file=bfile, bundle_sha256=PC.sha256_file(bfile),
        code_sha256={"pk_silver.py": PC.sha256_file(os.path.abspath(__file__)), "pk_common.py": PC.sha256_file(os.path.join(HERE, "pk_common.py")),
                     "kpct_executor.py": PC.sha256_file(os.path.join(EXEC_DIR, "kpct_executor.py")),
                     "kpct_core.py": PC.sha256_file(os.path.join(EXEC_DIR, "kpct_core.py"))},
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), secs=round(time.time() - t0, 1))
    man["sha256"] = PC.sha_recipe(man)
    json.dump(man, open(os.path.join(a.out, f"silver_{tag}_manifest.json"), "w"), indent=1)
    print(json.dumps({k_: man[k_] for k_ in ("llm_list", "n_rows", "n_kept", "kept_fraction", "structural_flag_executes_to_gold_rate",
                                             "serialisation_top10_identical_rate", "roundtrip_invalid", "secs")}))


if __name__ == "__main__":
    main()
