#!/usr/bin/env python3
"""pk_clean_silver.py -- clean-silver filter of the P_k silver targets (multitq_p1_clean_silver_rule.json, filter).

For every row of the frozen silver build (parser/silver/silver_<list>_all.jsonl) the stored serialised target is
re-executed exactly as pk_silver.py did (validate_compact -> kpct_executor.compact_to_parse(c, q, None) ->
Executor(<subset>).execute(probs='given')), and three conditions are recorded:
  C0 frozen keep   : valid AND official hit@1 of the PADDED RULE list (= pk_silver's keep; replay-checked vs the stored row)
  C1 nonempty pool : pool_size > 0 (candidate pool of the operators BEFORE output padding)
  C2 unpadded top-1: the first pool candidate in RULE order (execute()['candidates'][0], unpadded) is a gold answer
                     under official MultiTQ matching (multitq_eval.hit_at_k on the one-element list)
  CLEAN keep = valid AND C1 AND C2.
Diagnostics (not used for filtering): strict exact-string membership of the unpadded top-1 in the gold list;
target qtype vs released qtype; target granularity vs released time_level (all rows / time-answer rows).
Gold is read from bundles/<subset>/train.jsonl only for quids of the registered list (asserted).
STOP_REPLAY: any row whose re-executed C0 differs from the stored keep -> exit 2 before writing a train file.
"""
import argparse, collections, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import pk_common as PC

EXEC_DIR = os.environ.get("KPCT_EXECUTOR", "/kpct/executor")
sys.path.insert(0, EXEC_DIR)
_W = {}


def work(rows):
    import kpct_executor as X
    import kpct_core as K
    ME = K.ME
    ex = X.get(_W["k"])
    out = []
    for r in rows:
        g = _W["gold"][int(r["quid"])]
        gq = {"answers": g["gold"], "answer_type": g["gold_meta"]["answer_type"]}
        o = dict(quid=int(r["quid"]), qtype=r.get("qtype"), answer_type=r.get("answer_type"), stored_kept=bool(r["kept"]),
                 stored_hit1=r.get("rt_hit1"), stored_pool=r.get("rt_pool"))
        valid, pool, padded_hit, top1, top1_hit, strict = False, 0, 0, None, 0, False
        tq = tg = None
        if r.get("target"):
            try:
                c = PC.validate_compact(json.loads(r["target"]))
                tq, tg = c["qtype"], c["granularity"]
                res = ex.execute(X.compact_to_parse(c, r["question"], None), question=r["question"], probs="given")
                valid = bool(res["valid"])
                if valid:
                    pool = int(res["pool_size"])
                    padded_hit = int(ME.hit_at_k(gq, res["rule"], 1))
                    if pool > 0 and res["candidates"]:
                        top1 = res["candidates"][0]["answer"]
                        top1_hit = int(ME.hit_at_k(gq, [top1], 1))
                        strict = top1 in set(g["gold"])
            except Exception as e_:
                o["error"] = f"{type(e_).__name__}: {e_}"
        o.update(valid=valid, pool_size=pool, padded_hit1=padded_hit, unpadded_top1=top1, unpadded_top1_hit=top1_hit,
                 unpadded_top1_strict_member=bool(strict),
                 C0=bool(valid and padded_hit == 1), C1=bool(pool > 0), C2=bool(top1_hit == 1))
        o["clean_keep"] = bool(valid and o["C1"] and o["C2"])
        o["replay_ok"] = (o["C0"] == o["stored_kept"])
        o["target_qtype"], o["target_granularity"] = tq, tg
        o["meta_qtype"], o["meta_time_level"] = g["gold_meta"]["qtype"], g["gold_meta"].get("time_level")
        out.append(o)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True); ap.add_argument("--list", default=None)
    ap.add_argument("--silver_dir", default="/kpct/parser/silver")
    ap.add_argument("--manifest", default="/kpct/parser/code/multitq_kpct_subsets.json")
    ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--write_train", action="store_true", help="write the clean training file (k01_s0 only by the rule)")
    a = ap.parse_args()
    t0 = time.time()
    lst = a.list or a.subset
    allf = os.path.join(a.silver_dir, f"silver_{lst}_all.jsonl")
    manf = os.path.join(a.silver_dir, f"silver_{lst}_manifest.json")
    trf = os.path.join(a.silver_dir, f"silver_{lst}_train.jsonl")
    keep = set(PC.load_list(a.manifest, lst))
    rows = PC.read_jsonl(allf)
    assert {int(r["quid"]) for r in rows} == keep, "silver rows != registered list"
    gold = {}
    for r in PC.iter_jsonl(f"/kpct/bundles/{a.subset}/train.jsonl"):
        q = int(r["quid"])
        if q in keep:
            gold[q] = dict(gold=r["gold"], gold_meta=r["gold_meta"])
    assert set(gold) == keep, "bundle gold rows != registered list"
    import kpct_executor as X
    _W.update(k=a.subset, gold=gold)
    X.get(a.subset)
    import multiprocessing as mp
    chunks = [rows[i:i + 50] for i in range(0, len(rows), 50)]
    with mp.get_context("fork").Pool(a.workers) as pool:
        res = [x for ch in pool.map(work, chunks) for x in ch]
    res.sort(key=lambda r: r["quid"])
    n_replay_bad = sum(not r["replay_ok"] for r in res)
    kept0 = [r for r in res if r["C0"]]
    zero_pool = [r for r in kept0 if not r["C1"]]
    extra = [r for r in kept0 if r["C1"] and not r["C2"]]
    clean = [r for r in res if r["clean_keep"]]
    clean_not_frozen = [r for r in clean if not r["C0"]]
    by_qt = collections.defaultdict(lambda: [0, 0, 0])
    for r in res:
        if r["C0"]:
            by_qt[r["qtype"]][0] += 1
            by_qt[r["qtype"]][1 if r["clean_keep"] else 2] += 1
    tm = [r for r in clean if r["answer_type"] == "time"]
    diag = dict(
        clean_rows_unpadded_top1_not_strict_member=sum(1 for r in clean if not r["unpadded_top1_strict_member"]),
        clean_rows_qtype_disagree_metadata=sum(1 for r in clean if r["target_qtype"] != r["meta_qtype"]),
        clean_rows_granularity_disagree_metadata_all=sum(1 for r in clean if r["target_granularity"] != r["meta_time_level"]),
        clean_time_rows=len(tm),
        clean_time_rows_granularity_disagree_metadata=sum(1 for r in tm if r["target_granularity"] != r["meta_time_level"]),
        frozen_rows_qtype_disagree_metadata=sum(1 for r in kept0 if r["target_qtype"] != r["meta_qtype"]),
        frozen_time_rows_granularity_disagree_metadata=sum(1 for r in kept0 if r["answer_type"] == "time" and r["target_granularity"] != r["meta_time_level"]),
        zero_pool_rows_qtype=dict(collections.Counter(r["qtype"] for r in zero_pool)),
        zero_pool_rows_target_qtype=dict(collections.Counter(r["target_qtype"] for r in zero_pool)),
        note="diagnostics only; not used by the clean filter")
    os.makedirs(a.out, exist_ok=True)
    fa = os.path.join(a.out, f"silver_{lst}_clean_all.jsonl")
    PC.write_jsonl(fa, res)
    files = {os.path.basename(fa): PC.sha256_file(fa)}
    if n_replay_bad:
        print(json.dumps({"STOP_REPLAY": n_replay_bad}))
    if a.write_train and not n_replay_bad:
        src = {int(r["quid"]): r for r in PC.read_jsonl(trf)}
        assert set(src) == {r["quid"] for r in kept0}, "frozen train file != replayed C0 set"
        ft = os.path.join(a.out, f"silver_{lst}_clean_train.jsonl")
        PC.write_jsonl(ft, [dict(quid=r["quid"], question=src[r["quid"]]["question"], target=src[r["quid"]]["target"]) for r in clean])
        files[os.path.basename(ft)] = PC.sha256_file(ft)
    man = dict(name=f"clean_silver_{lst}", rule="multitq_p1_clean_silver_rule.json", subset=a.subset, llm_list=lst, list_n=len(keep),
               list_sha256=PC.list_sha256(keep), n_rows=len(res),
               frozen_silver=dict(all_file=allf, all_sha256=PC.sha256_file(allf), train_file=trf, train_sha256=PC.sha256_file(trf),
                                  manifest_sha256=PC.sha256_file(manf)),
               replay=dict(rows_replay_mismatch=n_replay_bad, STOP_REPLAY=bool(n_replay_bad),
                           invalid_rows=sum(1 for r in res if not r["valid"])),
               counts=dict(frozen_kept=len(kept0), dropped_zero_pool=len(zero_pool), dropped_unpadded_top1_not_gold=len(extra),
                           clean_kept=len(clean), clean_not_in_frozen=len(clean_not_frozen),
                           frozen_kept_fraction=len(kept0) / max(1, len(res)), clean_kept_fraction=len(clean) / max(1, len(res)),
                           LOW_SILVER_clean=bool(len(clean) / max(1, len(res)) < 0.5)),
               dropped_quids=dict(zero_pool=[r["quid"] for r in zero_pool], unpadded_top1_not_gold=[r["quid"] for r in extra]),
               by_qtype={k: dict(frozen_kept=v[0], clean_kept=v[1], dropped=v[2]) for k, v in sorted(by_qt.items())},
               clean_quids_sha256=PC.list_sha256([r["quid"] for r in clean]),
               diagnostics=diag, files=files,
               code_sha256={"pk_clean_silver.py": PC.sha256_file(os.path.abspath(__file__)), "pk_common.py": PC.sha256_file(os.path.join(HERE, "pk_common.py")),
                            "kpct_executor.py": PC.sha256_file(os.path.join(EXEC_DIR, "kpct_executor.py")),
                            "kpct_core.py": PC.sha256_file(os.path.join(EXEC_DIR, "kpct_core.py"))},
               created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), secs=round(time.time() - t0, 1))
    man["sha256"] = PC.sha_recipe(man)
    json.dump(man, open(os.path.join(a.out, f"silver_{lst}_clean_manifest.json"), "w"), indent=1)
    print(json.dumps(dict(list=lst, **man["counts"], replay_mismatch=n_replay_bad, secs=man["secs"])))
    sys.exit(2 if n_replay_bad else 0)


if __name__ == "__main__":
    main()
