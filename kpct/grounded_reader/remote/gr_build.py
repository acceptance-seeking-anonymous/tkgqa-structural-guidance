#!/usr/bin/env python3
"""gr_build.py -- grounded_reader track (rule multitq_robustness_grounded_reader_rule.json): Parser-grounded Reader
bundles on CPU. No LLM inference, never reads an answer.

For one level k and one split, every row's CACHED P_k generation (the frozen Parser's raw output) is validated exactly as
the frozen pk_execute.work (pk_common.extract_json + validate_compact, then check_names) and converted with the frozen
kpct_executor.compact_to_parse under the two registered bundle sources:
  PALONE : compact_to_parse(c, q, None)                     Parser hypothesis only
  PK     : compact_to_parse(c, q, structural bundle parse)  = the frozen P_k parse (hypothesis + deduplicated
                                                              structural alternatives)
Each parse is executed exactly as kpct_executor.Executor.execute(probs='classifier') does (rec_info_from_parse,
classifier_probs, anchor_mention override, build_info, predict_chunk with the 3 S_k LightGBM seeds) and the bundle is
built by the frozen structural bundle code used for every G_k bundle (rob_build.w_infer / kpct_run):
  parse = kpct_core.parse_json(rec, info) (+ anchor_mention when the parse has one; not rendered)
  core  = kpct_core.bundle_core(o, rec, S seed-0 scores)   -> top-20 candidates in S seed-0 order with RULE rank
          (position in the RULE-ordered pool of the Parser parse), S seed-0 score, operator flag and support facts;
          20 provenance facts; operator trace; pool size.
A row FALLS BACK (P_k rule, tau off): invalid_json / unknown_name (both sources) or execution error / invalid
execution / empty pool (source-specific). Fallback rows keep the frozen G_k output (handled by gr_assemble.py).
same_prompt: the rendered G user turn (frozen gk_common.g_user of the normalised bundle) equals that of the frozen
structural bundle of the same row -> the frozen G_k output is reused (registered prompt-identity reuse).

Outputs under --out (per source m in PALONE, PK):
  bundles_<m>.jsonl       bundle rows of status-ok rows (structural bundle schema)
  s0_<m>.jsonl            {"quid", "ranked"}: S seed-0 top-50 of the Parser pool (ranked-list tail of the Reader)
  status_<m>.jsonl        {quid, status, reason, detail, pool_size, n_sat, same_prompt}
  pk_seed{b}.jsonl        PK only: S seed-b top-50 (fallback rows: S_k's own list) = identity check I1a vs frozen P_k
  stats.json              counts, identity checks (I1a, I1b), sha256 of inputs and outputs, timings
--selfcheck (I1b): the frozen structural bundle parse is re-executed through the same builder with probs='given'; the
rendered G prompt must equal the frozen structural bundle's rendered prompt.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, collections, gzip, hashlib, json, sys, time

PK_CODE = os.environ.get("PK_CODE", "/kpct/parser/code")
EXEC_DIR = os.environ.get("KPCT_EXECUTOR", "/kpct/executor")
GK_CODE = os.environ.get("GK_CODE", "/kpct/guided/code")
for p_ in (GK_CODE, PK_CODE, EXEC_DIR):
    sys.path.insert(0, p_)
import pk_common as PC          # noqa: E402  frozen parser code
import gk_common as GC          # noqa: E402  frozen guided-reader code (renderer)

MODES = ("PALONE", "PK")
_W = {}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                yield json.loads(l)


def read_ranked(p):
    d = {}
    for x in read_jsonl(p):
        rk = x["ranked"]
        d[int(x["quid"])] = json.loads(rk.replace("'", '"')) if isinstance(rk, str) else rk
    return d


def check_names(c, K):          # = pk_execute.check_names (frozen)
    kg = K.kg()
    if c["relation"].replace(" ", "_") not in kg.r2i:
        return "unknown_relation"
    for m in c["mentions"]:
        if K._norm_name(m["entity"]) not in kg.e2i:
            return "unknown_entity"
    if c["anchor"] is not None and K._norm_name(c["anchor"]) not in kg.e2i:
        return "unknown_anchor"
    if not c["mentions"]:
        return "no_mentions"
    return None


def exec_bundle(ex, K, p, q, probs, build_empty=False):
    """= kpct_executor.Executor.execute (same calls, same order) + the structural bundle code.
    -> (valid, error, payload) with payload = dict(bundle_parse, core, lists{seed0..2}, pool_size, n_sat)"""
    try:
        rec, info = K.rec_info_from_parse(p, q)
    except (K.ParseError, KeyError, TypeError) as e_:
        return False, f"{type(e_).__name__}: {e_}", None
    if probs == "classifier":
        info = ex.classifier_probs(rec, info)
    anc = p.get("anchor_mention")
    old = K.MC.anchor_index
    if anc is not None:
        K.MC.anchor_index = lambda ws, ments, qtext, _a=int(anc): (_a if 0 <= _a < len(ments) and len(ments) >= 2 else None)
    try:
        o = K.build_info(rec, info)
    finally:
        K.MC.anchor_index = old
    names = o["names"]
    if not names and not build_empty:
        return True, None, dict(pool_size=0)
    sc = K.predict_chunk(ex.boosters, [o])
    lists = {f"seed{b}": K.make_ranked([names[j] for j in K.s_order(sc[b][0])], o, ex.FB, k=50) for b in range(3)}
    core = K.bundle_core(o, rec, sc[0][0])
    pj = K.parse_json(rec, o["info"])
    if anc is not None:
        pj["anchor_mention"] = int(anc)
    return True, None, dict(bundle_parse=pj, core=core, lists=lists, pool_size=len(names),
                            n_sat=int(o["X"][:, K.MC.FIDX["sat_any"]].sum()) if names else 0)


def render(bundle_row):
    return GC.g_user(GC.normalize_bundle(bundle_row))


def make_row(b, k, split, mode, pay):
    return dict(quid=int(b["quid"]), question=b["question"], split=split, k_subset=k, template_unseen=None,
                producer=f"grounded_reader {mode} (Parser parse, frozen executor + bundle code)",
                parse=pay["bundle_parse"], **pay["core"])


def work(items):
    import kpct_executor as X
    import kpct_core as K
    ex = X.get(_W["k"])
    out = []
    for raw, b in items:
        q = b["question"]
        u_struct = render(b)
        base = dict(quid=int(b["quid"]), status="ok", reason=None, detail=None, pool_size=None, n_sat=None, same_prompt=None)
        res = {}
        if _W["selfcheck"]:
            ok, err, pay = exec_bundle(ex, K, b["parse"], q, "given", build_empty=True)   # structural bundles exist for empty pools
            if not ok:
                res["SELF"] = dict(base, status="fallback", reason="invalid", detail=err)
            else:
                rowb = make_row(b, _W["k"], _W["split"], "SELF", pay)
                res["SELF"] = dict(base, same_prompt=render(rowb) == u_struct,
                                   same_cands=[c["answer"] for c in rowb["candidates"]] == [c["answer"] for c in b["candidates"]],
                                   same_facts=[f["fid"] for f in rowb["facts"]] == [f["fid"] for f in b["facts"]],
                                   same_trace=rowb["operator_trace"] == b["operator_trace"])
        if raw is None:
            for m in MODES:
                res[m] = dict(base, status="fallback", reason="no_raw", detail=None)
            out.append((res, {})); continue
        try:
            c = PC.validate_compact(PC.extract_json(raw["text"]))
        except PC.PInvalid as e_:
            for m in MODES:
                res[m] = dict(base, status="fallback", reason="invalid_json", detail=str(e_)[:200])
            out.append((res, {})); continue
        bad = check_names(c, K)
        if bad:
            for m in MODES:
                res[m] = dict(base, status="fallback", reason="unknown_name", detail=bad)
            out.append((res, {})); continue
        rows = {}
        for m in MODES:
            o_ = dict(base)
            try:
                p = X.compact_to_parse(c, q, b["parse"] if m == "PK" else None)
                ok, err, pay = exec_bundle(ex, K, p, q, "classifier")
            except Exception as e_:
                o_.update(status="fallback", reason="invalid_json", detail=f"{type(e_).__name__}: {e_}"[:200]); res[m] = o_; continue
            if not ok:
                o_.update(status="fallback", reason="unknown_name" if "vocabulary" in str(err) else "invalid_json", detail=str(err)[:200])
            elif pay["pool_size"] == 0:
                o_.update(status="fallback", reason="empty_pool")
            else:
                rowb = make_row(b, _W["k"], _W["split"], m, pay)
                o_.update(pool_size=pay["pool_size"], n_sat=pay["n_sat"], same_prompt=bool(render(rowb) == u_struct))
                rows[m] = dict(bundle=rowb, lists=pay["lists"])
            res[m] = o_
        out.append((res, rows))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--bundles", required=True, help="frozen structural bundles of the rows (G_k input)")
    ap.add_argument("--vidx", default=None, help="comma list: keep only synthetic quids with quid %% 10 in this set")
    ap.add_argument("--quids", default=None, help="json list restricting the rows")
    ap.add_argument("--fallback_s", required=True, help="pattern with {b}: S_k seed-b list of the rows (P_k fallback)")
    ap.add_argument("--frozen_p", required=True, help="pattern with {b}: frozen P_k seed-b list (identity check I1a)")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    keep_v = set(int(x) for x in a.vidx.split(",")) if a.vidx else None
    keep_q = set(int(x) for x in json.load(open(a.quids))) if a.quids else None
    B = {}
    for r in read_jsonl(a.bundles):
        q = int(r["quid"])
        if (keep_v is None or q % 10 in keep_v) and (keep_q is None or q in keep_q):
            B[q] = r
    R = {int(r["quid"]): r for r in read_jsonl(a.raw) if int(r["quid"]) in B}
    quids = sorted(B)
    if keep_q is not None:
        assert set(quids) == keep_q, f"{len(keep_q - set(quids))} listed quids without a bundle"
    import kpct_executor as X
    import torch
    torch.set_num_threads(1)
    _W.update(k=a.k, split=a.split, selfcheck=a.selfcheck)
    X.get(a.k).parser()
    import multiprocessing as mp
    items = [(R.get(q), B[q]) for q in quids]
    chunks = [items[i:i + 40] for i in range(0, len(items), 40)]
    with mp.get_context("fork").Pool(a.workers) as pool:
        res = [x for ch in pool.map(work, chunks) for x in ch]
    secs_exec = round(time.time() - t0, 1)
    FB = {b: read_ranked(a.fallback_s.format(b=b)) for b in range(3)}
    st = dict(track="grounded_reader/build", k=a.k, split=a.split, n=len(res), n_raw_missing=sum(q not in R for q in quids),
              inputs={"raw": a.raw, "raw_sha256": sha256_file(a.raw), "bundles": a.bundles, "bundles_sha256": sha256_file(a.bundles),
                      **{f"fallback_s_seed{b}": [a.fallback_s.format(b=b), sha256_file(a.fallback_s.format(b=b))] for b in range(3)},
                      **{f"frozen_p_seed{b}": [a.frozen_p.format(b=b), sha256_file(a.frozen_p.format(b=b))] for b in range(3)}},
              code_sha256={n: sha256_file(os.path.join(d, n)) for d, n in
                           ((os.path.dirname(os.path.abspath(__file__)), "gr_build.py"), (PK_CODE, "pk_common.py"),
                            (EXEC_DIR, "kpct_executor.py"), (EXEC_DIR, "kpct_core.py"), (GK_CODE, "gk_common.py"))},
              modes={}, files={}, vidx=a.vidx, quids_file=a.quids)
    for m in MODES:
        S = [r[0][m] for r in res]
        st["modes"][m] = dict(n_ok=sum(s["status"] == "ok" for s in S),
                              reasons=dict(collections.Counter(s["reason"] for s in S if s["status"] != "ok")),
                              n_same_prompt=sum(bool(s["same_prompt"]) for s in S),
                              n_generate=sum(s["status"] == "ok" and not s["same_prompt"] for s in S))
        fb_ = os.path.join(a.out, f"bundles_{m}.jsonl"); fs = os.path.join(a.out, f"s0_{m}.jsonl"); ft = os.path.join(a.out, f"status_{m}.jsonl")
        with open(fb_, "w") as f1, open(fs, "w") as f2, open(ft, "w") as f3:
            for s, (_, rows) in zip(S, res):
                f3.write(json.dumps(s, ensure_ascii=False) + "\n")
                if m in rows:
                    f1.write(json.dumps(rows[m]["bundle"], ensure_ascii=False) + "\n")
                    f2.write(json.dumps({"quid": s["quid"], "ranked": rows[m]["lists"]["seed0"]}, ensure_ascii=False) + "\n")
        for p in (fb_, fs, ft):
            st["files"][os.path.basename(p)] = sha256_file(p)
    # I1a: PK lists (fallback -> S_k's own list) vs the frozen P_k files
    I1a = {}
    for b in range(3):
        fz = read_ranked(a.frozen_p.format(b=b))
        mine = {}
        for (S, rows) in res:
            q = S["PK"]["quid"]
            mine[q] = rows["PK"]["lists"][f"seed{b}"] if "PK" in rows else FB[b].get(q)
        pf = os.path.join(a.out, f"pk_seed{b}.jsonl")
        with open(pf, "w") as f:
            for q in quids:
                f.write(json.dumps({"quid": q, "ranked": mine[q]}, ensure_ascii=False) + "\n")
        st["files"][os.path.basename(pf)] = sha256_file(pf)
        I1a[f"seed{b}"] = dict(n=len(quids), n_found=sum(q in fz for q in quids),
                               full=sum(mine[q] == fz.get(q) for q in quids) / max(1, len(quids)),
                               top1=sum((mine[q] or [None])[:1] == (fz.get(q) or [None])[:1] for q in quids) / max(1, len(quids)))
    st["I1a_PK_vs_frozen_P"] = I1a
    if a.selfcheck:
        S = [r[0]["SELF"] for r in res]
        n = max(1, len(S))
        st["I1b_structural_selfcheck"] = dict(n=len(S), n_fallback=sum(s["status"] != "ok" for s in S),
                                              same_prompt=sum(bool(s.get("same_prompt")) for s in S) / n,
                                              same_cands=sum(bool(s.get("same_cands")) for s in S) / n,
                                              same_facts=sum(bool(s.get("same_facts")) for s in S) / n,
                                              same_trace=sum(bool(s.get("same_trace")) for s in S) / n,
                                              differing_quids=[s["quid"] for s in S if not s.get("same_prompt")][:50])
    st["secs_exec"] = secs_exec
    st["secs"] = round(time.time() - t0, 1)
    json.dump(st, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    print(json.dumps({"k": a.k, "split": a.split, "n": len(res), "secs": st["secs"],
                      "modes": {m: (v["n_ok"], v["n_same_prompt"], v["n_generate"]) for m, v in st["modes"].items()},
                      "I1a_top1": {s: round(v["top1"], 4) for s, v in I1a.items()},
                      "I1b": {k_: v for k_, v in st.get("I1b_structural_selfcheck", {}).items() if k_ != "differing_quids"}}), flush=True)


if __name__ == "__main__":
    main()
