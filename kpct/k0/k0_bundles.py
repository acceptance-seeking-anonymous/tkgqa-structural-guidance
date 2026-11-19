#!/usr/bin/env python3
"""k0_bundles.py -- k = 0 guidance bundles for G_0 (rule arms.G_0) from P_0's executed parses (selected design).

For every question of the split, P_0's mapped compact parse (p0_execute.py `parse_used`: exact KG name or base-linker
mapping, relation normalised) is converted with kpct_executor.compact_to_parse(c, question, None) -- no structural
hypotheses at k = 0 -- and run through the structural track's own bundle code with the k0 executor configuration:
kpct_core.rec_info_from_parse -> build_info (anchor_mention override exactly as Executor.execute) -> bundle_core(o, rec,
s0_score=None): candidates = the first 20 pool candidates in RULE order, facts = the 20 provenance facts
(mtq_export.provenance_facts recipe, following RULE order), operator trace. k0 has no S re-ranker, so every candidate
carries s_score = null and s_rank = its RULE-order rank (the RULE rank sits in the slot that S occupies at k > 0).
No S score, classifier, sector alias, block list or train statistic is used. Bundle parse = kpct_core.parse_json of the
executed record + "anchor" (P_0's anchor entity or null) + "anchor_mention" when P_0 fixed one.
Integrity check (asserted per row): make_ranked(RULE pool, k0 padding) == P_0's prediction list for the quid.
P_0 rows that are invalid (invalid JSON / unknown relation / entity / anchor) get an empty bundle (parse {}, no
candidates, no facts) and are rendered "Parse: none" by g0_prompts; their G_0 ranked list is [answer] + [] (P_0 list).
No answer field is read: inputs are the questions json (quid + question only), P_0 exec rows and P_0 prediction rows.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import argparse, collections, gzip, hashlib, json, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
EXEC_DIR = os.environ.get("KPCT_EXECUTOR", "/kpct/executor")
sys.path.insert(1, EXEC_DIR)
import gk_common as GC          # noqa: E402  (schema check of the written rows)

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
        return [json.loads(l) for l in f if l.strip()]


def build_row(X, K, ex, quid, question, er, split, design):
    base = dict(quid=quid, question=question, split=split, k_subset="k0", p0_design=design,
                s_rank_note="k0 has no S re-ranker: s_score is null and s_rank holds the RULE-order rank")
    if er["status"] != "ok":
        return dict(base, parse={}, candidates=[], facts=[], operator_trace=f"no usable parse (P_0 output invalid: {er['reason']})",
                    pool_size=0, n_operator_satisfied=0, p0_status=er["status"], p0_reason=er["reason"]), True
    c2 = er["parse_used"]
    p = X.compact_to_parse(c2, question, None)
    rec, info = K.rec_info_from_parse(p, question)
    anc = p.get("anchor_mention")
    old = K.MC.anchor_index
    if anc is not None:
        K.MC.anchor_index = lambda ws, ments, qtext, _a=int(anc): (_a if 0 <= _a < len(ments) and len(ments) >= 2 else None)
    try:
        o = K.build_info(rec, info)
        rule = K.make_ranked(o["names"], o, ex.FB, k=50)
        core = K.bundle_core(o, rec, None)
    finally:
        K.MC.anchor_index = old
    match = rule == er["ranked"]
    for i, c in enumerate(core["candidates"]):
        c["s_rank"] = i + 1                       # RULE-order rank in the S slot (k0: no S)
    pj = K.parse_json(rec, o["info"])
    pj["anchor"] = c2.get("anchor")
    if anc is not None:
        pj["anchor_mention"] = int(anc)
    return dict(base, parse=pj, **core, p0_status="ok", p0_reason=None), match


def work(items):
    import kpct_executor as X
    import kpct_core as K
    ex = X.get("k0")
    out = []
    for quid, question, er in items:
        row, match = build_row(X, K, ex, quid, question, er, _W["split"], _W["design"])
        out.append((row, match))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="dev_sel2000 | test | dry")
    ap.add_argument("--exec_rows", required=True, help="exec_P0_<split>_<d>.jsonl.gz (p0_execute.py)")
    ap.add_argument("--pred", required=True, help="pred_P0_<split>_<d>.jsonl (p0_execute.py)")
    ap.add_argument("--design", required=True, help="the selected P_0 design")
    ap.add_argument("--questions", required=True, help="questions json of the split (quid + question read)")
    ap.add_argument("--out", required=True, help="output bundle file <dir>/<split>.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    E = {int(r["quid"]): r for r in read_jsonl(a.exec_rows)}
    Pr = {int(r["quid"]): r["ranked"] for r in read_jsonl(a.pred)}
    assert set(E) == set(Pr), "exec rows and prediction rows cover different quids"
    for q, r in E.items():
        assert r["ranked"] == Pr[q], f"exec/pred mismatch {q}"
    allq = json.load(open(a.questions))
    Q = {int(q["quid"]): q["question"] for q in allq if int(q["quid"]) in E}
    del allq
    assert set(Q) == set(E)
    _W.update(split=a.split, design=a.design)
    import kpct_executor as X
    X.get("k0")
    quids = sorted(E)
    items = [(q, Q[q], E[q]) for q in quids]
    chunks = [items[i:i + 50] for i in range(0, len(items), 50)]
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.workers) as pool:
            res = [x for ch in pool.map(work, chunks) for x in ch]
    else:
        res = [x for ch in chunks for x in work(ch)]
    rows = [r for r, _ in res]
    n_mis = sum(1 for r, m in res if not m)
    assert n_mis == 0, f"{n_mis} rows: re-executed RULE list != P_0 prediction list"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, a.out)
    chk = GC.check_bundles(rows, a.split)       # schema (gk_common.normalize_bundle) + no gold on dev/test rows
    st = dict(split=a.split, n=len(rows), p0_design=a.design, exec_rows_sha256=sha256_file(a.exec_rows),
              pred_sha256=sha256_file(a.pred), bundle_file=os.path.basename(a.out), bundle_sha256=sha256_file(a.out),
              n_p0_invalid=sum(r["p0_status"] != "ok" for r in rows),
              p0_invalid_reasons=dict(collections.Counter(r["p0_reason"] for r in rows if r["p0_status"] != "ok")),
              n_empty_pool=sum(1 for r in rows if r["p0_status"] == "ok" and r["pool_size"] == 0),
              rule_list_match_rate=1.0, schema_check=chk,
              mean_cands=float(sum(len(r["candidates"]) for r in rows) / max(1, len(rows))),
              mean_facts=float(sum(len(r["facts"]) for r in rows) / max(1, len(rows))),
              code_sha256={f: sha256_file(os.path.join(p, f)) for p, f in
                           ((HERE, "k0_bundles.py"), (EXEC_DIR, "kpct_executor.py"), (EXEC_DIR, "kpct_core.py"))},
              k0_FB_sha256=sha256_file(os.path.join(EXEC_DIR, "models", "k0", "FB.json")), secs=round(time.time() - t0, 1))
    json.dump(st, open(os.path.join(os.path.dirname(os.path.abspath(a.out)), f"bundle_stats_{a.split}.json"), "w"), indent=1)
    print(json.dumps({k: st[k] for k in ("n", "n_p0_invalid", "n_empty_pool", "mean_cands", "mean_facts", "secs")}))
    os.system(f"chmod -R a+rX {os.path.dirname(os.path.abspath(a.out))}")


if __name__ == "__main__":
    main()
