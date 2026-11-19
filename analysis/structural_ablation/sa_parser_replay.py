#!/usr/bin/env python3
"""sa_parser_replay.py -- structural-ablation track, part (a): CPU replay of the CACHED P_k parser generations.

No LLM inference. For every question the cached generation (raw_P_<split>_<ckpt>.jsonl.gz) is validated exactly as in the
frozen pk_execute.work (pk_common.extract_json + validate_compact, then check_names), then executed under four parse
modes with the frozen executor (probs='classifier', the 3 S_k LightGBM seeds):
  PK     = kpct_executor.compact_to_parse(c, q, structural_parse=bundle parse)   (frozen P_k: hypothesis + alternatives)
  PALONE = kpct_executor.compact_to_parse(c, q, structural_parse=None)           (Parser hypothesis only)
  PREL   = PK with every mention truncated to its first alternative              (secondary: + relation alternatives only)
  PENT   = PK with the relation list truncated to its first entry                (secondary: + entity alternatives only)
Per mode a question FALLS BACK (same rule as P_k, tau off) on invalid_json / unknown_name (identical for all modes) or
on an execution error / invalid execution / empty pool (mode-specific).
Outputs per (k, split, mode):
  pred_<mode>_<k>_<split>_fb_seed{b}.jsonl      fallback = S_k's own list for that seed (registered handling)
  pred_<mode>_<k>_<split>_strict_seed{b}.jsonl  strict no-fallback: a fallback question gets ranked = []
  pool_<mode>_<k>_<split>.jsonl.gz              {quid, status, reason, detail, pool_size, n_sat, pool: full RULE-ordered pool}
plus replay_stats_<k>_<split>.json (counts by reason and mode, sha256 of every output, identity of PK vs the frozen
pred_P files when given). Never reads an answer.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import argparse, collections, copy, gzip, hashlib, json, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PK_CODE = os.environ.get("PK_CODE")
EXEC_DIR = os.environ.get("KPCT_EXECUTOR")
sys.path.insert(0, PK_CODE)
sys.path.insert(0, EXEC_DIR)
import pk_common as PC  # noqa: E402

MODES = ("PK", "PALONE", "PREL", "PENT")
_W = {}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


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


def make_parse(mode, c, q, sp, X):
    if mode == "PK":
        return X.compact_to_parse(c, q, sp)
    if mode == "PALONE":
        return X.compact_to_parse(c, q, None)
    p = X.compact_to_parse(c, q, sp)
    p = copy.deepcopy(p)
    if mode == "PREL":
        for m in p["mentions"]:
            m["alternatives"] = m["alternatives"][:1]
    elif mode == "PENT":
        p["relations"] = p["relations"][:1]
    return p


def work(items):
    import kpct_executor as X
    import kpct_core as K
    ex = X.get(_W["k"])
    out = []
    for raw, b in items:
        q = b["question"]
        base = dict(quid=int(raw["quid"]), status="ok", reason=None, detail=None)
        try:
            c = PC.validate_compact(PC.extract_json(raw["text"]))
        except PC.PInvalid as e_:
            base.update(status="fallback", reason="invalid_json", detail=str(e_))
            out.append({m: dict(base) for m in MODES}); continue
        bad = check_names(c, K)
        if bad:
            base.update(status="fallback", reason="unknown_name", detail=bad)
            out.append({m: dict(base) for m in MODES}); continue
        res = {}
        for m in MODES:
            o = dict(base)
            try:
                p = make_parse(m, c, q, b["parse"], X)
                r = ex.execute(p, question=q, probs="classifier", top=50, details=1000)
            except Exception as e_:
                o.update(status="fallback", reason="invalid_json", detail=f"{type(e_).__name__}: {e_}"); res[m] = o; continue
            if not r["valid"]:
                o.update(status="fallback", reason="unknown_name" if "vocabulary" in str(r["error"]) else "invalid_json", detail=r["error"])
            elif r["pool_size"] == 0:
                o.update(status="fallback", reason="empty_pool")
            else:
                o.update(s=r["s"], rule=r["rule"], pool_size=r["pool_size"], n_sat=r["n_operator_satisfied"],
                         pool=[x["answer"] for x in r["candidates"]])
                assert len(o["pool"]) == r["pool_size"]
            res[m] = o
        out.append(res)
    return out


def read_ranked(p):
    d = {}
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt") as f:
        for l in f:
            if l.strip():
                x = json.loads(l); d[int(x["quid"])] = x["ranked"]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--bundles", required=True)
    ap.add_argument("--fallback_dir", required=True, help="dir with <split>_S_seed{b}.jsonl of this k")
    ap.add_argument("--fallback_prefix", default=None, help="file prefix (default = split)")
    ap.add_argument("--frozen_pred", default=None, help="frozen P_k pred file pattern with {b} (identity check)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="pilot: first N quids only, identity check only")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    R = {int(r["quid"]): r for r in PC.read_jsonl(a.raw)}
    quids = sorted(R)
    if a.limit:
        quids = quids[:a.limit]
    need = set(quids)
    B = {}
    for r in PC.iter_jsonl(a.bundles):
        q = int(r["quid"])
        if q in need:
            B[q] = dict(quid=q, question=r["question"], parse=r["parse"])
    assert set(B) == need, f"{len(need - set(B))} raw quids without a bundle row"
    pref = a.fallback_prefix or a.split
    FB = {}
    fb_sha = {}
    for s in range(3):
        f = os.path.join(a.fallback_dir, f"{pref}_S_seed{s}.jsonl")
        fb_sha[f"S_seed{s}"] = sha256_file(f)
        for q, rk in read_ranked(f).items():
            if q in need:
                FB.setdefault(q, {})[s] = rk
    assert all(q in FB and len(FB[q]) == 3 for q in quids), "missing fallback lists"
    import kpct_executor as X
    _W.update(k=a.k)
    X.get(a.k)
    import torch
    torch.set_num_threads(1)
    X.get(a.k).parser()
    import multiprocessing as mp
    items = [(R[q], B[q]) for q in quids]
    chunks = [items[i:i + 50] for i in range(0, len(items), 50)]
    with mp.get_context("fork").Pool(a.workers) as pool:
        res = [x for ch in pool.map(work, chunks) for x in ch]
    secs_exec = round(time.time() - t0, 1)
    stats = dict(track="structural_ablation/parser_replay", k=a.k, split=a.split, n=len(res), raw=a.raw,
                 raw_sha256=sha256_file(a.raw), bundles=a.bundles, fallback_source=fb_sha, modes={}, files={},
                 code_sha256={"sa_parser_replay.py": sha256_file(os.path.abspath(__file__)),
                              "pk_common.py": sha256_file(os.path.join(PK_CODE, "pk_common.py")),
                              "kpct_executor.py": sha256_file(os.path.join(EXEC_DIR, "kpct_executor.py")),
                              "kpct_core.py": sha256_file(os.path.join(EXEC_DIR, "kpct_core.py"))},
                 limit=a.limit)
    tag = f"{a.k}_{a.split}" + (f"_pilot{a.limit}" if a.limit else "")
    for m in MODES:
        rows = [r[m] for r in res]
        st = dict(reasons=dict(collections.Counter(r["reason"] for r in rows if r["status"] != "ok")),
                  details=dict(collections.Counter(str(r.get("detail")).split(":")[0] for r in rows if r["status"] != "ok").most_common(12)),
                  n_ok=sum(r["status"] == "ok" for r in rows))
        st["fallback_rate"] = 1 - st["n_ok"] / max(1, len(rows))
        stats["modes"][m] = st
        if a.limit:
            continue
        for s in range(3):
            for var in ("fb", "strict"):
                pf = os.path.join(a.out, f"pred_{m}_{a.k}_{a.split}_{var}_seed{s}.jsonl")
                with open(pf, "w") as f:
                    for r in rows:
                        rk = r["s"][f"seed{s}"] if r["status"] == "ok" else (FB[r["quid"]][s] if var == "fb" else [])
                        f.write(json.dumps({"quid": r["quid"], "ranked": rk}, ensure_ascii=False) + "\n")
                stats["files"][os.path.basename(pf)] = sha256_file(pf)
        pp = os.path.join(a.out, f"pool_{m}_{a.k}_{a.split}.jsonl.gz")
        with gzip.open(pp, "wt", compresslevel=6) as f:
            for r in rows:
                f.write(json.dumps({k_: r.get(k_) for k_ in ("quid", "status", "reason", "detail", "pool_size", "n_sat", "pool")},
                                   ensure_ascii=False) + "\n")
        stats["files"][os.path.basename(pp)] = sha256_file(pp)
    # identity of PK (fallback version) with the frozen P_k prediction files (predictions only, no gold)
    if a.frozen_pred:
        idn = {}
        for s in range(3):
            fz = read_ranked(a.frozen_pred.format(b=s))
            mine = {r["PK"]["quid"]: (r["PK"]["s"][f"seed{s}"] if r["PK"]["status"] == "ok" else FB[r["PK"]["quid"]][s]) for r in res}
            idn[f"seed{s}"] = dict(n=len(mine), n_found=sum(q in fz for q in mine),
                                   full=sum(mine[q] == fz.get(q) for q in mine) / len(mine),
                                   top1=sum(mine[q][:1] == (fz.get(q) or [None])[:1] for q in mine) / len(mine),
                                   frozen_sha256=sha256_file(a.frozen_pred.format(b=s)))
        stats["identity_PK_vs_frozen"] = idn
    stats["secs_exec"] = secs_exec
    stats["secs"] = round(time.time() - t0, 1)
    json.dump(stats, open(os.path.join(a.out, f"replay_stats_{tag}.json"), "w"), indent=1)
    print(json.dumps({"k": a.k, "split": a.split, "n": len(res), "secs": stats["secs"],
                      "fallback": {m: round(stats["modes"][m]["fallback_rate"], 4) for m in MODES},
                      "identity": {s: (v["full"], v["top1"]) for s, v in stats.get("identity_PK_vs_frozen", {}).items()}}), flush=True)


if __name__ == "__main__":
    main()
