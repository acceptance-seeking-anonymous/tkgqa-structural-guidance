#!/usr/bin/env python3
"""pk_execute.py -- execute P_k parses with the deterministic operators and the S_k re-rankers, with the registered
fallback to the structural parse (rule arms.P_k.execution / arms.P_k.fallback).

Per question:
  1. the generation must be one compact P JSON (pk_common.extract_json + validate_compact) -> else FALLBACK invalid_json;
  2. every name must be in the KG vocabulary (relation; each mention entity; a non-null anchor) -> else FALLBACK
     unknown_name (an anchor that is a KG name but matches no mention entity is ignored by compact_to_parse);
  3. P parse = hypothesis rank 0 followed by the structural parse's remaining hypotheses (duplicates removed):
     kpct_executor.compact_to_parse(compact, question, structural_parse = bundle parse of S_k);
  4. executed with probs='classifier' (S features use the k% classifier probabilities of P's choices), re-ranked by
     each of the 3 S_k LightGBM seeds -> FALLBACK empty_pool if the operator pool is empty (or execution invalid);
  5. FALLBACK low_confidence if tau is set and the mean token log-prob of the JSON < tau.
FALLBACK = S_k's own output for that seed (preds/<k>/<split>_S_seed{b}.jsonl; --fallback exec re-executes the bundle
parse instead, used only for train dry runs where no S prediction file exists).
Outputs, for every adapter x tau in {off, -0.30, -0.10, -0.03}:
  pred_P_<k>_<split>_<adapter>_tau<t>_seed{0,1,2}.jsonl ({"quid", "ranked"}; P_k hit = mean over the 3 seeds)
and exec_P_<k>_<split>_<adapter>.jsonl.gz (per-question status, reason, mean log-prob, P RULE / S lists) plus
exec_stats_P_<k>_<split><tag>.json (invalid / fallback rates by reason and tau, sha256 of every file).
Never reads an answer.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"   # process-parallel executor: single-threaded OpenMP (fork safety)
os.environ["MKL_NUM_THREADS"] = "1"
import argparse, collections, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pk_common as PC

EXEC_DIR = os.environ.get("KPCT_EXECUTOR", "/kpct/executor")
sys.path.insert(0, EXEC_DIR)
_W = {}


def tau_name(t):
    return "off" if t is None else f"{t:+.2f}".replace("+", "p").replace("-", "m").replace(".", "")


def check_names(c, K):
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


def work(items):
    import kpct_executor as X
    import kpct_core as K
    ex = X.get(_W["k"])
    out = []
    for raw, b in items:
        q = b["question"]
        o = dict(quid=int(raw["quid"]), mean_logprob=raw.get("mean_logprob"), status="ok", reason=None)
        try:
            c = PC.validate_compact(PC.extract_json(raw["text"]))
        except PC.PInvalid as e_:
            o.update(status="fallback", reason="invalid_json", detail=str(e_)); out.append(o); continue
        bad = check_names(c, K)
        if bad:
            o.update(status="fallback", reason="unknown_name", detail=bad); out.append(o); continue
        try:
            p = X.compact_to_parse(c, q, b["parse"])
            r = ex.execute(p, question=q, probs=_W["probs"])
        except Exception as e_:
            o.update(status="fallback", reason="invalid_json", detail=f"{type(e_).__name__}: {e_}"); out.append(o); continue
        if not r["valid"]:
            o.update(status="fallback", reason="unknown_name" if "vocabulary" in str(r["error"]) else "invalid_json", detail=r["error"])
        elif r["pool_size"] == 0:
            o.update(status="fallback", reason="empty_pool", detail=None)
        else:
            o.update(rule=r["rule"], s=r["s"], s_mean=r["s_mean"], pool_size=r["pool_size"], n_sat=r["n_operator_satisfied"])
        out.append(o)
    return out


def exec_structural(items):
    """fallback lists by re-executing the bundle (structural) parse -- train dry runs only"""
    import kpct_executor as X
    ex = X.get(_W["k"])
    out = {}
    for raw, b in items:
        r = ex.execute(b["parse"], question=b["question"], probs="given")
        out[int(b["quid"])] = {f"seed{s}": (r["s"][f"seed{s}"] if r["s"] else r["rule"]) for s in range(3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", required=True, help="executor subset (k01_s0 ... k100)")
    ap.add_argument("--split", required=True)
    ap.add_argument("--raw", required=True, help="name=raw_P_<split>_<adapter>.jsonl.gz[,name=...]")
    ap.add_argument("--bundles", default=None, help="default /kpct/bundles/<k>/<split>.jsonl")
    ap.add_argument("--preds", default="/kpct/preds", help="S_k prediction dir (preds/<k>/<split>_S_seed{b}.jsonl)")
    ap.add_argument("--fallback", default="preds", choices=["preds", "exec"])
    ap.add_argument("--taus", default="off,-0.30,-0.10,-0.03")
    ap.add_argument("--probs", default="classifier", choices=["classifier", "given"])
    ap.add_argument("--out", required=True); ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    taus = [None if x == "off" else float(x) for x in a.taus.split(",")]
    bfile = a.bundles or f"/kpct/bundles/{a.k}/{a.split}.jsonl"
    raws = [x.split("=", 1) for x in a.raw.split(",")]
    R = {n: {int(r["quid"]): r for r in PC.read_jsonl(p)} for n, p in raws}
    need = set().union(*[set(v) for v in R.values()])
    B = {}
    for r in PC.iter_jsonl(bfile):
        q = int(r["quid"])
        if q in need:
            B[q] = dict(quid=q, question=r["question"], parse=r["parse"])
    assert set(B) == need, f"{len(need - set(B))} raw quids without a bundle row"
    import kpct_executor as X
    _W.update(k=a.k, probs=a.probs if a.k != "k0" else "given")
    X.get(a.k)
    import torch
    torch.set_num_threads(1)
    if _W["probs"] == "classifier":
        X.get(a.k).parser()                 # load the k% classifier BEFORE forking (no torch.load in forked workers)
    import multiprocessing as mp
    pool = mp.get_context("fork").Pool(a.workers) if a.workers > 1 else None
    pmap = pool.map if pool else (lambda f, xs: [f(x) for x in xs])
    quids = sorted(need)
    # ---------------------------------------------------------------- fallback lists (S_k's own output)
    if a.fallback == "preds":
        FB = {}
        for s in range(3):
            f = os.path.join(a.preds, a.k, f"{a.split}_S_seed{s}.jsonl")
            for r in PC.iter_jsonl(f):
                q = int(r["quid"])
                if q in need:
                    FB.setdefault(q, {})[f"seed{s}"] = r["ranked"]
        fb_src = {f"S_seed{s}": PC.sha256_file(os.path.join(a.preds, a.k, f"{a.split}_S_seed{s}.jsonl")) for s in range(3)}
    else:
        it = [(None, B[q]) for q in quids]
        FB = {}
        for d in pmap(exec_structural, [it[i:i + 50] for i in range(0, len(it), 50)]):
            FB.update(d)
        fb_src = {"exec": "bundle parse re-executed (dry run)"}
    assert all(q in FB and len(FB[q]) == 3 for q in quids), "missing S_k fallback lists"
    stats = dict(arm="P", k=a.k, split=a.split, n=len(quids), probs=_W["probs"], taus=[tau_name(t) for t in taus],
                 fallback_source=fb_src, bundle_file=bfile, adapters={},
                 code_sha256={f: PC.sha256_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), f)) for f in ("pk_execute.py", "pk_common.py")}
                 | {f: PC.sha256_file(os.path.join(EXEC_DIR, f)) for f in ("kpct_executor.py", "kpct_core.py")})
    for name, _ in raws:
        items = [(R[name][q], B[q]) for q in quids]
        res = [x for ch in pmap(work, [items[i:i + 50] for i in range(0, len(items), 50)]) for x in ch]
        ef = os.path.join(a.out, f"exec_P_{a.k}_{a.split}_{name}{a.tag}.jsonl.gz")
        PC.write_jsonl(ef, res)
        st = dict(exec_file=os.path.basename(ef), exec_sha256=PC.sha256_file(ef),
                  reasons=dict(collections.Counter(r["reason"] for r in res if r["status"] != "ok")),
                  details=dict(collections.Counter(str(r.get("detail")).split(":")[0] for r in res if r["status"] != "ok").most_common(12)),
                  invalid_rate=sum(r["reason"] in ("invalid_json", "unknown_name") for r in res) / max(1, len(res)),
                  structural_fallback_rate_before_tau=sum(r["status"] != "ok" for r in res) / max(1, len(res)), tau={})
        for t in taus:
            use_p = [r["status"] == "ok" and (t is None or (r["mean_logprob"] is not None and r["mean_logprob"] >= t)) for r in res]
            tn = tau_name(t)
            st["tau"][tn] = dict(fallback_rate=1 - sum(use_p) / max(1, len(res)),
                                 low_confidence=sum(1 for r, u in zip(res, use_p) if r["status"] == "ok" and not u), files={})
            for s in range(3):
                rows = [dict(quid=r["quid"], ranked=(r["s"][f"seed{s}"] if r["s"] else r["rule"]) if u else FB[r["quid"]][f"seed{s}"])
                        for r, u in zip(res, use_p)]
                pf = os.path.join(a.out, f"pred_P_{a.k}_{a.split}_{name}_tau{tn}_seed{s}{a.tag}.jsonl")
                PC.write_jsonl(pf, rows)
                st["tau"][tn]["files"][os.path.basename(pf)] = PC.sha256_file(pf)
        stats["adapters"][name] = st
        print(name, json.dumps({k_: st[k_] for k_ in ("reasons", "invalid_rate", "structural_fallback_rate_before_tau")}), flush=True)
    if pool:
        pool.close()
    stats["secs"] = round(time.time() - t0, 1)
    json.dump(stats, open(os.path.join(a.out, f"exec_stats_P_{a.k}_{a.split}{a.tag}.json"), "w"), indent=1)
    os.system(f"chmod -R a+rX {a.out}")


if __name__ == "__main__":
    main()
