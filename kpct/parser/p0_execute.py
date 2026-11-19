#!/usr/bin/env python3
"""p0_execute.py -- execute P_0 parses with the k = 0 executor: operators + candidate retrieval + RULE ordering only
(no re-ranker, no classifier, no train-fit linker artefact; padding from KG statistics only) (rule arms.P_0).

Entities (rule arms.P_0.entities): an exact KG name is used directly (exact match, else a unique case-insensitive match
after '_' -> ' ' and whitespace collapsing); otherwise the mention is mapped by the BASE linker
(kpct_core.make_linker(): stemming, demonyms; no sector aliases, no block list, no train statistics): first the
generated entity string, then the mention text; the linked mention with the widest token span wins (ties: first),
its top alternative is used. The anchor is mapped the same way (or takes the mapped entity of the mention whose
generated entity string it repeats). Relation: exact KG relation name (space/underscore and case normalised).
Invalid JSON or an unknown name -> no prediction (empty ranked list = a miss); invalid rate reported (rule
arms.P_0.invalid). A parse whose operator pool is empty keeps the executor's KG-statistics padding.
Output: pred_P0_<split>_<design>.jsonl ({"quid", "ranked"}) + exec_P0_<split>_<design>.jsonl.gz + exec_stats_P0_<split>.json.
--verify_answers: ONLY for the 8 train exemplars (asserts the quids are the exemplar quids) -- hit@1 of the hand parses.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"   # process-parallel executor: single-threaded OpenMP (fork safety)
os.environ["MKL_NUM_THREADS"] = "1"
import argparse, collections, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pk_common as PC

EXEC_DIR = os.environ.get("KPCT_EXECUTOR", "/kpct/executor")
sys.path.insert(0, EXEC_DIR)
_W = {}


class Mapper:
    def __init__(self, K):
        self.K = K
        kg = K.kg()
        self.e2i = kg.e2i; self.names = kg.ent_names
        cf = collections.defaultdict(list)
        for n in kg.ent_names:
            cf[self.norm(n).casefold()].append(n)
        self.cf = {k: v[0] for k, v in cf.items() if len(v) == 1}
        self.rel = {}
        for r in kg.rel_names:
            self.rel[r] = r; self.rel.setdefault(r.casefold(), r)
            self.rel.setdefault(r.replace("_", " ").casefold(), r)
        self.LK = K.make_linker()          # base linker: no aliases, no block list

    @staticmethod
    def norm(s):
        return re.sub(r"\s+", " ", str(s).replace("_", " ")).strip()

    def exact(self, s):
        n = self.norm(s)
        if n in self.e2i:
            return n
        return self.cf.get(n.casefold())

    def link(self, s):
        if not s or not s.strip():
            return None
        ms, ws = self.LK.link(s)
        ms = [m for m in ms if m["alts"]]
        if not ms:
            return None
        m = sorted(ms, key=lambda m: (-(m["hi"] - m["lo"]), m["lo"]))[0]
        return self.names[m["alts"][0]["e"]]

    def entity(self, ent, text):
        e = self.exact(ent)
        if e is not None:
            return e, "exact"
        for src, s in (("link_entity", ent), ("link_text", text)):
            e = self.link(s)
            if e is not None:
                return e, src
        return None, "unknown"

    def relation(self, r):
        r = str(r).strip()
        return self.rel.get(r) or self.rel.get(r.casefold()) or self.rel.get(r.replace("_", " ").casefold())


def work(items):
    import kpct_executor as X
    import kpct_core as K
    ex = X.get("k0")
    M = _W.get("M") or Mapper(K)
    _W["M"] = M
    out = []
    for raw, q in items:
        o = dict(quid=int(raw["quid"]), status="ok", reason=None, map_sources=[])
        try:
            c = PC.validate_compact(PC.extract_json(raw["text"]))
        except PC.PInvalid as e_:
            o.update(status="invalid", reason="invalid_json", detail=str(e_), ranked=[]); out.append(o); continue
        rel = M.relation(c["relation"])
        if rel is None:
            o.update(status="invalid", reason="unknown_relation", ranked=[]); out.append(o); continue
        ments, bad, by_raw = [], False, {}
        for m in c["mentions"]:
            e, src = M.entity(m["entity"], m["text"])
            o["map_sources"].append(src)
            if e is None:
                bad = True; break
            ments.append({"text": m["text"], "entity": e}); by_raw[M.norm(m["entity"])] = e
        if bad or not ments:
            o.update(status="invalid", reason="unknown_entity" if bad else "no_mentions", ranked=[]); out.append(o); continue
        anc = None
        if c["anchor"] is not None:
            anc = by_raw.get(M.norm(c["anchor"])) or M.entity(c["anchor"], "")[0]
            if anc is None:
                o.update(status="invalid", reason="unknown_anchor", ranked=[]); out.append(o); continue
        c2 = dict(c, relation=rel, mentions=ments, anchor=anc)
        try:
            p = X.compact_to_parse(c2, q, None)
            r = ex.execute(p, question=q, probs="given")
        except Exception as e_:
            o.update(status="invalid", reason="invalid_json", detail=f"{type(e_).__name__}: {e_}", ranked=[]); out.append(o); continue
        if not r["valid"]:
            o.update(status="invalid", reason="unknown_name", detail=r["error"], ranked=[])
        else:
            o.update(ranked=r["rule"], pool_size=r["pool_size"], n_sat=r["n_operator_satisfied"], parse_used=c2)
        out.append(o)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--raw", required=True, help="name=raw_P0_<split>_<design>.jsonl.gz[,...]")
    ap.add_argument("--questions", required=True, help="questions json of the split (quid + question read)")
    ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--verify_answers", action="store_true"); ap.add_argument("--exemplars", default=None)
    a = ap.parse_args()
    t0 = time.time(); os.makedirs(a.out, exist_ok=True)
    raws = [x.split("=", 1) for x in a.raw.split(",")]
    R = {n: {int(r["quid"]): r for r in PC.read_jsonl(p)} for n, p in raws}
    need = set().union(*[set(v) for v in R.values()])
    allq = json.load(open(a.questions))
    Q = {int(q["quid"]): q["question"] for q in allq if int(q["quid"]) in need}
    gold = None
    if a.verify_answers:
        exq = set(json.load(open(a.exemplars))["quids"])
        assert need <= exq, "--verify_answers is only allowed for the 8 P_0 exemplar quids"
        gold = {int(q["quid"]): q for q in allq if int(q["quid"]) in need}
    del allq
    assert set(Q) == need
    import kpct_executor as X
    import kpct_core as K
    X.get("k0"); _W["M"] = Mapper(K)
    import multiprocessing as mp
    pool = mp.get_context("fork").Pool(a.workers) if a.workers > 1 else None
    pmap = pool.map if pool else (lambda f, xs: [f(x) for x in xs])
    quids = sorted(need)
    stats = dict(arm="P_0", split=a.split, n=len(quids), executor="k0 (operators + retrieval + RULE; KG-statistics padding)", out={},
                 code_sha256={f: PC.sha256_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), f)) for f in ("p0_execute.py", "pk_common.py")})
    for name, _ in raws:
        items = [(R[name][q], Q[q]) for q in quids]
        res = [x for ch in pmap(work, [items[i:i + 50] for i in range(0, len(items), 50)]) for x in ch]
        ef = os.path.join(a.out, f"exec_P0_{a.split}_{name}.jsonl.gz"); PC.write_jsonl(ef, res)
        pf = os.path.join(a.out, f"pred_P0_{a.split}_{name}.jsonl")
        PC.write_jsonl(pf, [dict(quid=r["quid"], ranked=r["ranked"]) for r in res])
        st = dict(exec_sha256=PC.sha256_file(ef), pred_file=os.path.basename(pf), pred_sha256=PC.sha256_file(pf),
                  invalid_rate=sum(r["status"] != "ok" for r in res) / max(1, len(res)),
                  reasons=dict(collections.Counter(r["reason"] for r in res if r["status"] != "ok")),
                  map_sources=dict(collections.Counter(s for r in res for s in r["map_sources"])),
                  empty_pool=sum(1 for r in res if r["status"] == "ok" and r.get("pool_size") == 0))
        if gold is not None:
            sys.path.insert(0, os.path.join(EXEC_DIR, "struct"))
            import multitq_eval as ME
            st["verify_hit1"] = {str(r["quid"]): int(ME.hit_at_k(gold[r["quid"]], r["ranked"], 1)) for r in res}
            st["verify_hit10"] = {str(r["quid"]): int(ME.hit_at_k(gold[r["quid"]], r["ranked"], 10)) for r in res}
            st["verify_top3"] = {str(r["quid"]): r["ranked"][:3] for r in res}
            st["verify_gold"] = {str(q): gold[q]["answers"] for q in quids}
        stats["out"][name] = st
        print(name, json.dumps({k_: st[k_] for k_ in ("invalid_rate", "reasons", "map_sources")}), flush=True)
    if pool:
        pool.close()
    stats["secs"] = round(time.time() - t0, 1)
    json.dump(stats, open(os.path.join(a.out, f"exec_stats_P0_{a.split}.json"), "w"), indent=1)
    os.system(f"chmod -R a+rX {a.out}")


if __name__ == "__main__":
    main()
