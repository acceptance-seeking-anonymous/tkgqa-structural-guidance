#!/usr/bin/env python3
"""gu_retrieve.py -- registered GU_k retriever (rule arms.GU_k.retriever). Not trained or tuned; identical for every k.

BAAI/bge-large-en-v1.5, normalised embeddings, cosine (inner product of unit vectors); passages = one verbalised fact
each "<subject> <relation, _ -> space> <object> on <YYYY-MM-DD>" for all 461,329 full-KG facts (kg/full.txt order);
query = "Represent this sentence for searching relevant passages: " + question; top-20 facts in retrieval order
(score descending, ties by fact index).

  --mode facts    embed all facts -> <out>/fact_emb_f16.npy (+ timing)
  --mode queries  embed the questions of --qfile restricted to --quids and retrieve top-20 -> <out>/<name>.jsonl.gz
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gk_common as GC

MODEL = "BAAI/bge-large-en-v1.5"
QPFX = "Represent this sentence for searching relevant passages: "
TOPK = 20


def verbalise(f):
    h, r, t, d = f
    return f"{h} {r.replace('_', ' ')} {t} on {d}"


def load_model():
    import torch
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL, device="cuda:0")
    m.half()
    return m, torch


def embed(m, texts, bs):
    return m.encode(texts, batch_size=bs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float16)


def retrieve(torch, F, Qe, k=TOPK, chunk=2048, over=64):
    Ft = torch.from_numpy(F).to("cuda:0")
    ids, scs = [], []
    for s in range(0, len(Qe), chunk):
        q = torch.from_numpy(Qe[s:s + chunk]).to("cuda:0")
        sc = (q.float() @ Ft.float().T)
        v, i = torch.topk(sc, over, dim=1)
        v, i = v.cpu().numpy(), i.cpu().numpy()
        for vr, ir in zip(v, i):
            o = np.lexsort((ir, -vr))[:k]       # score desc, ties by fact index (deterministic)
            ids.append(ir[o].tolist()); scs.append(np.round(vr[o], 5).tolist())
    return ids, scs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["facts", "queries"])
    ap.add_argument("--kg", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--emb", default=None, help="fact_emb_f16.npy for --mode queries")
    ap.add_argument("--qfile", default=None, help="questions json or jsonl(.gz) with quid/question")
    ap.add_argument("--quids", default=None, help="json list of quids (default: all in qfile)")
    ap.add_argument("--name", default=None); ap.add_argument("--bs", type=int, default=512)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    kg = GC.load_kg(a.kg)
    assert len(kg) == 461329, len(kg)
    t0 = time.time()
    m, torch = load_model()
    t_load = time.time() - t0
    st = dict(model=MODEL, mode=a.mode, load_s=round(t_load, 1))
    if a.mode == "facts":
        P = [verbalise(f) for f in kg]
        t1 = time.time()
        F = embed(m, P, a.bs)
        torch.cuda.synchronize(); dt = time.time() - t1
        p = os.path.join(a.out, "fact_emb_f16.npy")
        np.save(p, F)
        st.update(n_passages=len(P), embed_s=round(dt, 1), passages_per_s=round(len(P) / dt, 1), dim=int(F.shape[1]),
                  sha256=GC.sha256_file(p), passage_example=P[0], kg_sha256=GC.sha256_file(a.kg))
    else:
        F = np.load(a.emb)
        assert F.shape[0] == len(kg)
        if a.qfile.endswith(".json"):
            Q = json.load(open(a.qfile))
        else:
            Q = GC.read_jsonl(a.qfile)
        keep = set(int(x) for x in json.load(open(a.quids))) if a.quids else None
        Q = [q for q in Q if keep is None or int(q["quid"]) in keep]
        Q.sort(key=lambda q: int(q["quid"]))
        if keep is not None:
            assert len(Q) == len(keep), (len(Q), len(keep))
        t1 = time.time()
        Qe = embed(m, [QPFX + q["question"] for q in Q], a.bs)
        torch.cuda.synchronize(); dq = time.time() - t1
        t2 = time.time()
        ids, scs = retrieve(torch, F, Qe)
        dr = time.time() - t2
        rows = [dict(quid=int(q["quid"]), fact_ids=i, scores=s, facts=[list(kg[j]) for j in i]) for q, i, s in zip(Q, ids, scs)]
        p = os.path.join(a.out, f"{a.name}.jsonl.gz")
        GC.write_jsonl(p, rows)
        st.update(name=a.name, n_queries=len(Q), query_embed_s=round(dq, 1), queries_per_s=round(len(Q) / max(dq, 1e-6), 1),
                  search_s=round(dr, 1), file=os.path.basename(p), sha256=GC.sha256_file(p), quid_list_sha256=GC.list_sha256([q["quid"] for q in Q]),
                  emb_sha256=GC.sha256_file(a.emb))
    st["total_s"] = round(time.time() - t0, 1)
    json.dump(st, open(os.path.join(a.out, f"stats_{a.mode}{'_' + a.name if a.name else ''}.json"), "w"), indent=1)
    print(json.dumps(st), flush=True)


if __name__ == "__main__":
    main()
