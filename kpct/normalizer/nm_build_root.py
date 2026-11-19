"""nm_build_root.py -- normalizer track: apply the guard to raw normalizer outputs, decide per item which prediction it
uses (nm_common.reuse_source), and write the pipeline input root (texts needing a run + 200 identity-control rows).
Gold-free: reads item ids / texts and raw normalizer outputs only.

build(items, raws, configs, controls) ->
  decisions: {config_key: {iid: dict(status, reasons, text, source, squid)}}
  variants:  [{quid: synthetic quid, question}] sorted by synthetic quid (controls 10q+0, runs 10q+v)
"""
import gzip, json
import numpy as np
import nm_common as NC


def read_jsonl(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def config_key(ck, g, tau):
    return f"{ck}|{g}|{tau:g}"


def control_quids(all_quids, seed_tail, n=200):
    qs = sorted(int(q) for q in all_quids)
    idx = np.random.default_rng([20261118, seed_tail]).choice(len(qs), n, replace=False)
    return sorted(qs[i] for i in idx)


def build(items, raws, configs, controls, orig_text, RC, parse_time):
    """items: rows {iid, quid, vidx, text}; raws: {ckpt: {iid: (out, finish_reason)}}; configs: [(ckpt, guard, tau)];
    controls: quids for identity-control rows; orig_text: {quid: original question text}"""
    dec = {}
    need = {}
    for ck, g, tau in configs:
        key = config_key(ck, g, tau); D = {}
        for it in items:
            out, fr = raws[ck][it["iid"]]
            x = it["text"]; o = orig_text[it["quid"]]
            y, st, rs = NC.final_text(x, out, fr, g, tau, RC, parse_time)
            src = NC.reuse_source(x, y, o)
            D[it["iid"]] = dict(status=st, reasons=rs, text=y, source=src)
            if src == "run":
                need.setdefault(it["quid"], set()).add(y)
        dec[key] = D
    sq = NC.assign_squids(need)
    for key, D in dec.items():
        for iid, d in D.items():
            q = int(iid.split(":")[1])
            d["squid"] = sq[(q, d["text"])] if d["source"] == "run" else None
    variants = [dict(quid=10 * q, question=orig_text[q]) for q in controls]
    variants += [dict(quid=s, question=t) for (q, t), s in sq.items()]
    variants.sort(key=lambda r: r["quid"])
    assert len({r["quid"] for r in variants}) == len(variants)
    return dec, variants


def rates(D, iids):
    n = len(iids)
    c = {"unchanged": 0, "changed": 0, "fallback": 0}
    reasons = {}
    src = {"input": 0, "original": 0, "run": 0}
    for i in iids:
        d = D[i]; c[d["status"]] += 1; src[d["source"]] += 1
        for r in d["reasons"]:
            reasons[r] = reasons.get(r, 0) + 1
    return dict(n=n, **{f"{k}_rate": v / max(1, n) for k, v in c.items()}, counts=c, fallback_reasons=reasons, source_counts=src)
