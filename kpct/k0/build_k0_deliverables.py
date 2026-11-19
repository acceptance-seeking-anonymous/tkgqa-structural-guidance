#!/usr/bin/env python3
"""build_k0_deliverables.py -- LOCAL: assemble multitq_kpct_k0_freeze.json, multitq_kpct_k0_dev_results.json and
multitq_kpct_k0_report.md from the harvested out/ folders of the P_0 and G_0 jobs (k0_run.sh p0 / g0).
Reads no answer and computes no score: every number comes from the on-host selection / freeze / stats files.
Test prediction files are hash-verified (gunzipped harvest copy vs the freeze sha256) but never scored.
"""
import argparse, collections, gzip, hashlib, json, os, time


def sha_recipe(o):
    d = {k: v for k, v in o.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(d, indent=1, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def jl(p):
    return json.load(open(p)) if p and os.path.exists(p) else None


def gz_sha(p):
    h = hashlib.sha256()
    with gzip.open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def usage_log(dirs):
    rows, seen = [], set()
    for d in dirs:
        p = os.path.join(d, "usage_log_k0.tsv")
        if os.path.exists(p):
            for l in open(p):
                f = l.rstrip("\n").split("\t")
                if len(f) >= 10 and f[9] not in seen:
                    seen.add(f[9]); rows.append(dict(track=f[0], cards=f[1], n_cards=int(f[2]), start_utc=f[3], end_utc=f[4],
                                                     seconds=int(f[5]), gpu_h=float(f[6]), rc=int(f[7]), tag=f[8], container=f[9]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0_out", required=True); ap.add_argument("--g0_out", default=None)
    ap.add_argument("--registration", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    reg = json.load(open(a.registration))
    dirs = [d for d in (a.p0_out, a.g0_out) if d]
    S = {arm: jl(os.path.join(d, f"select_{arm}.json")) for arm, d in (("P0", a.p0_out), ("G0", a.g0_out or a.p0_out))}
    F = {arm: jl(os.path.join(d, f"freeze_{arm}.json")) for arm, d in (("P0", a.p0_out), ("G0", a.g0_out or a.p0_out))}
    L = usage_log(dirs)
    gpu = dict(rows=L, total_gpu_h=round(sum(r["gpu_h"] for r in L), 4),
               by_arm={arm: round(sum(r["gpu_h"] for r in L if r["tag"].startswith(arm)), 4) for arm in ("P0", "G0")})
    # ---------------------------------------------------------------- test hash verification (no scoring)
    tv = {}
    for arm, d in (("P0", a.p0_out), ("G0", a.g0_out)):
        fz = F.get(arm)
        if not d or not fz or not fz.get("test_predictions"):
            continue
        pf = fz["test_prediction_file"]; gzp = os.path.join(d, pf + ".gz")
        tv[arm] = dict(file=pf, freeze_sha256=fz["test_predictions"][pf],
                       harvest_copy_sha256=(gz_sha(gzp) if os.path.exists(gzp) else None))
        tv[arm]["match"] = tv[arm]["harvest_copy_sha256"] == tv[arm]["freeze_sha256"]
    # ---------------------------------------------------------------- primary LLM arm at k = 0 (rule criteria.primary_llm_arm)
    prim = None
    if S.get("P0") and S.get("G0") and F.get("G0"):
        p, g = S["P0"]["selected_hits1"], S["G0"]["selected_hits1"]
        prim = dict(rule="P_0 vs G_0 on dev_sel2000 Hits@1; |difference| < 0.0005 -> G_0", P_0=p, G_0=g, diff_G_minus_P=g - p,
                    primary=("G_0" if abs(g - p) < 0.0005 or g > p else "P_0"))
    elif S.get("P0"):
        prim = dict(rule="G_0 not frozen -> P_0", primary="P_0")
    dry = any(bool(f and f.get("dry_run")) for f in F.values())
    fz = dict(name="multitq_kpct_k0_freeze", track="k0", dry_run=dry, rule=reg["rule"], subsets_manifest_sha256=reg["subsets_manifest_sha256"],
              registration_sha256=reg["sha256"], amendments=reg["amendments_in_force"],
              arms={"P_0": F.get("P0"), "G_0": F.get("G0")},
              test_prediction_files={k.replace("0", "_0"): dict(file=v["file"], sha256=v["freeze_sha256"]) for k, v in tv.items()},
              test_hash_verification=tv, test_scored=False, primary_llm_arm_k0=prim, gpu=gpu,
              NOT_RUN=[x for x, f in (("P_0", F.get("P0")), ("G_0", F.get("G0"))) if not (f and f.get("test_predictions"))],
              built_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    fz["sha256"] = sha_recipe(fz)
    json.dump(fz, open(os.path.join(a.out, "multitq_kpct_k0_freeze.json"), "w"), indent=1)
    # ---------------------------------------------------------------- dev results
    ex = {s: jl(os.path.join(a.p0_out, f"exec_stats_P0_{s}.json")) for s in ("dev_sel2000", "test")}
    po = {s: jl(os.path.join(a.g0_out, f"post_stats_G0_{s}.json")) if a.g0_out else None for s in ("dev_sel2000", "test")}
    inf = {f"{arm}_{s}": jl(os.path.join(d, f"infer_stats_{arm}_{s}.json")) for arm, d in (("P0", a.p0_out), ("G0", a.g0_out or a.p0_out))
           for s in ("dev_sel2000", "test")}
    bst = {s: jl(os.path.join(a.p0_out, f"bundle_stats_{s}.json")) for s in ("dev_sel2000", "test")}
    dev = dict(name="multitq_kpct_k0_dev_results", dry_run=dry, split="dev_sel2000 (n = 2,000; selection only)",
               caveat="optimistic: each arm's figure is the best of 3 prompt designs chosen on the same 2,000 questions; descriptive input for the k = 0 primary-arm choice only",
               P_0=dict(selection=S.get("P0"), exec_dev=ex["dev_sel2000"], exec_test_rates=(ex["test"] or {}).get("out")),
               G_0=dict(selection=S.get("G0"), post_dev=po["dev_sel2000"], post_test_rates=(po["test"] or {}).get("out") if po["test"] else None),
               bundles=bst, inference=inf, primary_llm_arm_k0=prim, gpu=gpu)
    dev["sha256"] = sha_recipe(dev)
    json.dump(dev, open(os.path.join(a.out, "multitq_kpct_k0_dev_results.json"), "w"), indent=1)
    print(json.dumps(dict(primary=prim, gpu_h=gpu["total_gpu_h"], test_hash_ok={k: v["match"] for k, v in tv.items()}, dry=dry)))


if __name__ == "__main__":
    main()
