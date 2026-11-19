#!/usr/bin/env python3
"""gr_prep.py -- grounded_reader track: generation plans for gr_infer.py from the gr_build.py status files (gold-free).

Rows to GENERATE for (level k, split, source m) = status ok AND not same_prompt; same_prompt rows reuse the frozen G_k
output and fallback rows keep it (gr_assemble.py). Build dirs: /kpct/grounded_reader/build/<k>/<split>/.
  stage dev   : per level in --levels: dev (dev_sel2000 ORIG) and devpara (llm root, vidx 5 = DEV_ACCEPTED) for BOTH
                sources, then the I2 reuse check (frozen structural bundles of at most 150 dev ORIG + 150 DEV_ACCEPTED rows
                that are same_prompt under at least one source; default_rng([20261118, 11]) / ([20261118, 12]) choice
                over the ascending eligible quids).
  stage final : selected source per level (--selection json {k: source | 'G_k'}): registered cut order
                [r3rw_k, testr3orig_k for k in levels] -> [r1p3_k, r2rw_k, testr12orig_k for k in levels] -> [testrest_k].
Writes <out>/plan_<stage>.json, quid lists under <out>/quids/, and <out>/counts_<stage>.json.
"""
import argparse, json, os
import numpy as np

GR = "/kpct/grounded_reader"
ADAPTERS = {"k01_s0": "/kpct/guided/adapters/G_k01_s0_ordS_s0/ep2", "k05_s0": "/kpct/guided/adapters/G_k05_s0_ordS_s0/ep2",
            "k100": "/kpct/guided/adapters/G_k100_ordS_s0/ep1"}
FROZEN_BUNDLES = {"dev": "/kpct/bundles/{k}/dev_sel2000.jsonl", "devpara": "/kpct/robustness_r3/llm/bundles/{k}/variants.jsonl"}
FROZEN_S0 = {("dev", "k100"): "/mtq/phase1_structural/preds/dev_S_seed0.jsonl", ("dev", None): "/kpct/preds/{k}/dev_sel2000_S_seed0.jsonl",
             ("devpara", None): "/kpct/robustness_r3/llm/preds/{k}/variants_S_seed0.jsonl"}
SEED = 20261118


def status(k, split, m):
    p = f"{GR}/build/{k}/{split}/status_{m}.jsonl"
    return [json.loads(l) for l in open(p) if l.strip()]


def need_gen(k, split, m, filt=None):
    return sorted(s["quid"] for s in status(k, split, m) if s["status"] == "ok" and not s["same_prompt"] and (filt is None or filt(s["quid"])))


def wq(out, name, qs):
    os.makedirs(f"{out}/quids", exist_ok=True)
    p = f"{out}/quids/{name}.json"
    json.dump(sorted(int(q) for q in qs), open(p, "w"))
    return p


GEN_ROOT = {}


def item(out, stage, k, m, split_build, label, qs):
    return dict(name=f"{label}_{k}_{m}", adapter=f"G_{k}", bundles=f"{GR}/build/{k}/{split_build}/bundles_{m}.jsonl",
                quids=wq(out, f"{stage}_{label}_{k}_{m}", qs), s_list=f"{GR}/build/{k}/{split_build}/s0_{m}.jsonl",
                out=f"{GEN_ROOT['root']}/{k}/{m}", split=label, n=len(qs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["dev", "final"])
    ap.add_argument("--levels", required=True)
    ap.add_argument("--selection", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu_mem", type=float, default=0.40)
    ap.add_argument("--budget_s", type=float, required=True)
    ap.add_argument("--sec_per_prompt", type=float, default=0.075)
    ap.add_argument("--gen_root", default=None, help="output root of the generated files (default <GR>/gen/<stage>)")
    a = ap.parse_args()
    GEN_ROOT["root"] = a.gen_root or f"{GR}/gen/{a.stage}"
    levels = a.levels.split(",")
    items, counts = [], {}
    if a.stage == "dev":
        for k in levels:
            for m in ("PALONE", "PK"):
                qd = need_gen(k, "dev", m)
                qp = need_gen(k, "llm", m, lambda q: q % 10 == 5)
                items.append(item(a.out, "dev", k, m, "dev", "dev", qd))
                items.append(item(a.out, "dev", k, m, "llm", "devpara", qp))
                counts[f"{k}_{m}"] = dict(dev=len(qd), devpara=len(qp))
        for k in levels:
            for split_b, label, vfilt, sd in (("dev", "dev", None, 11), ("llm", "devpara", 5, 12)):
                same = set()
                for m in ("PALONE", "PK"):
                    same |= {s["quid"] for s in status(k, split_b, m) if s["same_prompt"] and (vfilt is None or s["quid"] % 10 == vfilt)}
                el = sorted(same)
                pick = [el[i] for i in np.random.default_rng([SEED, sd]).choice(len(el), size=min(150, len(el)), replace=False)] if el else []
                s0 = FROZEN_S0.get((label, k)) or FROZEN_S0[(label, None)]
                items.append(dict(name=f"reuse_{label}_{k}", adapter=f"G_{k}", bundles=FROZEN_BUNDLES[label].format(k=k),
                                  quids=wq(a.out, f"dev_reuse_{label}_{k}", pick), s_list=s0.format(k=k),
                                  out=f"{GEN_ROOT['root']}/{k}/REUSECHECK", split=f"reuse{label}", n=len(pick)))
                counts[f"reuse_{label}_{k}"] = dict(eligible=len(el), sampled=len(pick))
    else:
        sel = json.load(open(a.selection))
        lv = [k for k in levels if sel.get(k) in ("PALONE", "PK")]
        r3 = {s["quid"] // 10 for s in status(lv[0], "llm", "PK") if s["quid"] % 10 == 4} if lv else set()
        r1 = {s["quid"] // 10 for s in status(lv[0], "r1", "PK")} if lv else set()
        r2 = {s["quid"] // 10 for s in status(lv[0], "r2", "PK")} if lv else set()
        assert not (r3 & r1) and not (r3 & r2) and not (r1 & r2), "probe samples must be disjoint"
        for k in lv:
            m = sel[k]
            items.append(item(a.out, "final", k, m, "llm", "r3rw", need_gen(k, "llm", m, lambda q: q % 10 == 4)))
            items.append(item(a.out, "final", k, m, "test", "testr3orig", need_gen(k, "test", m, lambda q: q in r3)))
        for k in lv:
            m = sel[k]
            items.append(item(a.out, "final", k, m, "r1", "r1p3", need_gen(k, "r1", m)))
            items.append(item(a.out, "final", k, m, "r2", "r2rw", need_gen(k, "r2", m)))
            items.append(item(a.out, "final", k, m, "test", "testr12orig", need_gen(k, "test", m, lambda q: q in r1 or q in r2)))
        for k in lv:
            m = sel[k]
            items.append(item(a.out, "final", k, m, "test", "testrest", need_gen(k, "test", m, lambda q: q not in r1 and q not in r2 and q not in r3)))
        counts = {it["name"]: it["n"] for it in items}
        counts["probe_quids"] = dict(r3=len(r3), r1=len(r1), r2=len(r2))
    plan = dict(adapters={f"G_{k}": ADAPTERS[k] for k in levels}, gpu_mem=a.gpu_mem, budget_s=a.budget_s,
                sec_per_prompt=a.sec_per_prompt, items=items)
    json.dump(plan, open(f"{a.out}/plan_{a.stage}.json", "w"), indent=1)
    tot = sum(it["n"] for it in items)
    counts["total_prompts"] = tot
    counts["projected_s"] = round(tot * a.sec_per_prompt, 1)
    json.dump(counts, open(f"{a.out}/counts_{a.stage}.json", "w"), indent=1)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
