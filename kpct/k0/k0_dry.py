#!/usr/bin/env python3
"""k0_dry.py -- CPU plumbing dry run of the k = 0 pipeline (no GPU, no engine, nothing scored for real).

  prep : the first N dev_sel2000 quids (sorted) -> mock question file (quid, question, answers = ["__PLACEHOLDER__"],
         answer_type "entity"; question text taken from the structural k01_s0 dev_sel2000 bundle rows, which carry no
         gold, so no dev answer is read) + stand-in P_0 generations: the k01_s0 structural top-1 parse of each question
         serialised to the compact P JSON (pk_common.compact_from_parse; plumbing input only -- the real P_0 never sees
         a k01 artefact), the last row replaced by non-JSON text to exercise the invalid path.
  fake : stand-in generations in the exact raw format of k0_infer_70b.py, without an engine:
         P0 -> the prepared compact JSON texts; G0 -> cycle over {RULE-1 candidate, "The answer is <c>.", lower-case /
         underscore variant of <c>, a date, empty string} to exercise answer canonicalisation and the invalid path.
"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.environ.get("KPCT_PARSER_CODE", "/kpct/k0/code/parser"))
import pk_common as PC          # noqa: E402
import gk_common as GC          # noqa: E402


def prep(a):
    keep = sorted(PC.load_list(a.manifest, "dev_sel2000"))[:a.n]
    rows = {int(r["quid"]): r for r in GC.read_jsonl(a.struct_bundles) if int(r["quid"]) in set(keep)}
    assert set(rows) == set(keep)
    assert not any("gold" in r for r in rows.values())
    mock = [dict(quid=q, question=rows[q]["question"], answers=["__PLACEHOLDER__"], answer_type="entity") for q in keep]
    json.dump(mock, open(os.path.join(a.out, "mock_questions.json"), "w"), indent=1)
    json.dump(keep, open(os.path.join(a.out, "dry_quids.json"), "w"))
    texts = {}
    for i, q in enumerate(keep):
        p = rows[q]["parse"]
        t = json.loads(json.dumps(p)); t["mentions"] = [dict(m, alternatives=m["alternatives"][:1]) for m in t["mentions"]]
        t["relations"] = t["relations"][:1]
        texts[q] = PC.dumps_compact(PC.compact_from_parse(t, rows[q]["question"])) if i < len(keep) - 1 else "I cannot parse this question."
    json.dump({str(k): v for k, v in texts.items()}, open(os.path.join(a.out, "p0_standin_texts.json"), "w"), indent=1)
    print(json.dumps(dict(n=len(keep), quids=keep)))


def fake(a):
    os.makedirs(a.out, exist_ok=True)
    for d in a.designs.split(","):
        rows = []
        if a.arm == "P0":
            T = {int(k): v for k, v in json.load(open(a.texts)).items()}
            for q in sorted(T):
                rows.append(dict(quid=q, text=T[q], n_out=len(T[q].split()), mean_logprob=-0.05, finish_reason="stop"))
        else:
            bl = sorted([GC.normalize_bundle(r) for r in GC.read_jsonl(a.bundles)], key=lambda b: b["quid"])
            for i, b in enumerate(bl):
                c = b["candidates"][0]["candidate"] if b["candidates"] else "Barack Obama"
                txt = [c, f"The answer is {c}.", c.lower().replace(" ", "_"), "2012-05", ""][i % 5]
                rows.append(dict(quid=b["quid"], text=txt, n_out=len(txt.split()), finish_reason="stop", prompt_tok=0, n_groups_dropped=0))
        f = os.path.join(a.out, f"raw_{a.arm}_{a.split}_{d}.jsonl.gz")
        PC.write_jsonl(f, rows)
        open(f + ".done", "w").write(PC.sha256_file(f))
        json.dump(dict(arm=a.arm, split=a.split, n=len(rows), dry_standin=True, out={d: dict(raw_file=os.path.basename(f))}),
                  open(os.path.join(a.out, f"infer_stats_{a.arm}_{a.split}_standin_{d}.json"), "w"), indent=1)
        print(a.arm, d, len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "fake"])
    ap.add_argument("--manifest", default="/kpct/k0/code/multitq_kpct_subsets.json")
    ap.add_argument("--struct_bundles", default="/kpct/bundles/k01_s0/dev_sel2000.jsonl")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--arm", choices=["P0", "G0"]); ap.add_argument("--split", default="dev_sel2000")
    ap.add_argument("--designs", default="d1"); ap.add_argument("--texts"); ap.add_argument("--bundles")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    prep(a) if a.cmd == "prep" else fake(a)


if __name__ == "__main__":
    main()
