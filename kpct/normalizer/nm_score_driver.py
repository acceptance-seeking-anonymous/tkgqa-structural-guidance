"""nm_score_driver.py -- runs the ONE registered scoring pass of the normalizer track (nm_score.score + analyse) after the
freeze (multitq_normalizer_freeze.json) is saved; writes the results JSON and the per-question hits CSV.
usage: python nm_score_driver.py <freeze.json> <out_dir>
The freeze lists every input file with its sha256; the driver re-verifies them before reading any gold."""
import gzip, json, os, sys
import numpy as np
import pandas as pd
import nm_common as NC
import nm_score as NSC
import multitq_eval as ME


def main():
    fz = json.load(open(sys.argv[1])); out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    bad = [p for p, s in fz["files_sha256"].items() if NC.sha256_file(p) != s]
    assert not bad, f"freeze mismatch: {bad[:5]}"
    items = [json.loads(l) for l in open(fz["paths"]["test_items"]) if l.strip()]
    dec = json.load(open(fz["paths"]["decisions"]))
    test = json.load(open(fz["paths"]["test_json"]))
    gold = {int(q["quid"]): q for q in test}
    th = pd.read_csv(fz["paths"]["test_hits"]); th["quid"] = th["quid"].astype(int); th = th.set_index("quid")
    res = NSC.score(items, dec, gold, th, fz["paths"]["new_root"], fz["paths"]["frozen_rw_roots"], ME, fz["run_arms"])
    an = NSC.analyse(res)
    an["not_run"] = res["not_run"]
    json.dump(an, open(os.path.join(out, "multitq_normalizer_results.json"), "w"), indent=1, default=float)
    # per-question hits
    rows = []
    test_q = [it["quid"] for it in items if it["set"] == "TEST"]
    for name, A in res["arms"].items():
        for at in (1, 10):
            t = A[f"test@{at}"]
            for q, b, m in zip(test_q, t["base"], t["method"]):
                rows.append(dict(set="TEST", quid=q, arm=name, at=at, base_orig=b, method_orig=m))
            for s in ("R3", "R1", "R2"):
                x = A[f"{s}@{at}"]
                for i, q in enumerate(x["quids"]):
                    rows.append(dict(set=s, quid=q, arm=name, at=at, base_orig=x["o_b"][i], base_rw=x["r_b"][i], method_orig=x["o_m"][i], method_rw=x["r_m"][i]))
    pd.DataFrame(rows).to_csv(os.path.join(out, "multitq_normalizer_per_question_hits.csv.gz"), index=False)
    print(json.dumps({k: v["delta"]["delta"] for k, v in an["full_test"].items()}))


if __name__ == "__main__":
    main()
