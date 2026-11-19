#!/usr/bin/env python3
"""gk_check_bundles.py -- schema, coverage and STOP_OOF_LEAK checks for guidance bundles before any G_k dev_sel2000 score.

Checks for bundles/<subset>/{train,dev_sel2000,test}.jsonl(.gz):
  schema    every row normalises (gk_common.normalize_bundle); <= 20 candidates and <= 20 facts; gold on train rows only
  coverage  train quids == the registered k% LLM list; dev_sel2000 quids == the registered 2,000; test == all 54,584
  gold      train gold == train.json answers
  OOF       (rule STOP_OOF_LEAK) for k < 100 every train quid must be absent from every training list of the models that
            produced its bundle. Producer lists are read from <bundles>/producers.json (any nesting: every list of ints
            under fold0 / fold1 is a training list of a fold-f producer) and the row's fold from its 'oof_fold' / 'fold'
            field or, if absent, from the registered outer split (multitq_kpct_subsets.json oof_splits). Without
            producers.json the check reports NOT_CHECKABLE and exits 2 (no dev score may be computed).
            k = 100: train60k must be disjoint from s_train100k (S training list); classifier probabilities are the frozen
            2-fold OOF export (linker aliases / block list in-sample, disclosed).
"""
import argparse, collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gk_common as GC


def int_lists(x, out):
    if isinstance(x, list) and x and all(isinstance(v, int) for v in x):
        out.append(set(x))
    elif isinstance(x, dict):
        for v in x.values():
            int_lists(v, out)
    elif isinstance(x, list):
        for v in x:
            int_lists(v, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", required=True, help="dir with train/dev_sel2000/test.jsonl(.gz)")
    ap.add_argument("--subset", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--data", required=True, help="MultiTQ data dir (questions/)")
    ap.add_argument("--s_train100k", default=None); ap.add_argument("--out", required=True)
    a = ap.parse_args()

    def find(split):
        for ext in (".jsonl", ".jsonl.gz"):
            p = os.path.join(a.bundles, split + ext)
            if os.path.exists(p):
                return p
        return None
    rep = dict(subset=a.subset, bundles=a.bundles, files={}, status="PASS", problems=[])
    m = json.load(open(a.manifest))
    lists = dict(train=set(GC.load_list(a.manifest, a.subset)), dev_sel2000=set(GC.load_list(a.manifest, "dev_sel2000")),
                 test=set(q["quid"] for q in json.load(open(os.path.join(a.data, "questions", "test.json")))))
    Qtr = {q["quid"]: q["answers"] for q in json.load(open(os.path.join(a.data, "questions", "train.json")))}
    rows_train = None
    for split in ("train", "dev_sel2000", "test"):
        p = find(split)
        if p is None:
            rep["problems"].append(f"{split}: file missing"); rep["status"] = "FAIL"; continue
        rows = GC.read_jsonl(p)
        st = GC.check_bundles(rows, split)
        q = set(int(r["quid"]) for r in rows)
        st.update(file=os.path.basename(p), sha256=GC.sha256_file(p), coverage_ok=(q == lists[split]) and len(rows) == len(q),
                  missing=len(lists[split] - q), extra=len(q - lists[split]))
        if not st["coverage_ok"]:
            rep["problems"].append(f"{split}: coverage mismatch (missing {st['missing']}, extra {st['extra']}, dup {len(rows) - len(q)})"); rep["status"] = "FAIL"
        if split == "train":
            bad = [r["quid"] for r in rows if list(r.get("gold") or []) != Qtr[int(r["quid"])]]
            st["gold_mismatch"] = len(bad)
            if bad:
                rep["problems"].append(f"train: {len(bad)} gold != train.json answers"); rep["status"] = "FAIL"
            rows_train = rows
        rep["files"][split] = st
    # ---------------------------------------------------------------- STOP_OOF_LEAK
    if a.subset == "k100":
        s100 = set(json.load(open(a.s_train100k)))
        inter = len(lists["train"] & s100)
        oo = collections.Counter(str((r.get("oof") or {}).get("stack_trained_on")) for r in rows_train or [])
        rep["oof"] = dict(rule="k=100: reader_train60k disjoint from s_train100k; classifier OOF (frozen export)", overlap=inter,
                          status="PASS" if inter == 0 else "FAIL", row_oof_field=dict(oo))
    else:
        prod = os.path.join(a.bundles, "producers.json")
        has_oof = bool(rows_train) and all(isinstance(r.get("oof"), dict) and "outer_fold" in r["oof"] for r in rows_train)
        if not os.path.exists(prod) and has_oof:
            # per-row `oof` field: {outer_fold f, stack_trained_on "<subset>/fold<1-f>"}; the producing stack's training
            # list is the registered outer fold 1-f (its inner splits are subsets of it), sha256-checked against the manifest
            folds = m["oof_splits"][a.subset]
            fl = {}
            for f in (0, 1):
                q = [int(x) for x in folds[f"fold{f}"]["quids"]]
                assert GC.list_sha256(q) == folds[f"fold{f}"]["sha256"], f"fold{f} sha256 mismatch"
                fl[f] = set(q)
            assert not (fl[0] & fl[1]) and (fl[0] | fl[1]) == lists["train"], "outer folds do not partition the k% list"
            leaks, bad_meta = [], []
            for r in rows_train:
                qd, o = int(r["quid"]), r["oof"]
                f = o.get("outer_fold")
                if f not in (0, 1) or qd not in fl[f]:
                    bad_meta.append((qd, f"outer_fold {f} does not contain the quid")); continue
                if o.get("stack_trained_on") != f"{a.subset}/fold{1 - f}":
                    bad_meta.append((qd, f"stack_trained_on {o.get('stack_trained_on')!r}")); continue
                if qd in fl[1 - f]:
                    leaks.append((qd, f"in the training fold of its producer"))
            rep["oof"] = dict(method="per-row oof field + registered outer folds (manifest sha256 checked)",
                              status="PASS" if not leaks and not bad_meta else "FAIL", n_checked=len(rows_train), n_leaks=len(leaks),
                              n_bad_meta=len(bad_meta), sample=(leaks + bad_meta)[:10],
                              fold_sha256={f"fold{f}": folds[f"fold{f}"]["sha256"] for f in (0, 1)},
                              n_by_fold={f"fold{f}": sum(1 for r in rows_train if r["oof"].get("outer_fold") == f) for f in (0, 1)})
        elif not os.path.exists(prod):
            rep["oof"] = dict(status="NOT_CHECKABLE", reason="neither producers.json nor a per-row oof field")
        else:
            P = json.load(open(prod))
            folds = m["oof_splits"][a.subset]
            fold_of = {int(q): f for f in (0, 1) for q in folds[f"fold{f}"]["quids"]}
            prod_lists = {f: int_lists(P.get(f"fold{f}", {}), []) for f in (0, 1)}
            leaks, nofold = [], 0
            for r in rows_train or []:
                qd = int(r["quid"]); f = r.get("oof_fold", r.get("fold"))
                if f is None:
                    f = fold_of.get(qd); nofold += 1
                if f is None or not prod_lists[int(f)]:
                    leaks.append((qd, "no producer list")); continue
                if any(qd in L for L in prod_lists[int(f)]):
                    leaks.append((qd, f"in a training list of its fold-{f} producer"))
            rep["oof"] = dict(status="PASS" if not leaks else "FAIL", n_checked=len(rows_train or []), n_leaks=len(leaks),
                              leaks_sample=leaks[:10], fold_from_manifest=nofold,
                              n_producer_lists={f: len(v) for f, v in prod_lists.items()})
    if rep["oof"]["status"] != "PASS":
        rep["status"] = "FAIL" if rep["oof"]["status"] == "FAIL" else "NOT_CHECKABLE"
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps(rep, indent=1)[:3000])
    sys.exit(0 if rep["status"] == "PASS" else (2 if rep["status"] == "NOT_CHECKABLE" else 1))


if __name__ == "__main__":
    main()
