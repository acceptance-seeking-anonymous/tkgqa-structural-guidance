#!/usr/bin/env python3
"""cronquestions_eval.py -- CronQuestions scorer reproducing the official CronKGQA semantics.

Reference implementation: apoorvumang/CronKGQA @ 41a93de (2022-02-23), train_qa_model.py::eval
(lines 145-238; the scoring loop is lines 181-233) and qa_datasets.py::getAnswersFromScores
(lines 128-142).

Official semantics (reproduced exactly):
  * predicted = topk_answers[i][:k]  -- the model's ranked answers, i.e. the top-k ids of one
    score vector over (entities + timestamps); entity ids are decoded to Wikidata ids
    ('Q42'), timestamp ids to the integer year (id2ts[t][0]).  The official list therefore
    never contains duplicates and mixes entities and years freely.
  * hit@k = 1 iff set(question['answers']) & set(predicted) is non-empty  (any gold in top k).
    Gold entity answers are Wikidata id strings, gold time answers are Python ints (years).
  * Hits@k = mean over ALL questions of the split, in pickle order (a missing prediction = miss).
  * Strata: question['type'] (simple_entity, simple_time, before_after, first_last, time_join);
    'simple' if 'simple' in type else 'complex'; question['answer_type'] (entity / time).
    k in {1, 10}.

Only deviation option (default ON, documented): a prediction that is a canonical decimal
string ('1995', '-50', '0') is converted to int before matching, so that JSONL files may
carry years as strings.  Gold time answers are ints, every Wikidata entity id starts with
'Q', and the conversion is a bijection on canonical decimal strings, so it is a format
conversion, not a relaxation.  Pass --raw / normalize_years=False to disable it (then a
year must be emitted as a JSON integer to match, exactly as in the official code).
Booleans are rejected (JSON true/false must not match years 1/0).

Question identifier: qid = 0-based position of the question in the split pickle
(questions/{train,valid,test}.pickle).  The dataset's 'uniq_id' field is NOT unique within a
split and is not used.  v1 (data.zip) and v2 (data_v2.zip) pickles are position-aligned and
carry identical 'answers', 'type', 'answer_type' and 'template' fields; they differ only in
'question' and 'paraphrases' for the {tail2} rows.

Prediction file formats accepted:
  * JSONL: one {"qid": int, "ranked": [str|int, ...]} per line   (primary format)
  * JSON : {"<qid>": [...], ...}  or  [{"qid":..., "ranked": [...]}, ...]

CLI:
  python cronquestions_eval.py score --gold data/wikidata_big/questions/test.pickle \
         --pred preds.jsonl [--k 1 10] [--subsets probes.json] [--json out.json]
         [--per-question out.jsonl] [--strict] [--raw]
  python cronquestions_eval.py selftest --gold <split.pickle> --ent-id <ent_id> --ts-id <ts_id>
         [--regimes oracle random partial] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
from collections import OrderedDict, defaultdict

__version__ = "1.0.0"

QTYPES = ["simple_entity", "simple_time", "before_after", "first_last", "time_join"]
_CANON_INT = re.compile(r"^-?(0|[1-9][0-9]*)$")


# ---------------------------------------------------------------------------
# gold loading
# ---------------------------------------------------------------------------
def load_gold(path: str):
    """Return the list of question dicts in pickle order (qid = list index)."""
    with open(path, "rb") as f:
        qs = pickle.load(f)
    assert isinstance(qs, list) and qs and isinstance(qs[0], dict), "unexpected gold format"
    return qs


def simple_complex(q: dict) -> str:
    return "simple" if "simple" in q["type"] else "complex"   # official rule, line 191


# ---------------------------------------------------------------------------
# core scoring
# ---------------------------------------------------------------------------
def _norm(x, normalize_years: bool):
    if isinstance(x, bool):
        raise TypeError("boolean in a ranked list")
    if normalize_years and isinstance(x, str) and _CANON_INT.match(x):
        return int(x)
    return x


def hit_at_k(question: dict, ranked, k: int, normalize_years: bool = True) -> int:
    """Official hit for one question (1/0)."""
    if ranked is None:
        return 0
    predicted = [_norm(x, normalize_years) for x in list(ranked)[:k]]
    actual_answers = question["answers"]
    return 1 if len(set(actual_answers).intersection(set(predicted))) > 0 else 0


def load_predictions(path: str) -> dict:
    """Return {qid(int): ranked list}."""
    with open(path, "r", encoding="utf-8") as f:
        s = f.read().strip()
    preds = {}
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        obj = None  # -> JSONL
    if isinstance(obj, dict):
        if "qid" in obj and "ranked" in obj:
            preds[int(obj["qid"])] = list(obj["ranked"])
        else:
            for k, v in obj.items():
                preds[int(k)] = list(v)
        return preds
    if isinstance(obj, list):
        rows = obj
    else:
        rows = [json.loads(line) for line in s.splitlines() if line.strip()]
    for r in rows:
        q = int(r["qid"])
        if q in preds:
            raise ValueError(f"duplicate qid {q} in prediction file")
        preds[q] = list(r["ranked"])
    return preds


def score(questions, preds: dict, ks=(1, 10), normalize_years: bool = True,
          strict: bool = False, subsets: dict | None = None):
    """Score a split. Returns (summary dict, per-question list).

    questions: gold list (pickle order).  preds: {qid: ranked list}; missing qids score 0
    (the official loop scores every question); strict=True raises on missing/extra qids.
    subsets: optional {name: [qid, ...]} -> extra strata (e.g. pre-registered probes).
    """
    n = len(questions)
    missing = [i for i in range(n) if i not in preds]
    extra = [q for q in preds if not (0 <= q < n)]
    if strict and (missing or extra):
        raise ValueError(f"coverage error: {len(missing)} missing, {len(extra)} extra qids")
    n_dup = 0
    short = defaultdict(int)
    for i in range(n):
        r = preds.get(i)
        if r is None:
            continue
        rr = [_norm(x, normalize_years) for x in r[:max(ks)]]
        if len(set(rr)) < len(rr):
            n_dup += 1
        for k in ks:
            if len(r) < k:
                short[k] += 1
    sub_of = defaultdict(list)
    if subsets:
        for name, qids in subsets.items():
            for q in qids:
                sub_of[int(q)].append(name)

    strata = OrderedDict()
    for k in ks:
        strata[k] = {"overall": [], "type": defaultdict(list), "simple_complex": defaultdict(list),
                     "answer_type": defaultdict(list), "type_answer_type": defaultdict(list),
                     "subset": defaultdict(list)}
    per_q = []
    for i, q in enumerate(questions):
        r = preds.get(i)
        row = {"qid": i}
        for k in ks:
            h = hit_at_k(q, r, k, normalize_years)
            row[f"hit@{k}"] = h
            S = strata[k]
            S["overall"].append(h)
            S["type"][q["type"]].append(h)
            S["simple_complex"][simple_complex(q)].append(h)
            S["answer_type"][q["answer_type"]].append(h)
            S["type_answer_type"][q["type"] + "-" + q["answer_type"]].append(h)
            for name in sub_of.get(i, ()):
                S["subset"][name].append(h)
        per_q.append(row)

    def agg(v):
        return {"hits": (sum(v) / len(v)) if v else float("nan"), "n": len(v), "correct": int(sum(v))}

    summary = OrderedDict()
    summary["scorer"] = f"cronquestions_eval.py v{__version__} (official CronKGQA eval semantics)"
    summary["normalize_years"] = normalize_years
    summary["n_questions"] = n
    summary["n_missing_predictions"] = len(missing)
    summary["n_extra_predictions"] = len(extra)
    summary["n_questions_with_duplicate_items_in_topk"] = n_dup
    summary["n_short_lists"] = {f"<{k}": short[k] for k in ks}
    for k in ks:
        S = strata[k]
        block = OrderedDict()
        block["overall"] = agg(S["overall"])
        for name in ["simple_complex", "answer_type", "type", "type_answer_type", "subset"]:
            block[name] = OrderedDict((key, agg(S[name][key])) for key in sorted(S[name], key=str.lower))
        summary[f"hits@{k}"] = block
    return summary, per_q


def headline_row(summary: dict, k: int = 1) -> OrderedDict:
    """TECQA-style layout: Overall | Simple, Complex | Entity, Time."""
    b = summary[f"hits@{k}"]
    g = lambda d, key: d.get(key, {}).get("hits", float("nan"))
    return OrderedDict([("Overall", b["overall"]["hits"]),
                        ("Simple", g(b["simple_complex"], "simple")), ("Complex", g(b["simple_complex"], "complex")),
                        ("Entity", g(b["answer_type"], "entity")), ("Time", g(b["answer_type"], "time"))])


# ---------------------------------------------------------------------------
# verbatim port of the official loop (train_qa_model.py lines 181-233), for identity checks
# ---------------------------------------------------------------------------
def official_eval_reference(data, topk_answers, k_list=(1, 10)):
    """Line-for-line port of the official scoring loop; topk_answers[i] is the decoded top-10
    list of question i (Wikidata ids / int years).  Returns {k: {'all': acc, 'type': {...},
    'simple_complex': {...}, 'answer_type': {...}}} with unrounded values."""
    out = {}
    for k in k_list:
        hits_at_k = 0
        total = 0
        question_types_count = defaultdict(list)
        simple_complex_count = defaultdict(list)
        entity_time_count = defaultdict(list)
        for i, question in enumerate(data):
            actual_answers = question['answers']
            question_type = question['type']
            if 'simple' in question_type:
                simple_complex_type = 'simple'
            else:
                simple_complex_type = 'complex'
            entity_time_type = question['answer_type']
            predicted = topk_answers[i][:k]
            if len(set(actual_answers).intersection(set(predicted))) > 0:
                val_to_append = 1
                hits_at_k += 1
            else:
                val_to_append = 0
            question_types_count[question_type].append(val_to_append)
            simple_complex_count[simple_complex_type].append(val_to_append)
            entity_time_count[entity_time_type].append(val_to_append)
            total += 1
        out[k] = {"all": hits_at_k / total,
                  "type": {a: sum(v) / len(v) for a, v in question_types_count.items()},
                  "simple_complex": {a: sum(v) / len(v) for a, v in simple_complex_count.items()},
                  "answer_type": {a: sum(v) / len(v) for a, v in entity_time_count.items()}}
    return out


# ---------------------------------------------------------------------------
# paired Poisson(1) bootstrap over questions (house protocol, B = 2000)
# ---------------------------------------------------------------------------
def paired_poisson_bootstrap(hits_a, hits_b, B: int = 2000, seed: int = 0, alpha: float = 0.05):
    """delta = mean(a) - mean(b); Poisson(1) weights per question shared by both arms."""
    import numpy as np
    a = np.asarray(hits_a, dtype=float)
    b = np.asarray(hits_b, dtype=float)
    assert a.shape == b.shape and a.ndim == 1
    rng = np.random.default_rng(seed)
    d = a - b
    deltas = np.empty(B)
    for i in range(B):
        w = rng.poisson(1.0, size=d.shape[0])
        sw = w.sum()
        deltas[i] = (w * d).sum() / sw if sw > 0 else 0.0
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return {"delta": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_one_sided_le0": float((deltas <= 0).mean()), "B": B, "seed": seed}


# ---------------------------------------------------------------------------
# self-test: synthetic prediction regimes, port vs verbatim official loop
# ---------------------------------------------------------------------------
def _answer_space(ent_id_path, ts_id_path):
    with open(ent_id_path, "rb") as f:
        ent2id = pickle.load(f)
    with open(ts_id_path, "rb") as f:
        ts2id = pickle.load(f)
    id2ent = {v: k for k, v in ent2id.items()}
    id2ts = {v: k for k, v in ts2id.items()}
    # official decoding order: ids 0..E-1 entities, E..E+T-1 timestamps -> year int
    space = [id2ent[i] for i in range(len(id2ent))] + [id2ts[i][0] for i in range(len(id2ts))]
    return space


def synthetic_predictions(questions, space, regime: str, seed: int = 0, K: int = 10):
    """Return list of decoded top-K lists (official style: distinct items, years as int)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    N = len(space)
    out = []
    for q in questions:
        gold = sorted(q["answers"], key=str)
        gold_set = set(gold)
        if regime == "oracle":
            lst = list(gold[:K])
            while len(lst) < K:
                c = space[int(rng.integers(N))]
                if c not in gold_set and c not in lst:
                    lst.append(c)
        elif regime == "random":
            lst = []
            while len(lst) < K:
                c = space[int(rng.integers(N))]
                if c not in lst:
                    lst.append(c)
        elif regime == "partial":   # one gold answer at a uniform rank in 0..19 (absent if >= K)
            r = int(rng.integers(20))
            g = gold[int(rng.integers(len(gold)))]
            lst = []
            while len(lst) < K:
                c = space[int(rng.integers(N))]
                if c not in gold_set and c not in lst:
                    lst.append(c)
            if r < K:
                lst[r] = g
        else:
            raise ValueError(regime)
        out.append(lst)
    return out


def selftest(questions, space, regimes=("oracle", "random", "partial"), seed=0):
    res = OrderedDict()
    for regime in regimes:
        topk = synthetic_predictions(questions, space, regime, seed=seed)
        ref = official_eval_reference(questions, topk)
        preds = {i: lst for i, lst in enumerate(topk)}
        summ, _ = score(questions, preds, ks=(1, 10))
        # same predictions with years written as strings (exercise the documented conversion)
        preds_s = {i: [str(x) if isinstance(x, int) else x for x in lst] for i, lst in enumerate(topk)}
        summ_s, _ = score(questions, preds_s, ks=(1, 10))
        summ_raw, _ = score(questions, preds_s, ks=(1, 10), normalize_years=False)
        ident = True
        for k in (1, 10):
            b = summ[f"hits@{k}"]
            ident &= abs(b["overall"]["hits"] - ref[k]["all"]) < 1e-12
            for name in ("type", "simple_complex", "answer_type"):
                for key, v in ref[k][name].items():
                    ident &= abs(b[name][key]["hits"] - v) < 1e-12
            ident &= abs(summ_s[f"hits@{k}"]["overall"]["hits"] - ref[k]["all"]) < 1e-12
        res[regime] = OrderedDict([
            ("identical_to_official_loop", bool(ident)),
            ("hits@1", headline_row(summ, 1)), ("hits@10", headline_row(summ, 10)),
            ("hits@1_years_as_strings_raw_mode", summ_raw["hits@1"]["overall"]["hits"]),
            ("by_type_hits@1", {k2: v["hits"] for k2, v in summ["hits@1"]["type"].items()}),
        ])
    return res


def _fmt(x):
    return "  -  " if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--gold", required=True, help="questions/<split>.pickle (v1 or v2)")
    s.add_argument("--pred", required=True)
    s.add_argument("--k", type=int, nargs="+", default=[1, 10])
    s.add_argument("--subsets", default=None, help="JSON {name: [qid,...]} of extra strata")
    s.add_argument("--json", default=None)
    s.add_argument("--per-question", default=None)
    s.add_argument("--strict", action="store_true")
    s.add_argument("--raw", action="store_true", help="disable digit-string -> int year conversion")
    t = sub.add_parser("selftest")
    t.add_argument("--gold", required=True)
    t.add_argument("--ent-id", required=True)
    t.add_argument("--ts-id", required=True)
    t.add_argument("--regimes", nargs="+", default=["oracle", "random", "partial"])
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    questions = load_gold(a.gold)
    if a.cmd == "selftest":
        res = selftest(questions, _answer_space(a.ent_id, a.ts_id), a.regimes, a.seed)
        print(json.dumps(res, indent=1))
        if a.json:
            with open(a.json, "w") as f:
                json.dump(res, f, indent=1)
        return 0 if all(r["identical_to_official_loop"] for r in res.values()) else 1
    preds = load_predictions(a.pred)
    subsets = json.load(open(a.subsets)) if a.subsets else None
    if subsets and "subsets" in subsets:
        subsets = subsets["subsets"]
    summary, per_q = score(questions, preds, ks=tuple(a.k), normalize_years=not a.raw,
                           strict=a.strict, subsets=subsets)
    for k in a.k:
        row = headline_row(summary, k)
        print(f"Hits@{k}: " + " | ".join(f"{n} {_fmt(v)}" for n, v in row.items()))
    print(f"missing={summary['n_missing_predictions']} extra={summary['n_extra_predictions']} "
          f"dup_topk={summary['n_questions_with_duplicate_items_in_topk']}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(summary, f, indent=1)
    if a.per_question:
        with open(a.per_question, "w") as f:
            for r in per_q:
                f.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
