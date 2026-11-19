#!/usr/bin/env python3
"""g0_prompts.py -- the 3 zero-shot prompt designs of G_0 (rule arms.G_0), fixed in writing with sha256 BEFORE any
G_0 dev_sel2000 score exists (written in the CPU-preparation stage, before any P_0 or G_0 generation).

G_0 = Llama-3.1-70B-Instruct, zero-shot (no exemplars, no training), guided by P_0's k = 0 bundle:
P_0 parse, top-20 RULE-ordered candidates with RULE ranks and NO S scores, 20 supporting facts, operator trace.
Every design keeps the registered G_k layout (gk_common.g_user: Question; Parse; Candidates; Facts grouped losslessly by
(subject, relation, object) with chronological dates; Operator trace), with the S-score column removed because k = 0
has no re-ranker. Candidates are listed in RULE order (k0 bundles carry the RULE rank in the s_rank slot).
  g1: the registered G_k instruction (gk_common.G_HEAD, unchanged) + the bundle
  g2: g1 + a fixed glossary of the bundle fields (what RULE rank, "satisfies operator", fact groups and the trace mean;
      date granularity convention)
  g3: g2 + a fixed closing decision instruction (start from RULE rank 1, change it only when the facts show it fails the
      question's condition or another candidate fits better; copy one name or one date; no explanation) ending "Answer:"
Fixed a priori (not selected): greedy decoding, max 48 new tokens, K = 20 candidates, N = 20 facts, the 2,048-token cap
with trailing fact groups dropped first (as gk_common.fit_g_prompt), the Llama-3.1 chat template with date_string
"26 Jul 2024", one user turn.
Selection on dev_sel2000: best Hits@1; tie (equal hit count) -> shortest prompt (g1 < g2 < g3 by construction).
Bundles whose P_0 parse is unusable (invalid JSON / unknown name) render "Parse: none ..." with no candidates/facts.
"""
import argparse, inspect, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gk_common as GC

GK_COMMON_SHA256 = "3bbb684f484da4e9ccff5d7ed8ca644ccb5e8b6a918d50bd4b4ab04378e36b44"   # final guided-track version on the GPU host
MAX_LEN = GC.MAX_LEN          # 2,048 tokens prompt + answer (as G_k)
MAX_NEW = GC.MAX_NEW_G        # 48
DESIGNS = ("g1", "g2", "g3")

HEAD = GC.G_HEAD              # registered G_k instruction, unchanged (sha256 67d08028... in amendment B1)

GLOSS = (
    "\nHow to read the pipeline output:\n"
    "- Parse: the entities, relation, question type, answer type, granularity, time constraint and anchor that the pipeline "
    "read from the question.\n"
    "- Candidates: the pipeline's candidate answers in its rule-based order; RULE rank 1 is the pipeline's own answer. "
    "\"satisfies operator\" says whether the candidate meets the question's temporal condition (time window, before/after, "
    "first/last) under the parsed relation. \"supporting facts\" points to the fact groups below.\n"
    "- Facts: knowledge-graph facts (subject | relation | object | dates); one group lists every date of the same triple. "
    "They are a selection, not all facts of the graph.\n"
    "- Operator trace: how the temporal operator was applied to the top candidate.\n"
    "- Dates are written YYYY-MM-DD. If the question asks for a year or a month, answer YYYY or YYYY-MM.\n")

TAIL = (
    "\nDecide as follows: work out what the question asks for (an entity, or a date at the asked granularity). Start from "
    "the RULE rank 1 candidate and keep it unless the facts show that it does not satisfy the question's condition or that "
    "another candidate or fact fits the question better. Answer with one entity name copied exactly from the candidates or "
    "facts, or with one date. Give no explanation.\n"
    "Answer:")

NO_PARSE = "- none (the parser output for this question could not be used)"


def fixed_parts(design):
    return {"g1": (HEAD, "", ""), "g2": (HEAD, GLOSS, ""), "g3": (HEAD, GLOSS, TAIL)}[design]


def g0_user(b, design, max_fact_groups=None):
    """G_0 user turn for a normalised k0 bundle (gk_common.normalize_bundle). Candidates in RULE order (s_rank = RULE
    rank at k0); no S-score column. max_fact_groups: 2,048-token truncation (drop the last fact groups first)."""
    head, gloss, tail = fixed_parts(design)
    groups, f2g = GC.group_facts(b["facts"])
    if max_fact_groups is not None:
        groups = groups[:max_fact_groups]
    has_parse = bool(b["parse"])
    s = head + gloss + f"\nQuestion: {b['question']}\n\nParse:\n{GC.render_parse(b['parse']) if has_parse else NO_PARSE}\n"
    s += "\nCandidates (candidate | RULE rank | satisfies operator | supporting facts):\n"
    for c in b["candidates"]:                      # normalize_bundle sorts by s_rank = RULE rank at k0
        sat = c["operator_satisfied"]
        sat = "yes" if sat in (True, 1, "yes") else ("no" if sat in (False, 0, "no") else "n/a")
        fg = sorted({f2g[i] for i in c["fact_ids"] if i in f2g and f2g[i] <= len(groups)})
        rr = c["rule_rank"]
        rr = "n/a" if rr is None else str(int(rr) + 1)          # canonical rule_rank is 0-based
        s += f"- {c['candidate']} | {rr} | {sat} | " + (", ".join(f"F{g}" for g in fg) or "none") + "\n"
    if not b["candidates"]:
        s += "- none\n"
    s += "\nFacts (subject | relation | object | dates):\n"
    s += "".join(f"F{i + 1}. {h} | {GC.rel_text(r)} | {t} | {', '.join(ds)}\n" for i, (h, r, t, ds) in enumerate(groups)) or "none\n"
    s += f"\nOperator trace: {GC.render_trace(b['operator_trace']) if has_parse else 'no parse'}\n"
    s += tail
    return s


def fit_g0_prompt(tok, b, design, max_len=MAX_LEN):
    """-> (prompt_ids, n_groups_dropped); drops trailing fact groups until prompt + 48 new tokens <= 2,048."""
    ng = GC.n_fact_groups(b)
    p = None
    for keep in range(ng, -1, -1):
        p = GC.encode_prompt(tok, g0_user(b, design, None if keep == ng else keep))
        if len(p) + MAX_NEW <= max_len:
            return p, ng - keep
    return p[-(max_len - MAX_NEW):], ng


# fixed synthetic bundle (not a MultiTQ question): its rendering hash pins the exact layout of every design
EXAMPLE_BUNDLE = {
    "quid": 0, "question": "Who did Country A criticise first after Person B?", "k_subset": "k0",
    "parse": {"schema": "kpct-parse-v1", "tokens": ["who", "did", "country", "a", "criticise", "first", "after", "person", "b"],
              "mentions": [{"span": [2, 3], "text": "country a", "alternatives": [{"entity": "Country A", "score": 1.0, "alias": 0, "full": 1}]},
                           {"span": [7, 8], "text": "person b", "alternatives": [{"entity": "Person B", "score": 1.0, "alias": 0, "full": 1}]}],
              "time": None, "relations": [{"relation": "Criticize_or_denounce", "direction": "topic_or_first_mention_is_head", "prob": 1.0}],
              "qtype": {"label": "after_first", "prob": 1.0}, "answer_type": {"label": "entity", "prob": 1.0},
              "granularity": {"label": "day", "prob": 1.0},
              "operator_keywords": {"before_after": "after", "first_last": "first", "same_granularity": None},
              "anchor": "Person B", "anchor_mention": 1},
    "candidates": [{"answer": "Person C", "rule_rank": 1, "s_score": None, "s_rank": 1, "operator_satisfied": True, "support_fids": [11], "support_fact_idx": [1]},
                   {"answer": "Person D", "rule_rank": 2, "s_score": None, "s_rank": 2, "operator_satisfied": False, "support_fids": [12], "support_fact_idx": [2]}],
    "facts": [{"fid": 10, "head": "Country A", "relation": "Criticize_or_denounce", "tail": "Person B", "date": "2010-03-01"},
              {"fid": 11, "head": "Country A", "relation": "Criticize_or_denounce", "tail": "Person C", "date": "2010-03-05"},
              {"fid": 12, "head": "Country A", "relation": "Criticize_or_denounce", "tail": "Person D", "date": "2009-01-02"}],
    "operator_trace": "qtype after_first; answer entity; keyword after; keyword first; 1 of 2 pool candidates satisfy the operator",
    "pool_size": 2, "n_operator_satisfied": 1}


def renderer_source_sha256():
    src = "".join(inspect.getsource(f) for f in (fixed_parts, g0_user, fit_g0_prompt))
    return GC.sha256_text(src)


def manifest():
    assert GC.sha256_file(GC.__file__) == GK_COMMON_SHA256, "gk_common.py differs from the final guided-track version"
    ex = GC.normalize_bundle(EXAMPLE_BUNDLE)
    man = dict(name="g0_prompts", arm="G_0", model=GC.MODEL_70B, zero_shot=True, chat_date=GC.CHAT_DATE,
               decoding="greedy, max 48 new tokens", max_len=MAX_LEN, k_cands=GC.K_CANDS, n_facts=GC.N_FACTS,
               candidate_order="RULE (k0 bundles: s_rank holds the RULE rank; no S scores)",
               gk_common_sha256=GK_COMMON_SHA256, gk_renderers_sha256=GC.prompt_texts_sha256()["renderers"],
               g0_renderer_source_sha256=renderer_source_sha256(),
               fixed_texts={"HEAD": GC.sha256_text(HEAD), "GLOSS": GC.sha256_text(GLOSS), "TAIL": GC.sha256_text(TAIL),
                            "NO_PARSE": GC.sha256_text(NO_PARSE)},
               tie_break="equal dev_sel2000 hit count -> shortest prompt (fixed-text chars: g1 < g2 < g3)", designs={})
    for d in DESIGNS:
        h, g, t = fixed_parts(d)
        txt = g0_user(ex, d)
        man["designs"][d] = dict(fixed_text_sha256=GC.sha256_text(h + "\x00" + g + "\x00" + t), fixed_chars=len(h) + len(g) + len(t),
                                 example_render_sha256=GC.sha256_text(txt), example_render_chars=len(txt),
                                 parts=[k for k, v in (("HEAD", h), ("GLOSS", g), ("TAIL", t)) if v])
    man["sha256"] = GC.sha256_text(json.dumps({k: v for k, v in man.items() if k != "sha256"}, indent=1, sort_keys=True, default=str))
    return man


def verify(man_path):
    """assert that the code in this file reproduces the registered manifest (called by every G_0 job)"""
    reg = json.load(open(man_path))
    cur = manifest()
    assert reg["sha256"] == GC.sha256_text(json.dumps({k: v for k, v in reg.items() if k != "sha256"}, indent=1, sort_keys=True, default=str)), "manifest self-hash"
    assert cur["sha256"] == reg["sha256"], "G_0 prompt code differs from the registered designs"
    return reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="dir for g0_prompts_manifest.json + example renderings")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    man = manifest()
    json.dump(man, open(os.path.join(a.out, "g0_prompts_manifest.json"), "w"), indent=1)
    ex = GC.normalize_bundle(EXAMPLE_BUNDLE)
    for d in DESIGNS:
        open(os.path.join(a.out, f"g0_prompt_{d}_example.txt"), "w", encoding="utf-8").write(g0_user(ex, d))
    print(json.dumps({d: (v["fixed_text_sha256"][:12], v["fixed_chars"]) for d, v in man["designs"].items()}), man["sha256"])


if __name__ == "__main__":
    main()
