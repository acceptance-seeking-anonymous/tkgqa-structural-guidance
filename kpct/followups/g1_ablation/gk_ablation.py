#!/usr/bin/env python3
"""gk_ablation.py -- G_1 component ablation (rule multitq_g1_ablation_rule.json): inference-time bundle variants.

Every variant uses the frozen G_1 adapter, the frozen G_HEAD and the frozen chat encoding (gk_common). Removed
components are omitted with their section heading; the remaining sections keep the frozen order and wording.

  A_full       frozen bundle, frozen rendering (= gk_common.g_user(b, "S")); asserted byte-identical
  A_no_trace   A_full without the "Parse:" section and the "Operator trace:" line
  A_facts_cand facts (fact groups in a seeded random order) + candidate strings only (no RULE rank, S score,
               operator flag or fact links), candidates in the k% rule's registered random order
               (default_rng(int(quid)).permutation(n)); no parse, no trace
  A_facts      the structural facts only, fact groups in the same seeded random order as A_facts_cand
  A_facts_ord  (supplementary) the structural facts only, in the frozen (S seed-0 provenance) order
  A_GUfacts    A_full format with the structural facts replaced by the first n GU dense-retrieved facts
               (n = the bundle's number of structural facts), grouped as in g_user; each candidate's fact links are
               recomputed by string match (entity = subject or object; time = a date with the candidate as prefix),
               at most 3 groups, ascending; parse, candidates (S order, ranks, scores, operator flag) and trace kept

The 2,048-token cap is handled as in gk_common.fit_g_prompt: trailing displayed fact groups are dropped.
Fact-group shuffle seed: default_rng([20261028, int(quid)]).permutation(n_groups).
"""
import numpy as np
import gk_common as GC

VARIANTS = ["A_full", "A_no_trace", "A_facts_cand", "A_facts", "A_GUfacts", "A_facts_ord"]
SHUF_SEED = 20261028
CAND_HEAD = "\nCandidates (candidate | RULE rank | S score | satisfies operator | supporting facts):\n"
FACT_HEAD = "\nFacts (subject | relation | object | dates):\n"


def _groups(facts, order, quid):
    """-> (displayed groups [(h, r, t, dates)], fact_index -> displayed group number 1-based)"""
    groups, f2g = GC.group_facts(facts)                 # first-appearance order, f2g 1-based
    n = len(groups)
    if order == "frozen":
        perm = list(range(n))
    elif order == "shuffled":
        perm = [int(i) for i in np.random.default_rng([SHUF_SEED, int(quid)]).permutation(n)] if n else []
    else:
        raise ValueError(order)
    pos = {old: new for new, old in enumerate(perm)}    # old group idx (0-based) -> displayed position
    disp = [groups[old] for old in perm]
    f2d = {i: pos[g - 1] + 1 for i, g in f2g.items()}
    return disp, f2d


def _cand_rows_full(cands, f2d, ngroups):
    s = ""
    for c in cands:                                     # S order (frozen)
        sat = c["operator_satisfied"]
        sat = "yes" if sat in (True, 1, "yes") else ("no" if sat in (False, 0, "no") else "n/a")
        fg = sorted({f2d[i] for i in c["fact_ids"] if i in f2d and f2d[i] <= ngroups})
        rr = c["rule_rank"]
        rr = "n/a" if rr is None or (isinstance(rr, float) and np.isnan(rr)) else str(int(rr) + 1)
        s += f"- {c['candidate']} | {rr} | {GC._fmt_score(c['s_score'])} | {sat} | " + (", ".join(f"F{g}" for g in fg) or "none") + "\n"
    return s or "- none\n"


def _facts_rows(groups):
    return "".join(f"F{i + 1}. {h} | {GC.rel_text(r)} | {t} | {', '.join(ds)}\n" for i, (h, r, t, ds) in enumerate(groups)) or "none\n"


def gu_links(cands, groups, cap=3):
    """candidate -> displayed GU group numbers (1-based) whose subject/object equals it (entity) or a date starts with it (time)"""
    out = []
    for c in cands:
        x = c["candidate"]
        tm = GC.is_time(x)
        g = [i + 1 for i, (h, r, t, ds) in enumerate(groups) if (any(d.startswith(x) for d in ds) if tm else (h == x or t == x))]
        out.append(g[:cap])
    return out


def ab_user(b, variant, gu_facts=None, max_fact_groups=None):
    """user turn of one variant; gu_facts: [(h, r, t, d)] in retrieval order (A_GUfacts only)"""
    q = b["question"]
    if variant == "A_full":
        return GC.g_user(b, "S", None, max_fact_groups)
    if variant in ("A_no_trace", "A_GUfacts"):
        if variant == "A_GUfacts":
            n = len(b["facts"])
            gf = [tuple(f) for f in gu_facts[:n]]
            groups, _ = GC.group_facts(gf)
            if max_fact_groups is not None:
                groups = groups[:max_fact_groups]
            links = gu_links(b["candidates"], groups)
            s = GC.G_HEAD + f"\nQuestion: {q}\n\nParse:\n{GC.render_parse(b['parse'])}\n" + CAND_HEAD
            for c, fg in zip(b["candidates"], links):
                sat = c["operator_satisfied"]
                sat = "yes" if sat in (True, 1, "yes") else ("no" if sat in (False, 0, "no") else "n/a")
                rr = c["rule_rank"]
                rr = "n/a" if rr is None or (isinstance(rr, float) and np.isnan(rr)) else str(int(rr) + 1)
                s += f"- {c['candidate']} | {rr} | {GC._fmt_score(c['s_score'])} | {sat} | " + (", ".join(f"F{g}" for g in fg) or "none") + "\n"
            if not b["candidates"]:
                s += "- none\n"
            s += FACT_HEAD + _facts_rows(groups)
            s += f"\nOperator trace: {GC.render_trace(b['operator_trace'])}\n"
            return s
        groups, f2d = _groups(b["facts"], "frozen", b["quid"])
        if max_fact_groups is not None:
            groups = groups[:max_fact_groups]
        s = GC.G_HEAD + f"\nQuestion: {q}\n" + CAND_HEAD + _cand_rows_full(b["candidates"], f2d, len(groups))
        s += FACT_HEAD + _facts_rows(groups)
        return s
    if variant in ("A_facts", "A_facts_ord", "A_facts_cand"):
        groups, _ = _groups(b["facts"], "frozen" if variant == "A_facts_ord" else "shuffled", b["quid"])
        if max_fact_groups is not None:
            groups = groups[:max_fact_groups]
        s = GC.G_HEAD + f"\nQuestion: {q}\n"
        if variant == "A_facts_cand":
            n = len(b["candidates"])
            perm = [int(i) for i in np.random.default_rng(int(b["quid"])).permutation(n)]
            s += "\nCandidates:\n" + ("".join(f"- {b['candidates'][j]['candidate']}\n" for j in perm) or "- none\n")
        s += FACT_HEAD + _facts_rows(groups)
        return s
    raise ValueError(variant)


def shown_groups(b, variant, gu_facts=None, max_fact_groups=None):
    """displayed fact groups (after truncation) -- for coverage"""
    if variant == "A_GUfacts":
        groups, _ = GC.group_facts([tuple(f) for f in gu_facts[:len(b["facts"])]])
    else:
        groups, _ = _groups(b["facts"], "shuffled" if variant in ("A_facts", "A_facts_cand") else "frozen", b["quid"])
    return groups if max_fact_groups is None else groups[:max_fact_groups]


def n_groups(b, variant, gu_facts=None):
    return len(shown_groups(b, variant, gu_facts))


def fit_ab_prompt(tok, b, variant, gu_facts=None, max_len=GC.MAX_LEN):
    """-> (prompt_ids, n_groups_dropped, n_groups_shown); same cap rule as gk_common.fit_g_prompt (no answer)"""
    ng = n_groups(b, variant, gu_facts)
    for keep in range(ng, -1, -1):
        u = ab_user(b, variant, gu_facts, None if keep == ng else keep)
        p = GC.encode_prompt(tok, u)
        if len(p) + GC.MAX_NEW_G <= max_len:
            return p, ng - keep, keep
    return p[-(max_len - GC.MAX_NEW_G):], ng, 0


def gold_in_groups(gold, groups):
    """accepted gold string occurs in the displayed facts (gu_target semantics: entity = subject/object;
    time = a fact date truncated to the gold granularity)"""
    G = set(gold)
    L = GC.match_len(gold)
    if GC.is_time(gold[0]):
        return any((d[:L] if L else d) in G for (h, r, t, ds) in groups for d in ds)
    return any(h in G or t in G for (h, r, t, ds) in groups)
