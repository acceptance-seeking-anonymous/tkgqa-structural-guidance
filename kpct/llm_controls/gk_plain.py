#!/usr/bin/env python3
"""gk_plain.py -- GPLAIN_1 control (amendment LC1 to multitq_kpct_decision_rule.json): the G_1 evidence rendered as a
plain list, for TRAINING and inference of a new 1% reader.

Same QA rows, targets (gk_common.g_target), candidate identities (bundle candidates[:20]) and facts (bundle facts[:20],
lossless grouping by triple with chronological dates, gk_common.group_facts) as G_1. Removed: the parse, RULE ranks,
S scores, operator flags, candidate-fact links (F numbers) and the operator trace. Neutral fixed orders, so the S
(Ranker) order is not leaked:
  candidates  alphabetical: key (gk_common.norm_answer(c), c)   [casefolded, _ -> space, whitespace-collapsed; ties raw]
  fact groups alphabetical: key (norm(subject), norm(relation), norm(object), subject, relation, object); dates inside a
              group chronological (as group_facts)
The 2,048-token cap is handled as in gk_common.fit_g_prompt: trailing displayed fact groups are dropped.
Ranked output list (Hits@10 only; Hits@1 = the generated answer): [answer] + the displayed alphabetical candidates,
deduplicated (no S tail).

Use: `import gk_plain; gk_plain.patch()` BEFORE gk_train.main / gk_run_arm.main / gk_infer.main with --arm G. patch()
replaces gk_common.fit_g_prompt, gk_common.g_ranked and gk_common.prompt_texts_sha256 in-process; the frozen training
loop, vLLM session, selection and freeze code run unchanged.
"""
import inspect, sys
import gk_common as GC

PLAIN_HEAD = ("Answer the temporal question about the ICEWS knowledge graph using the candidate answers and facts below.\n"
              "Reply with the answer only: one entity name exactly as written, or a date written YYYY, YYYY-MM or YYYY-MM-DD.\n")
_ORIG = {}


def alpha_key(x):
    return (GC.norm_answer(str(x)), str(x))


def plain_groups(facts):
    groups, _ = GC.group_facts(facts)
    return sorted(groups, key=lambda g: (GC.norm_answer(g[0]), GC.norm_answer(g[1]), GC.norm_answer(g[2]), g[0], g[1], g[2]))


def plain_cands(b):
    return sorted([c["candidate"] for c in b["candidates"]], key=alpha_key)


def plain_user(b, max_fact_groups=None):
    groups = plain_groups(b["facts"])
    if max_fact_groups is not None:
        groups = groups[:max_fact_groups]
    s = PLAIN_HEAD + f"\nQuestion: {b['question']}\n"
    s += "\nCandidate answers (alphabetical order):\n" + ("".join(f"- {c}\n" for c in plain_cands(b)) or "- none\n")
    s += "\nFacts (subject | relation | object | dates; alphabetical order):\n"
    s += "".join(f"- {h} | {GC.rel_text(r)} | {t} | {', '.join(ds)}\n" for (h, r, t, ds) in groups) or "none\n"
    return s


def fit_plain_prompt(tok, b, answer=None, order="S", bundle_from=None, max_len=GC.MAX_LEN):
    """drop-in for gk_common.fit_g_prompt -> (prompt_ids, answer_ids, n_groups_dropped). order is ignored (fixed
    alphabetical order); bundle_from is not supported (no SHUF control for GPLAIN)."""
    assert bundle_from is None, "GPLAIN has no SHUF control"
    ng = len(plain_groups(b["facts"]))
    a_len = 0
    for keep in range(ng, -1, -1):
        u = plain_user(b, None if keep == ng else keep)
        p = GC.encode_prompt(tok, u)
        a = GC.encode_answer(tok, u, answer, p) if answer is not None else []
        a_len = len(a) if answer is not None else GC.MAX_NEW_G
        if len(p) + a_len <= max_len:
            return p, a, ng - keep
    return p[-(max_len - a_len):], a, ng


def g_ranked_plain(answer, tail):
    return _ORIG["g_ranked"](answer, sorted(tail, key=alpha_key))


def plain_texts_sha256():
    mod = sys.modules[__name__]
    src = "".join(inspect.getsource(getattr(mod, f)) for f in ("alpha_key", "plain_groups", "plain_cands", "plain_user",
                                                                 "fit_plain_prompt", "g_ranked_plain"))
    d = dict(_ORIG["prompt_texts_sha256"]())
    d.update(PLAIN_HEAD=GC.sha256_text(PLAIN_HEAD), plain_renderers=GC.sha256_text(src), arm="GPLAIN")
    return d


def patch():
    if _ORIG:
        return
    _ORIG.update(fit_g_prompt=GC.fit_g_prompt, g_ranked=GC.g_ranked, prompt_texts_sha256=GC.prompt_texts_sha256)
    GC.fit_g_prompt = fit_plain_prompt
    GC.g_ranked = g_ranked_plain
    GC.prompt_texts_sha256 = plain_texts_sha256
