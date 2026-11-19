"""a2_coverage.py -- A2 of protocol/multitq_output_analyses_plan.json: displayed-coverage flags per test question.
Bundle coverage (Reader G_k display): F = gold string in the displayed provenance facts (subject / object / date),
C = gold string in the displayed top-20 candidates; official matching via multitq_eval.hit_at_k.
Renderer: kpct/reader/gk_common.py (normalize_bundle, group_facts), as used by the Reader runs.
"""
import importlib.util, json, os, sys


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def fact_strings(facts):
    out = []
    for (h, r, t, d) in facts:
        out += [str(h), str(t), str(d)]
    return out


def bundle_coverage(bundle_path, gold, ME, GC, keep_groups=None):
    """-> {quid: dict(F, C, n_facts, n_groups, n_cands, F_groups_shown)}; keep_groups: {quid: groups kept after the
    2,048-token cap} (None = all groups displayed)"""
    out = {}
    with open(bundle_path) as f:
        for line in f:
            if not line.strip():
                continue
            b = GC.normalize_bundle(json.loads(line))
            q = b['quid']
            groups, f2g = GC.group_facts(b['facts'])
            ng = len(groups)
            kg = ng if keep_groups is None else keep_groups.get(q, ng)
            shown = [fa for i, fa in enumerate(b['facts']) if f2g[i] <= kg]
            fs = fact_strings(shown)
            cs = [c['candidate'] for c in b['candidates']]
            out[q] = dict(F=ME.hit_at_k(gold[q], fs, len(fs)) if fs else 0,
                          F_all=ME.hit_at_k(gold[q], fact_strings(b['facts']), 3 * len(b['facts'])) if b['facts'] else 0,
                          C=ME.hit_at_k(gold[q], cs, 20) if cs else 0,
                          n_facts=len(b['facts']), n_groups=ng, groups_shown=kg, n_cands=len(cs))
    return out


def gu_coverage(gu_rows, gold, ME, keep_facts=None):
    """GU_F per quid; gu_rows: iterable of {quid, facts[20]}; keep_facts: {quid: facts kept after the cap}"""
    out = {}
    for r in gu_rows:
        q = int(r['quid'])
        facts = [tuple(x) for x in r['facts']]
        kf = len(facts) if keep_facts is None else keep_facts.get(q, len(facts))
        fs = fact_strings(facts[:kf])
        out[q] = dict(GU_F=ME.hit_at_k(gold[q], fs, len(fs)) if fs else 0, GU_n_facts=len(facts), GU_facts_shown=kf)
    return out
