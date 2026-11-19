"""gk_common.py -- MultiTQ k% experiment, GUIDED REASONER track (G_k, GU_k, SHUF_G1, QONLY_G1).

Shared code for training (gk_train.py), vLLM inference (gk_infer.py), the k100 export converter and the
benchmark. Rule: multitq_kpct_decision_rule.json v1.0 (sha256 0f5f7604...).

Bundle schema (written by the structural k% track at $WORK_DIR/bundles/<subset>/{train,dev_sel2000,test}.jsonl):
  quid, question, parse{...}, candidates[top-20: candidate, rule_rank, s_score, operator_satisfied, fact ids],
  facts[top-20 (h, r, t, date)], operator_trace, gold (train rows only).
`normalize_bundle` maps the field-name variants listed in KEYMAP / CAND_KEYS onto one canonical form and raises
BundleSchemaError naming the missing fields, so a schema drift fails loudly instead of silently changing prompts.
"""
import gzip, hashlib, json, os, re, datetime
import numpy as np

# ----------------------------------------------------------------------------------------------- constants
MODEL_8B = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_70B = "meta-llama/Llama-3.1-70B-Instruct"
CHAT_DATE = "26 Jul 2024"          # fixed date_string for the Llama-3.1 chat template (else strftime_now -> today's date)
K_CANDS, N_FACTS = 20, 20          # fixed a priori (rule)
MAX_LEN = 2048                     # max sequence length (prompt + answer), rule lora_recipe
MAX_NEW_G = 48                     # greedy, max 48 new tokens (G_k / GU_k decoding)
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
OPT = dict(lr=2e-4, weight_decay=0.0, warmup=0.03, grad_clip=1.0, eff_batch=16)
E_MAX = {"k01": 3, "k05": 2, "k10": 1, "k100": 1, "cap20k_k10": 1, "cap20k_k100": 1}
DATE_LO, DATE_HI = datetime.date(2005, 1, 1), datetime.date(2015, 12, 31)


class BundleSchemaError(ValueError):
    pass


# ----------------------------------------------------------------------------------------------- io
def read_jsonl(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p, rows):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def list_sha256(quids):
    """registered list recipe (multitq_kpct_subsets.json list_sha256_recipe)"""
    return hashlib.sha256(json.dumps(sorted(int(q) for q in quids), separators=(",", ":")).encode("utf-8")).hexdigest()


def load_list(manifest_path, name):
    """quid list of a registered subset (k01_s0 ... ; k100 = reader_train60k for the LLM; cap20k_*; dev_sel2000), sha256-checked."""
    m = json.load(open(manifest_path))
    if name == "dev_sel2000":
        ent = m["dev_selection"]
    elif name in m["fallback_cap_lists"]:
        ent = m["fallback_cap_lists"][name]
    elif name == "k100":
        ent = m["llm_train_lists"]["k100"]
    else:
        ent = m["subsets"][name]
    q = [int(x) for x in ent["quids"]]
    assert list_sha256(q) == ent["sha256"], f"list sha256 mismatch for {name}"
    assert len(q) == int(ent["n"])
    return q


def k_of(subset):
    """'k01_s0' -> 'k01'; 'k100' -> 'k100'; cap lists keep their name"""
    return subset if subset.startswith("cap") else subset.split("_")[0]


# ----------------------------------------------------------------------------------------------- KG vocabulary
def kg_name(x):
    return x.replace("_", " ")


def load_kg(full_txt):
    """facts (h, r, t, date) in space form (relations keep underscores); CSV-unquotes the one quoted entity."""
    out = []
    with open(full_txt, encoding="utf-8") as f:
        for l in f:
            p = l.rstrip("\n").split("\t")
            if len(p) != 4:
                continue
            h, r, t, d = p
            h, t = [x[1:-1].replace('""', '"') if x.startswith('"') and x.endswith('"') else x for x in (h, t)]
            out.append((kg_name(h), r, kg_name(t), d))
    return out


def norm_answer(s):
    """rule: stripping, whitespace collapsing, case folding and _ -> space"""
    return re.sub(r"\s+", " ", s.replace("_", " ").strip()).casefold()


def entity_norm_map(entity_names):
    m = {}
    for e in entity_names:
        m.setdefault(norm_answer(e), e)
    return m


_TIME = re.compile(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$")


def is_time(s):
    return bool(_TIME.match(s or ""))


def valid_time(s):
    mm = _TIME.match(s)
    if not mm:
        return False
    y, mo, d = int(mm.group(1)), mm.group(2), mm.group(3)
    try:
        if d is not None:
            dt = datetime.date(y, int(mo), int(d)); return DATE_LO <= dt <= DATE_HI
        if mo is not None:
            return 1 <= int(mo) <= 12 and (2005, 1) <= (y, int(mo)) <= (2015, 12)
        return 2005 <= y <= 2015
    except ValueError:
        return False


def canon_answer(text, enorm):
    """-> (answer_string, valid). Valid entity answers are mapped to the KG name they normalise to; valid times kept;
    invalid outputs are returned (whitespace-collapsed) and stay at rank 1 as a miss (no fallback)."""
    s = re.sub(r"\s+", " ", text.strip())
    n = norm_answer(s)
    if n in enorm:
        return enorm[n], True
    if valid_time(s):
        return s, True
    return s, False


def match_len(answers):
    """official semantics: predictions are truncated to len(answers[0]) for year (4) / month (7) gold"""
    L = len(answers[0])
    return L if (L in (4, 7) and is_time(answers[0])) else None


def gold_hit(cand, answers):
    L = match_len(answers)
    return (cand[:L] if L else cand) in set(answers)


# ----------------------------------------------------------------------------------------------- bundle schema
KEYMAP = {
    "candidates": ["candidates", "cands", "top_candidates"],
    "facts": ["facts", "supporting_facts", "top20_facts"],
    "parse": ["parse", "parser", "parse_info"],
    "operator_trace": ["operator_trace", "trace", "op_trace"],
    "question": ["question", "q", "text"],
}
CAND_KEYS = {"candidate": ["candidate", "name", "answer", "c"], "rule_rank": ["rule_rank", "rank_RULE", "rank_rule"],
             "s_score": ["s_score", "S", "score", "S_seed0"], "operator_satisfied": ["operator_satisfied", "sat", "satisfied", "op_sat"],
             "fact_ids": ["support_fact_idx", "fact_ids", "facts", "support", "supporting_facts"], "s_rank": ["s_rank", "rank", "rank_S"]}
# structural k% bundles (kpct-parse-v1): candidates carry 1-based rule_rank / s_rank and support_fact_idx (0-based
# indices into facts; support_fids are KG line ids and are not used); facts are {fid, head, relation, tail, date}.


def _get(d, names, default=KeyError):
    for n in names:
        if n in d:
            return d[n]
    if default is KeyError:
        raise KeyError(names[0])
    return default


def _fact_tuple(f):
    if isinstance(f, dict):
        return (_get(f, ["h", "subject", "head"]), _get(f, ["r", "relation"]), _get(f, ["t", "object", "tail"]),
                _get(f, ["date", "d", "time", "ts"]))
    if isinstance(f, (list, tuple)) and len(f) >= 4:
        return tuple(f[:4])
    raise BundleSchemaError(f"fact not (h, r, t, date): {str(f)[:120]}")


def normalize_bundle(b, require_gold=False):
    """canonical bundle: quid:int, question:str, parse:dict, candidates:[{candidate, s_rank, rule_rank, s_score,
    operator_satisfied, fact_ids}] in S order, facts:[(h, r, t, date)], operator_trace:str|dict, gold:list|None"""
    miss = [k for k, v in KEYMAP.items() if not any(n in b for n in v)]
    if "quid" not in b:
        miss.append("quid")
    if miss:
        raise BundleSchemaError(f"bundle {b.get('quid')} missing fields {miss}; present {sorted(b)}")
    cands = []
    raw_c = _get(b, KEYMAP["candidates"]) or []
    one_based = any(isinstance(c, dict) and "support_fact_idx" in c for c in raw_c)     # structural schema: 1-based ranks
    for i, c in enumerate(raw_c):
        if isinstance(c, str):
            c = {"candidate": c}
        fid = _get(c, CAND_KEYS["fact_ids"], [])
        rr = _get(c, CAND_KEYS["rule_rank"], None)
        if one_based and rr is not None:
            rr = int(rr) - 1                                   # canonical form: 0-based (rendered + 1)
        cands.append(dict(candidate=str(_get(c, CAND_KEYS["candidate"])), s_rank=int(_get(c, CAND_KEYS["s_rank"], i)),
                          rule_rank=rr, s_score=_get(c, CAND_KEYS["s_score"], None),
                          operator_satisfied=_get(c, CAND_KEYS["operator_satisfied"], None),
                          fact_ids=[int(x) for x in (fid or [])]))
    cands.sort(key=lambda c: c["s_rank"])
    facts = [_fact_tuple(f) for f in (_get(b, KEYMAP["facts"]) or [])]
    gold = b.get("gold", b.get("answers"))
    if require_gold and not gold:
        raise BundleSchemaError(f"train bundle {b['quid']} has no gold")
    return dict(quid=int(b["quid"]), question=_get(b, KEYMAP["question"]), parse=_get(b, KEYMAP["parse"]) or {},
                candidates=cands[:K_CANDS], facts=facts[:N_FACTS], operator_trace=_get(b, KEYMAP["operator_trace"]) or "",
                gold=gold)


def check_bundles(rows, split):
    """schema check over a whole file -> stats dict; raises on the first malformed row"""
    st = dict(n=len(rows), n_cands=[], n_facts=[], gold_rows=0)
    for r in rows:
        b = normalize_bundle(r, require_gold=(split == "train"))
        st["n_cands"].append(len(b["candidates"])); st["n_facts"].append(len(b["facts"]))
        st["gold_rows"] += int(bool(b["gold"]))
        if split != "train" and b["gold"]:
            raise BundleSchemaError(f"{split} bundle {b['quid']} carries a gold field (train rows only)")
    return dict(n=st["n"], gold_rows=st["gold_rows"], mean_cands=float(np.mean(st["n_cands"])) if rows else 0.0,
                mean_facts=float(np.mean(st["n_facts"])) if rows else 0.0)


# ----------------------------------------------------------------------------------------------- prompts
G_HEAD = ("Answer the temporal question about the ICEWS knowledge graph. An automatic pipeline has parsed the question, "
          "retrieved supporting facts and ranked candidate answers. Its output may be wrong: check it against the facts "
          "and give a different answer when the facts support one.\n"
          "Reply with the answer only: one entity name exactly as written, or a date written YYYY, YYYY-MM or YYYY-MM-DD.\n")
GU_HEAD = ("Answer the temporal question about the ICEWS knowledge graph using the retrieved facts below.\n"
           "Reply with the answer only: one entity name exactly as written, or a date written YYYY, YYYY-MM or YYYY-MM-DD.\n")


def rel_text(r):
    return str(r).replace("_", " ")


def _role_text(direction):
    d = str(direction or "")
    if d in ("topic_or_first_mention_is_head", "head", "object"):
        return "first entity is the subject; answer is the object"
    if d in ("topic_or_first_mention_is_tail", "tail", "subject"):
        return "first entity is the object; answer is the subject"
    return d or "unknown"


def _alt_name(a):
    if isinstance(a, dict):
        return a.get("entity") or a.get("name")
    if isinstance(a, (list, tuple)):
        return a[0] if a else None
    return a


def _label(x):
    return x.get("label") if isinstance(x, dict) else x


def render_parse(p):
    """parse dict -> lines. Accepts the structural kpct-parse-v1 schema (mentions[].alternatives[{entity}], relations
    [{relation, direction, prob}] sorted by prob, qtype/answer_type/granularity {label, prob}, operator_keywords,
    time {level, value}) and the k100-converted form. Probabilities are not shown."""
    L = []
    ents = p.get("entities") or p.get("mentions") or []
    es = []
    for e in ents:
        if isinstance(e, str):
            es.append(e); continue
        alts = [_alt_name(a) for a in (e.get("alternatives") or [])]
        name = e.get("entity") or e.get("name") or (alts[0] if alts else None)
        if name is None:
            continue
        alts = [a for a in alts if a and a != name][:2]
        es.append(f"{name}" + (f" (alternatives: {', '.join(alts)})" if alts else ""))
    L.append("- entities: " + ("; ".join(es) if es else "none"))
    rel, dr = p.get("relation"), p.get("direction") or p.get("answer_role")
    if not rel and p.get("relations"):
        r0 = p["relations"][0]
        rel, dr = (r0.get("relation"), r0.get("direction")) if isinstance(r0, dict) else (r0[0], r0[1])
    if rel:
        L.append(f"- relation: {rel_text(rel)} ({_role_text(dr)})")
    qt = _label(p.get("qtype") or p.get("qtype_pred"))
    op = p.get("operator")
    kw = p.get("operator_keywords") or {}
    kws = [v for v in (kw.get("before_after"), kw.get("first_last"), kw.get("same_granularity")) if v]
    L.append(f"- question type: {qt}" + (f"; operator: {op}" if op and op != qt else "") + (f"; keyword: {', '.join(kws)}" if kws else ""))
    L.append(f"- answer type: {_label(p.get('answer_type') or p.get('answer_type_pred'))}; granularity: "
             f"{_label(p.get('granularity') or p.get('granularity_pred'))}")
    tc = p.get("time")
    if isinstance(tc, dict):
        tc = tc.get("value")
    L.append(f"- time constraint: {tc if tc else 'none'}" + (f"; anchor: {p.get('anchor') or 'none'}" if "anchor" in p else ""))
    return "\n".join(L)


def group_facts(facts):
    """lossless grouping by (h, r, t); groups in first-appearance order, dates chronological inside a group.
    Returns (groups [(h, r, t, [dates])], fact_index -> group number 1-based)."""
    groups, order, f2g = {}, [], {}
    for i, (h, r, t, d) in enumerate(facts):
        k = (h, r, t)
        if k not in groups:
            groups[k] = []; order.append(k)
        groups[k].append(d); f2g[i] = order.index(k) + 1
    return [(h, r, t, sorted(groups[(h, r, t)])) for (h, r, t) in order], f2g


def _fmt_score(s):
    try:
        return f"{float(s):.2f}"
    except (TypeError, ValueError):
        return "n/a"


def render_trace(tr):
    if isinstance(tr, dict):
        return "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in tr.items())
    if isinstance(tr, list):
        return "; ".join(map(str, tr))
    return str(tr)


def cand_order(b, order):
    n = len(b["candidates"])
    if order == "S":
        return list(range(n))
    if order == "random":   # fixed per-question permutation: the only permitted use of quid (rule inputs_allowed)
        return [int(i) for i in np.random.default_rng(int(b["quid"])).permutation(n)]
    raise ValueError(order)


def g_user(b, order="S", bundle_from=None, max_fact_groups=None):
    """G prompt (user turn). bundle_from: SHUF -- whole bundle (parse, candidates, facts, trace) of another question.
    max_fact_groups: truncation for the 2,048-token cap (drop the last fact groups first)."""
    q = b["question"]
    src = bundle_from if bundle_from is not None else b
    groups, f2g = group_facts(src["facts"])
    if max_fact_groups is not None:
        groups = groups[:max_fact_groups]
    s = G_HEAD + f"\nQuestion: {q}\n\nParse:\n{render_parse(src['parse'])}\n"
    s += "\nCandidates (candidate | RULE rank | S score | satisfies operator | supporting facts):\n"
    for j in cand_order(src, order):
        c = src["candidates"][j]
        sat = c["operator_satisfied"]
        sat = "yes" if sat in (True, 1, "yes") else ("no" if sat in (False, 0, "no") else "n/a")
        fg = sorted({f2g[i] for i in c["fact_ids"] if i in f2g and f2g[i] <= len(groups)})
        rr = c["rule_rank"]
        rr = "n/a" if rr is None or (isinstance(rr, float) and np.isnan(rr)) else str(int(rr) + 1)
        s += f"- {c['candidate']} | {rr} | {_fmt_score(c['s_score'])} | {sat} | " + (", ".join(f"F{g}" for g in fg) or "none") + "\n"
    if not src["candidates"]:
        s += "- none\n"
    s += "\nFacts (subject | relation | object | dates):\n"
    s += "".join(f"F{i + 1}. {h} | {rel_text(r)} | {t} | {', '.join(ds)}\n" for i, (h, r, t, ds) in enumerate(groups)) or "none\n"
    s += f"\nOperator trace: {render_trace(src['operator_trace'])}\n"
    return s


def n_fact_groups(b):
    return len(group_facts(b["facts"])[0])


def qonly_user(b):
    """QONLY_G1: G prompt with the whole bundle removed (question only)"""
    return G_HEAD + f"\nQuestion: {b['question']}\n"


def gu_user(question, facts, max_facts=None):
    """GU prompt: question + the top-20 dense-retrieved facts in retrieval order (no parse, candidates, scores, trace)"""
    fx = facts if max_facts is None else facts[:max_facts]
    s = GU_HEAD + f"\nQuestion: {question}\n\nFacts (subject | relation | object | date):\n"
    s += "".join(f"{h} | {rel_text(r)} | {t} | {d}\n" for (h, r, t, d) in fx) or "none\n"
    return s


def prompt_texts_sha256():
    """sha256 of the fixed prompt headers + renderer source, recorded in the amendment before any dev_sel2000 score"""
    import inspect, sys
    mod = sys.modules[__name__]
    src = "".join(inspect.getsource(getattr(mod, f)) for f in
                  ("render_parse", "_alt_name", "_label", "group_facts", "render_trace", "cand_order", "g_user", "qonly_user", "gu_user",
                   "_role_text", "normalize_bundle"))
    return dict(G_HEAD=sha256_text(G_HEAD), GU_HEAD=sha256_text(GU_HEAD), renderers=sha256_text(src), chat_date=CHAT_DATE)


# ----------------------------------------------------------------------------------------------- chat encoding
def encode_prompt(tok, user):
    """token ids of the one-user-turn chat prompt incl. the assistant header (fixed date_string)"""
    txt = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
                                  date_string=CHAT_DATE)
    return tok(txt, add_special_tokens=False)["input_ids"]


def encode_answer(tok, user, answer, prompt_ids=None):
    """answer tokens (incl. <|eot_id|>) exactly as the chat template appends them after the generation prompt"""
    full = tok.apply_chat_template([{"role": "user", "content": user}, {"role": "assistant", "content": answer}],
                                   tokenize=False, date_string=CHAT_DATE)
    p = prompt_ids if prompt_ids is not None else encode_prompt(tok, user)
    f = tok(full, add_special_tokens=False)["input_ids"]
    assert f[:len(p)] == p, "chat template: prompt is not a prefix of prompt+answer"
    return f[len(p):]


def fit_g_prompt(tok, b, answer=None, order="S", bundle_from=None, max_len=MAX_LEN):
    """G/SHUF prompt ids (+ answer ids) under the 2,048-token cap; drops trailing fact groups if needed.
    -> (prompt_ids, answer_ids, n_groups_dropped)"""
    src = bundle_from if bundle_from is not None else b
    ng = n_fact_groups(src)
    a_len = 0
    for keep in range(ng, -1, -1):
        u = g_user(b, order, bundle_from, None if keep == ng else keep)
        p = encode_prompt(tok, u)
        a = encode_answer(tok, u, answer, p) if answer is not None else []
        a_len = len(a) if answer is not None else MAX_NEW_G
        if len(p) + a_len <= max_len:
            return p, a, ng - keep
    return p[-(max_len - a_len):], a, ng


def fit_gu_prompt(tok, question, facts, answer=None, max_len=MAX_LEN):
    for keep in range(len(facts), -1, -1):
        u = gu_user(question, facts, keep)
        p = encode_prompt(tok, u)
        a = encode_answer(tok, u, answer, p) if answer is not None else []
        if len(p) + (len(a) if answer is not None else MAX_NEW_G) <= max_len:
            return p, a, len(facts) - keep
    return p, a, len(facts)


# ----------------------------------------------------------------------------------------------- targets
def g_target(b):
    """rule G_k target: the gold answer ranked highest in the bundle candidate list; answers[0] if none in the list.
    -> (target string, gold_in_list)"""
    gold = b["gold"]
    L = match_len(gold)
    G = set(gold)
    for c in b["candidates"]:           # S order
        x = c["candidate"][:L] if L else c["candidate"]
        if x in G:
            return x, True
    return gold[0], False


def gu_target(gold, facts):
    """rule GU_k target: the gold answer occurring earliest in the retrieved facts; answers[0] if none occurs"""
    G = set(gold)
    L = match_len(gold)
    timeq = is_time(gold[0])
    for (h, r, t, d) in facts:
        if timeq:
            x = d[:L] if L else d
            if x in G:
                return x, True
        else:
            if h in G:
                return h, True
            if t in G:
                return t, True
    return gold[0], False


# ----------------------------------------------------------------------------------------------- ranked lists
def g_ranked(answer, s_list):
    out, seen = [], set()
    for x in [answer] + list(s_list):
        if x not in seen:
            seen.add(x); out.append(x)
    return out


def gu_ranked(answer, facts, pad, k=20):
    """[answer] + entities and dates of the retrieved facts in retrieval order (dates at the granularity of the answer
    if it is a time string), deduplicated, padded with the most frequent k% train answers"""
    L = len(answer) if is_time(answer) else 10
    out, seen = [answer], {answer}
    for (h, r, t, d) in facts:
        for x in (h, t, d[:L]):
            if x not in seen:
                seen.add(x); out.append(x)
    for x in pad:
        if len(out) >= k:
            break
        if x not in seen:
            seen.add(x); out.append(x)
    return out


def derangement_map(quids, seed=0):
    """SHUF: default_rng(0) permutation over the split's quids sorted ascending, fixed points swapped with the next index.
    Returns {quid: donor quid}."""
    qs = sorted(int(q) for q in quids)
    n = len(qs)
    p = np.random.default_rng(seed).permutation(n)
    for i in range(n):
        if p[i] == i:
            j = (i + 1) % n
            p[i], p[j] = p[j], p[i]
    assert not np.any(p == np.arange(n))
    return {qs[i]: qs[int(p[i])] for i in range(n)}
