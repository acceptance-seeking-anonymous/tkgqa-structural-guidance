"""pk_common.py -- MultiTQ k% experiment, LLM PARSER track (P_k, P_0).

Rule: multitq_kpct_decision_rule.json v1.0 (sha256 0f5f760450356349a7bf353301520d76de3617f1dfec0b4e4b1430c97b1d6cd7),
arms.P_k / arms.P_0.

Compact P JSON (rule arms.P_k.json_schema), keys in this fixed order:
  {"mentions": [{"text", "entity"}], "relation", "answer_role": subject|object, "qtype", "answer_type": entity|time,
   "granularity": day|month|year, "time": string|null, "anchor": KG entity|null, "order": before|after|first|last|null}
Serialised with json.dumps(obj, ensure_ascii=False) (default separators), identical to the stage-0 benchmark targets.

This module never reads a dev/test answer. Train answers are read only by pk_silver.py for the quids of the
registered LLM train list of the arm (assertion there).
"""
import gzip, hashlib, json, os, re, datetime, unicodedata

# ----------------------------------------------------------------------------------------------- fixed recipe
MODEL_8B = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_70B = "meta-llama/Llama-3.1-70B-Instruct"
CHAT_DATE = "26 Jul 2024"          # fixed date_string of the Llama-3.1 chat template (same as the guided track)
MAX_LEN = 2048                     # rule lora_recipe (as G_k)
MAX_NEW_P = 160                    # rule arms.P_k.decoding / arms.P_0.selection: greedy, max 160 new tokens
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
OPT = dict(lr=2e-4, weight_decay=0.0, warmup=0.03, grad_clip=1.0, eff_batch=16)
E_MAX = {"k01": 3, "k05": 2, "k10": 1, "k100": 1, "cap20k_k10": 1, "cap20k_k100": 1}
TAUS = [None, -0.30, -0.10, -0.03]  # rule arms.P_k.fallback: tau in {off, -0.30, -0.10, -0.03}; tie -> off
QTYPES = ['equal', 'before_after', 'first_last', 'equal_multi', 'after_first', 'before_last']
LEVELS = ['day', 'month', 'year']
ATYPES = ['entity', 'time']
ROLES = ['subject', 'object']
ORDERS = ['before', 'after', 'first', 'last']
KEYS = ["mentions", "relation", "answer_role", "qtype", "answer_type", "granularity", "time", "anchor", "order"]
DIR_HEAD = 'topic_or_first_mention_is_head'   # answer is the object of (topic, relation, ?)
DIR_TAIL = 'topic_or_first_mention_is_tail'   # answer is the subject of (?, relation, topic)
DATE_LO, DATE_HI = datetime.date(2005, 1, 1), datetime.date(2015, 12, 31)

# P_k prompt: identical to the P-style prompt of the stage-0 benchmark (guided/code/gk_train.py P_HEAD), one user turn
P_HEAD = ("Parse the temporal question about the ICEWS knowledge graph into JSON with the keys mentions (list of "
          "{text, entity}), relation, answer_role (subject|object), qtype, answer_type (entity|time), granularity "
          "(day|month|year), time, anchor, order. Use exact KG entity and relation names.\n")


def p_user(question):
    return P_HEAD + f"\nQuestion: {question}\n"


# ----------------------------------------------------------------------------------------------- io
def read_jsonl(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def iter_jsonl(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                yield json.loads(l)


def write_jsonl(p, rows):
    op = gzip.open if p.endswith(".gz") else open
    tmp = p + ".tmp"
    with op(tmp, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, p)


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
    """quid list of a registered subset (LLM list for k100 = reader_train60k), sha256-checked."""
    m = json.load(open(manifest_path))
    if name == "dev_sel2000":
        ent = m["dev_selection"]
    elif name in m.get("fallback_cap_lists", {}):
        ent = m["fallback_cap_lists"][name]
    else:      # LLM train lists: k<100 entries are "= subsets.<name>"; k100 = reader_train60k (a full entry)
        e_ = m.get("llm_train_lists", {}).get(name)
        ent = e_ if isinstance(e_, dict) and "quids" in e_ else m["subsets"][name]
    q = [int(x) for x in ent["quids"]]
    assert list_sha256(q) == ent["sha256"], f"list sha256 mismatch for {name}"
    assert len(q) == int(ent["n"]), f"list size mismatch for {name}"
    return q


def k_of(subset):
    """'k01_s0' -> 'k01'; 'k100' -> 'k100'; cap lists keep their name"""
    return subset if subset.startswith("cap") else subset.split("_")[0]


def sha_recipe(obj):
    """rule sha256_recipe: SHA-256 of json.dumps({all keys except sha256}, indent=1, sort_keys=True, default=str)"""
    d = {k: v for k, v in obj.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(d, indent=1, sort_keys=True, default=str).encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------------------------- compact JSON
def dumps_compact(c):
    return json.dumps({k: c.get(k) for k in KEYS}, ensure_ascii=False)


def _strip_acc(s):
    return ''.join(ch for ch in unicodedata.normalize('NFKD', s) if not unicodedata.combining(ch))


def _norm_map(s):
    """accent-stripped lower-case copy of s with a char index map back to s (same normalisation as mtq_lib.toks)"""
    out, idx = [], []
    for i, ch in enumerate(s):
        t = _strip_acc(ch).lower().replace("\u2019", "'")
        for c2 in t:
            out.append(c2); idx.append(i)
    return "".join(out), idx


def surface_text(question, ws, lo, hi):
    """Original-case surface string of linker tokens ws[lo..hi] in the question (k-th occurrence if repeated);
    falls back to the lower-case token text."""
    toks = ws[lo:hi + 1]
    tok_text = " ".join(toks)
    if not toks or "timex" in toks:
        return tok_text
    occ = sum(1 for i in range(0, lo) if ws[i:i + len(toks)] == toks)   # occurrences before this one
    n, idx = _norm_map(question)
    pat = r"(?<![a-z0-9])" + r"(?:'s)?[^a-z0-9]+".join(re.escape(t) for t in toks) + r"(?![a-z0-9])"
    ms = list(re.finditer(pat, n))
    if len(ms) <= occ:
        return tok_text
    m = ms[occ]
    return question[idx[m.start()]: idx[m.end() - 1] + 1]


def order_from_keywords(qtype, answer_type, ba, fl):
    """One `order` slot for the two structural operator keywords. Priority = the keyword the frozen operator reads:
    time answers read first/last (mtq_cand.time_candidates); entity before_after reads before/after; entity
    first_last / equal_multi read first/last; equal / after_first / before_last read neither (before/after first)."""
    if answer_type == 'time':
        return fl or ba
    if qtype == 'before_after':
        return ba or fl
    if qtype in ('first_last', 'equal_multi'):
        return fl or ba
    return ba or fl


def compact_from_parse(p, question, anchor_entity=None):
    """Serialise the top-1 hypothesis of a kpct-parse-v1 parse to the compact P JSON.
    p: parse with relations[0] = top relation-direction, mentions[*].alternatives[0] = top entity.
    anchor_entity: KG name of the anchor mention (mtq_cand.anchor_index on the same record) or None."""
    ws = p['tokens']
    ments = []
    for m in sorted(p['mentions'], key=lambda m: m['span'][0]):
        if not m.get('alternatives'):
            continue
        lo, hi = m['span']
        ments.append({"text": surface_text(question, ws, lo, hi), "entity": m['alternatives'][0]['entity']})
    r0 = p['relations'][0]
    kw = p.get('operator_keywords') or {}
    c = {"mentions": ments,
         "relation": r0['relation'],
         "answer_role": "object" if r0['direction'] in (DIR_HEAD, 1) else "subject",
         "qtype": p['qtype']['label'],
         "answer_type": p['answer_type']['label'],
         "granularity": p['granularity']['label'],
         "time": (str(p['time']['value']) if p.get('time') else None),
         "anchor": anchor_entity,
         "order": order_from_keywords(p['qtype']['label'], p['answer_type']['label'], kw.get('before_after'),
                                      kw.get('first_last'))}
    return c


# ----------------------------------------------------------------------------------------------- LLM output parsing
_TIME = re.compile(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$")


def valid_time(s):
    mm = _TIME.match(s or "")
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


class PInvalid(ValueError):
    """the generated text is not a valid compact parse (reason in args[0])"""


def extract_json(text):
    """Strict: the stripped generation must be one JSON object; if not, the span from the first '{' to the last '}'
    is tried once (fixed before any dev_sel2000 score)."""
    s = text.strip()
    try:
        o = json.loads(s)
    except Exception:
        a, b = s.find("{"), s.rfind("}")
        if a < 0 or b <= a:
            raise PInvalid("no_json")
        try:
            o = json.loads(s[a:b + 1])
        except Exception:
            raise PInvalid("bad_json")
    if not isinstance(o, dict):
        raise PInvalid("not_object")
    return o


def validate_compact(o):
    """Schema check of a compact P JSON (values only; KG names are checked by the executor side).
    Returns a cleaned dict with the 9 keys."""
    for k in KEYS:
        if k not in o:
            raise PInvalid(f"missing_{k}")
    if not isinstance(o["mentions"], list):
        raise PInvalid("bad_mentions")
    ments = []
    for m in o["mentions"]:
        if not isinstance(m, dict) or not isinstance(m.get("entity"), str) or not m["entity"].strip():
            raise PInvalid("bad_mention")
        ments.append({"text": str(m.get("text") or ""), "entity": m["entity"].strip()})
    if not isinstance(o["relation"], str) or not o["relation"].strip():
        raise PInvalid("bad_relation")
    if o["answer_role"] not in ROLES:
        raise PInvalid("bad_answer_role")
    if o["qtype"] not in QTYPES:
        raise PInvalid("bad_qtype")
    if o["answer_type"] not in ATYPES:
        raise PInvalid("bad_answer_type")
    if o["granularity"] not in LEVELS:
        raise PInvalid("bad_granularity")
    t = o["time"]
    if t is not None:
        t = str(t).strip()
        if not valid_time(t):
            raise PInvalid("bad_time")
    an = o["anchor"]
    if an is not None and (not isinstance(an, str) or not an.strip()):
        raise PInvalid("bad_anchor")
    if o["order"] is not None and o["order"] not in ORDERS:
        raise PInvalid("bad_order")
    return {"mentions": ments, "relation": o["relation"].strip(), "answer_role": o["answer_role"], "qtype": o["qtype"],
            "answer_type": o["answer_type"], "granularity": o["granularity"], "time": t,
            "anchor": an.strip() if isinstance(an, str) else None, "order": o["order"]}


def mean_logprob(token_ids, logprobs, stop_ids):
    """mean token log-prob of the generated JSON: the sampled tokens' log-probs, stop/eos tokens excluded."""
    vals = [lp for t, lp in zip(token_ids, logprobs) if t not in stop_ids and lp is not None]
    return (sum(vals) / len(vals)) if vals else float("-inf")


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


def prompt_sha256():
    return dict(P_HEAD=sha256_text(P_HEAD), p_user_template=sha256_text(p_user("{question}")), chat_date=CHAT_DATE)
