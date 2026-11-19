"""rob_common.py -- MultiTQ robustness probe (track 'robustness'): sampling, deterministic perturbations, paraphrase
prompts and the mechanical paraphrase checks. Fixed before registration; its sha256 is recorded in
multitq_robustness_rule.json. Reads question TEXT, quid and qtype only (qtype for stratification of the probe sample;
no arm ever sees it). Never reads answers.
"""
import json, hashlib, re
import numpy as np

SEED = 20261028
N_SAMPLE = 2000
N_P3 = 600
PREFIX = "Please answer: "
VARIANTS = ("ORIG", "P1", "P2", "P3")          # variant index = position (synthetic quid = quid * 10 + index)


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def list_sha(quids):
    return sha256_text(json.dumps(sorted(int(x) for x in quids)))


def largest_remainder(counts, n):
    tot = sum(counts.values())
    raw = {k: n * v / tot for k, v in counts.items()}
    base = {k: int(np.floor(x)) for k, x in raw.items()}
    rem = n - sum(base.values())
    for k in sorted(counts, key=lambda k: (-(raw[k] - base[k]), k))[:rem]:
        base[k] += 1
    return base


def draw_sample(test_rows):
    """qtype-stratified n = 2,000 test sample (largest-remainder allocation; within qtype: quids sorted ascending,
    numpy default_rng(20261028).choice without replacement, qtypes in sorted order) and the qtype-stratified
    600-question P3 subsample of it (default_rng([20261028, 3]), same recipe)."""
    by = {}
    for q in test_rows:
        by.setdefault(q["qtype"], []).append(int(q["quid"]))
    for k in by:
        by[k].sort()
    alloc = largest_remainder({k: len(v) for k, v in by.items()}, N_SAMPLE)
    rng = np.random.default_rng(SEED)
    samp = {}
    for qt in sorted(by):
        idx = rng.choice(len(by[qt]), alloc[qt], replace=False)
        samp[qt] = sorted(by[qt][i] for i in idx)
    alloc3 = largest_remainder({k: len(v) for k, v in samp.items()}, N_P3)
    rng3 = np.random.default_rng([SEED, 3])
    p3 = {}
    for qt in sorted(samp):
        idx = rng3.choice(len(samp[qt]), alloc3[qt], replace=False)
        p3[qt] = sorted(samp[qt][i] for i in idx)
    return samp, p3, alloc, alloc3


# ------------------------------------------------------------------ deterministic perturbations
def p1_prefix(q):
    return PREFIX + q


def p2_strip_qmark(q):
    """remove a trailing '?' where present (trailing whitespace stripped first); otherwise unchanged"""
    s = q.rstrip()
    return s[:-1] if s.endswith("?") else q


# ------------------------------------------------------------------ P3 prompts (fixed text; sha256 in the rule)
GEN_PROMPT = """You rewrite questions about a database of dated political events (who did what to whom, and when).
Rewrite the question below so that it asks for exactly the same thing in different words.

Rules:
1. Copy every name of a person, country, organisation, group or role (for example "Danish Ministry of Defence and Security", "the Ukrainian military", "Citizen (India)", "Kitti Wasinondh") character for character. Do not translate, shorten, expand or re-order any name, and do not turn a possessive or adjective form into a different form.
2. Copy every date or time expression (for example "23 Feb 2005", "June 2008", "2008") character for character.
3. Keep the meaning exact: the same set of correct answers, the same kind of answer (a person, organisation or country, or a year, month or date), the same action, the same direction of the action (who acts towards whom), and the same temporal condition (before / after, first / last, on a given date, in the same day / month / year, and which event is the reference point).
4. Change the wording or sentence structure noticeably; do not return the original with only punctuation changed.
5. Write one grammatical English question ending with "?".

Question: {question}

Reply with JSON only: {{"paraphrase": "<rewritten question>", "kept_verbatim": ["<each name and time expression copied verbatim>"]}}"""

GEN_RETRY_SUFFIX = """

An earlier rewrite was rejected because it did not keep the names, dates or meaning exactly, or was too close to the original: "{previous}". Write a new rewrite that follows every rule."""

JUDGE_PROMPT = """You are verifying a question rewrite for a benchmark about dated political events (ICEWS). Compare the two questions.

Original: {original}
Rewrite: {paraphrase}

Answer each item strictly with true or false:
- same_answer_set: any answer that is correct for the original is correct for the rewrite, and vice versa (the expected kind of answer is also the same: an actor, or a year, month or date).
- same_entities: the rewrite mentions exactly the same named actors (people, countries, organisations, groups), with the names written identically, and adds none.
- same_time: an explicit date, month or year in the original appears unchanged in the rewrite; if the original has none, the rewrite has none.
- same_relation: the action, its direction (who acts towards whom) and the temporal condition (before / after, first / last, on or in a time, same day / month / year, and the reference event) are unchanged.
- grammatical: the rewrite is a single fluent, grammatical English question.

Reply with JSON only: {{"same_answer_set": true/false, "same_entities": true/false, "same_time": true/false, "same_relation": true/false, "grammatical": true/false, "note": "<at most 15 words>"}}"""

JUDGE_KEYS = ("same_answer_set", "same_entities", "same_time", "same_relation", "grammatical")
GEN_MAX_TOKENS = 300
JUDGE_MAX_TOKENS = 200
MAX_ATTEMPTS = 2


def extract_json(txt):
    """first {...} object in a model reply"""
    if txt is None:
        return None
    t = txt.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    a = t.find("{")
    b = t.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        return json.loads(t[a:b + 1])
    except Exception:
        return None


# ------------------------------------------------------------------ mechanical checks (gold-free)
_STOP_FIRST = {"who", "whom", "whose", "when", "which", "what", "where", "in", "on", "after", "before", "with", "to",
               "from", "for", "at", "during", "by", "since", "until", "of", "the", "a", "an", "how", "did", "was",
               "is", "were", "please", "answer", "who's", "whom's", "prior", "following", "once", "between",
               "could", "can", "would", "will", "do", "does", "tell", "has", "had", "have", "are", "as", "throughout",
               "against", "about"}
_CONNECT = {"of", "and", "the", "for", "de", "del", "la", "le", "du", "von", "van", "al", "el", "da", "do", "dos",
            "y", "&", "-", "'s", "’s", "(", ")"}
_TOK = re.compile(r"\(|\)|[A-Za-z0-9\u00C0-\u024F][A-Za-z0-9\u00C0-\u024F.\-]*(?:['’]s)?|&|-")


def _is_cap(t):
    return bool(t) and (t[0].isupper() or t[0].isdigit())


def protected_chunks(q):
    """maximal runs of capitalised tokens (allowing connectors between capitalised tokens) in the ORIGINAL question;
    a run consisting only of a sentence-initial function word is dropped. Each chunk must reappear verbatim."""
    toks = [(m.group(0), m.start(), m.end()) for m in _TOK.finditer(q)]
    chunks, i, n = [], 0, len(toks)
    while i < n:
        t = toks[i][0]
        if not _is_cap(t) or (t.lower() in _STOP_FIRST and not any(c.isdigit() for c in t)):
            i += 1
            continue
        j = i
        k = i + 1
        while k < n:
            tk = toks[k][0]
            if _is_cap(tk) and not (tk.lower() in _STOP_FIRST and tk.lower() not in ("of", "the")):
                j = k
                k += 1
            elif tk.lower() in _CONNECT:
                k += 1
            else:
                break
        # include a closing parenthesis directly after the run, and possessive already in the token
        end = toks[j][2]
        if j + 1 < n and toks[j + 1][0] == ")" and "(" in q[toks[i][1]:end]:
            end = toks[j + 1][2]
        start = toks[i][1]
        if start > 0 and q[start - 1] == "(":
            start -= 1
        chunks.append(q[start:end])
        i = j + 1
    return [c for c in chunks if c.strip()]


def _norm_cmp(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def mechanical_check(original, paraphrase, kept, parse_time):
    """returns (ok, reasons). parse_time = mtq_lib.parse_time (frozen time regex)."""
    r = []
    if not isinstance(paraphrase, str) or not paraphrase.strip():
        return False, ["empty"]
    p = paraphrase.strip()
    if "\n" in p:
        r.append("multiline")
    if not p.endswith("?"):
        r.append("no_qmark")
    if len(p) > 2.5 * len(original):
        r.append("too_long")
    if _norm_cmp(p) == _norm_cmp(original):
        r.append("identical")
    to, tp = parse_time(original), parse_time(p)
    if to is not None:
        if original[to[2][0]:to[2][1]] not in p:
            r.append("time_not_verbatim")
        if tp is None or (tp[0], tp[1]) != (to[0], to[1]):
            r.append("time_parse_changed")
    elif tp is not None:
        r.append("time_added")
    for c in protected_chunks(original):
        if c not in p:
            r.append("chunk_missing:" + c)
    if isinstance(kept, list):
        for s in kept:
            if isinstance(s, str) and s.strip() and (s not in original or s not in p):
                r.append("kept_not_verbatim:" + s)
    return (len(r) == 0), r
