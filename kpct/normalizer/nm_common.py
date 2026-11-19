"""nm_common.py -- question normalizer N (track 'normalizer'): prompt, training pairs, guard, item /
synthetic-quid bookkeeping, identity-reuse rule and the dev selection objective. Fixed before registration; sha256 in
multitq_normalizer_rule.json. Reads question TEXT and quids only, except score_* helpers, which take gold explicitly
(used only in the registered dev scoring and the single test scoring pass).

Re-uses unchanged: pk_common (frozen parser code, sha256 0b2cc2fa...) for the chat-template encoding with the fixed
date_string; rob_common (R1, sha256 de4e76ee...) protected_chunks; mtq_lib (frozen, sha256 471da9d3...) parse_time;
r2_common (sha256 5622ea2c...) summarise / holm; r3_common (sha256 d80d811e...) method_tests.
"""
import json, re, hashlib
import numpy as np

MODEL_8B = "meta-llama/Llama-3.1-8B-Instruct"
CHAT_DATE = "26 Jul 2024"                      # = pk_common.CHAT_DATE / gk_common (fixed date_string)
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
OPT = dict(lr=2e-4, weight_decay=0.0, warmup=0.03, grad_clip=1.0, eff_batch=16)
E_MAX = 2                                      # = pk_common.E_MAX['k05'] (the 5% list recipe)
SAVE_AT_EPOCHS = (0.5, 1.0, 2.0)               # candidate checkpoints ep0.5, ep1, ep2
MAX_LEN = 512
MAX_NEW = 128
TRAIN_TOK = 768                                # micro-batch padded-token limit (24 GB card; = parser seed-track setting)
SEED = 0

N_HEAD = ("Rewrite the question below into the standard wording of the question templates of a benchmark about dated "
          "political events. Copy every name and every date or time expression exactly as written. If the question "
          "already uses the standard wording, return it unchanged. Reply with the question only.")


def n_user(question):
    return N_HEAD + f"\nQuestion: {question}\n"


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def prompt_sha256():
    return sha256_text(N_HEAD + "|" + n_user("{question}"))


def encode_prompt(tok, user):
    """= pk_common.encode_prompt (one user turn, assistant header, fixed date_string)"""
    txt = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
                                  date_string=CHAT_DATE)
    return tok(txt, add_special_tokens=False)["input_ids"]


def encode_answer(tok, user, answer, prompt_ids=None):
    """= pk_common.encode_answer (answer tokens incl. <|eot_id|>)"""
    full = tok.apply_chat_template([{"role": "user", "content": user}, {"role": "assistant", "content": answer}],
                                   tokenize=False, date_string=CHAT_DATE)
    p = prompt_ids if prompt_ids is not None else encode_prompt(tok, user)
    f = tok(full, add_special_tokens=False)["input_ids"]
    assert f[:len(p)] == p, "chat template: prompt is not a prefix of prompt+answer"
    return f[len(p):]


# ------------------------------------------------------------------ training pairs (5% list only)
def training_pairs(train_rows, para_rows):
    """train_rows: {quid: original question text} of the k05_s0 list (questions/train.json); para_rows: rows of
    multitq_train5pct_paraphrase_set.json. Pairs: (paraphrase -> original) for every mech_ok row, plus the identity pair
    (original -> original) for the SAME questions; ratio 1 : 1. Sorted by (quid, kind) with kind 0 = paraphrase,
    1 = identity; ex_id = 10 * quid + kind."""
    out = []
    for r in para_rows:
        if not r.get("mech_ok"):
            continue
        q = int(r["quid"]); o = train_rows[q]
        out.append(dict(ex_id=10 * q + 0, quid=q, kind="para", input=r["paraphrase"].strip(), target=o))
        out.append(dict(ex_id=10 * q + 1, quid=q, kind="identity", input=o, target=o))
    out.sort(key=lambda e: e["ex_id"])
    return out


# ------------------------------------------------------------------ guard (gold-free, text only)
def _toks(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def jaccard(a, b):
    A, B = _toks(a), _toks(b)
    return len(A & B) / max(1, len(A | B))


def name_pieces(s, RC):
    """protected name strings of s: rob_common.protected_chunks(s); a trailing possessive ('s / ’s) is removed and a chunk
    joined by ' and ' is split into its parts (protected_chunks joins 'A and B' into one run)"""
    out = []
    for c in RC.protected_chunks(s):
        c = re.sub(r"(?:'s|’s)$", "", c.strip())
        for part in c.split(" and "):
            part = part.strip()
            if part:
                out.append(part)
    return out


def guard(x, y, finish_reason, variant, tau, RC, parse_time):
    """returns (ok, reasons) for raw normalizer output y of input x.
    W (well-formed, 'parseable'): non-empty, one line, ends with '?', len <= 2.5 * len(x), finish_reason == 'stop';
    E (names): every name piece of x (name_pieces: rob_common.protected_chunks, trailing possessive 's removed, split on
       ' and ') appears verbatim in y;
    T (dates): if mtq_lib.parse_time(x) is not None: the matched time string of x appears verbatim in y and parse_time(y)
       has the same (granularity, value); if parse_time(x) is None: parse_time(y) is None (no time added);
    variant 'B' adds N (no new names): every name piece of y appears verbatim in x;
    tau > 0 adds O (overlap floor): token Jaccard(x, y) >= tau."""
    r = []
    if not isinstance(y, str) or not y.strip():
        return False, ["empty"]
    y = y.strip()
    if "\n" in y:
        r.append("multiline")
    if not y.endswith("?"):
        r.append("no_qmark")
    if len(y) > 2.5 * len(x):
        r.append("too_long")
    if finish_reason != "stop":
        r.append("truncated")
    for c in name_pieces(x, RC):
        if c not in y:
            r.append("chunk_missing")
            break
    tx, ty = parse_time(x), parse_time(y)
    if tx is not None:
        if x[tx[2][0]:tx[2][1]] not in y:
            r.append("time_not_verbatim")
        if ty is None or (ty[0], ty[1]) != (tx[0], tx[1]):
            r.append("time_parse_changed")
    elif ty is not None:
        r.append("time_added")
    if variant == "B":
        for c in name_pieces(y, RC):
            if c not in x:
                r.append("chunk_added")
                break
    if tau and jaccard(x, y) < tau:
        r.append("low_overlap")
    return (len(r) == 0), r


GUARD_GRID = [("A", 0.0), ("A", 0.2), ("B", 0.0), ("B", 0.2)]


def final_text(x, raw, finish_reason, variant, tau, RC, parse_time):
    """(text fed to the unchanged pipeline, status in {'unchanged', 'changed', 'fallback'}, guard reasons).
    unchanged: the stripped output equals the input; fallback: output differs but fails the guard -> input used."""
    y = raw.strip() if isinstance(raw, str) else ""
    if y == x.strip():
        return x, "unchanged", []
    ok, rs = guard(x, raw, finish_reason, variant, tau, RC, parse_time)
    return (y, "changed", []) if ok else (x, "fallback", rs)


# ------------------------------------------------------------------ identity reuse
def reuse_source(x, y, o):
    """which prediction an item uses. x: input text, y: final text, o: the question's ORIGINAL text.
    'input'   : y == x  -> the frozen prediction of the input (the pipeline and its input are unchanged);
    'original': y == o != x (a rewrite normalized back to the exact original) -> the frozen ORIG prediction;
    'run'     : anything else -> a new run of the unchanged frozen pipeline on y."""
    if y == x:
        return "input"
    if y == o:
        return "original"
    return "run"


def assign_squids(texts_by_quid):
    """texts_by_quid: {quid: sorted list of distinct texts needing a run}. synthetic quid = 10 * quid + v, v = 1..9 in
    sorted-text order (v = 0 is reserved for the identity-control rows = original text)."""
    m = {}
    for q in sorted(texts_by_quid):
        ts = sorted(set(texts_by_quid[q]))
        assert len(ts) <= 9, f"quid {q}: {len(ts)} distinct texts"
        for v, t in enumerate(ts, 1):
            m[(q, t)] = 10 * q + v
    return m


# ------------------------------------------------------------------ dev selection objective (registered)
LOSS_CAP = 0.003


def objective(h):
    """h: {'P_5': {'orig': mean Hits@1 dev ORIG (2,000), 'para': mean Hits@1 DEV_ACCEPTED (1,762)}, 'G_5': {...},
    'base': {'P_5': {'orig': ...}, 'G_5': {...}}}. Returns (J, feasible, max_orig_loss)."""
    J = 0.5 * h["P_5"]["para"] + 0.5 * h["G_5"]["para"]
    loss = max(h["base"]["P_5"]["orig"] - h["P_5"]["orig"], h["base"]["G_5"]["orig"] - h["G_5"]["orig"])
    return J, loss <= LOSS_CAP + 1e-12, loss


def select(cands):
    """cands: list of dict(ckpt_epoch, guard, tau, J, feasible, loss) in registered grid order. Feasible candidates:
    max J; ties |dJ| < 0.0005 -> earlier checkpoint, then guard B before A, then larger tau (closer to the baseline).
    No feasible candidate: smallest loss (same tie order), flagged CONSTRAINT_UNMET."""
    def pref(c):
        return (c["ckpt_epoch"], 0 if c["guard"] == "B" else 1, -c["tau"])
    feas = [c for c in cands if c["feasible"]]
    if feas:
        best = max(c["J"] for c in feas)
        tied = sorted([c for c in feas if best - c["J"] < 0.0005], key=pref)
        return tied[0], "SELECTED"
    lo = min(c["loss"] for c in cands)
    tied = sorted([c for c in cands if c["loss"] - lo < 0.0005], key=pref)
    return tied[0], "CONSTRAINT_UNMET"
