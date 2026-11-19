"""r3_common.py -- MultiTQ robustness probe R3 (third generator family) and the dev / training paraphrase sets.
Fixed before registration; sha256 recorded in multitq_robustness_rule_R3.json.

Re-uses rob_common.py of R1 (sha256 de4e76ee...) unchanged for GEN_PROMPT, GEN_RETRY_SUFFIX, JUDGE_PROMPT, JUDGE_KEYS,
extract_json, mechanical_check, largest_remainder, list_sha; and r2_common.py (sha256 5622ea2c...) unchanged for the
paired Poisson(1) bootstrap (poisson_boot / summarise) and Holm.
Reads question TEXT, quid and qtype only for sampling (qtype only to stratify; no arm sees it). Never reads answers.
"""
import json
import numpy as np
import rob_common as RC
import r2_common as R2C

SEED_R3 = 20261118
N_SAMPLE = 2000
N_POOL = 600
VIDX = {'ORIG': 0, 'RW3': 4, 'DEVPARA': 5}     # synthetic quid = 10 * quid + vidx (R1 used 1-3, R2 used 3)
B_SECONDARY = 10000
BOOT_SEED = 20261118
ALPHA = 0.05

# ------------------------------------------------------------------ R3 generator (Google Gemini API)
GEMINI_MODEL = 'gemini-2.5-flash'
GEMINI_ENDPOINT = 'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'
GEMINI_GENERATION_CONFIG = {'temperature': 0.0, 'candidateCount': 1, 'maxOutputTokens': RC.GEN_MAX_TOKENS, 'seed': 0,
                            'thinkingConfig': {'thinkingBudget': 0}}
GEMINI_SAFETY = [{'category': c, 'threshold': 'BLOCK_NONE'} for c in
                 ('HARM_CATEGORY_HARASSMENT', 'HARM_CATEGORY_HATE_SPEECH', 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
                  'HARM_CATEGORY_DANGEROUS_CONTENT')]
GEMINI_TRANSPORT_RETRIES = 6          # HTTP 429 / 5xx / timeout: same request re-sent (not a rewrite attempt)


def gemini_body(prompt):
    """one user turn = the prompt text (R1 GEN_PROMPT, optionally + GEN_RETRY_SUFFIX); no system instruction"""
    return {'contents': [{'role': 'user', 'parts': [{'text': prompt}]}],
            'generationConfig': GEMINI_GENERATION_CONFIG, 'safetySettings': GEMINI_SAFETY}


def gemini_text(js):
    """(text or None, finishReason, modelVersion, usageMetadata, blockReason) from a generateContent response;
    thought parts (if any) are excluded; a missing candidate / missing parts gives text None"""
    js = js or {}
    cands = js.get('candidates') or []
    block = (js.get('promptFeedback') or {}).get('blockReason')
    if not cands:
        return None, None, js.get('modelVersion'), js.get('usageMetadata'), block
    c0 = cands[0]
    parts = ((c0.get('content') or {}).get('parts')) or []
    txt = ''.join(p.get('text', '') for p in parts if not p.get('thought'))
    return (txt if parts else None), c0.get('finishReason'), js.get('modelVersion'), js.get('usageMetadata'), block


# ------------------------------------------------------------------ sampling
def draw_sample_r3(test_rows, r1_sample_quids, r2_sample_quids):
    """Fresh qtype-stratified 2,000-question sample of the official test split, DISJOINT from the R1 and R2 samples.
    Candidate set = test minus (R1 2,000 quids union R2 2,000 quids). Same procedure as R1 / R2 (rob_common.draw_sample,
    r2_common.draw_sample_r2): largest-remainder allocation proportional to the candidate set's qtype counts; within
    each qtype (qtypes in sorted order) the quids are sorted ascending and ONE numpy default_rng(20261118) generator
    draws choice(n_qtype, alloc, replace=False). Pool of 600 = R1 pool procedure: largest remainder over the sample's
    qtype counts, default_rng([20261118, 3]).choice within each qtype."""
    excl = set(int(q) for q in r1_sample_quids) | set(int(q) for q in r2_sample_quids)
    by = {}
    for q in test_rows:
        if int(q['quid']) in excl:
            continue
        by.setdefault(q['qtype'], []).append(int(q['quid']))
    for k in by:
        by[k].sort()
    alloc = RC.largest_remainder({k: len(v) for k, v in by.items()}, N_SAMPLE)
    rng = np.random.default_rng(SEED_R3)
    samp = {}
    for qt in sorted(by):
        idx = rng.choice(len(by[qt]), alloc[qt], replace=False)
        samp[qt] = sorted(by[qt][i] for i in idx)
    alloc_p = RC.largest_remainder({k: len(v) for k, v in samp.items()}, N_POOL)
    rng3 = np.random.default_rng([SEED_R3, 3])
    pool = {}
    for qt in sorted(samp):
        idx = rng3.choice(len(samp[qt]), alloc_p[qt], replace=False)
        pool[qt] = sorted(samp[qt][i] for i in idx)
    return samp, pool, alloc, alloc_p


def training_order(k01_quids, k05_quids):
    """registered processing order of the 5% training list (nested subset k05_s0, 19,339 quids): the nested 1% list
    k01_s0 first (ascending quid), then the remaining k05_s0 quids in the order
    numpy default_rng([20261118, 5]).permutation over them sorted ascending. A budget cut keeps a prefix of this order,
    so a partial set always contains the complete 1% list first and then a uniform random subset of the rest."""
    k01 = sorted(int(q) for q in k01_quids)
    s01 = set(k01)
    rest = sorted(int(q) for q in k05_quids if int(q) not in s01)
    assert s01 <= set(int(q) for q in k05_quids)
    perm = np.random.default_rng([SEED_R3, 5]).permutation(len(rest))
    return k01 + [rest[i] for i in perm]


# ------------------------------------------------------------------ secondary analysis on R3 (all method tracks)
def contrasts(o_m, r_m, o_b, r_b):
    """per-question contrasts of a method m against its baseline b on the R3 PRIMARY questions (same order):
    c_drop = drop(b) - drop(m) (> 0: m loses less), c_rw = Hits@1_RW(m) - Hits@1_RW(b) (> 0: m scores higher on R3);
    drop = Hits@1(original question) - Hits@1(R3 rewrite)"""
    o_m, r_m, o_b, r_b = (np.asarray(x, float) for x in (o_m, r_m, o_b, r_b))
    return {'c_drop': (o_b - r_b) - (o_m - r_m), 'c_rw': r_m - r_b}


def method_tests(per_level, B=B_SECONDARY, seed=BOOT_SEED, alpha=ALPHA):
    """per_level: {level: dict(o_m, r_m, o_b, r_b)} for ONE method. For every level and both contrasts: paired
    Poisson(1) bootstrap of the mean contrast (r2_common.summarise, B = 10,000, seed 20261118; one-sided p = share of
    bootstrap means <= 0); Holm over all tests of this method (levels x {c_drop, c_rw}); label SUPPORTED iff the
    Holm-adjusted one-sided p < 0.05, else NOT_SUPPORTED. Also descriptive Hits@1 ORIG / RW / drop per arm."""
    res, keys, ps = {}, [], []
    for lev, d in per_level.items():
        C = contrasts(d['o_m'], d['r_m'], d['o_b'], d['r_b'])
        res[lev] = dict(n=int(len(d['o_m'])),
                        method=dict(orig=float(np.mean(d['o_m'])), rw=float(np.mean(d['r_m'])), drop=float(np.mean(np.asarray(d['o_m'], float) - np.asarray(d['r_m'], float)))),
                        baseline=dict(orig=float(np.mean(d['o_b'])), rw=float(np.mean(d['r_b'])), drop=float(np.mean(np.asarray(d['o_b'], float) - np.asarray(d['r_b'], float)))))
        for cn, cv in C.items():
            s = R2C.summarise(cv, B, seed)
            res[lev][cn] = s
            keys.append((lev, cn)); ps.append(s['p_one_sided'])
    adj = R2C.holm(ps) if ps else []
    for (lev, cn), pa in zip(keys, adj):
        res[lev][cn]['p_holm_one_sided'] = pa
        res[lev][cn]['label'] = 'SUPPORTED' if pa < alpha else 'NOT_SUPPORTED'
    return res


def audit_selection(quids, per_question_contrasts, n_random=100):
    """targeted + random audit set (registered; built only after the method tracks' freeze).
    quids: sorted PRIMARY quids; per_question_contrasts: {name: array over quids} for EVERY method x level x contrast.
    T = quids with any contrast > 0. R = sorted(quids)[i] for i in default_rng([20261118, 7]).choice(n, min(100, n),
    replace=False). Audit set = T union R, presented in the order default_rng([20261118, 8]).permutation over the audit
    set sorted ascending; ids A001, A002, ..."""
    quids = sorted(int(q) for q in quids)
    n = len(quids)
    M = np.vstack([np.asarray(v, float) for v in per_question_contrasts.values()]) if per_question_contrasts else np.zeros((0, n))
    T = {quids[i] for i in range(n) if M.shape[0] and (M[:, i] > 0).any()}
    R = {quids[i] for i in np.random.default_rng([SEED_R3, 7]).choice(n, size=min(n_random, n), replace=False)}
    A = sorted(T | R)
    order = np.random.default_rng([SEED_R3, 8]).permutation(len(A))
    rows = [dict(audit_id=f'A{j + 1:03d}', quid=A[i], in_targeted=A[i] in T, in_random100=A[i] in R) for j, i in enumerate(order)]
    return rows, T, R


def bound_analysis(per_method_levels, quids, rejected, reviewed, B=B_SECONDARY, seed=BOOT_SEED):
    """BOUND (as R2 amendment A2): recompute method_tests on the PRIMARY quids minus the human-rejected rows.
    per_method_levels: {method: {level: dict(o_m, r_m, o_b, r_b) over `quids`}}. Asserts that every UNREVIEWED row has
    every contrast <= 0 (so unreviewed rows cannot create positive evidence). BOUND_HOLDS per test iff Holm-adjusted
    one-sided p < 0.05 after removal, else BOUND_FAILS."""
    quids = [int(q) for q in quids]
    rej = set(int(q) for q in rejected); rev = set(int(q) for q in reviewed)
    keep = np.array([q not in rej for q in quids])
    out = {}
    for m, levs in per_method_levels.items():
        for lev, d in levs.items():
            C = contrasts(d['o_m'], d['r_m'], d['o_b'], d['r_b'])
            for cn, cv in C.items():
                bad = [q for q, c in zip(quids, cv) if q not in rev and c > 0]
                assert not bad, f'unreviewed rows with positive {cn} for {m} {lev}: {bad[:5]}'
        sub = {lev: {k: np.asarray(v, float)[keep] for k, v in d.items()} for lev, d in levs.items()}
        t = method_tests(sub, B, seed)
        for lev in t:
            for cn in ('c_drop', 'c_rw'):
                t[lev][cn]['bound_label'] = 'BOUND_HOLDS' if t[lev][cn]['p_holm_one_sided'] < ALPHA else 'BOUND_FAILS'
        out[m] = t
    return out


def wilson(x, n, z=1.959964):
    if n == 0:
        return [float('nan'), float('nan')]
    p = x / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]
