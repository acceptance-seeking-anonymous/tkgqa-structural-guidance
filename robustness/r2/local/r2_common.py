"""r2_common.py -- MultiTQ robustness probe R2 (confirmatory replication of the parser-vs-reader robustness contrast).
Fixed before registration; sha256 recorded in multitq_robustness_rule_R2.json.

Re-uses rob_common.py of R1 (sha256 de4e76ee...) unchanged for: GEN_PROMPT, GEN_RETRY_SUFFIX, JUDGE_PROMPT,
JUDGE_KEYS, extract_json, mechanical_check, largest_remainder, list_sha.
Reads question TEXT, quid and qtype only for sampling (qtype only to stratify; no arm sees it). Never reads answers.
"""
import json
import numpy as np
import rob_common as RC

SEED_R2 = 20261029
N_SAMPLE = 2000
N_POOL = 600
VARIANTS = ('ORIG', 'RW')            # synthetic quid = 10 * quid + vidx; ORIG 0, RW (R2 rewrite) 3 (same slot as R1 P3)
VIDX = {'ORIG': 0, 'RW': 3}
B_PRIMARY = 10000
BOOT_SEED_PRIMARY = 20261029
ALPHA = 0.05


def draw_sample_r2(test_rows, r1_sample_quids):
    """Fresh qtype-stratified 2,000-question sample of the official test split, DISJOINT from the R1 sample.
    Candidate set = test minus the R1 2,000 quids. Same procedure as R1 (rob_common.draw_sample): largest-remainder
    allocation proportional to the candidate set's qtype counts; within each qtype (qtypes in sorted order) the quids
    are sorted ascending and ONE numpy default_rng(20261029) generator draws choice(n_qtype, alloc, replace=False).
    Pool of 600 = R1 pool procedure: largest remainder over the sample's qtype counts, default_rng([20261029, 3])."""
    excl = set(int(q) for q in r1_sample_quids)
    by = {}
    for q in test_rows:
        if int(q['quid']) in excl:
            continue
        by.setdefault(q['qtype'], []).append(int(q['quid']))
    for k in by:
        by[k].sort()
    alloc = RC.largest_remainder({k: len(v) for k, v in by.items()}, N_SAMPLE)
    rng = np.random.default_rng(SEED_R2)
    samp = {}
    for qt in sorted(by):
        idx = rng.choice(len(by[qt]), alloc[qt], replace=False)
        samp[qt] = sorted(by[qt][i] for i in idx)
    alloc_p = RC.largest_remainder({k: len(v) for k, v in samp.items()}, N_POOL)
    rng3 = np.random.default_rng([SEED_R2, 3])
    pool = {}
    for qt in sorted(samp):
        idx = rng3.choice(len(samp[qt]), alloc_p[qt], replace=False)
        pool[qt] = sorted(samp[qt][i] for i in idx)
    return samp, pool, alloc, alloc_p


# ------------------------------------------------------------------ statistics (used by the rescoring script)
def poisson_boot(d, B, seed):
    """paired Poisson(1) bootstrap of mean(d): the multitq_eval.paired_poisson_bootstrap stream (one weight vector
    per draw, rng = default_rng(seed), w = rng.poisson(1.0, n)); returns the array of bootstrap means."""
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    for i in range(B):
        w = rng.poisson(1.0, size=d.shape[0])
        sw = w.sum()
        out[i] = (w * d).sum() / sw if sw > 0 else 0.0
    return out


def summarise(d, B, seed):
    D = poisson_boot(d, B, seed)
    lo, hi = np.quantile(D, [0.025, 0.975])
    return dict(delta=float(np.mean(d)), ci95=[float(lo), float(hi)], lower_one_sided_95=float(np.quantile(D, 0.05)),
                p_one_sided=float((D <= 0).mean()), p_two=float(min(1.0, 2 * min((D <= 0).mean(), (D >= 0).mean()))),
                B=B, seed=seed, n=int(len(d)))


def holm(ps):
    m = len(ps); order = np.argsort(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj.tolist()


def primary_tests(drop_G, drop_P, B=B_PRIMARY, seed=BOOT_SEED_PRIMARY, alpha=ALPHA):
    """H1-H3: for each k, d_i = drop(G_k)_i - drop(P_k)_i (drop = Hits@1 original - Hits@1 rewrite per question;
    P seed-averaged over its 3 S seeds); one-sided p = share of bootstrap means <= 0; Holm over the three k;
    CONFIRMED iff Holm-adjusted one-sided p < alpha. drop_G / drop_P: {k: per-question arrays (same question order)}."""
    ks = list(drop_G)
    res = {k: summarise(np.asarray(drop_G[k]) - np.asarray(drop_P[k]), B, seed) for k in ks}
    adj = holm([res[k]['p_one_sided'] for k in ks])
    for k, pa in zip(ks, adj):
        res[k]['p_holm_one_sided'] = pa
        res[k]['label'] = 'CONFIRMED' if pa < alpha else 'NOT_CONFIRMED'
    return res


def drop_label(d):
    """R1 per-arm label (B = 2000, seed 0, two-sided 95% CI): BRITTLE iff drop >= 0.02 and CI low > 0;
    ROBUST iff CI high < 0.02; else INCONCLUSIVE"""
    s = summarise(d, 2000, 0)
    return 'BRITTLE' if (s['delta'] >= 0.02 and s['ci95'][0] > 0) else ('ROBUST' if s['ci95'][1] < 0.02 else 'INCONCLUSIVE'), s
