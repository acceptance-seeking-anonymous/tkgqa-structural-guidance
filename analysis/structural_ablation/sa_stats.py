"""sa_stats.py -- shared scoring helpers for the structural-ablation track.
Bootstrap = multitq_eval.paired_poisson_bootstrap (B = 2000, seed 0): the Poisson(1) weight matrix is drawn once per
(n, B, seed) with numpy.random.default_rng(seed).poisson(1.0, size=(B, n)), which is the same stream as the evaluator's
per-replicate loop (checked in sa_score_*.py against the evaluator for one contrast)."""
import json, gzip
import numpy as np

_WC = {}


def weights(n, B=2000, seed=0):
    key = (n, B, seed)
    if key not in _WC:
        rng = np.random.default_rng(seed)
        W = np.empty((B, n), np.int8)
        for i in range(B):
            W[i] = rng.poisson(1.0, size=n)
        _WC[key] = (W, W.sum(1).astype(np.float64))
    return _WC[key]


def _wdot(W, v, chunk=250):
    out = np.empty(W.shape[0], np.float64)
    for i in range(0, W.shape[0], chunk):
        out[i:i + chunk] = W[i:i + chunk].astype(np.float64) @ v
    return out


def boot_mean(h, B=2000, seed=0):
    h = np.asarray(h, np.float64)
    W, sw = weights(len(h), B, seed)
    m = _wdot(W, h) / np.where(sw > 0, sw, 1)
    lo, hi = np.quantile(m, [0.025, 0.975])
    return dict(mean=float(h.mean()), ci_low=float(lo), ci_high=float(hi), n=int(len(h)))


def boot_diff(a, b, B=2000, seed=0):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    d = a - b
    W, sw = weights(len(d), B, seed)
    dl = _wdot(W, d) / np.where(sw > 0, sw, 1)
    dl = np.where(sw > 0, dl, 0.0)
    lo, hi = np.quantile(dl, [0.025, 0.975])
    p2 = min(1.0, 2 * min(float((dl <= 0).mean()), float((dl >= 0).mean())))
    return dict(delta=float(d.mean()), ci_low=float(lo), ci_high=float(hi), p_two_sided=p2,
                p_one_sided_le0=float((dl <= 0).mean()), n=int(len(d)), B=B, seed=seed)


def holm(pvals):
    """Holm step-down adjusted p-values (dict name -> p)."""
    items = sorted(pvals.items(), key=lambda x: x[1])
    m = len(items); adj = {}; run = 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        adj[k] = run
    return adj


def read_ranked(p):
    op = gzip.open if p.endswith('.gz') else open
    d = {}
    with op(p, 'rt') as f:
        for l in f:
            if l.strip():
                x = json.loads(l); d[int(x['quid'])] = x['ranked']
    return d


def label(c, adj_p, pos='HELP', neg='HURT', none='NO_DETECTABLE_DIFFERENCE', thr=0.001):
    if c['ci_low'] > 0 and c['delta'] >= thr and adj_p < 0.05:
        return pos
    if c['ci_high'] < 0 and abs(c['delta']) >= thr and adj_p < 0.05:
        return neg
    return none
