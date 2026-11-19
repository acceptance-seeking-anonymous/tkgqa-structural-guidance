"""a2_tables.py -- coverage-conditioned Hits@1 tables (A2 of protocol/multitq_output_analyses_plan.json)."""
import numpy as np
import pandas as pd

B, SEED = 2000, 0


def boot_means(X, B=B, seed=SEED):
    """X: n x m array of per-question values; returns (means m, lo m, hi m) with Poisson(1) weights shared across the
    m columns (paired), percentile 95% CI."""
    X = np.asarray(X, float)
    n = X.shape[0]
    if n == 0:
        nan = np.full(X.shape[1], np.nan)
        return nan, nan, nan
    rng = np.random.default_rng(seed)
    out = np.empty((B, X.shape[1]))
    for i in range(B):
        w = rng.poisson(1.0, size=n)
        sw = w.sum()
        out[i] = (w @ X) / sw if sw > 0 else 0.0
    lo, hi = np.quantile(out, [0.025, 0.975], axis=0)
    return X.mean(0), lo, hi


def cell_rows(df, mask, cols, label, extra=None, contrasts=()):
    """one row per arm column (+ contrasts) for the questions in mask"""
    sub = df.loc[mask]
    X = [sub[c].values for c in cols]
    names = list(cols)
    for a, b in contrasts:
        X.append(sub[a].values - sub[b].values); names.append(f'{a} - {b}')
    X = np.column_stack(X) if len(sub) else np.zeros((0, len(names)))
    m, lo, hi = boot_means(X)
    rows = []
    for j, nm in enumerate(names):
        r = dict(cell=label, n=int(mask.sum()), share_of_test=float(mask.mean()), quantity=nm, value=float(m[j]) if len(sub) else np.nan,
                 lo=float(lo[j]), hi=float(hi[j]), successes=(float(sub[nm].sum()) if nm in sub else np.nan))
        if extra:
            r.update(extra)
        rows.append(r)
    return rows
