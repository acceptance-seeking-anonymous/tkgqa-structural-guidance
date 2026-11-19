"""cron_rank.py -- S re-ranker (LightGBM lambdarank) training, registered dev grid, seeds, scoring."""
import time, numpy as np, lightgbm as lgb

S_BASE = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1], min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=20, num_threads=10, verbose=-1)
S_GRID = [(nl, lr) for nl in (31, 63, 127) for lr in (0.05, 0.1)]


def train_one(Xtr, ytr, gtr, Xdv, ydv, gdv, num_leaves, lr, seed, rounds=None):
    p = dict(S_BASE, num_leaves=num_leaves, learning_rate=lr, seed=seed, bagging_seed=seed, feature_fraction_seed=seed,
             data_random_seed=seed)
    dtr = lgb.Dataset(Xtr, ytr, group=gtr, free_raw_data=False)
    if rounds is None:
        ddv = lgb.Dataset(Xdv, ydv, group=gdv, reference=dtr)
        m = lgb.train(p, dtr, num_boost_round=2000, valid_sets=[ddv],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    else:
        m = lgb.train(p, dtr, num_boost_round=rounds)
    return m


def rerank(keys, X, model, num_iteration=None):
    """score all pool candidates; ties broken by RULE order (pool order)"""
    if len(keys) == 0:
        return keys, np.zeros(0)
    s = model.predict(X, num_iteration=num_iteration)
    o = np.lexsort((np.arange(len(keys)), -s))
    return [keys[i] for i in o], s
