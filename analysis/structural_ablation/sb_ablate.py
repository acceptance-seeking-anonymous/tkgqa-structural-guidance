#!/usr/bin/env python3
"""sb_ablate.py -- structural-ablation track, part (b): Ranker feature-group ablation with the frozen S recipe.

Runs from the structural tree that holds kpct_run.py / kpct_core.py (unchanged). Requires a finished stage fitA in
$KPCT_OUT/<k>/ (run through kpct_retrain_fixed_vocab.py with a fixed PYTHONHASHSEED and the preserved vocab orders).

  python sb_ablate.py train <k> [arm ...]      LightGBM, own process (no fork): for every arm, the frozen stage_trainS
                                              recipe on the kept feature columns; boosters -> $KPCT_OUT/<k>/ablate/<arm>/
  python sb_ablate.py infer <k> <outdir> [--extra name=dir ...]
                                              fork pool, never trains: frozen stage_infer path (parse_info + build_info)
                                              for dev_sel2000 and test; every arm's 3 seeds score the kept columns;
                                              writes pred_<arm>_<k>_<split>_seed{s}.jsonl and RULE_<k>_<split>.jsonl
Arms: full, minus_<G> for the 6 registered groups, minus_C_RULEKEY, minus_C_TEMPORAL (rule multitq_structural_ablation).
Test answers are never read (test_text() keeps quid + question only); dev_sel2000 gold is read for selection only.
"""
import os, sys, json, pickle, time, hashlib
os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import kpct_run as KR
import kpct_core as K

GROUPS = {
    'G_OPSAT': ['sat_top', 'sat_any', 'sat_anchor_last', 'anchor_found', 'q_ncand_sat'],
    'G_TPOS': ['pos', 'pos_rel', 'gap', 'dist', 'pos_anchor_last', 'first_day', 'last_day'],
    'G_RELP': ['best_rank', 'best_relp', 'sum_relp_sat', 'relp_conn', 'rel1_p', 'rel_margin', 'dir_top', 'qt_p', 'atype_p', 'level_p'],
    'G_LINK': ['alt_rank', 'alt_eff_ratio', 'anchor_alt', 'role_swapped', 'is_qent', 'n_ment'],
    'G_PROV': ['n_sat', 'n_conn', 'n_pair_all'],
    'G_PRIOR': ['is_time', 'log_deg', 'log_prior'],
}
COMPOSITES = {
    'C_RULEKEY': ['sat_any', 'best_rank', 'alt_rank', 'role_swapped', 'pos', 'pos_rel', 'n_conn'],
    'C_TEMPORAL': GROUPS['G_OPSAT'] + GROUPS['G_TPOS'],
}
FEATS = K.MC.FEATS
assert sorted(f for g in GROUPS.values() for f in g) == sorted(FEATS) and len(FEATS) == 34


def keep_cols(arm):
    if arm == 'full':
        return list(range(len(FEATS)))
    g = arm[len('minus_'):]
    drop = set(GROUPS.get(g) or COMPOSITES[g])
    return [i for i, f in enumerate(FEATS) if f not in drop]


ARMS = ['full'] + [f'minus_{g}' for g in GROUPS] + [f'minus_{c}' for c in COMPOSITES]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ train (= kpct_run.stage_trainS on kept columns)
def train(nm, arms):
    import lightgbm as lgb
    d = os.path.join(KR.OUT, nm)
    tr = np.load(f'{d}/S_train.npz')
    Xtr, ytr, gtr = tr['X'], tr['y'], tr['grp']
    dd = pickle.load(open(f'{d}/S_dev.pkl', 'rb')); PD = dd['P']
    _, dgold = KR.dev_sel(); gold = [dgold[q] for q in dd['quids']]
    FB = json.load(open(f'{d}/FB.json'))
    nz = [p for p in PD if len(p['names'])]
    Xd_full = np.concatenate([p['X'] for p in nz]); yd = np.concatenate([p['y'] for p in nz]).astype(np.float32)
    gd = np.array([len(p['names']) for p in nz])
    sizes = [len(p['names']) for p in PD]; offs = np.cumsum([0] + sizes)
    Xall_full = np.concatenate([p['X'] for p in PD])
    for arm in arms:
        t0 = time.time()
        kc = keep_cols(arm)
        ad = os.path.join(d, 'ablate', arm); os.makedirs(ad, exist_ok=True)
        X = Xtr if arm == 'full' else np.ascontiguousarray(Xtr[:, kc])
        dtrain = KR._lgb_dataset(X, ytr, gtr)
        Xd = Xd_full if arm == 'full' else np.ascontiguousarray(Xd_full[:, kc])
        dvalid = KR._lgb_dataset(Xd, yd, gd, ref=dtrain)
        Xall = Xall_full if arm == 'full' else np.ascontiguousarray(Xall_full[:, kc])
        grid = []
        for nl, lr in KR.S_GRID:
            prm = dict(KR.S_BASE, num_leaves=nl, learning_rate=lr, seed=0, num_threads=KR.LGB_THREADS)
            t1 = time.time()
            bst = lgb.train(prm, dtrain, num_boost_round=2000, valid_sets=[dvalid], callbacks=[lgb.early_stopping(100, verbose=False)])
            bi = int(bst.best_iteration) or int(bst.current_iteration())
            sc = bst.predict(Xall, num_iteration=bi, num_threads=KR.LGB_THREADS)
            h1, h10 = KR.dev_hits_from_scores(PD, gold, [sc[offs[i]:offs[i + 1]] for i in range(len(PD))], FB)
            nd = float(bst.best_score['valid_0']['ndcg@1'])
            grid.append(dict(num_leaves=nl, learning_rate=lr, best_iter=bi, dev_sel2000_ndcg1=nd, dev_sel2000_hits1=h1,
                             dev_sel2000_hits10=h10, secs=round(time.time() - t1)))
        order = sorted(range(len(grid)), key=lambda i: (-grid[i]['dev_sel2000_hits1'], -grid[i]['dev_sel2000_ndcg1'], i))
        b = grid[order[0]]
        sel = {'arm': arm, 'kept_features': [FEATS[i] for i in kc], 'n_features': len(kc), 'grid': grid,
               'selected': {k: b[k] for k in ('num_leaves', 'learning_rate', 'best_iter')},
               'rule': 'frozen kpct_run.stage_trainS: max dev_sel2000 Hits@1 at LightGBM seed 0 (ties: higher ndcg@1, then grid order); seeds 0,1,2 retrained with the seed-0 best iteration',
               'S_train_rows': int(len(Xtr)), 'S_train_groups': int(len(gtr)), 'hash_seed': os.environ.get('PYTHONHASHSEED'),
               'S_train_sha256': sha256_file(f'{d}/S_train.npz')}
        sel['seed_dev_sel2000_hits1'] = {}
        for s in (0, 1, 2):
            prm = dict(KR.S_BASE, num_leaves=b['num_leaves'], learning_rate=b['learning_rate'], seed=s, num_threads=KR.LGB_THREADS)
            bst = lgb.train(prm, dtrain, num_boost_round=b['best_iter'])
            p = f'{ad}/S_seed{s}.txt'
            bst.save_model(p)
            sc = bst.predict(Xall, num_threads=KR.LGB_THREADS)
            h1, h10 = KR.dev_hits_from_scores(PD, gold, [sc[offs[i]:offs[i + 1]] for i in range(len(PD))], FB)
            sel['seed_dev_sel2000_hits1'][f'seed{s}'] = h1
            sel[f'S_seed{s}_sha256'] = sha256_file(p)
        sel['secs'] = round(time.time() - t0)
        KR.jdump(sel, f'{ad}/selection.json')
        KR.log(nm, arm, sel['selected'], sel['seed_dev_sel2000_hits1'], sel['secs'], 's')


# ------------------------------------------------------------------ infer (= kpct_run.stage_infer path, forks, no training)
_B = {}


def _boosters(paths):
    import lightgbm as lgb
    key = tuple(paths)
    if key not in _B:
        _B[key] = [lgb.Booster(model_file=p) for p in paths]
    return _B[key]


def w_infer(idx):
    cfg = KR._G['cfg']
    out = []
    builds = []
    for i in idx:
        rec = cfg['recs'][i]
        info = K.MC.parse_info(rec, KR._probs_i(cfg['probs'], i), cfg['qtext'][i])
        builds.append(K.build_info(rec, info))
    sizes = [len(o['names']) for o in builds]; offs = np.cumsum([0] + sizes)
    Xall = np.concatenate([o['X'] for o in builds if len(o['names'])]) if sum(sizes) else np.zeros((0, len(FEATS)), np.float32)
    scores = {}
    for arm, (paths, kc) in cfg['arms'].items():
        bst = _boosters(paths)
        Xa = Xall if len(kc) == len(FEATS) else np.ascontiguousarray(Xall[:, kc])
        scores[arm] = [b.predict(Xa, num_threads=1) if len(Xa) else np.zeros(0) for b in bst]
    for n, (i, o) in enumerate(zip(idx, builds)):
        res = {'quid': cfg['recs'][i]['quid'], 'rule': K.make_ranked(o['names'], o, cfg['FB'])}
        for arm in cfg['arms']:
            res[arm] = [K.make_ranked([o['names'][j] for j in K.s_order(sc[offs[n]:offs[n + 1]])], o, cfg['FB']) for sc in scores[arm]]
        out.append(res)
    return out


def infer(nm, outdir, extras):
    import torch
    d = os.path.join(KR.OUT, nm)
    t0 = time.time()
    K.kg()
    dtext, _ = KR.dev_sel(); ttext = KR.test_text()
    FB = json.load(open(f'{d}/FB.json'))
    E = json.load(open(f'{d}/selection_E.json'))['E_selected']
    voc, net = K.load_parser(f'{d}/parser_full_E{E}.pt')
    drec = pickle.load(open(f'{d}/link_dev_sel2000.pkl', 'rb')); trec = pickle.load(open(f'{d}/link_test.pkl', 'rb'))
    arms = {}
    for arm in ARMS:
        ad = os.path.join(d, 'ablate', arm)
        if os.path.exists(f'{ad}/selection.json'):
            arms[arm] = ([f'{ad}/S_seed{s}.txt' for s in range(3)], keep_cols(arm))
    for name, root in extras:          # e.g. full_hs7=<other KPCT_OUT>
        ad = os.path.join(root, nm, 'ablate', 'full')
        arms[name] = ([f'{ad}/S_seed{s}.txt' for s in range(3)], keep_cols('full'))
    os.makedirs(outdir, exist_ok=True)
    stats = dict(k=nm, E=E, arms={a: dict(boosters=[os.path.relpath(p, HERE) for p in v[0]], booster_sha256=[sha256_file(p) for p in v[0]],
                                          n_features=len(v[1])) for a, v in arms.items()}, files={})
    for split, recs, text in (('dev_sel2000', drec, dtext), ('test', trec, ttext)):
        torch.set_num_threads(1)
        probs = K.parser_probs(voc, net, recs)
        KR._G['cfg'] = dict(recs=recs, probs=probs, qtext=[q['question'] for q in text], FB=FB, arms=arms)
        res = KR._pool(w_infer, list(range(len(recs))), 100)
        assert [r['quid'] for r in res] == [q['quid'] for q in text]
        files = {'RULE': lambda r: r['rule']}
        p = os.path.join(outdir, f'RULE_{nm}_{split}.jsonl')
        with open(p, 'w') as f:
            for r in res:
                f.write(json.dumps({'quid': r['quid'], 'ranked': r['rule']}) + '\n')
        stats['files'][os.path.basename(p)] = sha256_file(p)
        for arm in arms:
            for s in range(3):
                p = os.path.join(outdir, f'pred_SA_{arm}_{nm}_{split}_seed{s}.jsonl')
                with open(p, 'w') as f:
                    for r in res:
                        f.write(json.dumps({'quid': r['quid'], 'ranked': r[arm][s]}) + '\n')
                stats['files'][os.path.basename(p)] = sha256_file(p)
        KR.log(nm, 'infer', split, len(res), round(time.time() - t0), 's')
    stats['secs'] = round(time.time() - t0)
    json.dump(stats, open(os.path.join(outdir, f'infer_stats_{nm}.json'), 'w'), indent=1)


if __name__ == '__main__':
    cmd, nm = sys.argv[1], sys.argv[2]
    if cmd == 'train':
        train(nm, sys.argv[3:] or ARMS)
    elif cmd == 'infer':
        outdir = sys.argv[3]
        extras = [x.split('=', 1) for x in sys.argv[4:] if '=' in x]
        infer(nm, outdir, extras)
    else:
        raise SystemExit(__doc__)
