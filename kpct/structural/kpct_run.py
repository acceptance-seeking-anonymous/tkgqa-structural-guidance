"""kpct_run.py -- structural k% track: retrain every train-fit artefact on one k% subset, OOF stacks, inference,
guidance bundles and prediction files.

usage: python kpct_run.py all <subset>          (stages fitA -> trainS -> fold0/fold1 -> trainS_fold0/1 -> infer)
       python kpct_run.py <stage> <subset> [fold]
Stages that train LightGBM run in their own process (no fork after OpenMP); stages that fork never train LightGBM.
Every read of a train answer is restricted to the subset (AnswerGuard); dev answers are read only for dev_sel2000
selection / reporting; test questions are loaded as {quid, question} only.
"""
import os, sys, json, pickle, time, subprocess, collections, gzip, hashlib
os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import kpct_core as K

NPROC = int(os.environ.get('KPCT_NPROC', '8'))
LGB_THREADS = int(os.environ.get('KPCT_LGB_THREADS', str(NPROC)))
OUT = os.environ.get('KPCT_OUT', os.path.join(HERE, 'out'))
SUBF = os.path.join(HERE, 'subsets', 'multitq_kpct_subsets.json')
E_GRID = (6, 24)
S_GRID = [(nl, lr) for nl in (31, 63, 127) for lr in (0.05, 0.1)]
S_BASE = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1], min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=20, verbose=-1)
TRAIN_CAP = 200


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def jdump(o, p):
    tmp = p + '.tmp'
    json.dump(o, open(tmp, 'w'), indent=1, default=str)
    os.replace(tmp, p)


def pdump(o, p):
    tmp = p + '.tmp'
    pickle.dump(o, open(tmp, 'wb'), protocol=5)
    os.replace(tmp, p)


# ------------------------------------------------------------------ data access with guards
_SUB = None


def manifest():
    global _SUB
    if _SUB is None:
        _SUB = json.load(open(SUBF))
    return _SUB


def subset_quids(name):
    return list(manifest()['subsets'][name]['quids'])


def folds(name):
    o = manifest()['oof_splits'][name]
    return {'fold0': o['fold0']['quids'], 'fold1': o['fold1']['quids'], **{f'inner{k}': v['quids'] for k, v in o['inner'].items()}}


def train_questions(name):
    """train questions of the subset only (all fields); every other train question is dropped immediately."""
    keep = set(subset_quids(name))
    qs = [q for q in K.load_questions('train') if q['quid'] in keep]
    assert len(qs) == len(keep), (len(qs), len(keep))
    assert K.list_sha([q['quid'] for q in qs]) == manifest()['subsets'][name]['sha256']
    return qs


def dev_sel():
    keep = set(manifest()['dev_selection']['quids'])
    qs = [q for q in K.load_questions('dev') if q['quid'] in keep]
    assert len(qs) == 2000 and K.list_sha([q['quid'] for q in qs]) == manifest()['dev_selection']['sha256']
    text = [{'quid': q['quid'], 'question': q['question']} for q in qs]
    gold = {q['quid']: q for q in qs}          # selection / reporting only
    return text, gold


def test_text():
    qs = K.load_questions('test')
    assert len(qs) == 54584
    return [{'quid': q['quid'], 'question': q['question']} for q in qs]   # no answers / metadata kept


# ------------------------------------------------------------------ multiprocessing (fork; globals in _G)
_G = {}


def _pool(fn, items, chunk, ordered=True):
    import multiprocessing as mp
    chunks = [items[i:i + chunk] for i in range(0, len(items), chunk)]
    if NPROC <= 1 or len(chunks) <= 1:
        return [x for c in chunks for x in fn(c)]
    with mp.get_context('fork').Pool(min(NPROC, len(chunks))) as pool:
        res = pool.map(fn, chunks)
    return [x for c in res for x in c]


def w_alias(qs):
    out = []
    for q in qs:
        _, d, n = K.A.alias_evidence(q)
        out.append((q['quid'], d, n))
    return out


def w_silver(qs):
    return [K.silver_rec(q) for q in qs]


def w_link(items):
    return [K.link_rec(_G['LK'], it['quid'], it['question']) for it in items]


def _probs_i(probs, i):
    return {k: v[i] for k, v in probs.items()}


def w_rule(idx):
    out = []
    for i in idx:
        o = K.MC.build(_G['recs'][i], _probs_i(_G['probs'], i), _G['qtext'][i])
        out.append(K.make_ranked(o['names'], o, _G['FB']))
    return out


def w_pool(idx):
    """pool features (+ labels when _G['gold'] is set) for S training / selection."""
    out = []
    for i in idx:
        o = K.MC.build(_G['recs'][i], _probs_i(_G['probs'], i), _G['qtext'][i])
        K.cap_pool(o, _G.get('cap'))
        q = _G['gold'][i] if _G.get('gold') is not None else None
        y = K.labels_for(_G['guard'].check(q), o['names']) if q is not None else None
        out.append(dict(X=o['X'], y=y, names=o['names'], info=o['info'], prior=o['prior']))
    return out


# ------------------------------------------------------------------ stage helpers
def fit_linker(Tqs, ev, guard, nm):
    """sector aliases, block list and v3 silver records fit on the training list Tqs only."""
    for q in Tqs:
        guard.check(q)
    den = collections.Counter(); num = collections.Counter()
    T = {q['quid'] for q in Tqs}
    for quid, d, n in ev:
        if quid in T:
            den.update(d); num.update(n)
    alias = K.alias_rules_from_counts(den, num)
    LK1 = K.make_linker(alias, None); K.A.init(K.kg(), LK1)
    v1 = _pool(w_silver, Tqs, 250)
    block = K.block_list(Tqs, v1)
    LK3 = K.make_linker(alias, block); K.A.init(K.kg(), LK3)
    v3 = _pool(w_silver, Tqs, 250)
    log(nm, 'linker: alias rules', len(alias), 'block', len(block), 'strong silver', sum(r['strong'] for r in v3), '/', len(v3))
    return alias, block, v3, LK3


def link_all(LK, items):
    _G['LK'] = LK
    return _pool(w_link, items, 500)


def rule_dev_hits(recs, probs, qtext, FB, gold_list):
    _G.update(recs=recs, probs=probs, qtext=qtext, FB=FB)
    ranked = _pool(w_rule, list(range(len(recs))), 100)
    return float(np.mean([K.hit1(q, r) for q, r in zip(gold_list, ranked)]))


def pools(recs, probs, qtext, gold=None, cap=None, guard=None):
    _G.update(recs=recs, probs=probs, qtext=qtext, gold=gold, cap=cap, guard=guard)
    out = _pool(w_pool, list(range(len(recs))), 100)
    _G.update(gold=None, cap=None, guard=None)
    return out


def oof_probs(Tqs, Trecs, split_lists, E, guard, nm):
    """2-fold out-of-fold classifier probabilities for Tqs (vocabulary + classifier fit on the other half)."""
    pos = {q['quid']: i for i, q in enumerate(Tqs)}
    probs = {'rel': np.zeros((len(Tqs), K.NR * 2), np.float32), 'qtype': np.zeros((len(Tqs), 6), np.float32),
             'atype': np.zeros((len(Tqs), 2), np.float32), 'level': np.zeros((len(Tqs), 3), np.float32)}
    covered = np.zeros(len(Tqs), bool)
    for a, b in ((0, 1), (1, 0)):
        tr_ids = [pos[x] for x in split_lists[a]]; ho_ids = [pos[x] for x in split_lists[b]]
        assert not set(tr_ids) & set(ho_ids)
        qs_tr = [guard.check(Tqs[i]) for i in tr_ids]
        voc, net = K.fit_parser(qs_tr, [Trecs[i] for i in tr_ids], E)
        pr = K.parser_probs(voc, net, [Trecs[i] for i in ho_ids])
        for k in probs:
            probs[k][ho_ids] = pr[k]
        covered[ho_ids] = True
        log(nm, f'OOF half {a}->{b}: train {len(tr_ids)} vocab {len(voc.itos)} heldout {len(ho_ids)}')
    assert covered.all()
    return probs


def stack_S_data(P):
    keep = [p for p in P if p['y'] is not None and p['y'].any()]
    X = np.concatenate([p['X'] for p in keep]).astype(np.float32)
    y = np.concatenate([p['y'] for p in keep]).astype(np.float32)
    grp = np.array([len(p['y']) for p in keep], np.int32)
    return X, y, grp, len(P) - len(keep)


# ------------------------------------------------------------------ stage fitA: full-subset stack (no LightGBM)
def stage_fitA(nm):
    d = os.path.join(OUT, nm); os.makedirs(d, exist_ok=True)
    t0 = time.time()
    trq = train_questions(nm)
    guard = K.AnswerGuard([q['quid'] for q in trq], nm)
    dtext, dgold = dev_sel(); dgold_list = [dgold[q['quid']] for q in dtext]
    ttext = test_text()
    K.kg()
    # per-question alias evidence with the base linker (question-local; aggregated per training list)
    K.A.init(K.kg(), K.make_linker())
    ev = _pool(w_alias, [guard.check(q) for q in trq], 250)
    pdump(ev, f'{d}/alias_evidence.pkl')
    alias, block, v3, LK3 = fit_linker(trq, ev, guard, nm)
    jdump({'alias_rules': alias, 'ent_block': block, 'train_list': nm, 'n_train': len(trq)}, f'{d}/linker_full.json')
    pdump(v3, f'{d}/parse_train_full.pkl')
    drec = link_all(LK3, dtext); trec = link_all(LK3, ttext)
    pdump(drec, f'{d}/link_dev_sel2000.pkl'); pdump(trec, f'{d}/link_test.pkl')
    log(nm, 'linked dev_sel2000', len(drec), 'test', len(trec), round(time.time() - t0), 's')
    FB = K.build_FB([guard.check(q) for q in trq]); jdump(FB, f'{d}/FB.json')
    # classifier epochs E in {6, 24}: dev_sel2000 RULE_k Hits@1 (tie -> 6)
    sel = {'E_grid': {}}
    dq = [q['question'] for q in dtext]
    for E in E_GRID:
        voc, net = K.fit_parser([guard.check(q) for q in trq], v3, E)
        K.save_parser(f'{d}/parser_full_E{E}.pt', voc, net)
        pr = K.parser_probs(voc, net, drec)
        h = rule_dev_hits(drec, pr, dq, FB, dgold_list)
        sel['E_grid'][E] = {'dev_sel2000_RULE_hits1': h, 'vocab': len(voc.itos)}
        log(nm, f'E={E} vocab {len(voc.itos)} dev_sel2000 RULE Hits@1 {h:.4f}')
        jdump(sel, f'{d}/selection_E.json')
    best = max(E_GRID, key=lambda E: (round(sel['E_grid'][E]['dev_sel2000_RULE_hits1'], 12), -E))
    sel['E_selected'] = best; sel['rule'] = 'max dev_sel2000 RULE_k Hits@1; tie -> 6'
    jdump(sel, f'{d}/selection_E.json')
    voc, net = K.load_parser(f'{d}/parser_full_E{best}.pt')
    # OOF probabilities for the S training rows: outer 2-fold split, linker artefacts = full-subset ones
    F = folds(nm)
    oof = oof_probs(trq, v3, [F['fold0'], F['fold1']], best, guard, nm)
    np.savez_compressed(f'{d}/oof_probs_train.npz', quid=np.array([q['quid'] for q in trq]), **oof)
    tq = [q['question'] for q in trq]
    P = pools(v3, oof, tq, gold=trq, cap=TRAIN_CAP, guard=guard)
    X, y, grp, ndrop = stack_S_data(P)
    np.savez(f'{d}/S_train.npz', X=X, y=y, grp=grp)
    log(nm, 'S train rows', len(X), 'groups', len(grp), 'dropped (no positive)', ndrop)
    pr = K.parser_probs(voc, net, drec)
    PD = pools(drec, pr, dq, gold=dgold_list, cap=None, guard=K.AnswerGuard([q['quid'] for q in dgold_list], 'dev_sel2000'))
    pdump({'P': PD, 'quids': [q['quid'] for q in dtext]}, f'{d}/S_dev.pkl')
    jdump({'stage': 'fitA', 'secs': round(time.time() - t0), 'S_rows': int(len(X)), 'S_groups': int(len(grp)),
           'S_dropped_no_positive': int(ndrop), 'answer_reads_train': guard.n_reads}, f'{d}/fitA_done.json')
    log(nm, 'fitA done', round(time.time() - t0), 's')


# ------------------------------------------------------------------ stage trainS: LightGBM (own process)
def _lgb_dataset(X, y, grp, ref=None):
    import lightgbm as lgb
    return lgb.Dataset(X, y, group=grp, reference=ref, free_raw_data=False)


def dev_hits_from_scores(PD, gold, scores, FB):
    hs1, hs10 = [], []
    for p, q, s in zip(PD, gold, scores):
        names = [p['names'][j] for j in K.s_order(s)] if len(s) else []
        r = K.make_ranked(names, p, FB)
        hs1.append(K.hit1(q, r)); hs10.append(K.hit10(q, r))
    return float(np.mean(hs1)), float(np.mean(hs10))


def stage_trainS(nm):
    import lightgbm as lgb
    d = os.path.join(OUT, nm)
    t0 = time.time()
    tr = np.load(f'{d}/S_train.npz')
    dtrain = _lgb_dataset(tr['X'], tr['y'], tr['grp'])
    dd = pickle.load(open(f'{d}/S_dev.pkl', 'rb')); PD = dd['P']
    _, dgold = dev_sel(); gold = [dgold[q] for q in dd['quids']]
    FB = json.load(open(f'{d}/FB.json'))
    nz = [p for p in PD if len(p['names'])]
    Xd = np.concatenate([p['X'] for p in nz]); yd = np.concatenate([p['y'] for p in nz]).astype(np.float32)
    dvalid = _lgb_dataset(Xd, yd, np.array([len(p['names']) for p in nz]), ref=dtrain)
    sizes = [len(p['names']) for p in PD]; offs = np.cumsum([0] + sizes)
    Xall = np.concatenate([p['X'] for p in PD])
    grid = []
    for nl, lr in S_GRID:
        prm = dict(S_BASE, num_leaves=nl, learning_rate=lr, seed=0, num_threads=LGB_THREADS)
        t1 = time.time()
        bst = lgb.train(prm, dtrain, num_boost_round=2000, valid_sets=[dvalid], callbacks=[lgb.early_stopping(100, verbose=False)])
        bi = int(bst.best_iteration) or int(bst.current_iteration())
        sc = bst.predict(Xall, num_iteration=bi, num_threads=LGB_THREADS)
        h1, h10 = dev_hits_from_scores(PD, gold, [sc[offs[i]:offs[i + 1]] for i in range(len(PD))], FB)
        nd = float(bst.best_score['valid_0']['ndcg@1'])
        grid.append(dict(num_leaves=nl, learning_rate=lr, best_iter=bi, dev_sel2000_ndcg1=nd, dev_sel2000_hits1=h1,
                         dev_sel2000_hits10=h10, secs=round(time.time() - t1)))
        log(nm, 'S grid', grid[-1])
        jdump({'grid': grid}, f'{d}/selection_S.json')
    order = sorted(range(len(grid)), key=lambda i: (-grid[i]['dev_sel2000_hits1'], -grid[i]['dev_sel2000_ndcg1'], i))
    b = grid[order[0]]
    sel = {'grid': grid, 'selected': {k: b[k] for k in ('num_leaves', 'learning_rate', 'best_iter')},
           'rule': 'max dev_sel2000 Hits@1 at LightGBM seed 0 (ties: higher dev_sel2000 ndcg@1, then grid order); seeds 0,1,2 retrained with the seed-0 best iteration'}
    for s in (0, 1, 2):
        prm = dict(S_BASE, num_leaves=b['num_leaves'], learning_rate=b['learning_rate'], seed=s, num_threads=LGB_THREADS)
        bst = lgb.train(prm, dtrain, num_boost_round=b['best_iter'])
        bst.save_model(f'{d}/S_seed{s}.txt')
    jdump(sel, f'{d}/selection_S.json')
    log(nm, 'trainS done', round(time.time() - t0), 's', sel['selected'])


# ------------------------------------------------------------------ fold stacks (train-time OOF bundles)
def stage_fold(nm, f):
    """outer fold f: the whole structural stack retrained on fold 1-f; produces the bundles of fold f."""
    d = os.path.join(OUT, nm); fd = os.path.join(d, f'fold{f}'); os.makedirs(fd, exist_ok=True)
    t0 = time.time()
    F = folds(nm)
    trq = train_questions(nm)
    byq = {q['quid']: q for q in trq}
    Tl = F[f'fold{1 - f}']; Hl = F[f'fold{f}']
    assert not set(Tl) & set(Hl)
    guard = K.AnswerGuard(Tl, f'{nm}/fold{1 - f}')
    Tqs = [byq[x] for x in sorted(Tl)]
    ev = pickle.load(open(f'{d}/alias_evidence.pkl', 'rb'))
    K.kg()
    alias, block, v3T, LK3 = fit_linker(Tqs, ev, guard, f'{nm}/fold{f}')
    E = json.load(open(f'{d}/selection_E.json'))['E_selected']
    voc, net = K.fit_parser([guard.check(q) for q in Tqs], v3T, E)
    K.save_parser(f'{fd}/parser.pt', voc, net)
    Hitems = [{'quid': x, 'question': byq[x]['question']} for x in sorted(Hl)]   # text only
    Hrec = link_all(LK3, Hitems)
    jdump({'alias_rules': alias, 'ent_block': block, 'train_list': f'{nm}/fold{1 - f}', 'n_train': len(Tqs)}, f'{fd}/linker.json')
    pdump(Hrec, f'{fd}/link_heldout.pkl')
    # inner OOF probabilities for the stack's S training rows
    inner = [F[f'inner{1 - f}0'], F[f'inner{1 - f}1']]
    assert sorted(inner[0] + inner[1]) == sorted(Tl)
    oofT = oof_probs(Tqs, v3T, inner, E, guard, f'{nm}/fold{f}')
    P = pools(v3T, oofT, [q['question'] for q in Tqs], gold=Tqs, cap=TRAIN_CAP, guard=guard)
    X, y, grp, ndrop = stack_S_data(P)
    np.savez(f'{fd}/S_train.npz', X=X, y=y, grp=grp)
    jdump({'stage': f'fold{f}', 'train_list': f'fold{1 - f}', 'heldout_list': f'fold{f}', 'n_train': len(Tqs), 'n_heldout': len(Hl),
           'E': E, 'S_rows': int(len(X)), 'S_groups': int(len(grp)), 'S_dropped_no_positive': int(ndrop),
           'secs': round(time.time() - t0), 'answer_reads_train': guard.n_reads,
           'train_quids_sha256': K.list_sha(Tl), 'heldout_quids_sha256': K.list_sha(Hl)}, f'{fd}/fold_done.json')
    log(nm, f'fold{f} done', round(time.time() - t0), 's')


def stage_trainS_fold(nm, f):
    import lightgbm as lgb
    d = os.path.join(OUT, nm); fd = os.path.join(d, f'fold{f}')
    sel = json.load(open(f'{d}/selection_S.json'))['selected']
    tr = np.load(f'{fd}/S_train.npz')
    dtrain = _lgb_dataset(tr['X'], tr['y'], tr['grp'])
    prm = dict(S_BASE, num_leaves=sel['num_leaves'], learning_rate=sel['learning_rate'], seed=0, num_threads=LGB_THREADS)
    bst = lgb.train(prm, dtrain, num_boost_round=sel['best_iter'])
    bst.save_model(f'{fd}/S_seed0.txt')
    log(nm, f'fold{f} S trained', sel)


# ------------------------------------------------------------------ stage infer (forks; never trains LightGBM)
_BST = {}


def _boosters(paths):
    import lightgbm as lgb
    key = tuple(paths)
    if key not in _BST:
        _BST[key] = [lgb.Booster(model_file=p) for p in paths]
    return _BST[key]


def top1_parse(pj):
    """the parser's top-1 hypothesis: first alternative of every mention, top relation-direction only."""
    t = json.loads(json.dumps(pj))
    for m in t['mentions']:
        m['alternatives'] = m['alternatives'][:1]
    t['relations'] = t['relations'][:1]
    return t


def w_infer(items):
    """items: (kind, i). kind in {'dev', 'test', 'train'}; returns compact per-question results."""
    out = []
    by_kind = collections.defaultdict(list)
    for kind, i in items:
        by_kind[kind].append(i)
    for kind, idx in by_kind.items():
        cfg = _G['cfg'][kind]
        bst = _boosters(cfg['boosters'])
        builds = []
        for i in idx:
            rec = cfg['recs'][i]
            info = K.MC.parse_info(rec, _probs_i(cfg['probs'], i), cfg['qtext'][i])
            o = K.build_info(rec, info)
            builds.append(o)
        sc = K.predict_chunk(bst, builds)
        for n, (i, o) in enumerate(zip(idx, builds)):
            rec = cfg['recs'][i]
            res = {'kind': kind, 'i': i, 'quid': rec['quid']}
            res['rule'] = K.make_ranked(o['names'], o, cfg['FB'])
            res['S'] = [K.make_ranked([o['names'][j] for j in K.s_order(sc[b][n])], o, cfg['FB']) for b in range(len(bst))]
            if len(bst) == 3:
                mean = (sc[0][n] + sc[1][n] + sc[2][n]) / 3.0
                res['S_mean'] = K.make_ranked([o['names'][j] for j in K.s_order(mean)], o, cfg['FB'])
            pj = K.parse_json(rec, o['info'])
            core = K.bundle_core(o, rec, sc[0][n])
            res['bundle'] = dict(parse=pj, **core)
            # recall@20 of S seed 0 (dev only, computed by the parent from names)
            if kind.startswith('train'):
                # in-sample (non-OOF) parse of the full-subset parser, executed with operators + RULE order
                ins = cfg['insample']
                rec2 = ins['recs'][i]
                info2 = K.MC.parse_info(rec2, _probs_i(ins['probs'], i), cfg['qtext'][i])
                pj2 = K.parse_json(rec2, info2)
                o_full = K.build_info(rec2, info2)
                t1 = top1_parse(pj2)
                r_t, i_t = K.rec_info_from_parse(t1)
                o_top = K.build_info(r_t, i_t)
                res['insample'] = dict(parse=pj2, top1_parse=t1,
                                       rule_top10_full=K.make_ranked(o_full['names'], o_full, cfg['FB'])[:10],
                                       rule_top10_top1=K.make_ranked(o_top['names'], o_top, cfg['FB'])[:10],
                                       pool_size_top1=len(o_top['names']))
            out.append(res)
    return out


def _gold_rank(q, names):
    for j in range(len(names)):
        if K.ME.hit_at_k(q, names[j:j + 1], 1):
            return j + 1
    return None


def bootstrap_ci(h, B=1000, seed=0):
    h = np.asarray(h, float)
    rng = np.random.default_rng(seed)
    w = rng.poisson(1.0, size=(B, len(h)))
    m = (w * h).sum(1) / np.maximum(w.sum(1), 1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def template_sets(nm):
    """frozen-linker template keys (kpct_keys.py definition) -> set of train templates of the subset."""
    tk = pickle.load(open(os.path.join(OUT, 'template_keys.pkl'), 'rb'))
    keep = set(subset_quids(nm)) if nm != 'k100' else None
    seen = {t for quid, t in tk['train'].items() if keep is None or quid in keep}
    return seen, tk


def stage_infer(nm):
    d = os.path.join(OUT, nm)
    t0 = time.time()
    K.kg()
    trq = train_questions(nm); byq = {q['quid']: q for q in trq}
    guard = K.AnswerGuard(list(byq), nm)
    dtext, dgold = dev_sel(); ttext = test_text()
    FB = json.load(open(f'{d}/FB.json'))
    E = json.load(open(f'{d}/selection_E.json'))['E_selected']
    voc, net = K.load_parser(f'{d}/parser_full_E{E}.pt')
    drec = pickle.load(open(f'{d}/link_dev_sel2000.pkl', 'rb')); trec = pickle.load(open(f'{d}/link_test.pkl', 'rb'))
    v3 = pickle.load(open(f'{d}/parse_train_full.pkl', 'rb'))
    full_boosters = [f'{d}/S_seed{s}.txt' for s in range(3)]
    cfg = {
        'dev': dict(recs=drec, probs=K.parser_probs(voc, net, drec), qtext=[q['question'] for q in dtext], FB=FB, boosters=full_boosters),
        'test': dict(recs=trec, probs=K.parser_probs(voc, net, trec), qtext=[q['question'] for q in ttext], FB=FB, boosters=full_boosters),
    }
    # train rows: fold stacks (OOF) + in-sample full-subset parse
    F = folds(nm)
    tr_recs, tr_probs, tr_qtext, tr_meta, ins_recs = [], [], [], [], []
    v3_by = {r['quid']: r for r in v3}
    ins_probs_all = K.parser_probs(voc, net, v3)
    ins_idx = {r['quid']: i for i, r in enumerate(v3)}
    fold_boosters = {}
    for f in (0, 1):
        fd = os.path.join(d, f'fold{f}')
        fv, fn_ = K.load_parser(f'{fd}/parser.pt')
        Hrec = pickle.load(open(f'{fd}/link_heldout.pkl', 'rb'))
        pr = K.parser_probs(fv, fn_, Hrec)
        Tl = set(F[f'fold{1 - f}'])
        for j, r in enumerate(Hrec):
            assert r['quid'] not in Tl                       # STOP_OOF_LEAK: never in the producing stack's lists
            tr_recs.append(r); tr_probs.append({k: v[j] for k, v in pr.items()}); tr_qtext.append(byq[r['quid']]['question'])
            tr_meta.append(dict(fold=f, stack=f'fold{1 - f}')); ins_recs.append(v3_by[r['quid']])
        fold_boosters[f] = [f'{fd}/S_seed0.txt']
    stack = lambda L_: {k: np.stack([p[k] for p in L_]) for k in L_[0]}
    ins_probs = {k: v[[ins_idx[r['quid']] for r in ins_recs]] for k, v in ins_probs_all.items()}
    for f in (0, 1):
        ids = [i for i, m in enumerate(tr_meta) if m['fold'] == f]
        cfg[f'train{f}'] = dict(recs=[tr_recs[i] for i in ids], probs=stack([tr_probs[i] for i in ids]),
                                qtext=[tr_qtext[i] for i in ids], FB=FB, boosters=fold_boosters[f],
                                insample=dict(recs=[ins_recs[i] for i in ids], probs={k: v[ids] for k, v in ins_probs.items()}),
                                meta=[tr_meta[i] for i in ids])
    _G['cfg'] = cfg
    tseen, tk = template_sets(nm)
    bd = os.path.join(d, 'bundles'); pd_ = os.path.join(d, 'preds'); os.makedirs(bd, exist_ok=True); os.makedirs(pd_, exist_ok=True)
    import multiprocessing as mp
    results_dev = {}
    split_name = {'dev': 'dev_sel2000', 'test': 'test'}
    for kind in ('dev', 'test', 'train0', 'train1'):
        n = len(cfg[kind]['recs'])
        items = [(kind, i) for i in range(n)]
        chunks = [items[a:a + 50] for a in range(0, n, 50)]
        is_train = kind.startswith('train')
        split = 'train' if is_train else split_name[kind]
        bf = open(f'{bd}/{split}.jsonl.tmp', 'a' if (is_train and kind == 'train1') else 'w')
        pf = None if is_train else {a: open(f'{pd_}/{split}_{a}.jsonl', 'w') for a in ('RULE', 'S_seed0', 'S_seed1', 'S_seed2', 'S_mean')}
        cnt = 0
        with mp.get_context('fork').Pool(NPROC) as pool:
            for res in pool.imap(w_infer, chunks):
                for r in res:
                    quid = r['quid']
                    if is_train:
                        q = byq[quid]; guard.check(q)
                        m = cfg[kind]['meta'][r['i']]
                        ins = r['insample']
                        t_hit = bool(K.hit1(q, ins['rule_top10_top1'])) and ins['pool_size_top1'] > 0
                        f_hit = bool(K.hit1(q, ins['rule_top10_full']))
                        row = dict(quid=quid, question=q['question'], split='train', k_subset=nm,
                                   gold=q['answers'],
                                   gold_meta={'answer_type': q['answer_type'], 'time_level': q['time_level'], 'qtype': q['qtype'], 'qlabel': q['qlabel']},
                                   oof={'outer_fold': m['fold'], 'stack_trained_on': f"{nm}/{m['stack']}",
                                        'note': 'linker aliases + block list, silver labels, vocabulary, classifier and S of this row were fit on the other outer fold only'},
                                   **r['bundle'],
                                   in_sample_parse=dict(parse=ins['parse'], top1_parse=ins['top1_parse'],
                                                        rule_top10_top1=ins['rule_top10_top1'], rule_top10_full=ins['rule_top10_full'],
                                                        executes_to_gold=t_hit, executes_to_gold_full_parse=f_hit,
                                                        note='full-subset (non-OOF) parser; executes_to_gold = top-1 hypothesis executed with operators + RULE order returns a gold answer at rank 1 (official matching)'))
                    else:
                        row = dict(quid=quid, question=cfg[kind]['qtext'][r['i']], split=split, k_subset=nm,
                                   template_unseen=bool(tk[split].get(quid) not in tseen), **r['bundle'])
                        for a, lst in (('RULE', r['rule']), ('S_seed0', r['S'][0]), ('S_seed1', r['S'][1]), ('S_seed2', r['S'][2]), ('S_mean', r['S_mean'])):
                            pf[a].write(json.dumps({'quid': quid, 'ranked': lst}) + '\n')
                        if kind == 'dev':
                            results_dev[quid] = dict(rule=r['rule'], S=r['S'], S_mean=r['S_mean'],
                                                     cand20=[c['answer'] for c in r['bundle']['candidates']],
                                                     unseen=row['template_unseen'])
                    bf.write(json.dumps(row, ensure_ascii=False) + '\n'); cnt += 1
        bf.close()
        if pf:
            for v in pf.values():
                v.close()
        if kind in ('dev', 'test'):
            os.replace(f'{bd}/{split}.jsonl.tmp', f'{bd}/{split}.jsonl')
        log(nm, 'infer', kind, cnt, round(time.time() - t0), 's')
    os.replace(f'{bd}/train.jsonl.tmp', f'{bd}/train.jsonl')
    # dev_sel2000 summary (selection / reporting only)
    rows = []
    for q in dtext:
        g = dgold[q['quid']]; r = results_dev[q['quid']]
        rows.append(dict(quid=q['quid'], qtype=g['qtype'], unseen=r['unseen'],
                         RULE1=K.hit1(g, r['rule']), RULE10=K.hit10(g, r['rule']),
                         S1=[K.hit1(g, x) for x in r['S']], S10=[K.hit10(g, x) for x in r['S']],
                         Smean1=K.hit1(g, r['S_mean']), rec20=int(any(K.hit1(g, [c]) for c in r['cand20']))))
    pdump(rows, f'{d}/dev_rows.pkl')
    summ = summarize_dev(rows)
    # silver (answer-consistent) share of the in-sample parses
    n = ok = okf = 0
    for line in open(f'{bd}/train.jsonl'):
        x = json.loads(line); n += 1; ok += x['in_sample_parse']['executes_to_gold']; okf += x['in_sample_parse']['executes_to_gold_full_parse']
    summ['train_in_sample'] = {'n': n, 'executes_to_gold_top1_share': ok / n, 'executes_to_gold_full_parse_share': okf / n}
    jdump(summ, f'{d}/dev_summary.json')
    log(nm, 'infer done', round(time.time() - t0), 's', json.dumps({k: summ[k] for k in ('RULE_hits1', 'S_hits1_mean')}))


def summarize_dev(rows):
    R1 = [r['RULE1'] for r in rows]; R10 = [r['RULE10'] for r in rows]
    S1 = np.array([r['S1'] for r in rows], float); S10 = np.array([r['S10'] for r in rows], float)
    s1m = S1.mean(1); s10m = S10.mean(1)
    rec = np.array([r['rec20'] for r in rows], float)
    out = dict(n=len(rows), RULE_hits1=float(np.mean(R1)), RULE_hits1_ci95=bootstrap_ci(R1), RULE_hits10=float(np.mean(R10)),
               S_hits1_per_seed=[float(x) for x in S1.mean(0)], S_hits1_mean=float(s1m.mean()), S_hits1_mean_ci95=bootstrap_ci(s1m),
               S_hits10_per_seed=[float(x) for x in S10.mean(0)], S_hits10_mean=float(s10m.mean()),
               S_seedmean_score_hits1=float(np.mean([r['Smean1'] for r in rows])),
               S_seed0_recall20=float(rec.mean()), S_seed0_headroom20=float(rec.mean() - S1[:, 0].mean()),
               template_unseen_share=float(np.mean([r['unseen'] for r in rows])))
    for flag, nmf in ((True, 'unseen'), (False, 'seen')):
        idx = [i for i, r in enumerate(rows) if r['unseen'] == flag]
        out[f'{nmf}_n'] = len(idx)
        if idx:
            out[f'{nmf}_RULE_hits1'] = float(np.mean([R1[i] for i in idx])); out[f'{nmf}_S_hits1_mean'] = float(s1m[idx].mean())
    by = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by[r['qtype']].append(i)
    out['by_qtype'] = {k: dict(n=len(v), RULE_hits1=float(np.mean([R1[i] for i in v])), S_hits1_mean=float(s1m[v].mean())) for k, v in sorted(by.items())}
    return out


# ------------------------------------------------------------------ driver
STAGES = ['fitA', 'trainS', 'fold0', 'fold1', 'trainS_fold0', 'trainS_fold1', 'infer']


def run_stage(stage, nm):
    if stage == 'fitA':
        stage_fitA(nm)
    elif stage == 'trainS':
        stage_trainS(nm)
    elif stage in ('fold0', 'fold1'):
        stage_fold(nm, int(stage[-1]))
    elif stage in ('trainS_fold0', 'trainS_fold1'):
        stage_trainS_fold(nm, int(stage[-1]))
    elif stage == 'infer':
        stage_infer(nm)
    else:
        raise SystemExit(f'unknown stage {stage}')


if __name__ == '__main__':
    stage, nm = sys.argv[1], sys.argv[2]
    if stage == 'all':
        start = sys.argv[3] if len(sys.argv) > 3 else 'fitA'
        for st in STAGES[STAGES.index(start):]:
            log('=== stage', st, nm)
            r = subprocess.run([sys.executable, os.path.abspath(__file__), st, nm])
            if r.returncode != 0:
                raise SystemExit(f'stage {st} failed ({r.returncode})')
    else:
        run_stage(stage, nm)
