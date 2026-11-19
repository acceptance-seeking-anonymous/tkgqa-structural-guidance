"""run_cron_structural.py -- transcription of the executed steps of the CronQuestions structural comparison.
Executed on CPU (10 cores, env 'multitq': python 3.12, lightgbm 4.6.0, torch 2.14.1 CPU).
The frozen artefacts (sha256 in cronquestions_freeze.json) are authoritative; this file documents the code that
produced them (multithreaded LightGBM/torch may give bit-level differences on re-run).

Order:
  1. operator derivation on train with oracle roles (oracle_eval)                      -> addendum table
  2. parser: templates (train), surfaces (KG), parse train, alt scorer (2-fold OOF + full),
     classifier (2-fold OOF + full)                                                     -> models/  (frozen 08:45:07Z)
  3. S subsample (100,000 qids, type-stratified, RandomState(0))                        -> models/s_train100k_qids.json
  4. addendum registered (08:48:16Z) before any dev metric
  5. dev parse (OWN: paraphrases[0]; GOLD-ANN: question + entities + times), pools, RULE/PRIOR, dev metrics
  6. S train pools (OOF parser), registered grid on dev v2 (seed 0), seeds 0/1/2      -> models/S_{OWN,GOLD}_seed*.txt
  7. LOOKUP_TRAIN / DEV_LOOKUP, dev tables, failure stages, leakage self-check
  8. test loaded through the field filter; all test prediction files written once; exports; freeze
"""
import os, sys, json, pickle, math, time, copy, collections
import numpy as np, torch, lightgbm as lgb
import cron_kg, cron_text, cron_parse, cron_clf as CC, cron_cand as CD, cron_rank as RK, cron_pipeline as CP

D = os.environ.get('CRON_DATA', '../cronquestions_data')
OUT = 'out'


def oracle_eval(kg, qs):
    """gold roles + gold relation/type/answer type/adj/ba -> cron_cand.run_op; exact set / Hits@1 / recall"""
    res = collections.defaultdict(collections.Counter)
    for q in qs:
        a = q['annotation']; rel = next(iter(q['relations'])); r = kg.r2i[rel]
        ents = {k: kg.e2i[a[k]] for k in ('head', 'tail', 'event_head') if k in a}
        if q['type'] == 'time_join' and 'head' in a:
            t2 = set(q['entities']) - {a.get('head'), a.get('tail')}
            if len(t2) == 1:
                ents['tail2'] = kg.e2i[t2.pop()]
        t = int(a['time']) if 'time' in a else None
        out = CD.run_op(q['type'], q['answer_type'], a.get('adj'), a.get('type'), rel, r, ents, t)
        A = q['answers']; key = (q['type'], q['answer_type'], rel)
        sat = [k for k, v in out[0].items() if v['sat']] if out else []
        order = sorted(sat, key=lambda k: (-out[0][k]['n_support'], str(k[1])))
        items = [CD.key_out(k) for k in order]
        res[key]['n'] += 1; res[key]['exact'] += set(items) == A; res[key]['hit1'] += bool(items) and items[0] in A
        res[key]['recall'] += len(set(items) & A) > 0
    return res


def train_parser(kg, tr):
    """step 2: segmenter + alternative scorer + classifier, all on v2 train"""
    SF = cron_text.Surfaces(kg); TP = cron_text.Templates(tr); PA = cron_parse.Parser(kg, SF, TP)
    P = []
    for q in tr:
        text = cron_text.ws(q['paraphrases'][0]); hs = PA.hypotheses(text) or [PA.fallback(text)]
        P.append(dict(text=text, hyps=hs))
    fold = np.random.RandomState(0).randint(0, 2, len(tr))
    # alternative scorer rows (label = alternative in gold 'entities'); ambiguous slots only
    Xs, ys, qi, na = [], [], [], []
    for i, p in enumerate(P):
        h = p['hyps'][0]
        if h['tid'] < 0:
            continue
        hh = copy.deepcopy(h); rel = TP.items[h['tid']]['frame']['rel']
        cron_parse.attach_all_alts(kg, SF, hh)
        X, keys, ent_slots = cron_parse.alt_feature_rows(kg, SF, hh, rel)
        sid = {id(s): j for j, s in enumerate(ent_slots)}
        G = {kg.e2i[e] for e in tr[i]['entities']}
        for row, (s_, k) in zip(X, keys):
            s = ent_slots[sid[s_]]
            Xs.append(row); ys.append(int(s['all_alts'][k][0] in G)); qi.append(i); na.append(len(s['all_alts']))
    Xa, ya, qi, na = np.array(Xs), np.array(ys), np.array(qi), np.array(na)
    prm = dict(objective='binary', learning_rate=0.1, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.9,
               verbose=-1, num_threads=10, seed=0)
    amb = na >= 2
    alt = {f'fold{f}': lgb.train(prm, lgb.Dataset(Xa[(fold[qi] != f) & amb], ya[(fold[qi] != f) & amb]), 200) for f in (0, 1)}
    alt['full'] = lgb.train(prm, lgb.Dataset(Xa[amb], ya[amb]), 200)
    # classifier
    docs = [CC.ngrams(cron_parse.masked_text(p['text'], p['hyps'][0])) for p in P]
    voc = CC.Vocab(docs, 3); enc = [np.array(voc.encode(d)) for d in docs]
    labs = [CC.labels_of(q) for q in tr]; Y = {k: np.array([l[k] for l in labs]) for k in CC.HEADS}
    clf = {f'fold{f}': CC.train(enc, Y, np.where(fold != f)[0], len(voc.itos), seed=0) for f in (0, 1)}
    clf['full'] = CC.train(enc, Y, np.arange(len(tr)), len(voc.itos), seed=0)
    os.makedirs(f'{OUT}/models', exist_ok=True)
    pickle.dump(TP.items, open(f'{OUT}/models/templates.pkl', 'wb'))
    for k, m in alt.items():
        m.save_model(f'{OUT}/models/alt_{k}.txt')
    for k, n in clf.items():
        torch.save(n.state_dict(), f'{OUT}/models/clf_{k}.pt')
    pickle.dump(voc.itos, open(f'{OUT}/models/clf_vocab.pkl', 'wb')); np.save(f'{OUT}/models/train_fold.npy', fold)


def s_subsample(tr, N=100000):
    types = np.array([q['type'] for q in tr]); cnt = collections.Counter(types.tolist())
    quota = {k: v * N / len(tr) for k, v in cnt.items()}; base = {k: int(math.floor(v)) for k, v in quota.items()}
    for k in sorted(quota, key=lambda k: -(quota[k] - base[k]))[:N - sum(base.values())]:
        base[k] += 1
    rng = np.random.RandomState(0); sel = []
    for k in sorted(base):
        sel += sorted(rng.choice(np.where(types == k)[0], base[k], replace=False).tolist())
    return sorted(sel), base


def lab_of(keys, A):
    return np.array([CD.key_out(a) in A for a in keys], np.int8)


def s_train_matrix(PIPE, tr, qids, reg, fold):
    """step 6: OWN = OOF alternative scorer + OOF classifier; GOLD-ANN = gold parse of 'question' + OOF classifier"""
    own = [PIPE.parse_own(tr[i]['paraphrases'][0]) for i in qids]
    enc = [np.array(PIPE.voc.encode(CC.ngrams(cron_parse.masked_text(p['text'], p['hyps'][0])))) for p in own]
    fs = fold[qids]; probs = {k: np.zeros((len(qids), len(v)), np.float32) for k, v in CC.HEADS.items()}
    for f in (0, 1):
        ids = np.where(fs == f)[0]; pr = CC.predict(PIPE.clf[f'fold{f}'], [enc[j] for j in ids])
        for k in CC.HEADS:
            probs[k][ids] = pr[k]
    if reg == 'OWN':
        parses = [PIPE.rerank(p, f'fold{fs[j]}') for j, p in enumerate(own)]
    else:
        parses = [PIPE.parse_gold(tr[i]['question'], tr[i]['entities'], tr[i]['times']) for i in qids]
    Xs, ys, g = [], [], []
    for j, i in enumerate(qids):
        keys, X, meta = CD.candidates(parses[j], CP.probs_row(probs, j), PIPE.tmpl_freq)
        keys, X = keys[:200], X[:200]; y = lab_of(keys, tr[i]['answers'])
        if len(keys) and y.sum():
            Xs.append(X); ys.append(y); g.append(len(keys))
    return np.concatenate(Xs), np.concatenate(ys), np.array(g)


def ranked_lists(PIPE, pool, FB, S_models):
    keys, X, meta = pool
    out = {'RULE': PIPE.pad([CD.key_out(a) for a in keys], meta, FB),
           'PRIOR': PIPE.pad([CD.key_out(a) for a in meta['prior']], meta, FB)}
    for s, m in S_models.items():
        rk, _ = RK.rerank(keys, X, m)
        out[f'S_seed{s}'] = PIPE.pad([CD.key_out(a) for a in rk], meta, FB)
    return out


def lookup_train_table(tr, PIPE):
    tid = {it['t']: i for i, it in enumerate(PIPE.tp.items)}; c = collections.defaultdict(collections.Counter)
    for q in tr:
        c[q['template']].update(q['answers'])
    return {tid[t]: [a for a, _ in sorted(v.items(), key=lambda x: (-x[1], str(x[0])))[:50]] for t, v in c.items()}


def dev_lookup(dev_v2, eset, exclude=None, _cache={}):
    if 'by' not in _cache:
        by = collections.defaultdict(list)
        for j, q in enumerate(dev_v2):
            by[frozenset(q['entities'])].append(j)
        _cache['by'] = by
    c = collections.Counter()
    for j in _cache['by'].get(eset, ()):
        if j != exclude:
            c.update(dev_v2[j]['answers'])
    return [a for a, _ in sorted(c.items(), key=lambda x: (-x[1], str(x[0])))[:50]]


def fallback_answers(tr):
    fe, ft = collections.Counter(), collections.Counter()
    for q in tr:
        (fe if q['answer_type'] == 'entity' else ft).update(q['answers'])
    return {'entity': [a for a, _ in sorted(fe.items(), key=lambda x: (-x[1], str(x[0])))[:50]],
            'time': [int(a) for a, _ in sorted(ft.items(), key=lambda x: (-x[1], str(x[0])))[:50]]}
# Test pass (step 8): test pickles are loaded with CP.load_questions(..., filter_fields=True), which keeps only
# paraphrases / question / entities / times; v1 pools are recomputed only for the rows whose text differs from v2
# (1,448 dev, 1,407 test) and reused for the byte-identical rest (equality checked on samples of 300 rows).
