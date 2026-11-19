"""kpct_core.py -- MultiTQ k% data-efficiency experiment, structural track.

Re-uses the frozen structural code (struct/: mtq_lib, mtq_align, mtq_clf, mtq_cand, mtq_export, multitq_eval)
unchanged and adds:
  * train-fit artefact builders restricted to an explicit train-quid list (sector aliases, block list,
    gold-consistent silver relation labels, n-gram vocabulary, classifier heads, LightGBM S ranker);
  * a parse-JSON <-> internal-record bridge so that a parse can be executed from JSON (executor);
  * guidance-bundle construction (parse, top-20 candidates, top-20 facts, operator trace).
Nothing here reads a dev/test answer except `labels_for` / `hit1` on dev_sel2000 (selection only).
"""
import os, sys, json, pickle, collections, math, hashlib, time, re, datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.environ.get('MTQ_CODE', os.path.join(HERE, 'struct'))
if CODE not in sys.path:
    sys.path.insert(0, CODE)
import mtq_lib as L, mtq_align as A, mtq_clf as C, mtq_cand as MC, multitq_eval as ME, mtq_export as MX  # noqa

ROOT = os.environ.get('MTQ_ROOT', os.path.join(HERE, 'mtqdata'))
QTYPES = MC.QTYPES
LEVELS = MC.LEVELS
ATYPES = ['entity', 'time']
NR = 251

_KG = None


def kg():
    global _KG
    if _KG is None:
        _KG = L.KG(root=ROOT)
        MC.init(_KG)
    return _KG


def load_questions(split):
    return json.load(open(f'{ROOT}/questions/{split}.json'))


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def list_sha(quids):
    return hashlib.sha256(json.dumps(sorted(int(q) for q in quids), separators=(',', ':')).encode()).hexdigest()


# ------------------------------------------------------------------ guard: answers only for allowed quids
class AnswerGuard:
    """Every read of a train answer goes through .answers(q); asserts the quid is in the allowed list."""

    def __init__(self, allowed, name):
        self.allowed = set(int(x) for x in allowed)
        self.name = name
        self.n_reads = 0

    def check(self, q):
        assert int(q['quid']) in self.allowed, f'answer read outside {self.name}: quid {q["quid"]}'
        self.n_reads += 1
        return q


# ------------------------------------------------------------------ linker records
def make_linker(alias=None, block=None):
    return L.Linker(kg(), alias=alias, ent_block=block)


def pack(ms):
    return [(m['lo'], m['hi'], [(a['e'], round(a['eff'], 3), int(bool(a.get('alias'))), int(a['full'])) for a in m['alts']])
            for m in ms]


def link_rec(LK, quid, qtext):
    """Text-only linking (dev / test / any question at inference)."""
    (ms, ws), tp = LK.link_question(qtext)
    return dict(quid=quid, ments=pack(ms), ws=ws, tp=tp[:2] if tp else None)


def silver_rec(q):
    """Gold-consistent silver record for one TRAIN question (A.init must have been called with the linker)."""
    s, ws, ms = A.silver(q)
    tp = s['tp']
    return dict(quid=q['quid'], ments=pack(ms), ws=ws, tp=tp[:2] if tp else None,
                topic=s['res']['topic'], partner=s['res']['partner'],
                topic_linked=s['res']['topic_linked'], partner_linked=s['res']['partner_linked'],
                rel_labels=s['rel_labels'], strong=s['strong'], fl=s['fl'], ba=s['ba'])


def alias_rules_from_counts(den, num):
    """= pipeline_steps.build_alias_rules (count >= 5, precision >= 0.1, top-3 sectors per n-gram)."""
    byk = collections.defaultdict(list)
    for (k, S), cn in num.items():
        byk[k].append((cn, S))
    rules = {}
    for k, lst in byk.items():
        lst.sort(reverse=True)
        keep = [[S, round(cn / den[k], 4), cn] for cn, S in lst[:3] if cn >= 5 and cn / den[k] >= 0.1]
        if keep:
            rules[' '.join(k)] = keep
    return rules


def block_list(Qtr, Ptr):
    """= pipeline_steps.build_block_list (train link precision < 0.05 with n >= 10)."""
    K = kg()
    cnt = collections.Counter(); good = collections.Counter()
    for q, r in zip(Qtr, Ptr):
        gold = {r['topic'], r['partner']}
        Aset = {K.e2i.get(a) for a in q['answers']}
        for lo, hi, alts in r['ments']:
            e, eff, al, full = alts[0]
            k = (e, full, al); cnt[k] += 1; good[k] += (e in gold) or (e in Aset)
    return sorted([[int(k[0]), int(k[1]), int(k[2])] for k, c in cnt.items() if c >= 10 and good[k] / c < 0.05])


# ------------------------------------------------------------------ classifier (fastText-style, frozen recipe)
def clf_doc(rec):
    return C.ngrams(C.masked_tokens(rec))


class VocabK:
    """mtq_clf.Vocab (min_count 3) restricted to a training list; itos kept for serialisation."""

    def __init__(self, docs=None, itos=None, min_count=3):
        if itos is None:
            itos = C.Vocab(docs, min_count=min_count).itos
        self.itos = list(itos)
        self.stoi = {k: i + 1 for i, k in enumerate(self.itos)}

    def encode(self, d):
        return np.array([self.stoi[x] for x in d if x in self.stoi] or [0], dtype=np.int64)


def rel_targets(r):
    labs = []
    for rr, d in r['rel_labels']:
        labs += [rr * 2, rr * 2 + 1] if d == -1 else [rr * 2 + d]
    return labs


def train_clf(enc_tr, tr_rel, y_q, y_a, y_l, V, epochs=6, seed=0, threads=1):
    """pipeline_steps.train_clf generalised to E epochs: the first E//3 epochs use uniform soft relation
    targets, the remaining epochs EM-refined targets (E = 6 -> 2 + 4, exactly the frozen recipe)."""
    import torch, torch.nn as nn
    torch.set_num_threads(threads)
    torch.manual_seed(seed); np.random.seed(seed)
    net = C.Net(V, 96, {'rel': NR * 2, 'qtype': 6, 'atype': 2, 'level': 3})
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    n = len(enc_tr)
    tr_ids = np.arange(n)
    sft = {i: (np.array(tr_rel[i]), np.full(len(tr_rel[i]), 1 / len(tr_rel[i]), dtype=np.float32)) for i in tr_ids if tr_rel[i]}
    n_uniform = epochs // 3
    y_q = np.asarray(y_q); y_a = np.asarray(y_a); y_l = np.asarray(y_l)
    for ep in range(epochs):
        if ep >= n_uniform:
            pr = C.predict(net, [enc_tr[i] for i in tr_ids])['rel']
            for j, i in enumerate(tr_ids):
                if tr_rel[i]:
                    l = np.array(tr_rel[i]); p = pr[j, l] + 1e-6; sft[i] = (l, (p / p.sum()).astype(np.float32))
        net.train(); perm = tr_ids[np.random.permutation(n)]
        for s in range(0, n, 512):
            ids = perm[s:s + 512]; idx, off = C.batchify(enc_tr, ids); o = net(idx, off)
            loss = sum(nn.functional.cross_entropy(o[k], torch.tensor(y[ids])) for k, y in (('qtype', y_q), ('atype', y_a), ('level', y_l)))
            rows = [(j, sft[i]) for j, i in enumerate(ids) if i in sft]
            if rows:
                lp = torch.log_softmax(o['rel'], -1)
                jj = np.concatenate([[j] * len(t[0]) for j, t in rows]); ll = np.concatenate([t[0] for j, t in rows])
                ww = np.concatenate([t[1] for j, t in rows])
                loss = loss - (lp[torch.tensor(jj), torch.tensor(ll)] * torch.tensor(ww)).sum() / len(rows)
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


def clf_labels(qs):
    return (np.array([QTYPES.index(q['qtype']) for q in qs]), np.array([ATYPES.index(q['answer_type']) for q in qs]),
            np.array([LEVELS.index(q['time_level']) for q in qs]))


def fit_parser(train_qs, train_recs, epochs, threads=1, seed=0):
    """vocabulary + classifier on one training list (train_recs = silver records aligned with train_qs)."""
    docs = [clf_doc(r) for r in train_recs]
    voc = VocabK(docs)
    enc = [voc.encode(d) for d in docs]
    y_q, y_a, y_l = clf_labels(train_qs)
    tr_rel = [rel_targets(r) for r in train_recs]
    net = train_clf(enc, tr_rel, y_q, y_a, y_l, len(voc.itos), epochs=epochs, seed=seed, threads=threads)
    return voc, net


def parser_probs(voc, net, recs):
    enc = [voc.encode(clf_doc(r)) for r in recs]
    if not enc:
        return {'rel': np.zeros((0, NR * 2), np.float32), 'qtype': np.zeros((0, 6), np.float32),
                'atype': np.zeros((0, 2), np.float32), 'level': np.zeros((0, 3), np.float32)}
    return C.predict(net, enc)


def save_parser(path, voc, net):
    import torch
    torch.save({'itos': voc.itos, 'state_dict': net.state_dict()}, path)


def load_parser(path):
    import torch
    d = torch.load(path, map_location='cpu', weights_only=False)
    voc = VocabK(itos=d['itos'])
    net = C.Net(len(voc.itos), 96, {'rel': NR * 2, 'qtype': 6, 'atype': 2, 'level': 3})
    net.load_state_dict(d['state_dict']); net.eval()
    return voc, net


def load_frozen_parser(code=CODE):
    import torch
    itos = pickle.load(open(f'{code}/parser_vocab.pkl', 'rb'))
    voc = VocabK(itos=itos)
    net = C.Net(len(itos), 96, {'rel': NR * 2, 'qtype': 6, 'atype': 2, 'level': 3})
    net.load_state_dict(torch.load(f'{code}/parser_clf_seed0.pt', map_location='cpu')); net.eval()
    return voc, net


# ------------------------------------------------------------------ labels / padding / ranked lists
def labels_for(q, names):
    Aset = set(q['answers'])
    if q['answer_type'] == 'time':
        Lg = len(q['answers'][0])
        if Lg in (4, 7):
            return np.array([n[:Lg] in Aset for n in names], np.int8)
    return np.array([n in Aset for n in names], np.int8)


def build_FB(train_qs, n=50):
    """Output-list padding: most frequent train answers of each answer type (time: per answer granularity)."""
    ce = collections.Counter(); ct = {lv: collections.Counter() for lv in LEVELS}
    for q in train_qs:
        if q['answer_type'] == 'entity':
            ce.update(q['answers'])
        else:
            for a in q['answers']:
                lv = {4: 'year', 7: 'month', 10: 'day'}.get(len(a))
                if lv:
                    ct[lv][a] += 1
    top = lambda c: [k for k, _ in c.most_common(n)]   # ties: first occurrence in train-file order (frozen)
    return {'entity': top(ce), **{lv: top(ct[lv]) for lv in LEVELS}}


def build_FB_kg(n=50):
    """k = 0 padding from KG statistics only: highest-degree entities; most frequent fact dates per granularity."""
    K = kg()
    ents = [K.ent_names[i] for i in np.argsort(-K.degree, kind='stable')[:n]]
    out = {'entity': ents}
    for lv in LEVELS:
        ks = MC.gran_key(K.d, lv)
        u, c = np.unique(ks, return_counts=True)
        o = np.lexsort((u, -c))[:n]
        out[lv] = [MC.key_str(u[i], lv) for i in o]
    return out


def make_ranked(ordered, meta, FB, k=50):
    """= pipeline_steps.make_ranked: top-50 distinct strings; padded to >= 10 with PRIOR then FB."""
    out, seen = [], set()
    for n in ordered:
        if n not in seen:
            seen.add(n); out.append(n)
        if len(out) >= k:
            break
    if len(out) < 10:
        info = meta['info']; fb = FB['entity'] if info['atype'] == 'entity' else FB[info['level']]
        for n in list(meta['prior']) + fb:
            if n not in seen:
                seen.add(n); out.append(n)
            if len(out) >= 10:
                break
    return out


def hit1(q, ranked):
    return ME.hit_at_k(q, ranked, 1)


def hit10(q, ranked):
    return ME.hit_at_k(q, ranked, 10)


# ------------------------------------------------------------------ parse <-> internal record bridge
DIR_NAMES = {1: 'topic_or_first_mention_is_head', 0: 'topic_or_first_mention_is_tail'}
DIR_CODES = {v: k for k, v in DIR_NAMES.items()}
BA_NAMES = {0: 'before', 1: 'after', -1: None}
FL_NAMES = {0: 'first', 1: 'last', -1: None}
PARSE_SCHEMA = 'kpct-parse-v1'


def info_from_probs(rec, probs, qtext):
    """= mtq_cand.parse_info (frozen parser decisions)."""
    return MC.parse_info(rec, probs, qtext)


def parse_json(rec, info, qtext=None):
    """Serialise one parse (linker record + parser decisions) to the bundle / executor schema."""
    K = kg()
    ws = list(rec['ws'])
    ments = []
    for lo, hi, alts in rec['ments']:
        ments.append({'span': [int(lo), int(hi)], 'text': ' '.join(ws[lo:hi + 1]),
                      'alternatives': [{'entity': K.ent_names[e], 'score': float(eff), 'alias': int(al), 'full': int(fu)}
                                       for e, eff, al, fu in alts]})
    return {
        'schema': PARSE_SCHEMA,
        'tokens': ws,
        'mentions': ments,
        'time': ({'level': rec['tp'][0], 'value': rec['tp'][1]} if rec['tp'] else None),
        'relations': [{'relation': K.rel_names[r], 'direction': DIR_NAMES[d], 'prob': float(p)} for r, d, p in info['rels']],
        'qtype': {'label': info['qt'], 'prob': float(info['qt_p'])},
        'answer_type': {'label': info['atype'], 'prob': float(info['atype_p'])},
        'granularity': {'label': info['level'], 'prob': float(info['level_p'])},
        'operator_keywords': {'before_after': BA_NAMES[info['BA']], 'first_last': FL_NAMES[info['FL']],
                              'same_granularity': info['same']},
    }


class ParseError(ValueError):
    pass


def _norm_name(s):
    return s.replace('_', ' ').strip()


def _locate(ws, text, used):
    """token span of a mention text inside the linker tokens (first unused exact token match, else stem match)."""
    t = [w for w in L.toks(text)]
    if not t:
        return None
    n = len(t)
    for st in (lambda x: x, L.stem):
        tt = [st(w) for w in t]; ww = [st(w) for w in ws]
        for i in range(len(ws) - n + 1):
            if ww[i:i + n] == tt and not any(j in used for j in range(i, i + n)):
                return [i, i + n - 1]
    return None


def rec_info_from_parse(p, question=None):
    """Executor input: parse JSON (kpct-parse-v1) -> (rec, info). Raises ParseError on unknown names.
    Missing optional fields: tokens (recomputed from `question` with the fixed tokeniser / time regex),
    mention spans (located in the tokens), alternative score/alias/full (1.0/0/1), probabilities (1.0),
    operator_keywords (fixed keyword rules on `question`)."""
    K = kg()
    if p.get('schema', PARSE_SCHEMA) != PARSE_SCHEMA:
        raise ParseError(f"unknown schema {p.get('schema')}")
    tp = None
    if p.get('time'):
        lv, val = p['time']['level'], str(p['time']['value'])
        if lv not in LEVELS:
            raise ParseError(f'bad time level {lv}')
        tp = (lv, val)
        try:
            A.tspan(tp)
        except Exception as ex:
            raise ParseError(f'bad time value {val}: {ex}')
    ws = p.get('tokens')
    if ws is None:
        if question is None:
            raise ParseError('tokens missing and no question text given')
        q2 = question
        t_ = L.parse_time(question)
        if t_ is not None:
            a, b = t_[2]; q2 = question[:a] + ' timex ' + question[b:]
        ws = L.toks(q2)
    ws = list(ws)
    ments, used = [], set()
    for m in p.get('mentions', []):
        span = m.get('span')
        if span is None:
            span = _locate(ws, m.get('text', ''), used)
            if span is None:   # not locatable: place after the last token (keeps mention order semantics)
                span = [len(ws) + len(ments), len(ws) + len(ments)]
        lo, hi = int(span[0]), int(span[1])
        used.update(range(lo, hi + 1))
        alts = []
        for a in m.get('alternatives') or ([{'entity': m['entity']}] if m.get('entity') else []):
            e = K.e2i.get(_norm_name(a['entity']))
            if e is None:
                raise ParseError(f"entity not in KG vocabulary: {a['entity']}")
            alts.append((e, float(a.get('score', 1.0)), int(a.get('alias', 0)), int(a.get('full', 1))))
        ments.append((lo, hi, alts))
    ments.sort(key=lambda m: m[0])
    rels = []
    for r in p.get('relations', []):
        rid = K.r2i.get(r['relation'].replace(' ', '_'))
        if rid is None:
            raise ParseError(f"relation not in KG vocabulary: {r['relation']}")
        d = r['direction'] if isinstance(r['direction'], int) else DIR_CODES.get(r['direction'])
        if d not in (0, 1):
            raise ParseError(f"bad direction {r['direction']}")
        rels.append((rid, d, float(r.get('prob', 1.0))))
    rels = rels[:MC.KREL]
    def lab(key, allowed):
        v = p.get(key)
        if v is None:
            raise ParseError(f'{key} missing')
        lbl = v['label'] if isinstance(v, dict) else v
        if lbl not in allowed:
            raise ParseError(f'bad {key} {lbl}')
        return lbl, float(v.get('prob', 1.0)) if isinstance(v, dict) else 1.0
    qt, qt_p = lab('qtype', QTYPES)
    at, at_p = lab('answer_type', ATYPES)
    lv, lv_p = lab('granularity', LEVELS)
    kw = p.get('operator_keywords')
    if kw is None:
        if question is None:
            raise ParseError('operator_keywords missing and no question text given')
        BA, FL, same = L.kw_ba(question), L.kw_fl(question), MC.same_gran_word(question)
    else:
        BA = {'before': 0, 'after': 1, None: -1}[kw.get('before_after')]
        FL = {'first': 0, 'last': 1, None: -1}[kw.get('first_last')]
        same = kw.get('same_granularity')
        if same not in (None,) + tuple(LEVELS):
            raise ParseError(f'bad same_granularity {same}')
    rec = dict(quid=p.get('quid'), ments=ments, ws=ws, tp=tp)
    info = dict(qt=qt, qt_p=qt_p, atype=at, atype_p=at_p, level=lv, level_p=lv_p, rels=rels, BA=BA, FL=FL,
                same=same, tp=tp)
    return rec, info


# ------------------------------------------------------------------ candidate build from an explicit info
def build_info(rec, info):
    """= mtq_cand.build with the parser decisions supplied as `info` (identical body after parse_info)."""
    kg_ = kg()
    prior_list, prior_cnt = MC.prior_candidates(rec, info)
    if info['atype'] == 'entity':
        cand, retrieved, qents, margin, rel1 = MC.entity_candidates(rec, info, None)
        names_of = lambda c: kg_.ent_names[c]
        is_time = 0
    else:
        cand, retrieved, margin, rel1 = MC.time_candidates(rec, info, None)
        qents = set()
        names_of = lambda c: MC.key_str(c, info['level'])
        is_time = 1
    FIDX, FEATS = MC.FIDX, MC.FEATS
    keys = list(cand)
    keys.sort(key=lambda c: cand[c]['rule'])
    keys = keys[:MC.POOL_CAP]
    nsat = sum(1 for c in keys if cand[c]['sat_any'])
    X = np.zeros((len(keys), len(FEATS)), np.float32)
    for i, c in enumerate(keys):
        f = cand[c]
        row = X[i]
        row[FIDX['is_time']] = is_time
        for k in ('sat_top', 'sat_any', 'best_rank', 'best_relp', 'sum_relp_sat', 'relp_conn', 'alt_rank',
                  'alt_eff_ratio', 'n_sat', 'n_conn', 'pos', 'pos_rel', 'anchor_found', 'role_swapped', 'dir_top',
                  'anchor_alt', 'sat_anchor_last', 'pos_anchor_last'):
            row[FIDX[k]] = f[k]
        row[FIDX['gap']] = math.log1p(min(f['gap'], 1e6))
        row[FIDX['dist']] = math.log1p(min(f['dist'], 1e6))
        row[FIDX['n_pair_all']] = f.get('n_pair_all', 0)
        if not is_time:
            row[FIDX['is_qent']] = int(c in qents)
            row[FIDX['log_deg']] = math.log1p(kg_.degree[c])
            row[FIDX['first_day']] = f['first_day'] / kg_.ndays
            row[FIDX['last_day']] = f['last_day'] / kg_.ndays
        row[FIDX['log_prior']] = math.log1p(prior_cnt.get(c, 0))
        row[FIDX['q_ncand_sat']] = math.log1p(nsat)
        row[FIDX['qt_p']] = info['qt_p']; row[FIDX['atype_p']] = info['atype_p']; row[FIDX['level_p']] = info['level_p']
        row[FIDX['n_ment']] = len(rec['ments'])
        row[FIDX['rel1_p']] = rel1; row[FIDX['rel_margin']] = margin
    if not is_time and keys:
        tops = {}
        for c in keys:
            t = cand[c].get('topic')
            if t is not None:
                tops.setdefault(t, []).append(c)
        idx = {c: i for i, c in enumerate(keys)}
        for t, cs in tops.items():
            f = kg_.facts_of(t)
            oth = np.where(kg_.h[f] == t, kg_.t[f], kg_.h[f])
            u, n = np.unique(oth, return_counts=True)
            mp = dict(zip(u.tolist(), n.tolist()))
            for c in cs:
                X[idx[c], FIDX['n_pair_all']] = mp.get(c, 0)
    retrieved.sort(key=lambda x: x[0])
    seen, facts = set(), []
    for _, fid in retrieved:
        if fid not in seen:
            seen.add(fid); facts.append(fid)
        if len(facts) >= 20:
            break
    names = [names_of(c) for c in keys]
    rinfo = [(cand[c].get('r', -1), cand[c].get('d', -1)) for c in keys]
    return dict(names=names, X=X, info=info, facts=facts, prior=prior_list, rinfo=rinfo,
                topics=[cand[c].get('topic', -1) if not is_time else -1 for c in keys])


def cap_pool(o, cap):
    if cap:
        o['names'] = o['names'][:cap]; o['X'] = o['X'][:cap]; o['rinfo'] = o['rinfo'][:cap]; o['topics'] = o['topics'][:cap]
    return o


# ------------------------------------------------------------------ S scoring / orders
def s_order(score):
    """S order: descending score, ties by RULE order (= pool order)."""
    return np.argsort(-np.asarray(score, dtype=np.float64), kind='stable')


def predict_chunk(boosters, outs):
    """Score every pool candidate of a list of build outputs with each booster (one predict call per booster)."""
    Xs = [o['X'] for o in outs]
    sizes = [len(x) for x in Xs]
    res = [[None] * len(outs) for _ in boosters]
    if sum(sizes) == 0:
        for b in range(len(boosters)):
            for i in range(len(outs)):
                res[b][i] = np.zeros(0)
        return res
    Xall = np.concatenate([x for x in Xs if len(x)])
    offs = np.cumsum([0] + sizes)
    for b, bst in enumerate(boosters):
        sc = bst.predict(Xall, num_threads=1)
        for i in range(len(outs)):
            res[b][i] = sc[offs[i]:offs[i + 1]]
    return res


# ------------------------------------------------------------------ provenance / trace
def cand_support(o, rec, j, per_cand=3):
    """Supporting fact ids of pool candidate j (mtq_export.provenance_facts inner recipe, <= 3 facts)."""
    K = kg()
    info = o['info']; names = o['names']; rinfo = o['rinfo']; topics = o['topics']
    tp = info.get('tp')
    lo = hi = None
    if tp is not None:
        lo, hi = A.tspan((tp[0], tp[1]))
    r, d = rinfo[j]
    if info['atype'] == 'entity':
        t = topics[j]; c = K.e2i.get(names[j])
        if t is None or t < 0 or c is None or r < 0:
            return []
        fi, o_, dd = MC.rel_slice(t, r, d)
        sel = fi[o_ == c]
        if not len(sel):
            return []
        dsel = K.d[sel]
        if lo is not None:
            key = np.where(dsel < lo, lo - dsel, np.where(dsel > hi, dsel - hi, 0))
            sel = sel[np.argsort(key, kind='stable')]
        elif info['FL'] == 1 or info['qt'] == 'before_last':
            sel = sel[::-1]
    else:
        ents = [m[2][0][0] for m in rec['ments'] if m[2]]
        if len(ents) < 1 or r < 0:
            return []
        e1 = ents[0]
        f = K.facts_of(e1)
        if len(ents) >= 2:
            e2 = ents[1]
            f = f[((K.h[f] == e1) & (K.t[f] == e2)) | ((K.h[f] == e2) & (K.t[f] == e1))]
        f = f[K.r[f] == r]
        ks = MC.gran_key(K.d[f], info['level'])
        sel = f[[MC.key_str(k_, info['level']) == names[j] for k_ in ks]] if len(f) else f
    return [int(x) for x in sel[:per_cand]]


def _ds(i):
    return kg().day_str[int(i)]


def operator_trace(o, rec, order):
    """Short deterministic description of what the operator did for the top candidate of `order`."""
    K = kg(); info = o['info']; names = o['names']
    qt, at, lv = info['qt'], info['atype'], info['level']
    parts = [f"qtype {qt}", f"answer {at}" + (f" at {lv} granularity" if at == 'time' else '')]
    tp = info['tp']; lo = hi = None
    if tp:
        lo, hi = A.tspan(tp)
        parts.append(f"time constraint {tp[1]} = window {_ds(lo)}..{_ds(hi)}")
    if info['BA'] >= 0:
        parts.append(f"keyword {BA_NAMES[info['BA']]}")
    if info['FL'] >= 0:
        parts.append(f"keyword {FL_NAMES[info['FL']]}")
    if not names:
        return '; '.join(parts + ['no candidate retrieved (empty pool)'])
    nsat = int(o['X'][:, MC.FIDX['sat_any']].sum())
    parts.append(f"{nsat} of {len(names)} pool candidates satisfy the operator")
    j = int(order[0]); c = names[j]; r, d = o['rinfo'][j]
    sat = bool(o['X'][j, MC.FIDX['sat_any']])
    rel = K.rel_names[r] if r >= 0 else '(any relation)'
    if at == 'entity':
        t = o['topics'][j]
        tn = K.ent_names[t] if t is not None and t >= 0 else '?'
        role = 'object' if d == 1 else 'subject'
        parts.append(f"top candidate {c} = {role} of ({tn}, {rel})")
        anchored = qt in ('after_first', 'before_last') or (qt in ('before_after', 'equal_multi') and tp is None)
        ce = K.e2i.get(c)
        if r >= 0 and t is not None and t >= 0 and ce is not None:
            fi, oo, dd = MC.rel_slice(t, r, d)
            cd = dd[oo == ce]
            if anchored:
                anc = None
                for lo_, hi_, alts in rec['ments']:
                    for e, *_ in alts[:MC.KALT]:
                        if e != t and (oo == e).any():
                            anc = e; break
                    if anc is not None:
                        break
                if anc is None:
                    parts.append('no anchor fact found for this hypothesis')
                else:
                    t0 = int(dd[oo == anc].min())
                    parts.append(f"anchor {K.ent_names[anc]}: first ({tn}, {rel}, {K.ent_names[anc]}) fact on {_ds(t0)}")
                    if qt == 'after_first':
                        after = cd[cd > t0]
                        m_ = dd[(dd > t0) & (oo != anc)]
                        if len(after):
                            parts.append(f"after_first: earliest admissible event after the anchor on {_ds(m_.min())}; candidate's first event after the anchor on {_ds(after.min())}")
                    elif qt == 'before_last':
                        bef = cd[cd < t0]
                        m_ = dd[(dd < t0) & (oo != anc)]
                        if len(bef):
                            parts.append(f"before_last: latest admissible event before the anchor on {_ds(m_.max())}; candidate's last event before the anchor on {_ds(bef.max())}")
                    elif qt == 'before_after':
                        nn = int(((cd < t0) if info['BA'] != 1 else (cd > t0)).sum())
                        parts.append(f"{BA_NAMES[max(info['BA'], 0)] if info['BA'] != 1 else 'after'} the anchor: candidate has {nn} admissible facts")
                    else:
                        g = info['same'] or lv
                        parts.append(f"same {g} as an anchor fact")
            elif qt in ('equal', 'equal_multi') and tp:
                ins = cd[(cd >= lo) & (cd <= hi)]
                s = f"candidate has {len(ins)} facts inside the window"
                if qt == 'equal_multi' and info['FL'] >= 0 and len(ins):
                    s += f"; its {'first' if info['FL'] == 0 else 'last'} in-window fact on {_ds(ins.min() if info['FL'] == 0 else ins.max())}"
                parts.append(s)
            elif qt == 'before_after' and tp:
                adm = cd[cd < lo] if info['BA'] != 1 else cd[cd > hi]
                bound = _ds(lo) if info['BA'] != 1 else _ds(hi)
                parts.append(f"{'before' if info['BA'] != 1 else 'after'} {bound}: candidate has {len(adm)} admissible facts")
            elif qt == 'first_last' or qt == 'equal_multi':
                if len(dd):
                    ext = dd.max() if info['FL'] == 1 else dd.min()
                    mine = cd.max() if info['FL'] == 1 else cd.min()
                    parts.append(f"{'last' if info['FL'] == 1 else 'first'} ({tn}, {rel}) fact on {_ds(ext)}; candidate's on {_ds(mine)}")
    else:
        ents = [m[2][0][0] for m in rec['ments'] if m[2]]
        en = ', '.join(K.ent_names[e] for e in ents[:2]) or '?'
        parts.append(f"top candidate {c} = {lv} bucket of ({en}) facts under {rel}")
        if qt == 'first_last' or (info['FL'] >= 0 and qt != 'equal'):
            parts.append(f"{'latest' if info['FL'] == 1 else 'earliest'} bucket selected")
    parts.append(f"top candidate {'satisfies' if sat else 'does NOT satisfy'} the operator")
    return '; '.join(parts)


def fact_rows(fids):
    K = kg()
    out = []
    for f in fids:
        h, r, t, d = K.quad(int(f))
        out.append({'fid': int(f), 'head': h, 'relation': r, 'tail': t, 'date': d})
    return out


def bundle_core(o, rec, s0_score, rule_k=20):
    """Bundle parts that depend on the executed pool: S seed-0 order, top-20 candidates, top-20 facts, trace."""
    names = o['names']
    order = s_order(s0_score) if s0_score is not None else np.arange(len(names))
    rule_rank = {}
    for i, n in enumerate(names):
        rule_rank.setdefault(n, i + 1)
    facts = MX.provenance_facts(kg(), o, list(order), names, o['rinfo'], o['topics'], rec)
    fpos = {f: i for i, f in enumerate(facts)}
    cands, seen = [], set()
    for rank, j in enumerate(order):
        n = names[j]
        if n in seen:
            continue
        seen.add(n)
        sup = cand_support(o, rec, int(j))
        cands.append({'answer': n, 'rule_rank': int(j) + 1,
                      's_score': (float(s0_score[j]) if s0_score is not None else None),
                      's_rank': (len(cands) + 1 if s0_score is not None else None),
                      'operator_satisfied': bool(o['X'][j, MC.FIDX['sat_any']]),
                      'support_fids': sup, 'support_fact_idx': [fpos[f] for f in sup if f in fpos]})
        if len(cands) >= rule_k:
            break
    return dict(candidates=cands, facts=fact_rows(facts), operator_trace=operator_trace(o, rec, order),
                pool_size=len(names), n_operator_satisfied=int(o['X'][:, MC.FIDX['sat_any']].sum()) if len(names) else 0)
