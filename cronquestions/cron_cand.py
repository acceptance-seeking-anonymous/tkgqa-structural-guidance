"""cron_cand.py -- deterministic interval operators over inclusive [start, end] year facts, candidate
retrieval, RULE ordering and S features for CronQuestions.

Operator semantics were derived on TRAIN with oracle (gold) roles and relations (see the report):
  simple_entity : answer facts valid at year t (S <= t <= E)                       train exact 1.000
  simple_time   : union of the years of the (head, r, tail) intervals             train exact 1.000
  first_last    : first = minimum start, last = maximum end (entity or year)       train Hits@1 >= 0.999
  before_after  : anchor [S0, E0] = (min start, max end) of the (head, r, tail) facts or the event interval;
                  after  -> members valid at the first year >= E0+1 at which any member is valid,
                  before -> members valid at the last  year <= S0-1 at which any member is valid;
                  group = holders of the position (P39, events) or the head's other objects (P54/P108/P26)
  time_join     : members of (., r, tail2 or tail) whose interval overlaps (inclusive) any anchor interval;
                  the head itself counts as an answer for P54 and P39 but not for P108 (train-derived)
Inputs per question: frozen parser output only (no gold, no metadata)."""
import math, collections
import numpy as np

KSEG, KREL, KCOMBO = 2, 2, 6
POOL_CAP = 1000
BA_GROUP = {'P39': 'obj', 'P54': 'subj', 'P108': 'subj', 'P26': 'subj', 'P166': 'subj'}   # derived on train
TJ_SELF = {'P54': True, 'P39': True, 'P108': False, 'P26': True, 'P166': True}            # derived on train
RELS = ['P54', 'P39', 'P166', 'P26', 'P108']
TYPES = ['simple_entity', 'simple_time', 'before_after', 'first_last', 'time_join']

FEATS = ['is_time', 'sat_top', 'sat_any', 'best_seg', 'best_rel_rank', 'best_relp', 'sum_relp_sat', 'alt_rank',
         'alt_score', 'alt_margin', 'n_sat_hyp', 'q_ncand_sat', 'op_dist', 'ov_len', 's_rel', 'e_rel', 'pos',
         'pos_rel', 'n_support', 'is_qent', 'log_deg', 'log_prior', 'qt_p', 'atype_p', 'adj_p', 'ba_p', 'rel1_p',
         'rel_margin', 'n_ment', 'n_seg', 'is_fallback', 'tmpl_logfreq', 'is_start', 'is_end', 'cover_cnt',
         'anchor_found', 'side_ok', 'iv_len', 'grp_size']
FIDX = {f: i for i, f in enumerate(FEATS)}
_kg = None


def init(kg):
    global _kg
    _kg = kg


def _slice(e, r, side):
    """side 'subj': facts (e, r, x) -> answers x = T; side 'obj': facts (x, r, e) -> answers x = H"""
    kg = _kg
    if side == 'subj':
        f = kg.subj_facts(e, r); return f, kg.T[f]
    f = kg.obj_facts(e, r); return f, kg.H[f]


def _agg(out, a, sat, dist, **kw):
    """keep, per answer, the best (sat, -dist) record and count support"""
    cur = out.get(a)
    if cur is None:
        d = dict(sat=sat, dist=dist, n_support=1); d.update(kw); out[a] = d
        return
    cur['n_support'] += 1
    if (sat, -dist) > (cur['sat'], -cur['dist']):
        n = cur['n_support']; cur.clear(); cur.update(dict(sat=sat, dist=dist, n_support=n)); cur.update(kw)


# ---------------------------------------------------------------- operators
def op_simple_entity(ent, side, r, t):
    f, ans = _slice(ent, r, side)
    out = {}
    if not len(f):
        return out, dict(anchor=0, grp=0)
    S, E = _kg.S[f], _kg.E[f]
    for a, s, e in zip(ans.tolist(), S.tolist(), E.tolist()):
        sat = s <= t <= e
        dist = 0 if sat else min(abs(t - s), abs(t - e))
        _agg(out, ('e', a), sat, dist, s_rel=s - t, e_rel=e - t, ov_len=1 if sat else 0, iv_len=e - s + 1)
    return out, dict(anchor=1, grp=len(set(ans.tolist())))


def op_simple_time(head, tail, r):
    kg = _kg
    out = {}
    if head is not None:
        f = kg.subj_facts(head, r)
        pf = f[kg.T[f] == tail] if tail is not None else f
        other = f[kg.T[f] != tail] if tail is not None else f[:0]
    else:
        f = kg.obj_facts(tail, r); pf = f; other = f[:0]
    if not len(pf):
        return out, dict(anchor=0, grp=0)
    ivs = list(zip(kg.S[pf].tolist(), kg.E[pf].tolist()))
    cover = collections.Counter()
    for s, e in ivs:
        for y in range(s, min(e, s + 300) + 1):
            cover[y] += 1
    starts = {s for s, _ in ivs}; ends = {e for _, e in ivs}
    lo, hi = min(s for s, _ in ivs), max(e for _, e in ivs)
    for y, c in cover.items():
        out[('y', y)] = dict(sat=True, dist=0, n_support=c, cover_cnt=c, is_start=int(y in starts),
                             is_end=int(y in ends), s_rel=y - lo, e_rel=y - hi, iv_len=hi - lo + 1)
    near = set()
    for s, e in ivs:
        near.update([s - 1, e + 1, s - 2, e + 2])
    for s, e in zip(kg.S[other].tolist(), kg.E[other].tolist()):
        near.update([s, e])
    for y in near:
        if ('y', y) in out:
            continue
        d = min(min(abs(y - s), abs(y - e)) for s, e in ivs)
        out[('y', y)] = dict(sat=False, dist=d, n_support=0, cover_cnt=0, is_start=0, is_end=0,
                             s_rel=y - lo, e_rel=y - hi, iv_len=hi - lo + 1)
    return out, dict(anchor=1, grp=len(ivs))


def op_first_last_entity(ent, side, r, adj):
    f, ans = _slice(ent, r, side)
    out = {}
    if not len(f):
        return out, dict(anchor=0, grp=0)
    S, E = _kg.S[f], _kg.E[f]
    key = S if adj == 'first' else E
    ext = key.min() if adj == 'first' else key.max()
    best = {}
    for a, k, s, e in zip(ans.tolist(), key.tolist(), S.tolist(), E.tolist()):
        b = best.get(a)
        better = b is None or (k < b[0] if adj == 'first' else k > b[0])
        if better:
            best[a] = (k, s, e)
    keys = sorted({v[0] for v in best.values()}, reverse=(adj == 'last'))
    rank = {k: i for i, k in enumerate(keys)}
    cnt = collections.Counter(ans.tolist())
    for a, (k, s, e) in best.items():
        out[('e', a)] = dict(sat=(k == ext), dist=abs(int(k) - int(ext)), n_support=cnt[a], pos=rank[k],
                             s_rel=s - int(ext), e_rel=e - int(ext), iv_len=e - s + 1)
    return out, dict(anchor=1, grp=len(best))


def op_first_last_time(head, tail, r, adj):
    kg = _kg
    if head is not None:
        f = kg.subj_facts(head, r)
        if tail is not None:
            f = f[kg.T[f] == tail]
    elif tail is not None:
        f = kg.obj_facts(tail, r)
    else:
        return {}, dict(anchor=0, grp=0)
    out = {}
    if not len(f):
        return out, dict(anchor=0, grp=0)
    S, E = kg.S[f].tolist(), kg.E[f].tolist()
    ext = min(S) if adj == 'first' else max(E)
    ys = collections.Counter(S + E)
    order = sorted(ys, reverse=(adj == 'last'))
    rank = {y: i for i, y in enumerate(order)}
    sset, eset = set(S), set(E)
    for y, c in ys.items():
        out[('y', y)] = dict(sat=(y == ext), dist=abs(y - ext), n_support=c, pos=rank[y], is_start=int(y in sset),
                             is_end=int(y in eset), s_rel=y - ext, e_rel=y - ext)
    return out, dict(anchor=1, grp=len(ys))


def anchor_intervals(head, tail, event, r):
    kg = _kg
    if event is not None:
        iv = kg.event_iv.get(event)
        return [iv] if iv else []
    if head is None or tail is None:
        return []
    f = kg.subj_facts(head, r); f = f[kg.T[f] == tail]
    return list(zip(kg.S[f].tolist(), kg.E[f].tolist()))


def op_before_after(head, tail, event, r, rel, ba):
    kg = _kg
    ivs = anchor_intervals(head, tail, event, r)
    if not ivs:
        return {}, dict(anchor=0, grp=0)
    S0, E0 = min(s for s, _ in ivs), max(e for _, e in ivs)
    if event is not None or BA_GROUP.get(rel, 'subj') == 'obj':
        if tail is None:
            return {}, dict(anchor=0, grp=0)
        f, ans = _slice(tail, r, 'obj'); excl = head
    else:
        f, ans = _slice(head, r, 'subj'); excl = tail
    m = ans != (excl if excl is not None else -1)
    f, ans = f[m], ans[m]
    out = {}
    if not len(f):
        return out, dict(anchor=1, grp=0)
    S, E = kg.S[f], kg.E[f]
    if ba == 'after':
        y0 = E0 + 1; mm = E >= y0
        ys = int(np.maximum(S[mm], y0).min()) if mm.any() else None
    else:
        y0 = S0 - 1; mm = S <= y0
        ys = int(np.minimum(E[mm], y0).max()) if mm.any() else None
    ref = ys if ys is not None else (E0 if ba == 'after' else S0)
    for a, s, e in zip(ans.tolist(), S.tolist(), E.tolist()):
        sat = ys is not None and s <= ys <= e
        dist = 0 if sat else (min(abs(ref - s), abs(ref - e)) if not (s <= ref <= e) else 1)
        side = int(e >= E0 + 1) if ba == 'after' else int(s <= S0 - 1)
        _agg(out, ('e', a), sat, dist, s_rel=s - ref, e_rel=e - ref, side_ok=side, iv_len=e - s + 1)
    keys = sorted({v['dist'] for v in out.values()})
    rk = {k: i for i, k in enumerate(keys)}
    for v in out.values():
        v['pos'] = rk[v['dist']]
    return out, dict(anchor=1, grp=len(out))


def op_time_join(head, tail, tail2, event, r, rel):
    kg = _kg
    ivs = anchor_intervals(head, tail, event, r)
    if not ivs:
        return {}, dict(anchor=0, grp=0)
    grp_ent = tail2 if tail2 is not None else tail
    if grp_ent is None:
        return {}, dict(anchor=0, grp=0)
    f, ans = _slice(grp_ent, r, 'obj')
    out = {}
    if not len(f):
        return out, dict(anchor=1, grp=0)
    S, E = kg.S[f].tolist(), kg.E[f].tolist()
    self_ok = TJ_SELF.get(rel, True)
    for a, s, e in zip(ans.tolist(), S, E):
        best_ov, best_gap = -1, 10 ** 6
        for s0, e0 in ivs:
            ov = min(e, e0) - max(s, s0) + 1
            if ov > best_ov:
                best_ov = ov
            gap = 0 if ov > 0 else max(s - e0, s0 - e, 0)
            best_gap = min(best_gap, gap)
        sat = best_ov > 0 and (self_ok or a != head)
        _agg(out, ('e', a), sat, 0 if best_ov > 0 else best_gap, ov_len=max(best_ov, 0), s_rel=s - ivs[0][0],
             e_rel=e - ivs[0][1], iv_len=e - s + 1, side_ok=int(a != head))
    return out, dict(anchor=1, grp=len(out))


# ---------------------------------------------------------------- hypotheses
def slot_alts(hyp):
    d = {}
    for s in hyp['slots']:
        if s['role'] == 'time':
            d.setdefault('time', s['year'])
        elif s['alts']:
            d.setdefault(s['role'], s['alts'])
        else:
            d.setdefault(s['role'], None)
    return d


def role_variants(hyp):
    """explicit roles from the segmenter; for the dictionary fallback ('unk') enumerate assignments"""
    d = slot_alts(hyp)
    if 'unk' not in [s['role'] for s in hyp['slots']]:
        return [d]
    ents = [s['alts'] for s in hyp['slots'] if s['role'] == 'unk' and s['alts']]
    base = {k: v for k, v in d.items() if k == 'time'}
    out = []
    if len(ents) == 1:
        out += [dict(base, head=ents[0]), dict(base, tail=ents[0])]
    elif len(ents) >= 2:
        out += [dict(base, head=ents[0], tail=ents[1]), dict(base, head=ents[1], tail=ents[0]),
                dict(base, event_head=ents[0], tail=ents[1]), dict(base, tail=ents[0], event_head=ents[1])]
        if len(ents) >= 3:
            out += [dict(base, tail2=ents[0], head=ents[1], tail=ents[2])]
    return out or [base]


def combos(d, k=KCOMBO):
    roles = [r for r in ('head', 'tail', 'tail2', 'event_head') if d.get(r)]
    lists = [[(i, a) for i, a in enumerate(d[r][:3])] for r in roles]
    allc = [[]]
    for L in lists:
        allc = [c + [x] for c in allc for x in L]
    allc.sort(key=lambda c: (sum(i for i, _ in c), [i for i, _ in c]))
    res = []
    for c in allc[:k]:
        ents = {r: a[0] for r, (i, a) in zip(roles, c)}
        sc = min([a[2] for _, a in c], default=1.0)
        res.append((sum(i for i, _ in c), ents, sc))
    return res


def run_op(qt, atype, adj, ba, rel, r, ents, t):
    head, tail, tail2, event = ents.get('head'), ents.get('tail'), ents.get('tail2'), ents.get('event_head')
    if qt == 'simple_entity':
        if t is None:
            return None
        if head is not None:
            return op_simple_entity(head, 'subj', r, t)
        if tail is not None:
            return op_simple_entity(tail, 'obj', r, t)
        return None
    if qt == 'simple_time':
        if head is None and tail is None:
            return None
        return op_simple_time(head, tail, r)
    if qt == 'first_last':
        a = adj if adj in ('first', 'last') else 'first'
        if atype == 'time':
            return op_first_last_time(head, tail, r, a)
        if head is not None:
            return op_first_last_entity(head, 'subj', r, a)
        if tail is not None:
            return op_first_last_entity(tail, 'obj', r, a)
        return None
    if qt == 'before_after':
        b = ba if ba in ('before', 'after') else 'after'
        return op_before_after(head, tail, event, r, rel, b)
    if qt == 'time_join':
        return op_time_join(head, tail, tail2, event, r, rel)
    return None


def prior_counts(ents_primary, atype):
    """PRIOR: entity -> co-occurrence count with the linked entities; time -> facts covering the year"""
    kg = _kg
    c = collections.Counter()
    for e in ents_primary:
        f = kg.facts_of(e)
        if atype == 'time':
            for s, x in zip(kg.S[f].tolist(), kg.E[f].tolist()):
                for y in range(s, min(x, s + 200) + 1):
                    c[('y', y)] += 1
        else:
            for h, t in zip(kg.H[f].tolist(), kg.T[f].tolist()):
                c[('e', t if h == e else h)] += 1
    return c


def candidates(parse, probs, tmpl_freq):
    """parse: {'hyps': [...]} from cron_parse (alternatives already re-ranked).  probs: classifier row dict.
    Returns (keys, X, rule_order, meta)."""
    kg = _kg
    relp = probs['rel']; order = np.argsort(-relp)
    rels = [(RELS[i], float(relp[i])) for i in order[:KREL]]
    qt = TYPES[int(probs['qtype'].argmax())]
    atype = 'time' if probs['atype'][1] > probs['atype'][0] else 'entity'
    adj = ['none', 'first', 'last'][int(probs['adj'].argmax())]
    ba = ['none', 'before', 'after'][int(probs['ba'].argmax())]
    hyps = parse['hyps'][:KSEG]
    cand = {}
    nsat_top = 0
    anchor_found = 0
    sum_relp = collections.defaultdict(float)
    nsat = collections.Counter()
    sat_top = set()
    for si, h in enumerate(hyps):
        for d in role_variants(h):
            t = d.get('time')
            for ri, (rel, rp) in enumerate(rels):
                r = kg.r2i[rel]
                for ci, (crank, ents, ascore) in enumerate(combos(d)):
                    res = run_op(qt, atype, adj, ba, rel, r, ents, t)
                    if res is None:
                        continue
                    out, info = res
                    if si == 0 and ri == 0 and ci == 0:
                        anchor_found = info['anchor']
                        nsat_top = sum(1 for v in out.values() if v['sat'])
                    hk = (si, ri, crank)
                    for a, v in out.items():
                        if atype == 'time' and a[0] != 'y':
                            continue
                        if v['sat']:
                            sum_relp[a] += rp if ci == 0 else 0.0
                            nsat[a] += 1
                        rk = (0 if v['sat'] else 1, si, ri, crank, v['dist'], -v['n_support'])
                        if v['sat'] and hk == (0, 0, 0):
                            sat_top.add(a)
                        cur = cand.get(a)
                        if cur is None or rk < cur['rk']:
                            rec = dict(v); rec.update(rk=rk, seg=si, rel_rank=ri, relp=rp, alt_rank=crank,
                                                      alt_score=ascore, grp=info['grp'])
                            cand[a] = rec
    # PRIOR (primary links of the top hypothesis)
    prim = []
    if hyps:
        for s in hyps[0]['slots']:
            if s['role'] != 'time' and s['alts']:
                prim.append(s['alts'][0][0])
    pc = prior_counts(prim, atype)
    qents = {a[0] for h in hyps for s in h['slots'] if s['role'] != 'time' for a in s['alts']}
    keys = sorted(cand, key=lambda a: cand[a]['rk'] + (str(a[1]),))[:POOL_CAP]
    X = np.zeros((len(keys), len(FEATS)), np.float32)
    amargin = 0.0
    if hyps:
        sl = [s for s in hyps[0]['slots'] if s['role'] != 'time' and len(s['alts']) >= 2]
        if sl:
            amargin = min(s['alts'][0][2] - s['alts'][1][2] for s in sl)
    rel1 = rels[0][1] if rels else 0.0
    relm = rel1 - (rels[1][1] if len(rels) > 1 else 0.0)
    nment = len([s for s in hyps[0]['slots'] if s['role'] != 'time']) if hyps else 0
    tfreq = math.log1p(tmpl_freq.get(hyps[0]['tid'], 0)) if hyps else 0.0
    for i, a in enumerate(keys):
        c = cand[a]; row = X[i]
        row[FIDX['is_time']] = int(a[0] == 'y')
        row[FIDX['sat_top']] = int(a in sat_top)
        row[FIDX['sat_any']] = int(c['sat'])
        row[FIDX['best_seg']] = c['seg']; row[FIDX['best_rel_rank']] = c['rel_rank']; row[FIDX['best_relp']] = c['relp']
        row[FIDX['sum_relp_sat']] = sum_relp.get(a, 0.0)
        row[FIDX['alt_rank']] = c['alt_rank']; row[FIDX['alt_score']] = c['alt_score']; row[FIDX['alt_margin']] = amargin
        row[FIDX['n_sat_hyp']] = nsat.get(a, 0)
        row[FIDX['q_ncand_sat']] = math.log1p(nsat_top)
        row[FIDX['op_dist']] = math.log1p(min(max(c['dist'], 0), 10 ** 4))
        row[FIDX['ov_len']] = math.log1p(max(c.get('ov_len', 0), 0))
        row[FIDX['s_rel']] = float(np.clip(c.get('s_rel', 0), -200, 200))
        row[FIDX['e_rel']] = float(np.clip(c.get('e_rel', 0), -200, 200))
        row[FIDX['pos']] = min(c.get('pos', 0), 1000)
        row[FIDX['pos_rel']] = c.get('pos', 0) / max(c['grp'], 1)
        row[FIDX['n_support']] = math.log1p(c['n_support'])
        if a[0] == 'e':
            row[FIDX['is_qent']] = int(a[1] in qents)
            row[FIDX['log_deg']] = math.log1p(kg.degree[a[1]])
        row[FIDX['log_prior']] = math.log1p(pc.get(a, 0))
        row[FIDX['qt_p']] = float(probs['qtype'].max()); row[FIDX['atype_p']] = float(probs['atype'].max())
        row[FIDX['adj_p']] = float(probs['adj'].max()); row[FIDX['ba_p']] = float(probs['ba'].max())
        row[FIDX['rel1_p']] = rel1; row[FIDX['rel_margin']] = relm
        row[FIDX['n_ment']] = nment; row[FIDX['n_seg']] = len(parse['hyps'])
        row[FIDX['is_fallback']] = int(bool(hyps) and hyps[0]['tid'] < 0)
        row[FIDX['tmpl_logfreq']] = tfreq
        row[FIDX['is_start']] = c.get('is_start', 0); row[FIDX['is_end']] = c.get('is_end', 0)
        row[FIDX['cover_cnt']] = math.log1p(c.get('cover_cnt', 0))
        row[FIDX['anchor_found']] = anchor_found
        row[FIDX['side_ok']] = c.get('side_ok', 1)
        row[FIDX['iv_len']] = math.log1p(max(c.get('iv_len', 0), 0))
        row[FIDX['grp_size']] = math.log1p(c['grp'])
    meta = dict(qt=qt, atype=atype, adj=adj, ba=ba, rels=rels, prior=[a for a, _ in sorted(pc.items(), key=lambda x: (-x[1], str(x[0][1])))[:50]],
                prim=prim)
    return keys, X, meta


def key_out(a):
    """candidate key -> scorer item (Wikidata id str or int year)"""
    return _kg.ents[a[1]] if a[0] == 'e' else int(a[1])
