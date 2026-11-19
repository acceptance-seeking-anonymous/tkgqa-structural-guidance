"""mtq_cand.py -- candidate retrieval + deterministic temporal operators for MultiTQ (RULE / S / PRIOR).

Inputs per question: frozen parser output (mentions, parsed time, classifier probabilities, keyword
operators). No gold, no metadata.  Output: candidate list with features, RULE key, provenance facts.
"""
import re, math, collections
import numpy as np
import mtq_lib as L
from mtq_align import tspan

QTYPES = ['equal', 'before_after', 'first_last', 'equal_multi', 'after_first', 'before_last']
LEVELS = ['day', 'month', 'year']
KREL = 5          # relation-direction hypotheses considered
KALT = 3          # entity alternatives per mention
POOL_CAP = 1000

FEATS = ['is_time', 'sat_top', 'sat_any', 'best_rank', 'best_relp', 'sum_relp_sat', 'relp_conn', 'alt_rank',
         'alt_eff_ratio', 'n_sat', 'n_conn', 'n_pair_all', 'pos', 'pos_rel', 'gap', 'dist', 'anchor_found',
         'role_swapped', 'is_qent', 'log_deg', 'log_prior', 'q_ncand_sat', 'first_day', 'last_day',
         'qt_p', 'atype_p', 'level_p', 'n_ment', 'dir_top', 'anchor_alt', 'sat_anchor_last', 'pos_anchor_last',
         'rel1_p', 'rel_margin']
FIDX = {f: i for i, f in enumerate(FEATS)}

_kg = None


def init(kg):
    global _kg
    _kg = kg


def gran_key(days, g):
    kg = _kg
    if g == 'year':
        return kg.day_year[days]
    if g == 'month':
        return kg.day_month[days]
    return days


def key_str(k, g):
    if g == 'year':
        return str(int(k))
    if g == 'month':
        return f'{int(k) // 12}-{int(k) % 12 + 1:02d}'
    return _kg.day_str[int(k)]


def same_gran_word(s):
    m = re.search(r'same (day|month|year)', s.lower())
    return m.group(1) if m else None


def anchor_index(ws, ments, qtext):
    """Index of the anchor mention: first mention after the before/after/same keyword."""
    kws = [i for i, w in enumerate(ws) if w in ('before', 'after', 'same', 'prior', 'following', 'since', 'until')]
    if not kws or len(ments) < 2:
        return None
    k = kws[0]
    after = [j for j, m in enumerate(ments) if m[0] > k]
    if after:
        return after[0]
    before = [j for j, m in enumerate(ments) if m[1] < k]
    return before[-1] if before else None


def rel_slice(e, r, d):
    """facts of e with relation r; d=1: e is head (others = tails), d=0: e is tail. chronological."""
    kg = _kg
    f = kg.facts_of(e)
    if d == 1:
        m = (kg.h[f] == e) & (kg.r[f] == r)
        return f[m], kg.t[f[m]], kg.d[f[m]]
    m = (kg.t[f] == e) & (kg.r[f] == r)
    return f[m], kg.h[f[m]], kg.d[f[m]]


def parse_info(rec, probs, qtext):
    """Collect the frozen parser decisions for one question."""
    qt_p = probs['qtype']; at_p = probs['atype']; lv_p = probs['level']; rel_p = probs['rel']
    qt = QTYPES[int(qt_p.argmax())]
    atype = 'time' if at_p[1] > at_p[0] else 'entity'
    level = LEVELS[int(lv_p.argmax())]
    top = np.argsort(-rel_p)[:KREL]
    rels = [(int(x) // 2, int(x) % 2, float(rel_p[x])) for x in top]
    return dict(qt=qt, qt_p=float(qt_p.max()), atype=atype, atype_p=float(at_p.max()), level=level,
                level_p=float(lv_p.max()), rels=rels, BA=L.kw_ba(qtext), FL=L.kw_fl(qtext),
                same=same_gran_word(qtext), tp=rec['tp'])


def entity_candidates(rec, info, qtext):
    kg = _kg
    ments = rec['ments']
    ws = rec['ws']
    qt, tp, BA, FL = info['qt'], info['tp'], info['BA'], info['FL']
    has_time = tp is not None
    anchored = qt in ('after_first', 'before_last') or (qt in ('before_after', 'equal_multi') and not has_time)
    ai = anchor_index(ws, ments, qtext) if anchored else None
    qents = {a[0] for m in ments for a in m[2]}
    # role hypotheses: (topic mention index, anchor mention index, swapped flag)
    roles = []
    if ai is not None:
        others = [j for j in range(len(ments)) if j != ai]
        for j in others[:2]:
            roles.append((j, ai, 0))
        for j in others[:1]:
            roles.append((ai, j, 1))
    else:
        for j in range(len(ments)):
            roles.append((j, None, 0))
    if tp is not None:
        lo, hi = tspan((tp[0], tp[1]))
    g_same = info['same'] or info['level']
    cand = {}   # c -> feature dict (best over hypotheses)
    prov = collections.defaultdict(list)
    retrieved = []  # (priority tuple, fact id)
    rel1 = info['rels'][0][2] if info['rels'] else 0.0
    margin = rel1 - (info['rels'][1][2] if len(info['rels']) > 1 else 0.0)
    for (ti, aj, swapped) in roles:
        talts = ments[ti][2][:KALT]
        if not talts:
            continue
        eff0 = max(talts[0][1], 1e-6)
        aalts = ments[aj][2][:KALT] if aj is not None else []
        for ar, (et, eff, _, _) in enumerate(talts):
            for rr, (r, d, rp) in enumerate(info['rels']):
                fi, o, dd = rel_slice(et, r, d)
                if not len(fi):
                    continue
                # anchor times (facts (et, r, anchor) in the same orientation)
                t0f = t0l = None
                anchor_ent = None
                anc_alt = KALT
                if anchored:
                    for k_, (ea, _, _, _) in enumerate(aalts):
                        am = o == ea
                        if am.any():
                            t0f, t0l = int(dd[am].min()), int(dd[am].max())
                            anchor_ent = ea
                            anc_alt = k_
                            break
                uo, inv = np.unique(o, return_inverse=True)
                cnt = np.bincount(inv, minlength=len(uo))
                first = np.full(len(uo), 10 ** 6); np.minimum.at(first, inv, dd)
                last = np.full(len(uo), -1); np.maximum.at(last, inv, dd)
                sat = np.zeros(len(uo), bool); nsat = np.zeros(len(uo), int)
                key = np.zeros(len(uo), float)   # smaller = better position
                gap = np.zeros(len(uo), float); dist = np.zeros(len(uo), float)
                sat2 = np.zeros(len(uo), bool); key2 = np.zeros(len(uo), float)

                def window(mask, bucket_dist=None):
                    s_ = np.zeros(len(uo), int); np.add.at(s_, inv[mask], 1)
                    return s_

                if qt == 'equal' and has_time or (qt == 'equal_multi' and has_time):
                    m = (dd >= lo) & (dd <= hi)
                    nsat = window(m); sat = nsat > 0
                    dx = np.where(dd < lo, lo - dd, np.where(dd > hi, dd - hi, 0)).astype(float)
                    dist = np.full(len(uo), 1e6); np.minimum.at(dist, inv, dx)
                    if qt == 'equal_multi' and FL >= 0 and m.any():
                        # first / last within the window
                        fw = np.full(len(uo), 10 ** 6); lw = np.full(len(uo), -1)
                        np.minimum.at(fw, inv[m], dd[m]); np.maximum.at(lw, inv[m], dd[m])
                        if FL == 0:
                            key = np.where(sat, fw - dd[m].min(), 1e6)
                        else:
                            key = np.where(sat, dd[m].max() - lw, 1e6)
                        gap = key.copy()
                    else:
                        key = -nsat.astype(float)
                elif qt == 'before_after' and has_time:
                    m = (dd < lo) if BA != 1 else (dd > hi)
                    nsat = window(m); sat = nsat > 0
                    bound = lo if BA != 1 else hi
                    dx = np.abs(dd - bound).astype(float)
                    gp = np.full(len(uo), 1e6); np.minimum.at(gp, inv[m], dx[m]) if m.any() else None
                    gap = gp
                    dist = np.where(sat, 0, np.full(len(uo), 1e6))
                    dmin = np.full(len(uo), 1e6); np.minimum.at(dmin, inv, dx); dist = np.where(sat, 0, dmin)
                    key = -nsat.astype(float)
                elif qt == 'first_last' or (qt == 'equal_multi' and not anchored):
                    if FL == 1:
                        key = (dd.max() - last).astype(float)
                    else:
                        key = (first - dd.min()).astype(float)
                    sat = key == 0
                    nsat = np.where(sat, cnt, 0)
                    gap = key.copy()
                elif anchored:
                    if t0f is None:
                        sat = np.zeros(len(uo), bool)
                        key = -cnt.astype(float)
                    else:
                        excl = uo == anchor_ent
                        for which, t0 in ((0, t0f), (1, t0l)):
                            if qt == 'after_first':
                                m = dd > t0
                                fa = np.full(len(uo), 10 ** 6); np.minimum.at(fa, inv[m], dd[m]) if m.any() else None
                                s_ = (fa < 10 ** 6) & ~excl
                                k_ = np.where(s_, fa - t0, 1e6).astype(float)
                                if s_.any():
                                    k_ = np.where(s_, k_ - k_[s_].min(), 1e6)
                                ss = s_ & (k_ == 0)
                            elif qt == 'before_last':
                                m = dd < t0
                                la = np.full(len(uo), -1); np.maximum.at(la, inv[m], dd[m]) if m.any() else None
                                s_ = (la >= 0) & ~excl
                                k_ = np.where(s_, t0 - la, 1e6).astype(float)
                                if s_.any():
                                    k_ = np.where(s_, k_ - k_[s_].min(), 1e6)
                                ss = s_ & (k_ == 0)
                            elif qt == 'before_after':
                                m = (dd < t0) if BA != 1 else (dd > t0)
                                ns = window(m)
                                ss = (ns > 0) & ~excl
                                k_ = -ns.astype(float)
                            else:  # equal_multi, same granularity as anchor
                                ga = gran_key(np.array([t0]), g_same)[0]
                                am_days = dd[o == anchor_ent]
                                gks = set(gran_key(am_days, g_same).tolist())
                                m = np.isin(gran_key(dd, g_same), list(gks))
                                ns = window(m)
                                ss = (ns > 0) & ~excl
                                k_ = -ns.astype(float)
                            if which == 0:
                                sat, key = ss, k_
                                nsat = np.where(ss, cnt, 0)
                                gap = np.where(k_ < 1e6, k_, 1e6) if qt in ('after_first', 'before_last') else gap
                            else:
                                sat2, key2 = ss, k_
                else:
                    sat = np.ones(len(uo), bool); nsat = cnt.copy(); key = -cnt.astype(float)
                # position rank of each candidate under this hypothesis (0 = best by operator key)
                order = np.lexsort((-cnt, key))
                pos = np.empty(len(uo), int); pos[order] = np.arange(len(uo))
                order2 = np.lexsort((-cnt, key2))
                pos2 = np.empty(len(uo), int); pos2[order2] = np.arange(len(uo))
                hyp_top = (ar == 0 and rr == 0 and swapped == 0)
                for j in range(len(uo)):
                    c = int(uo[j])
                    if c == et:
                        continue
                    f = cand.get(c)
                    if f is None:
                        f = dict(sat_top=0, sat_any=0, best_rank=KREL, best_relp=0.0, sum_relp_sat=0.0, relp_conn=0.0,
                                 alt_rank=KALT, alt_eff_ratio=0.0, n_sat=0, n_conn=0, pos=10 ** 4, pos_rel=1.0,
                                 gap=1e6, dist=1e6, anchor_found=0, role_swapped=1, first_day=0, last_day=0,
                                 dir_top=0, anchor_alt=KALT, sat_anchor_last=0, pos_anchor_last=10 ** 4,
                                 rule=(1, KREL, KALT, 1, 1e9, 0, 0), topic=et)
                        cand[c] = f
                    s = bool(sat[j])
                    f['relp_conn'] = max(f['relp_conn'], rp)
                    f['n_conn'] = max(f['n_conn'], int(cnt[j]))
                    if s:
                        f['sat_any'] = 1
                        f['sum_relp_sat'] += rp if ar == 0 else 0.0
                        if hyp_top:
                            f['sat_top'] = 1
                    if sat2[j]:
                        f['sat_anchor_last'] = 1
                    f['pos_anchor_last'] = min(f['pos_anchor_last'], int(pos2[j]))
                    better = (not s, rr, ar, swapped) < (f['rule'][0], f['rule'][1], f['rule'][2], f['role_swapped'])
                    if better:
                        f['best_rank'] = rr if s else KREL
                        f['best_relp'] = rp if s else 0.0
                        f['alt_rank'] = ar
                        f['alt_eff_ratio'] = eff / eff0
                        f['n_sat'] = int(nsat[j])
                        f['pos'] = int(pos[j]); f['pos_rel'] = pos[j] / max(1, len(uo))
                        f['gap'] = float(gap[j]); f['dist'] = float(dist[j])
                        f['anchor_found'] = int(t0f is not None)
                        f['anchor_alt'] = anc_alt
                        f['role_swapped'] = swapped
                        f['first_day'] = int(first[j]); f['last_day'] = int(last[j])
                        f['dir_top'] = d
                        f['topic'] = et
                        f['rule'] = (int(not s), rr, ar, swapped, float(key[j]), -int(cnt[j]), c)
                        f['r'], f['d'] = r, d
                # provenance / retrieval: satisfying facts first
                for j_, fid in enumerate(fi[:400]):
                    c = int(o[j_])
                    retrieved.append(((rr, ar, swapped, 0 if sat[np.searchsorted(uo, c)] else 1), int(fid)))
    return cand, retrieved, qents, margin, rel1


def time_candidates(rec, info, qtext):
    kg = _kg
    ments = rec['ments']
    g = info['level']
    FL = info['FL']
    qt = info['qt']
    cand = {}
    retrieved = []
    rel1 = info['rels'][0][2] if info['rels'] else 0.0
    margin = rel1 - (info['rels'][1][2] if len(info['rels']) > 1 else 0.0)
    pairs = []
    for i in range(len(ments)):
        for j in range(i + 1, len(ments)):
            pairs.append((i, j))
    pairs = pairs[:3]
    single = len(ments) == 1
    for pi, (i, j) in enumerate(pairs if pairs else ([(0, None)] if single else [])):
        ialts = ments[i][2][:KALT]
        jalts = ments[j][2][:KALT] if j is not None else [(None, 1.0, 0, 0)]
        eff0 = max(ialts[0][1], 1e-6) + (max(jalts[0][1], 1e-6) if j is not None else 0)
        for ai_, (ei, effi, _, _) in enumerate(ialts):
            fe = kg.facts_of(ei)
            for aj_, (ej, effj, _, _) in enumerate(jalts):
                ar = ai_ + aj_
                if ej is not None:
                    mp = ((kg.h[fe] == ei) & (kg.t[fe] == ej)) | ((kg.h[fe] == ej) & (kg.t[fe] == ei))
                    ff = fe[mp]
                else:
                    ff = fe
                if not len(ff):
                    continue
                pair_all_keys = gran_key(kg.d[ff], g)
                for rr, (r, d, rp) in enumerate(info['rels']):
                    m = kg.r[ff] == r
                    if ej is not None:
                        # d=1: first-mentioned entity is head
                        m &= (kg.h[ff] == ei) if d == 1 else (kg.h[ff] == ej)
                    fr = ff[m]
                    if not len(fr):
                        continue
                    ks = gran_key(kg.d[fr], g)
                    uk, cnt = np.unique(ks, return_counts=True)
                    if qt == 'first_last' or FL >= 0 and qt != 'equal':
                        if FL == 1:
                            key = (uk.max() - uk).astype(float)
                        else:
                            key = (uk - uk.min()).astype(float)
                        sat = key == 0
                    else:
                        sat = np.ones(len(uk), bool)
                        key = -cnt.astype(float)
                    order = np.lexsort((-cnt, key))
                    pos = np.empty(len(uk), int); pos[order] = np.arange(len(uk))
                    hyp_top = (ar == 0 and rr == 0 and pi == 0)
                    for t_ in range(len(uk)):
                        c = int(uk[t_])
                        f = cand.get(c)
                        if f is None:
                            f = dict(sat_top=0, sat_any=0, best_rank=KREL, best_relp=0.0, sum_relp_sat=0.0, relp_conn=0.0,
                                     alt_rank=2 * KALT, alt_eff_ratio=0.0, n_sat=0, n_conn=0, pos=10 ** 4, pos_rel=1.0,
                                     gap=1e6, dist=0, anchor_found=int(ej is not None), role_swapped=pi,
                                     first_day=0, last_day=0, dir_top=0, anchor_alt=0, sat_anchor_last=0,
                                     pos_anchor_last=0, rule=(1, KREL, 2 * KALT, 9, 1e9, 0, 0))
                            cand[c] = f
                        s = bool(sat[t_])
                        f['relp_conn'] = max(f['relp_conn'], rp)
                        f['n_conn'] = max(f['n_conn'], int(cnt[t_]))
                        if s:
                            f['sat_any'] = 1
                            f['sum_relp_sat'] += rp if ar == 0 else 0.0
                            if hyp_top:
                                f['sat_top'] = 1
                        rk = (int(not s), rr, ar, pi, float(key[t_]), -int(cnt[t_]), c)
                        if rk < f['rule']:
                            f['rule'] = rk
                            f['best_rank'] = rr if s else KREL
                            f['best_relp'] = rp if s else 0.0
                            f['alt_rank'] = ar
                            f['alt_eff_ratio'] = (effi + (effj if ej is not None else 0)) / eff0
                            f['n_sat'] = int(cnt[t_]) if s else 0
                            f['pos'] = int(pos[t_]); f['pos_rel'] = pos[t_] / max(1, len(uk))
                            f['gap'] = float(key[t_]) if qt == 'first_last' else 0.0
                            f['dir_top'] = d
                            f['role_swapped'] = pi
                            f['anchor_found'] = int(ej is not None)
                            f['r'], f['d'] = r, d
                            f['n_pair_all'] = int((pair_all_keys == c).sum())
                    for fid in fr[:200]:
                        retrieved.append(((rr, ar, pi, 0), int(fid)))
                # fallback: any relation between the pair (no relation hypothesis matched)
                if ej is not None:
                    uk, cnt = np.unique(pair_all_keys, return_counts=True)
                    for t_ in range(len(uk)):
                        c = int(uk[t_])
                        if c in cand:
                            cand[c]['n_pair_all'] = max(cand[c].get('n_pair_all', 0), int(cnt[t_]))
                            continue
                        cand[c] = dict(sat_top=0, sat_any=0, best_rank=KREL, best_relp=0.0, sum_relp_sat=0.0,
                                       relp_conn=0.0, alt_rank=ar, alt_eff_ratio=0.0, n_sat=0, n_conn=0, pos=10 ** 4,
                                       pos_rel=1.0, gap=1e6, dist=0, anchor_found=1, role_swapped=pi, first_day=0,
                                       last_day=0, dir_top=0, anchor_alt=0, sat_anchor_last=0, pos_anchor_last=0,
                                       n_pair_all=int(cnt[t_]), rule=(1, KREL + 1, ar, pi, -float(cnt[t_]), 0, c))
                    for fid in ff[:100]:
                        retrieved.append(((KREL, ar, pi, 1), int(fid)))
    return cand, retrieved, margin, rel1


def prior_candidates(rec, info):
    """PRIOR arm: co-occurrence frequency (entity) / date frequency among linked entities (time)."""
    kg = _kg
    ents = [m[2][0][0] for m in rec['ments'] if m[2]]
    if info['atype'] == 'entity':
        c = collections.Counter()
        for e in ents:
            f = kg.facts_of(e)
            oth = np.where(kg.h[f] == e, kg.t[f], kg.h[f])
            u, n = np.unique(oth, return_counts=True)
            for a, b in zip(u.tolist(), n.tolist()):
                if a not in ents:
                    c[a] += b
        return [kg.ent_names[a] for a, _ in sorted(c.items(), key=lambda x: (-x[1], x[0]))[:50]], c
    g = info['level']
    c = collections.Counter()
    es = set(ents)
    if len(ents) >= 2:
        f = kg.facts_of(ents[0])
        oth = np.where(kg.h[f] == ents[0], kg.t[f], kg.h[f])
        f = f[np.isin(oth, list(es - {ents[0]}))]
    elif ents:
        f = kg.facts_of(ents[0])
    else:
        f = np.zeros(0, int)
    ks = gran_key(kg.d[f], g)
    u, n = np.unique(ks, return_counts=True)
    for a, b in zip(u.tolist(), n.tolist()):
        c[a] += b
    return [key_str(a, g) for a, _ in sorted(c.items(), key=lambda x: (-x[1], x[0]))[:50]], c


def build(rec, probs, qtext):
    """Full candidate set for one question. Returns dict with names, feature matrix, rule order, retrieved facts."""
    kg = _kg
    info = parse_info(rec, probs, qtext)
    prior_list, prior_cnt = prior_candidates(rec, info)
    if info['atype'] == 'entity':
        cand, retrieved, qents, margin, rel1 = entity_candidates(rec, info, qtext)
        names_of = lambda c: kg.ent_names[c]
        is_time = 0
    else:
        cand, retrieved, margin, rel1 = time_candidates(rec, info, qtext)
        qents = set()
        names_of = lambda c: key_str(c, info['level'])
        is_time = 1
    keys = list(cand)
    # cap pool: rule order
    keys.sort(key=lambda c: cand[c]['rule'])
    keys = keys[:POOL_CAP]
    nsat = sum(1 for c in keys if cand[c]['sat_any'])
    X = np.zeros((len(keys), len(FEATS)), np.float32)
    pmax = max(prior_cnt.values()) if prior_cnt else 1
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
            row[FIDX['log_deg']] = math.log1p(kg.degree[c])
            row[FIDX['first_day']] = f['first_day'] / kg.ndays
            row[FIDX['last_day']] = f['last_day'] / kg.ndays
            if 'topic' in f and f['topic'] is not None:
                pass
        row[FIDX['log_prior']] = math.log1p(prior_cnt.get(c, 0))
        row[FIDX['q_ncand_sat']] = math.log1p(nsat)
        row[FIDX['qt_p']] = info['qt_p']; row[FIDX['atype_p']] = info['atype_p']; row[FIDX['level_p']] = info['level_p']
        row[FIDX['n_ment']] = len(rec['ments'])
        row[FIDX['rel1_p']] = rel1; row[FIDX['rel_margin']] = margin
    # n_pair_all for entity answers: facts between primary topic and candidate (any relation)
    if not is_time and keys:
        tops = {}
        for c in keys:
            t = cand[c].get('topic')
            if t is not None:
                tops.setdefault(t, []).append(c)
        for t, cs in tops.items():
            f = kg.facts_of(t)
            oth = np.where(kg.h[f] == t, kg.t[f], kg.h[f])
            u, n = np.unique(oth, return_counts=True)
            mp = dict(zip(u.tolist(), n.tolist()))
            idx = {c: i for i, c in enumerate(keys)}
            for c in cs:
                X[idx[c], FIDX['n_pair_all']] = mp.get(c, 0)
    # retrieved facts: order by priority, unique
    retrieved.sort(key=lambda x: x[0])
    seen, facts = set(), []
    for _, fid in retrieved:
        if fid not in seen:
            seen.add(fid); facts.append(fid)
        if len(facts) >= 20:
            break
    prov = {}
    names = [names_of(c) for c in keys]
    rinfo = [(cand[c].get('r', -1), cand[c].get('d', -1)) for c in keys]
    return dict(names=names, X=X, info=info, facts=facts, prior=prior_list, rinfo=rinfo,
                topics=[cand[c].get('topic', -1) if not is_time else -1 for c in keys])
