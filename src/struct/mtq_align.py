"""mtq_align.py -- gold-consistent resolution of question entities on TRAIN (silver labels)
and learning of sector-phrase aliases ("the prime minister of <C>" -> "Head of Government").

Gold answers are used here ONLY on the train split (training labels) and on dev for reporting
silver parser accuracy; nothing in this file runs at dev/test inference.
"""
import re, collections, math
import numpy as np
import mtq_lib as L

_kg = None
_lk = None
_ctx = {}


def init(kg, lk):
    global _kg, _lk
    _kg, _lk = kg, lk
    names = kg.ent_names
    SC = {}
    countries = set()
    for i, e in enumerate(names):
        m = re.match(r'^(.*) \((.*)\)$', e)
        if m:
            SC[(m.group(1), m.group(2))] = i
            countries.add(m.group(2))
    cstems = {}
    for c in countries:
        cs = tuple(L.stem(w) for w in L.toks(c) if w not in L.STOP)
        if cs:
            cstems.setdefault(cs, c)
    first = collections.defaultdict(list)
    for cs, c in cstems.items():
        first[cs[0]].append((cs, c))
    _ctx.update(SC=SC, countries=countries, cstems=cstems, cfirst=first)


def country_mentions(ws):
    """Return list of (lo, hi, C) country mentions (by name stems or demonym) in token list ws."""
    st = [L.stem(w) for w in ws]
    idx = [i for i, w in enumerate(ws) if w not in L.STOP]
    seq = [st[i] for i in idx]
    out = []
    for k, s in enumerate(seq):
        for cs, c in _ctx['cfirst'].get(s, ()):
            if tuple(seq[k:k + len(cs)]) == cs:
                out.append((idx[k], idx[k + len(cs) - 1], c))
    for i, w in enumerate(ws):
        if w in L.DEMONYM_MANUAL and L.DEMONYM_MANUAL[w]:
            cs = tuple(L.stem(x) for x in L.toks(L.DEMONYM_MANUAL[w]) if x not in L.STOP)
            c = _ctx['cstems'].get(cs)
            if c:
                out.append((i, i, c))
    # keep longest non-overlapping
    out.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
    sel, used = [], set()
    for lo, hi, c in out:
        if any(j in used for j in range(lo, hi + 1)):
            continue
        sel.append((lo, hi, c))
        used.update(range(lo, hi + 1))
    return sorted(sel)


def ngram_keys(ws, lo, hi, maxlen=6):
    """n-grams (stemmed, country replaced by <C>) that contain the country mention [lo,hi]."""
    toks = [L.stem(w) for w in ws[:lo]] + ['<C>'] + [L.stem(w) for w in ws[hi + 1:]]
    p = lo
    keys = []
    for a in range(max(0, p - maxlen + 1), p + 1):
        for b in range(p, min(len(toks), a + maxlen)):
            if b - a + 1 < 2:
                continue
            keys.append(tuple(toks[a:b + 1]))
    return keys


def qstems(ws):
    s = set()
    for w in ws:
        if w in L.STOP:
            continue
        s.add(L.stem(w))
        if w in _lk.demonym:
            s.update(_lk.demonym[w])
    return s


def sim(e, qs):
    ss = list(dict.fromkeys(_lk.ent_stems[e]))
    if not ss:
        return 0.0
    tot = sum(_lk.idf[s] for s in ss)
    return sum(_lk.idf[s] for s in ss if s in qs) / tot


def tkey_arr(d, g):
    kg = _kg
    return kg.day_month[d] if g == 'month' else (kg.day_year[d] if g == 'year' else d)


def resolve(q):
    """Gold-consistent resolution of question entities for one train/dev question.
    Returns dict with resolved entity ids (may be None) and whether each was in the linker output."""
    kg, lk = _kg, _lk
    H, R, T, D = kg.h, kg.r, kg.t, kg.d
    (ms, ws), tp = lk.link_question(q['question'])
    E = [a['e'] for m in ms for a in m['alts']]
    Eset = set(E)
    qs = qstems(ws)
    res = dict(quid=q['quid'], linked=[[a['e'] for a in m['alts']] for m in ms], topic=None, partner=None,
               topic_linked=None, partner_linked=None)
    if q['answer_type'] == 'entity':
        A = {kg.e2i.get(a) for a in q['answers']}
        A.discard(None)
        # topic: an entity sharing facts with a gold answer
        best = None
        for e in E:
            f = kg.facts_of(e)
            m = ((H[f] == e) & np.isin(T[f], list(A))) | ((T[f] == e) & np.isin(H[f], list(A)))
            if m.any():
                key = (int(m.sum()), sim(e, qs))
                if best is None or key > best[0]:
                    best = (key, e, f[m])
        if best is None:
            # neighbourhood of gold answers sharing a stem with the question
            nb = collections.Counter()
            for a in A:
                f = kg.facts_of(a)
                oth = np.where(H[f] == a, T[f], H[f])
                for o in oth.tolist():
                    nb[o] += 1
            cand = [(sim(o, qs), c, o) for o, c in nb.items() if o not in A]
            cand = [x for x in cand if x[0] > 0]
            cand.sort(reverse=True)
            if cand and (len(cand) == 1 or cand[0][0] > cand[1][0]):
                res['topic'] = cand[0][2]
                res['topic_linked'] = False
        else:
            res['topic'] = best[1]
            res['topic_linked'] = True
        anchored = q['qtype'] in ('after_first', 'before_last') or (
            q['qtype'] in ('before_after', 'equal_multi') and tp is None)
        t = res['topic']
        if anchored and t is not None:
            f = kg.facts_of(t)
            mg = ((H[f] == t) & np.isin(T[f], list(A))) | ((T[f] == t) & np.isin(H[f], list(A)))
            rels = set(R[f[mg]].tolist())
            dirs = set(np.where(H[f[mg]] == t, 1, 0).tolist())
            mr = np.isin(R[f], list(rels))
            oth = np.where(H[f] == t, T[f], H[f])
            isdir = np.where(H[f] == t, 1, 0)
            mm = mr & np.isin(isdir, list(dirs)) & ~np.isin(oth, list(A)) & (oth != t)
            partners = set(oth[mm].tolist())
            inE = [e for e in E if e in partners]
            if inE:
                res['partner'] = max(inE, key=lambda e: sim(e, qs))
                res['partner_linked'] = True
            else:
                cand = sorted(((sim(o, qs), o) for o in partners), reverse=True)
                cand = [x for x in cand if x[0] > 0]
                if cand and (len(cand) == 1 or cand[0][0] > cand[1][0]):
                    res['partner'] = cand[0][1]
                    res['partner_linked'] = False
    else:
        g = q['time_level']
        gold = q['answers'][0]
        if g == 'year':
            gk = int(gold); keyarr = kg.day_year
        elif g == 'month':
            gk = int(gold[:4]) * 12 + int(gold[5:7]) - 1; keyarr = kg.day_month
        else:
            gk = kg.day_str.index(gold) if gold in kg.day_str else -1; keyarr = np.arange(kg.ndays)
        # pairs in linker output with a fact at gold date
        found = None
        for i1, e1 in enumerate(E):
            f = kg.facts_of(e1)
            ok = keyarr[D[f]] == gk
            if not ok.any():
                continue
            ff = f[ok]
            oth = np.where(H[ff] == e1, T[ff], H[ff])
            for e2 in E:
                if e2 != e1 and (oth == e2).any():
                    key = (sim(e1, qs) + sim(e2, qs))
                    if found is None or key > found[0]:
                        found = (key, e1, e2)
        if found:
            res['topic'], res['partner'] = found[1], found[2]
            res['topic_linked'] = res['partner_linked'] = True
        else:
            # one linked entity + best partner at gold date
            best = None
            for e1 in E:
                f = kg.facts_of(e1)
                if not len(f):
                    continue
                ok = keyarr[D[f]] == gk
                ff = f[ok]
                oth = set(np.where(H[ff] == e1, T[ff], H[ff]).tolist()) - {e1}
                cand = sorted(((sim(o, qs), o) for o in oth), reverse=True)
                cand = [x for x in cand if x[0] > 0]
                if cand and (len(cand) == 1 or cand[0][0] > cand[1][0]):
                    key = cand[0][0] + sim(e1, qs)
                    if best is None or key > best[0]:
                        best = (key, e1, cand[0][1])
            if best:
                res['topic'], res['partner'] = best[1], best[2]
                res['topic_linked'], res['partner_linked'] = True, False
    return res, ws, ms, tp


def tspan(tp):
    """(lo_day, hi_day) inclusive day-index span of a parsed time (level, str, span)."""
    import datetime as dt
    lvl, s = tp[0], tp[1]
    d0 = dt.date(2005, 1, 1)
    if lvl == 'year':
        lo, hi = dt.date(int(s), 1, 1), dt.date(int(s), 12, 31)
    elif lvl == 'month':
        y, m = map(int, s.split('-'))
        lo = dt.date(y, m, 1)
        hi = dt.date(y + (m == 12), m % 12 + 1, 1) - dt.timedelta(days=1)
    else:
        lo = hi = dt.date.fromisoformat(s)
    return (lo - d0).days, (hi - d0).days


def silver(q):
    """Silver labels for one question (train: training labels; dev: reporting only).
    rel_labels: list of (rel_id, dir) consistent with gold; dir=1 means topic is head (entity answers)
    or first-mentioned entity is head (time answers; -1 if order unknown).
    fl: 'first'/'last' (from gold position), ba: 'before'/'after' (from gold facts)."""
    kg = _kg
    H, R, T, D = kg.h, kg.r, kg.t, kg.d
    res, ws, ms, tp = resolve(q)
    out = dict(res=res, rel_labels=[], strong=False, fl=None, ba=None, tp=tp)
    t, p = res['topic'], res['partner']
    if t is None:
        return out, ws, ms
    qt = q['qtype']
    if q['answer_type'] == 'entity':
        A = np.array(sorted({kg.e2i[a] for a in q['answers'] if a in kg.e2i}), dtype=np.int32)
        f = kg.facts_of(t)
        isH = H[f] == t
        oth = np.where(isH, T[f], H[f])
        mg = np.isin(oth, A)
        cons, weak = set(), set()
        BA = L.kw_ba(q['question']); FL = L.kw_fl(q['question'])
        for r_, dh in set(zip(R[f[mg]].tolist(), isH[mg].astype(int).tolist())):
            weak.add((r_, dh))
            sel = (R[f] == r_) & (isH.astype(int) == dh)
            d_all, o_all = D[f[sel]], oth[sel]
            g_mask = np.isin(o_all, A)
            ok = False
            if tp is not None and qt in ('equal', 'equal_multi'):
                lo, hi = tspan(tp)
                ins = (d_all >= lo) & (d_all <= hi)
                ok = bool((g_mask & ins).any())
                if ok and qt == 'equal_multi' and FL >= 0:
                    dd = d_all[ins]
                    t_ = dd.min() if FL == 0 else dd.max()
                    ok = bool((g_mask & ins & (d_all == t_)).any())
            elif tp is not None and qt == 'before_after':
                lo, hi = tspan(tp)
                ok = bool((g_mask & ((d_all < lo) if BA == 0 else (d_all > hi))).any())
            elif qt == 'first_last':
                if len(d_all):
                    t_ = d_all.min() if FL == 0 else d_all.max()
                    ok = bool((g_mask & (d_all == t_)).any())
            elif p is not None:
                pm = o_all == p
                ok = False
                for t0 in sorted(set(d_all[pm].tolist()))[:3]:
                    if qt == 'after_first':
                        mm_ = (d_all > t0) & (o_all != p)
                        ok = bool(mm_.any()) and bool((g_mask & mm_ & (d_all == d_all[mm_].min())).any())
                    elif qt == 'before_last':
                        mm_ = (d_all < t0) & (o_all != p)
                        ok = bool(mm_.any()) and bool((g_mask & mm_ & (d_all == d_all[mm_].max())).any())
                    elif qt == 'before_after':
                        ok = bool((g_mask & ((d_all < t0) if BA == 0 else (d_all > t0))).any())
                    else:
                        ok = True
                    if ok:
                        break
            else:
                ok = True
            if ok:
                cons.add((r_, dh))
        out['rel_labels'] = sorted(cons) if cons else sorted(weak)
        out['strong'] = bool(cons)
    else:
        if p is None:
            return out, ws, ms
        g = q['time_level']; gold = q['answers'][0]
        if g == 'year':
            gk = int(gold); keyarr = kg.day_year
        elif g == 'month':
            gk = int(gold[:4]) * 12 + int(gold[5:7]) - 1; keyarr = kg.day_month
        else:
            gk = kg.day_str.index(gold); keyarr = np.arange(kg.ndays)
        # textual order of topic / partner
        pos = {}
        for m in ms:
            for a in m['alts']:
                pos.setdefault(a['e'], m['lo'])
        first_is_t = None
        if t in pos and p in pos:
            first_is_t = pos[t] <= pos[p]
        f = kg.facts_of(t)
        mp = ((H[f] == t) & (T[f] == p)) | ((H[f] == p) & (T[f] == t))
        ff = f[mp]
        cons = set()
        for r_ in set(R[ff].tolist()):
            for th in (1, 0):
                sel = ff[(R[ff] == r_) & ((H[ff] == t) == bool(th))]
                if not len(sel):
                    continue
                keys = keyarr[D[sel]]
                if not (keys == gk).any():
                    continue
                if qt == 'first_last':
                    FL = L.kw_fl(q['question'])
                    if (keys.min() if FL != 1 else keys.max()) != gk:
                        continue
                dirv = -1 if first_is_t is None else int(th == int(first_is_t))
                cons.add((r_, dirv))
        out['rel_labels'] = sorted(cons)
        out['strong'] = bool(cons)
    return out, ws, ms


def alias_evidence(q):
    """Return (denominator keys, numerator (key, S) pairs) for one train question."""
    res, ws, ms, tp = resolve(q)
    if tp is not None:
        a, b = tp[2]
        qq = q['question'][:a] + ' ' * (b - a) + q['question'][b:]
        ws = L.toks(qq)
    cms = country_mentions(ws)
    den, num = [], []
    resolved = [e for e in (res['topic'], res['partner']) if e is not None]
    rs = []
    for e in resolved:
        m = re.match(r'^(.*) \((.*)\)$', _kg.ent_names[e])
        if m:
            rs.append((m.group(1), m.group(2)))
    # positions covered by linker mentions that contain each resolved entity
    spans = {}
    for m in ms:
        for a in m['alts']:
            spans.setdefault(a['e'], set()).update(range(m['lo'], m['hi'] + 1))
    for lo, hi, c in cms:
        keys = ngram_keys(ws, lo, hi)
        den.extend(keys)
        for e in resolved:
            mm = re.match(r'^(.*) \((.*)\)$', _kg.ent_names[e])
            if not mm or mm.group(2) != c:
                continue
            # skip if this country mention lies inside the span of ANOTHER resolved entity
            other = [o for o in resolved if o != e]
            if any(lo in spans.get(o, ()) for o in other):
                continue
            # if e itself was linked, only attribute to the mention inside its own span
            if e in spans and lo not in spans[e]:
                continue
            num.extend((k, mm.group(1)) for k in keys)
    return res, den, num
