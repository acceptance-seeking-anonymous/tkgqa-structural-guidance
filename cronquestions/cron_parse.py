"""cron_parse.py -- frozen-parser stage 1: slot segmentation + entity alternatives + time parsing.

OWN regime: input = paraphrases[0] text only.
GOLD-ANN regime: input = the id-substituted 'question' text + gold 'entities' + gold 'times'
(mention positions from the Q-ids; slot roles from the same train-template segmenter).
No dev/test annotation, relation, template, type, answer_type or answers are read here."""
import math, collections, re
import numpy as np
from cron_text import norm, YEAR_RE, QID_RE, TOK_RE, tokens, ws

KALT = 5


class Parser:
    def __init__(self, kg, surfaces, templates):
        self.kg, self.sf, self.tp = kg, surfaces, templates
        self.r2i = kg.r2i

    # --- slot compatibility with the template frame relation (KG-only check) ---
    def role_ok(self, e, role, rel):
        kg = self.kg
        r = self.r2i[rel]
        if role == 'event_head':
            return e in kg.event_iv
        if role == 'head':
            return (e, r) in kg.sr
        if role in ('tail', 'tail2'):
            return (e, r) in kg.orr
        return True

    def rank_alts(self, alts, role, rel):
        """alts: [(e, is_label)] -> sorted [(e, is_label, role_ok)]"""
        kg = self.kg
        out = [(e, il, int(self.role_ok(e, role, rel))) for e, il in alts]
        out.sort(key=lambda x: (-x[2], -x[1], -int(kg.degree[x[0]]), x[0]))
        return out

    def hypotheses(self, text, gold=None, maxh=3):
        """Return ranked segmentation hypotheses.  gold = (entities set, times set) for GOLD-ANN."""
        kg, tp = self.kg, self.tp
        text = ws(text)
        hyps = []
        for i in tp.candidates(text):
            it = tp.items[i]
            rel = it['frame']['rel']
            for sp in tp.splits(i, text):
                slots = []
                bad = False
                n_unl = n_inc = 0
                for role, (a, b) in zip(it['slots'], sp):
                    f = text[a:b].strip()
                    if role == 'time':
                        if not YEAR_RE.match(f):
                            bad = True; break
                        slots.append(dict(role=role, span=(a, b), text=f, year=int(f), alts=[]))
                        continue
                    if gold is not None:
                        alts = [(kg.e2i[f], 1)] if (QID_RE.match(f) and f in kg.e2i) else []
                    else:
                        alts = self.sf.lookup(f)
                    if not alts:
                        n_unl += 1
                        slots.append(dict(role=role, span=(a, b), text=f, alts=[]))
                        continue
                    ra = self.rank_alts(alts, role, rel)
                    if not ra[0][2]:
                        n_inc += 1
                    slots.append(dict(role=role, span=(a, b), text=f, alts=ra[:KALT], n_alts=len(ra)))
                if bad:
                    continue
                # pairwise KG consistency (head, rel, tail) for the top alternatives
                pair = 0
                d = {s['role']: s for s in slots}
                if 'head' in d and 'tail' in d and d['head']['alts'] and d['tail']['alts']:
                    r = self.r2i[rel]
                    for eh, _, _ in d['head']['alts'][:3]:
                        fs = kg.subj_facts(eh, r)
                        tails = set(kg.T[fs].tolist())
                        if any(et in tails for et, _, _ in d['tail']['alts'][:3]):
                            pair = 1; break
                key = (n_unl, n_inc, -pair, -it['nlit'], -it['n'], i)
                hyps.append(dict(key=key, tid=i, slots=slots, n_unl=n_unl, n_inc=n_inc, pair=pair))
        hyps.sort(key=lambda h: h['key'])
        # de-duplicate identical segmentations (same spans+roles)
        out, seen = [], set()
        for h in hyps:
            sig = tuple((s['role'], s['span']) for s in h['slots'])
            if sig in seen:
                continue
            seen.add(sig); out.append(h)
            if len(out) >= maxh:
                break
        return out

    def fallback(self, text):
        """dictionary longest-match linker (used only when no template skeleton matches)"""
        kg, sf = self.kg, self.sf
        toks = list(TOK_RE.finditer(text))
        n = len(toks)
        cands = []
        for i in range(n):
            for j in range(min(n, i + sf.maxlen), i, -1):
                a, b = toks[i].start(), toks[j - 1].end()
                s = text[a:b]
                if not any(c.isupper() or c.isdigit() for c in s):
                    continue
                alts = sf.lookup(s)
                if alts:
                    cands.append((-(j - i), a, b, s, alts)); break
        cands.sort()
        used = []; slots = []
        for _, a, b, s, alts in cands:
            if any(not (b <= x or a >= y) for x, y in used):
                continue
            if YEAR_RE.match(s):
                continue
            used.append((a, b))
            slots.append(dict(role='unk', span=(a, b), text=s, alts=[(e, il, 1) for e, il in alts[:KALT]],
                              n_alts=len(alts)))
        for m in re.finditer(r'\b(\d{3,4})\b', text):
            if not any(x <= m.start() < y for x, y in used):
                slots.append(dict(role='time', span=(m.start(), m.end()), text=m.group(1), year=int(m.group(1)), alts=[]))
        slots.sort(key=lambda s: s['span'])
        return dict(key=(9,), tid=-1, slots=slots, n_unl=0, n_inc=0, pair=0)

    def parse(self, text, gold=None):
        hs = self.hypotheses(text, gold)
        if not hs or hs[0]['n_unl'] == len([s for s in hs[0]['slots'] if s['role'] != 'time']) and hs[0]['slots']:
            fb = self.fallback(text) if gold is None else None
            if fb is not None and fb['slots']:
                hs = hs + [fb] if hs else [fb]
        return hs


def masked_text(text, hyp):
    """entity spans -> ENT, time spans -> TIME, lower-cased word/punct tokens"""
    out, pos = [], 0
    for s in sorted(hyp['slots'], key=lambda s: s['span']):
        a, b = s['span']
        out += [t.lower() for t in tokens(text[pos:a])]
        out.append('TIME' if s['role'] == 'time' else 'ENT')
        pos = b
    out += [t.lower() for t in tokens(text[pos:])]
    return out


# ---------------------------------------------------------------------------
# learned alternative ranking (features from the KG only; weights fitted on TRAIN)
# ---------------------------------------------------------------------------
ALT_FEATS = ['is_label', 'log_deg', 'deg_rank', 'n_alts', 'role_ok', 'log_nrel', 'pair_ok', 'log_npair',
             'time_ok', 'event_ok', 'is_event', 'other_amb', 'label_rank']
NALT_FEAT = 10


def alt_feature_rows(kg, sf, hyp, rel):
    """rows for every (entity slot, alternative) of a hypothesis; uses the full alt lists (<= NALT_FEAT)."""
    r = kg.r2i[rel]
    ent_slots = [s for s in hyp['slots'] if s['role'] not in ('time', 'unk') and s.get('all_alts')]
    years = [s['year'] for s in hyp['slots'] if s['role'] == 'time']
    rows, keys = [], []
    for s in ent_slots:
        alts = s['all_alts'][:NALT_FEAT]
        degs = [int(kg.degree[e]) for e, _ in alts]
        order = np.argsort(-np.array(degs), kind='stable')
        drank = np.empty(len(alts), int); drank[order] = np.arange(len(alts))
        others = [o for o in ent_slots if o is not s]
        n_lab = 0
        for k, (e, il) in enumerate(alts):
            role = s['role']
            if role == 'head':
                fs = kg.subj_facts(e, r); other_col = kg.T
            elif role in ('tail', 'tail2'):
                fs = kg.obj_facts(e, r); other_col = kg.H
            else:
                fs = np.zeros(0, np.int32); other_col = kg.T
            pair_ok, npair = 0, 0
            if role in ('head', 'tail') and len(fs):
                partner = [o for o in others if o['role'] in ('head', 'tail') and o['role'] != role]
                if partner:
                    pset = {pe for pe, _ in partner[0]['all_alts'][:NALT_FEAT]}
                    m = np.isin(other_col[fs], list(pset))
                    npair = int(m.sum()); pair_ok = int(npair > 0)
            time_ok = 0
            if years and len(fs):
                y = years[0]
                time_ok = int(((kg.S[fs] <= y) & (kg.E[fs] >= y)).any())
            is_event = int(e in kg.event_iv)
            row = [il, math.log1p(degs[k]), drank[k], len(s['all_alts']), int(len(fs) > 0) if role != 'event_head' else is_event,
                   math.log1p(len(fs)), pair_ok, math.log1p(npair), time_ok, is_event if role == 'event_head' else 0,
                   is_event, max([len(o['all_alts']) for o in others], default=0), n_lab if il else -1]
            if il:
                n_lab += 1
            rows.append(row); keys.append((id(s), k))
    return np.array(rows, np.float32).reshape(-1, len(ALT_FEATS)), keys, ent_slots


def attach_all_alts(kg, sf, hyp, gold=False):
    for s in hyp['slots']:
        if s['role'] in ('time',):
            continue
        if gold:
            s['all_alts'] = [(a[0], a[1]) for a in s['alts']]
        else:
            s['all_alts'] = [(e, il) for e, il in sf.lookup(s['text'])] if s['text'] else []


def rerank_alts(kg, sf, hyp, rel, model, gold=False):
    """re-order each slot's alternatives by the learned scorer; keep KALT; store scores."""
    attach_all_alts(kg, sf, hyp, gold)
    X, keys, ent_slots = alt_feature_rows(kg, sf, hyp, rel)
    if len(X) == 0:
        for s in hyp['slots']:
            s.pop('all_alts', None)
        return
    p = model.predict(X) if model is not None else -X[:, 2]
    by = collections.defaultdict(list)
    for (sid, k), sc in zip(keys, p):
        by[sid].append((float(sc), k))
    for s in ent_slots:
        sc = sorted(by[id(s)], key=lambda x: (-x[0], x[1]))
        alts = s['all_alts']
        s['alts'] = [(alts[k][0], alts[k][1], round(v, 4)) for v, k in sc[:KALT]]
    for s in hyp['slots']:
        s.pop('all_alts', None)
