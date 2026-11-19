"""mtq_lib.py -- MultiTQ structural QA: KG index, text normalisation, time parsing, entity linker.

Inputs at dev/test: question text + MultiTQ KG only.
"""
import re, json, collections, unicodedata, datetime, math
import numpy as np

import os
DATA = os.environ.get('DATA_DIR', 'mtq/MultiTQ_94ad61c/data/MultiTQ')

# ---------------------------------------------------------------- KG
class KG:
    def __init__(self, root=DATA):
        rows = []
        for fn in ['train.txt', 'valid.txt', 'test.txt']:
            for line in open(f'{root}/kg/{fn}', encoding='utf-8'):
                p = line.rstrip('\n').split('\t')
                rows.append(p)
        ents = sorted({r[0] for r in rows} | {r[2] for r in rows})
        rels = sorted({r[1] for r in rows})
        self.ent_names = [e.replace('_', ' ') for e in ents]
        self.e2i = {e: i for i, e in enumerate(self.ent_names)}
        self.rel_names = rels
        self.r2i = {r: i for i, r in enumerate(rels)}
        d0 = datetime.date(2005, 1, 1)
        self.ndays = (datetime.date(2015, 12, 31) - d0).days + 1
        self.day_str = [(d0 + datetime.timedelta(days=i)).isoformat() for i in range(self.ndays)]
        ds2i = {s: i for i, s in enumerate(self.day_str)}
        self.h = np.array([self.e2i[r[0].replace('_', ' ')] for r in rows], dtype=np.int32)
        self.r = np.array([self.r2i[r[1]] for r in rows], dtype=np.int32)
        self.t = np.array([self.e2i[r[2].replace('_', ' ')] for r in rows], dtype=np.int32)
        self.d = np.array([ds2i[r[3]] for r in rows], dtype=np.int32)
        # order facts by day so per-entity lists are chronological
        o = np.lexsort((self.t, self.r, self.h, self.d))
        self.h, self.r, self.t, self.d = self.h[o], self.r[o], self.t[o], self.d[o]
        self.n = len(self.h)
        # month / year index of day
        self.day_month = np.array([int(s[:4]) * 12 + int(s[5:7]) - 1 for s in self.day_str], dtype=np.int32)
        self.day_year = np.array([int(s[:4]) for s in self.day_str], dtype=np.int32)
        # entity -> fact ids (chronological)
        by_e = collections.defaultdict(list)
        for i in range(self.n):
            by_e[int(self.h[i])].append(i)
            if self.t[i] != self.h[i]:
                by_e[int(self.t[i])].append(i)
        self.by_e = {k: np.array(v, dtype=np.int32) for k, v in by_e.items()}
        self.degree = np.zeros(len(self.ent_names), dtype=np.int32)
        for k, v in self.by_e.items():
            self.degree[k] = len(v)
        # relation verbalisation
        self.rel_text = [r.replace('_', ' ') for r in rels]

    def facts_of(self, e):
        return self.by_e.get(e, np.zeros(0, dtype=np.int32))

    def quad(self, i):
        return (self.ent_names[self.h[i]], self.rel_names[self.r[i]], self.ent_names[self.t[i]], self.day_str[self.d[i]])

    def verbalise(self, i):
        return f"{self.ent_names[self.h[i]]} | {self.rel_text[self.r[i]]} | {self.ent_names[self.t[i]]} | {self.day_str[self.d[i]]}"


# ---------------------------------------------------------------- normalisation
def strip_acc(s):
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))

STOP = {'the', 'of', 'and', 'a', 'an', 's', 'with', 'to', 'for', 'in', 'on', 'by', 'at', 'from', 'as'}

def toks(s):
    s = strip_acc(s).lower()
    s = s.replace("’", "'")
    s = re.sub(r"'s\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.split()

def stem(w):
    if len(w) >= 7:
        return w[:7]
    if len(w) > 3 and w.endswith('ies'):
        return w[:-3] + 'y'
    if len(w) > 3 and w.endswith('s') and not w.endswith('ss') and not w.endswith('us'):
        return w[:-1]
    return w

# Demonym exceptions the prefix rule cannot catch (manual, generic English knowledge, no data used)
DEMONYM_MANUAL = {
    'danish': 'denmark', 'dane': 'denmark', 'british': 'united kingdom', 'briton': 'united kingdom', 'uk': 'united kingdom',
    'english': 'united kingdom', 'french': 'france', 'frenchman': 'france', 'dutch': 'netherlands', 'swiss': 'switzerland',
    'american': 'united states', 'us': 'united states', 'usa': 'united states', 'u': None,
    'spanish': 'spain', 'spaniard': 'spain', 'finnish': 'finland', 'finn': 'finland', 'polish': 'poland', 'pole': 'poland',
    'burmese': 'myanmar', 'ivorian': 'cote d ivoire', 'emirati': 'united arab emirates', 'irish': 'ireland',
    'cypriot': 'cyprus', 'filipino': 'philippines', 'philippine': 'philippines', 'greek': 'greece', 'welsh': 'united kingdom',
    'scottish': 'united kingdom', 'kiwi': 'new zealand', 'zealander': 'new zealand', 'malagasy': 'madagascar',
    'peruvian': 'peru', 'argentine': 'argentina', 'argentinian': 'argentina', 'thai': 'thailand', 'lao': 'laos',
    'laotian': 'laos', 'congolese': 'congo', 'saudi': 'saudi arabia', 'kuwaiti': 'kuwait', 'iraqi': 'iraq',
    'israeli': 'israel', 'pakistani': 'pakistan', 'afghan': 'afghanistan', 'bangladeshi': 'bangladesh',
    'nepali': 'nepal', 'nepalese': 'nepal', 'yemeni': 'yemen', 'omani': 'oman', 'qatari': 'qatar', 'bahraini': 'bahrain',
    'somali': 'somalia', 'sudanese': 'sudan', 'lebanese': 'lebanon', 'chinese': 'china', 'japanese': 'japan',
    'vietnamese': 'vietnam', 'taiwanese': 'taiwan', 'portuguese': 'portugal', 'maltese': 'malta', 'senegalese': 'senegal',
    'togolese': 'togo', 'guinean': 'guinea', 'beninese': 'benin', 'gabonese': 'gabon', 'malian': 'mali',
    'nigerien': 'niger', 'chadian': 'chad', 'turkish': 'turkey', 'turk': 'turkey', 'uzbek': 'uzbekistan',
    'kazakh': 'kazakhstan', 'kyrgyz': 'kyrgyzstan', 'tajik': 'tajikistan', 'turkmen': 'turkmenistan',
    'azerbaijani': 'azerbaijan', 'azeri': 'azerbaijan', 'czech': 'czech republic', 'slovak': 'slovakia',
    'mexican': 'mexico', 'brazilian': 'brazil', 'colombian': 'colombia', 'venezuelan': 'venezuela', 'cuban': 'cuba',
    'chilean': 'chile', 'bolivian': 'bolivia', 'ecuadorian': 'ecuador', 'uruguayan': 'uruguay', 'paraguayan': 'paraguay',
    'haitian': 'haiti', 'canadian': 'canada', 'australian': 'australia', 'indian': 'india', 'indonesian': 'indonesia',
    'malaysian': 'malaysia', 'singaporean': 'singapore', 'cambodian': 'cambodia', 'korean': 'korea',
    'russian': 'russia', 'german': 'germany', 'italian': 'italy', 'egyptian': 'egypt', 'libyan': 'libya',
    'tunisian': 'tunisia', 'algerian': 'algeria', 'moroccan': 'morocco', 'syrian': 'syria', 'jordanian': 'jordan',
    'iranian': 'iran', 'palestinian': 'palestine', 'ukrainian': 'ukraine', 'belarusian': 'belarus', 'georgian': 'georgia',
    'armenian': 'armenia', 'romanian': 'romania', 'bulgarian': 'bulgaria', 'hungarian': 'hungary', 'serbian': 'serbia',
    'serb': 'serbia', 'croatian': 'croatia', 'croat': 'croatia', 'bosnian': 'bosnia and herzegovina', 'albanian': 'albania',
    'macedonian': 'macedonia', 'slovenian': 'slovenia', 'austrian': 'austria', 'belgian': 'belgium', 'swedish': 'sweden',
    'swede': 'sweden', 'norwegian': 'norway', 'icelandic': 'iceland', 'estonian': 'estonia', 'latvian': 'latvia',
    'lithuanian': 'lithuania', 'kenyan': 'kenya', 'nigerian': 'nigeria', 'ethiopian': 'ethiopia', 'eritrean': 'eritrea',
    'ugandan': 'uganda', 'rwandan': 'rwanda', 'burundian': 'burundi', 'tanzanian': 'tanzania', 'zambian': 'zambia',
    'zimbabwean': 'zimbabwe', 'mozambican': 'mozambique', 'angolan': 'angola', 'namibian': 'namibia',
    'botswanan': 'botswana', 'ghanaian': 'ghana', 'liberian': 'liberia', 'cameroonian': 'cameroon',
    'mauritanian': 'mauritania', 'sri': None, 'lankan': 'sri lanka', 'mongolian': 'mongolia', 'moldovan': 'moldova',
    'montenegrin': 'montenegro', 'kosovar': 'kosovo', 'fijian': 'fiji', 'jamaican': 'jamaica', 'guatemalan': 'guatemala',
    'honduran': 'honduras', 'salvadoran': 'el salvador', 'nicaraguan': 'nicaragua', 'panamanian': 'panama',
    'costa': None, 'rican': 'costa rica', 'dominican': 'dominican republic', 'european': 'europe', 'african': 'africa',
    'asian': 'asia', 'arab': 'arab', 'britain': 'united kingdom', 'kurdish': 'kurdistan', 'tibetan': 'tibet', 'chechen': 'chechnya',
}


def time_parse_init():
    MONS = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
            'november', 'december']
    ab = {m[:3]: i + 1 for i, m in enumerate(MONS)}
    ab['sept'] = 9
    full = {m: i + 1 for i, m in enumerate(MONS)}
    mon_map = {**ab, **full}
    MON = '(' + '|'.join(sorted(mon_map, key=len, reverse=True)) + r')\.?'
    Y = r'((?:19|20)\d{2})'
    D = r'(\d{1,2})(?:st|nd|rd|th)?'
    pats = [
        ('iso_day', re.compile(r'\b((?:19|20)\d{2})-(\d{1,2})-(\d{1,2})\b'), (3, 2, 1)),
        ('iso_month', re.compile(r'\b((?:19|20)\d{2})-(\d{1,2})\b'), (None, 2, 1)),
        ('day', re.compile(r'\b' + D + r'\s+(?:of\s+)?' + MON + r',?\s+(?:the\s+year\s+(?:of\s+)?|of\s+|in\s+)?' + Y + r'\b', re.I), (1, 2, 3)),
        ('day', re.compile(r'\b' + MON + r'\s+' + D + r',?\s+(?:the\s+year\s+(?:of\s+)?|of\s+|in\s+)?' + Y + r'\b', re.I), (2, 1, 3)),
        ('month', re.compile(r'\b' + MON + r',?\s+(?:the\s+year\s+(?:of\s+)?|of\s+(?:the\s+year\s+)?|in\s+(?:the\s+year\s+)?)?' + Y + r'\b', re.I), (None, 1, 2)),
        ('year', re.compile(r'\b' + Y + r'\b'), (None, None, 1)),
    ]
    return pats, mon_map

_TPATS, _MONMAP = time_parse_init()

def parse_time(q):
    """Return (level, string, span) of the first explicit time expression, or None."""
    for level, pat, (di, mi, yi) in _TPATS:
        m = pat.search(q)
        if not m:
            continue
        y = int(m.group(yi))
        if not (2000 <= y <= 2030):
            continue
        if level.startswith('iso'):
            mo = int(m.group(mi))
            if not 1 <= mo <= 12:
                continue
            if level == 'iso_month':
                return ('month', f'{y}-{mo:02d}', m.span())
            dd = int(m.group(di))
            try:
                datetime.date(y, mo, dd)
            except ValueError:
                continue
            return ('day', f'{y}-{mo:02d}-{dd:02d}', m.span())
        if level == 'year':
            return ('year', f'{y}', m.span())
        mo = _MONMAP[m.group(mi).lower().rstrip('.')]
        if level == 'month':
            return ('month', f'{y}-{mo:02d}', m.span())
        dd = int(m.group(di))
        if not 1 <= dd <= 31:
            continue
        try:
            datetime.date(y, mo, dd)
        except ValueError:
            continue
        return ('day', f'{y}-{mo:02d}-{dd:02d}', m.span())
    return None


# ---------------------------------------------------------------- entity linker
class Linker:
    """Stem-set matcher over KG entity names with demonym expansion and position-aware window selection."""

    def __init__(self, kg, max_extra=3, min_partial=0.5, alias=None, ent_block=None):
        self.kg = kg
        self.max_extra = max_extra
        self.min_partial = min_partial
        self.alt_ratio = 0.7
        self.ent_block = {tuple(x) for x in ent_block} if ent_block else set()   # (entity, full, alias) keys
        names = kg.ent_names
        self.ent_stems = []
        self.ent_cstem = []
        df = collections.Counter()
        for e in names:
            ss = self._ent_stems(e)
            self.ent_stems.append(ss)
            m_ = re.match(r'^(.*) \((.*)\)$', e)
            self.ent_cstem.append(set(stem(w) for w in toks(m_.group(2)) if w not in STOP) if m_ else None)
            df.update(set(ss))
        N = len(names)
        self.idf = {s: math.log(1 + N / c) for s, c in df.items()}
        self.inv = collections.defaultdict(list)
        for i, ss in enumerate(self.ent_stems):
            for s in set(ss):
                self.inv[s].append(i)
        # demonym map token -> stems of the country
        self.demonym = {}
        country_like = {}
        for i, e in enumerate(names):
            if '(' not in e:
                country_like[' '.join(toks(e))] = e
        for d, c in DEMONYM_MANUAL.items():
            if c is None:
                continue
            self.demonym[d] = [stem(w) for w in toks(c) if w not in STOP]
        # sector alias rules learned on train: stemmed n-gram with <C> -> (sector S, precision P)
        self.rules = {}
        if alias:
            for k, v in alias.items():
                self.rules[tuple(k.split(' '))] = [(s_, p_) for s_, p_, *_ in v]
        SC = {}
        countries = set()
        for i, e in enumerate(names):
            m = re.match(r'^(.*) \((.*)\)$', e)
            if m:
                SC[(m.group(1), m.group(2))] = i
                countries.add(m.group(2))
        self.SC = SC
        cst = {}
        for c in countries:
            cs = tuple(stem(w) for w in toks(c) if w not in STOP)
            if cs:
                cst.setdefault(cs, c)
        self.cstems = cst
        self.cfirst = collections.defaultdict(list)
        for cs, c in cst.items():
            self.cfirst[cs[0]].append((cs, c))

    def country_mentions(self, ws):
        st = [stem(w) for w in ws]
        idx = [i for i, w in enumerate(ws) if w not in STOP]
        seq = [st[i] for i in idx]
        out = []
        for k, s in enumerate(seq):
            for cs, c in self.cfirst.get(s, ()):
                if tuple(seq[k:k + len(cs)]) == cs:
                    out.append((idx[k], idx[k + len(cs) - 1], c))
        for i, w in enumerate(ws):
            if w in DEMONYM_MANUAL and DEMONYM_MANUAL[w]:
                cs = tuple(stem(x) for x in toks(DEMONYM_MANUAL[w]) if x not in STOP)
                c = self.cstems.get(cs)
                if c:
                    out.append((i, i, c))
        out.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
        sel, used = [], set()
        for lo, hi, c in out:
            if any(j in used for j in range(lo, hi + 1)):
                continue
            sel.append((lo, hi, c))
            used.update(range(lo, hi + 1))
        return sorted(sel)

    def alias_candidates(self, ws, maxlen=6):
        if not self.rules:
            return []
        out = []
        for lo, hi, c in self.country_mentions(ws):
            tk = [stem(w) for w in ws[:lo]] + ['<C>'] + [stem(w) for w in ws[hi + 1:]]
            p = lo
            best = None
            for a in range(max(0, p - maxlen + 1), p + 1):
                for b in range(p, min(len(tk), a + maxlen)):
                    if b - a + 1 < 2:
                        continue
                    r = self.rules.get(tuple(tk[a:b + 1]))
                    if r is None:
                        continue
                    es = [(self.SC[(s_, c)], p_) for s_, p_ in r if (s_, c) in self.SC]
                    if not es:
                        continue
                    key = (b - a + 1, es[0][1])
                    if best is None or key > best[0]:
                        A_ = a if a < p else lo
                        B_ = (b + (hi - lo)) if b > p else hi
                        best = (key, es, A_, B_)
            if best:
                _, es, A_, B_ = best
                used = {j for j in range(A_, B_ + 1) if ws[j] not in STOP}
                for e, P in es:
                    ss = list(dict.fromkeys(self.ent_stems[e]))
                    wt = sum(self.idf[s] for s in ss)
                    out.append(dict(e=e, score=P, full=True, nmatch=len(ss), lo=A_, hi=B_, ne=0, used=set(used), w=wt,
                                    eff=P * wt, alias=True))
        return out

    def _ent_stems(self, e):
        m = re.match(r'^(.*) \((.*)\)$', e)
        ws = toks(e)
        return tuple(stem(w) for w in ws if w not in STOP)

    def link(self, q, topn=5):
        ws = toks(q)
        st = [stem(w) for w in ws]
        # position lists per stem (incl. demonym expansions)
        pos = collections.defaultdict(list)
        for i, s in enumerate(st):
            if ws[i] in STOP:
                continue
            pos[s].append(i)
            if ws[i] in self.demonym:
                for s2 in self.demonym[ws[i]]:
                    pos[s2].append(i)
        # candidate entities sharing at least one stem
        cand = collections.Counter()
        for s in list(pos):
            for e in self.inv.get(s, ()):
                cand[e] += 1
        out = []
        for e, c in cand.items():
            ss = self.ent_stems[e]
            if not ss:
                continue
            uniq = list(dict.fromkeys(ss))
            wt = sum(self.idf[s] for s in uniq)
            matched = [s for s in uniq if s in pos]
            mw = sum(self.idf[s] for s in matched)
            score = mw / wt
            if score < self.min_partial:
                continue
            if len(matched) < len(uniq):
                # partial match: the rarest stem must be matched, and for "S (C)" names at least one S stem
                top = max(uniq, key=lambda s: self.idf[s])
                if top not in pos:
                    continue
                cs = self.ent_cstem[e]
                if cs is not None and not any(s in pos for s in uniq if s not in cs):
                    continue
            # minimal window over matched stems (choose nearest occurrences)
            best = None
            combos = [[]]
            for s in matched:
                combos = [c_ + [p] for c_ in combos for p in pos[s]][:64]
            for cmb in combos:
                lo, hi = min(cmb), max(cmb)
                span = hi - lo + 1
                extra = span - len(set(cmb))
                # count non-stop extra tokens inside window
                ne = sum(1 for j in range(lo, hi + 1) if j not in cmb and ws[j] not in STOP)
                key = (ne, span)
                if best is None or key < best[0]:
                    best = (key, lo, hi, set(cmb))
            (ne, span), lo, hi, used = best
            if ne > self.max_extra:
                continue
            full = len(matched) == len(uniq)
            eff = mw - 0.5 * (wt - mw) - 3.0 * ne
            out.append(dict(e=e, score=score, full=full, nmatch=len(matched), lo=lo, hi=hi, ne=ne, used=used,
                            w=mw, eff=eff))
        # learned sector-alias candidates
        out.extend(self.alias_candidates(ws))
        if self.ent_block:
            out = [c for c in out if (c['e'], int(c['full']), int(bool(c.get('alias')))) not in self.ent_block]
        # rank by penalised matched idf weight
        out.sort(key=lambda x: (-x['eff'], -x['full'], x['hi'] - x['lo'], x['e']))
        # greedy non-overlapping selection of mentions; overlapping near-competitors kept as alternatives
        mentions = []
        for c in out:
            ov = [m for m in mentions if c['used'] & m['used']]
            if not ov:
                mentions.append(dict(lo=c['lo'], hi=c['hi'], used=set(c['used']), alts=[c]))
                continue
            if len(ov) == 1:
                m = ov[0]
                if c['e'] not in {a['e'] for a in m['alts']} and len(m['alts']) < topn and c['eff'] >= self.alt_ratio * m['alts'][0]['eff'] and c['used'] <= m['used'] | c['used']:
                    m['alts'].append(c)
        mentions.sort(key=lambda m: m['lo'])
        return mentions, ws

    def link_question(self, q, topn=5):
        """Link after blanking the explicit time expression."""
        tp = parse_time(q)
        q2 = q
        if tp is not None:
            a, b = tp[2]
            q2 = q[:a] + ' timex ' + q[b:]
        return self.link(q2, topn=topn), tp


# ---------------------------------------------------------------- operator keywords (text only)
_BA_B = re.compile(r'\b(before|prior to|earlier than|until|preceding|ahead of)\b')
_BA_A = re.compile(r'\b(after|since|following|later than|subsequent)\b')
_FL_F = re.compile(r'\b(first|earliest|initial|initially|1st)\b')
_FL_L = re.compile(r'\b(last|latest|final|finally|most recent|most recently|lastly)\b')


def kw_ba(s):
    """0 = before, 1 = after, -1 = none (first cue wins when both occur)."""
    s = s.lower(); b = _BA_B.search(s); a = _BA_A.search(s)
    if b and not a: return 0
    if a and not b: return 1
    if a and b: return 0 if b.start() < a.start() else 1
    return -1


def kw_fl(s):
    """0 = first, 1 = last, -1 = none."""
    s = s.lower(); f = _FL_F.search(s); l = _FL_L.search(s)
    if f and not l: return 0
    if l and not f: return 1
    if f and l: return 0 if f.start() < l.start() else 1
    return -1
