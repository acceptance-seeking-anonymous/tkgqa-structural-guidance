"""cron_text.py -- normalisation, surface dictionary (KG labels + Wikidata aliases), train-template
skeleton segmenter and dictionary fallback linker for CronQuestions.

Everything here is built from the KG files and from TRAIN questions only (template strings and
their slot names are train labels).  Dev/test inputs: the question text (paraphrases[0] in OWN;
the id-substituted 'question' field in GOLD-ANN)."""
import re, collections, unicodedata
import numpy as np

SLOT_RE = re.compile(r'(\{head\}|\{tail\}|\{tail2\}|\{time\}|\{event_head\})')
SLOTS = ('head', 'tail', 'tail2', 'time', 'event_head')
YEAR_RE = re.compile(r'^\d{1,4}$')
TOK_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
QID_RE = re.compile(r'^Q\d+$')


def fold(s):
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return s


def norm(s):
    """lower-cased, accent-folded, punctuation-tokenised surface key"""
    return ' '.join(TOK_RE.findall(fold(s).lower()))


class Surfaces:
    """norm(label or alias) -> [(entity index, is_label)]"""
    def __init__(self, kg):
        self.kg = kg
        d = collections.defaultdict(dict)
        for q, lab in kg.label.items():
            e = kg.e2i.get(q)
            if e is None:
                continue
            if lab:
                d[norm(lab)][e] = 1
            for a in kg.aliases.get(q, ()):
                k = norm(a)
                if k and e not in d[k]:
                    d[k][e] = 0
        self.d = {k: sorted(v.items(), key=lambda x: (-x[1], -int(kg.degree[x[0]]), x[0])) for k, v in d.items()}
        self.maxlen = max(len(k.split()) for k in self.d)

    def lookup(self, s):
        return self.d.get(norm(s), [])


class Templates:
    """Train template inventory: literal skeleton, slot names, train frame and frequency."""
    def __init__(self, train_questions):
        cnt = collections.Counter(q['template'] for q in train_questions)
        frame = {}
        for q in train_questions:
            t = q['template']
            if t not in frame:
                a = q['annotation']
                frame[t] = dict(type=q['type'], atype=q['answer_type'], rel=next(iter(q['relations'])),
                                adj=a.get('adj'), ba=a.get('type'))
        self.items = []
        for t, n in cnt.items():
            parts = SLOT_RE.split(ws(t))
            lits = parts[0::2]
            lits[0] = lits[0].lstrip(); lits[-1] = lits[-1].rstrip()
            slots = [p[1:-1] for p in parts[1::2]]
            self.items.append(dict(t=t, lits=lits, slots=slots, n=n, frame=frame[t],
                                   nlit=sum(len(x.strip()) for x in lits)))
        self.by_first = collections.defaultdict(list)
        for i, it in enumerate(self.items):
            self.by_first[it['lits'][0][:6].lower()].append(i)

    def candidates(self, text):
        low = text.lower()
        out = set()
        for k in {low[:j] for j in range(0, 7)}:
            out.update(self.by_first.get(k, ()))
        return [i for i in out if low.startswith(self.items[i]['lits'][0].lower())
                and low.endswith(self.items[i]['lits'][-1].lower())]

    def splits(self, i, text, maxn=20):
        """all assignments of slot fillers (case-insensitive literal matching)"""
        it = self.items[i]
        lits, ns = it['lits'], len(it['slots'])
        low = text.lower()
        L0, Ln = len(lits[0]), len(lits[-1])
        end = len(text) - Ln
        if end < L0:
            return []
        res = []

        def rec(pos, k, acc):
            if len(res) >= maxn:
                return
            if k == ns - 1:
                f = text[pos:end]
                if f.strip():
                    res.append(acc + [(pos, end)])
                return
            lit = lits[k + 1].lower()
            j = low.find(lit, pos + 1)
            while j != -1 and j + len(lit) <= end:
                if text[pos:j].strip():
                    rec(j + len(lit), k + 1, acc + [(pos, j)])
                j = low.find(lit, j + 1)
        if ns == 0:
            return [[]] if low == lits[0].lower() else []
        rec(L0, 0, [])
        return res


def ws(s):
    """collapse whitespace runs and strip (applied to templates and to question texts)"""
    return ' '.join(s.split())


def tokens(s):
    return TOK_RE.findall(s)
