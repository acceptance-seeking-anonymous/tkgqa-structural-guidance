"""cron_kg.py -- CronQuestions Wikidata TKG index (full.txt, unique facts, inclusive year intervals)."""
import os, pickle, collections, numpy as np

DATA = os.environ.get('CRON_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cronquestions_data'))
EVENT_REL, EVENT_OBJ = 'P793', 'Q1190554'   # (event, significant event, occurrence, start, end)


class KG:
    def __init__(self, data=DATA):
        K = os.path.join(data, 'kg')
        seen = collections.Counter()
        rows = []
        for l in open(os.path.join(K, 'full.txt'), encoding='utf-8'):
            p = l.rstrip('\n').split('\t')
            if len(p) != 5:
                continue
            key = tuple(p)
            seen[key] += 1
            if seen[key] == 1:
                rows.append(key)
        self.dup = np.array([seen[k] for k in rows], np.int16)      # line multiplicity
        ents = sorted({x[0] for x in rows} | {x[2] for x in rows})
        rels = sorted({x[1] for x in rows})
        self.ents, self.rels = ents, rels
        self.e2i = {e: i for i, e in enumerate(ents)}
        self.r2i = {r: i for i, r in enumerate(rels)}
        self.H = np.array([self.e2i[x[0]] for x in rows], np.int32)
        self.R = np.array([self.r2i[x[1]] for x in rows], np.int16)
        self.T = np.array([self.e2i[x[2]] for x in rows], np.int32)
        self.S = np.array([int(x[3]) for x in rows], np.int32)
        self.E = np.array([int(x[4]) for x in rows], np.int32)
        self.nf, self.ne = len(rows), len(ents)
        # CSR by subject and by object
        self.sub_ptr, self.sub_idx = self._csr(self.H)
        self.obj_ptr, self.obj_idx = self._csr(self.T)
        self.degree = np.diff(self.sub_ptr) + np.diff(self.obj_ptr)
        # (entity, relation) -> fact ids, subject / object side
        self.sr = collections.defaultdict(list); self.orr = collections.defaultdict(list)
        for f in range(self.nf):
            self.sr[(int(self.H[f]), int(self.R[f]))].append(f)
            self.orr[(int(self.T[f]), int(self.R[f]))].append(f)
        self.sr = {k: np.array(v, np.int32) for k, v in self.sr.items()}
        self.orr = {k: np.array(v, np.int32) for k, v in self.orr.items()}
        # event intervals
        ev_r, ev_o = self.r2i[EVENT_REL], self.e2i.get(EVENT_OBJ, -1)
        self.event_iv = {}
        for f in self.sr_get_rel_obj(ev_r, ev_o):
            self.event_iv.setdefault(int(self.H[f]), (int(self.S[f]), int(self.E[f])))
        # texts
        self.label = {}
        for l in open(os.path.join(K, 'wd_id2entity_text.txt'), encoding='utf-8'):
            p = l.rstrip('\n').split('\t')
            self.label[p[0]] = p[1] if len(p) > 1 else ''
        self.rlabel = {}
        for l in open(os.path.join(K, 'wd_id2relation_text.txt'), encoding='utf-8'):
            p = l.rstrip('\n').split('\t')
            self.rlabel[p[0]] = p[1] if len(p) > 1 else ''
        self.aliases = pickle.load(open(os.path.join(K, 'wd_id_to_aliases.pickle'), 'rb'))

    def _csr(self, col):
        order = np.argsort(col, kind='stable').astype(np.int32)
        cnt = np.bincount(col, minlength=self.ne)
        ptr = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
        return ptr, order

    def sr_get_rel_obj(self, r, o):
        if o < 0:
            return np.zeros(0, np.int32)
        return self.orr.get((o, r), np.zeros(0, np.int32))

    def subj_facts(self, e, r):
        return self.sr.get((e, r), np.zeros(0, np.int32))

    def obj_facts(self, e, r):
        return self.orr.get((e, r), np.zeros(0, np.int32))

    def facts_of(self, e):
        return np.concatenate([self.sub_idx[self.sub_ptr[e]:self.sub_ptr[e + 1]],
                               self.obj_idx[self.obj_ptr[e]:self.obj_ptr[e + 1]]])
