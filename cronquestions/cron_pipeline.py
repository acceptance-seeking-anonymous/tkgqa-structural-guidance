"""cron_pipeline.py -- glue for the CronQuestions structural track: question loading (with the test field
filter), OWN / GOLD-ANN parsing, alternative re-ranking, classifier probabilities, candidate pools, RULE /
PRIOR / S / LOOKUP_TRAIN / DEV_LOOKUP ranked lists."""
import os, pickle, json, copy, collections, math
import numpy as np
import torch, lightgbm as lgb
import cron_kg, cron_text, cron_parse, cron_clf as CC, cron_cand as CD

ALLOWED_TEST_FIELDS = ('paraphrases', 'question', 'entities', 'times')


def load_questions(data, ver, split, filter_fields=False):
    name = {'train': 'train', 'dev': 'valid', 'test': 'test'}[split]
    qs = pickle.load(open(os.path.join(data, ver, 'questions', name + '.pickle'), 'rb'))
    if filter_fields:
        qs = [{k: q[k] for k in ALLOWED_TEST_FIELDS} for q in qs]
    return qs


class Pipe:
    def __init__(self, data, models):
        self.kg = cron_kg.KG(data)
        CD.init(self.kg)
        self.sf = cron_text.Surfaces(self.kg)
        self.tp = cron_text.Templates.__new__(cron_text.Templates)
        items = pickle.load(open(os.path.join(models, 'templates.pkl'), 'rb'))
        self.tp.items = items
        self.tp.by_first = collections.defaultdict(list)
        for i, it in enumerate(items):
            self.tp.by_first[it['lits'][0][:6].lower()].append(i)
        self.pa = cron_parse.Parser(self.kg, self.sf, self.tp)
        self.alt = {k: lgb.Booster(model_file=os.path.join(models, f'alt_{k}.txt')) for k in ('full', 'fold0', 'fold1')}
        itos = pickle.load(open(os.path.join(models, 'clf_vocab.pkl'), 'rb'))
        self.voc = CC.Vocab([]); self.voc.itos = itos; self.voc.stoi = {k: i + 1 for i, k in enumerate(itos)}
        self.clf = {}
        for k in ('full', 'fold0', 'fold1'):
            net = CC.Net(len(itos)); net.load_state_dict(torch.load(os.path.join(models, f'clf_{k}.pt'))); net.eval()
            self.clf[k] = net
        self.tmpl_freq = {i: it['n'] for i, it in enumerate(items)}

    # ---------------- parsing
    def parse_own(self, text):
        text = cron_text.ws(text)
        hs = self.pa.hypotheses(text)
        if not hs:
            hs = [self.pa.fallback(text)]
        return dict(text=text, hyps=hs)

    def parse_gold(self, q_text, ents, times):
        kg = self.kg
        text = cron_text.ws(q_text)
        hs = self.pa.hypotheses(text, gold=(ents, times))
        G = [kg.e2i[e] for e in sorted(ents) if e in kg.e2i]
        if not hs:
            slots = [dict(role='unk', span=(0, 0), text=kg.ents[e], alts=[(e, 1, 1.0)]) for e in G]
            slots += [dict(role='time', span=(0, 0), text=str(t), year=int(t), alts=[]) for t in sorted(times)]
            return dict(text=text, hyps=[dict(key=(9,), tid=-1, slots=slots, n_unl=0, n_inc=0, pair=0)])
        for h in hs:
            placed = {s['alts'][0][0] for s in h['slots'] if s['role'] != 'time' and s['alts']}
            unpl = [e for e in G if e not in placed]
            unl = [s for s in h['slots'] if s['role'] != 'time' and not s['alts']]
            if len(unl) == 1 and len(unpl) == 1:
                unl[0]['alts'] = [(unpl[0], 1, 1)]
            for s in h['slots']:
                if s['role'] != 'time':
                    s['alts'] = [(a[0], a[1], 1.0) for a in s['alts'][:1]]
                elif times and s['year'] not in times and len(times) == 1:
                    s['year'] = int(next(iter(times)))
        return dict(text=text, hyps=hs)

    def rerank(self, p, model_key, gold=False):
        pp = copy.deepcopy(p)
        if gold:
            return pp
        for h in pp['hyps']:
            if h['tid'] >= 0:
                rel = self.tp.items[h['tid']]['frame']['rel']
                cron_parse.rerank_alts(self.kg, self.sf, h, rel, self.alt[model_key])
            else:
                for s in h['slots']:
                    if s['role'] != 'time':
                        s['alts'] = [(a[0], a[1], 1.0 / (1 + k)) for k, a in enumerate(s['alts'])]
        return pp

    def clf_probs(self, own_parses, model_key):
        enc = [np.array(self.voc.encode(CC.ngrams(cron_parse.masked_text(p['text'], p['hyps'][0])))) for p in own_parses]
        return CC.predict(self.clf[model_key], enc)

    # ---------------- ranked lists
    def pad(self, items, meta, FB, k=50):
        out, seen = [], set()
        for x in items:
            if x not in seen:
                seen.add(x); out.append(x)
            if len(out) >= k:
                break
        if len(out) < 10:
            for x in [CD.key_out(a) for a in meta['prior']] + FB[meta['atype']]:
                if x not in seen:
                    seen.add(x); out.append(x)
                if len(out) >= 10:
                    break
        return out


def probs_row(P, i):
    return {k: P[k][i] for k in CC.HEADS}
