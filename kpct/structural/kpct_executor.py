"""kpct_executor.py -- execute a structured MultiTQ parse with the frozen temporal operators, candidate retrieval,
RULE ordering and the S_k LightGBM re-rankers of one k% subset.

    import kpct_executor as X
    r = X.execute(parse_json, 'k01_s0')            # -> dict, see Executor.execute
    ex = X.Executor('k01_s0'); r = ex.execute(parse_json)

k_subset in {k01_s0, k05_s0, k10_s0, k01_s1, k01_s2, k100, k0}:
  * k<100 : S_k models (3 LightGBM seeds), k%-trained classifier (only for probs='classifier' / parse_question),
            output padding from the k% train answers;
  * k100  : the frozen full-data artefacts (S_seed0-2, v3 classifier, full-train linker, full-train padding);
  * k0    : no train-fit artefact at all: operators + retrieval + RULE ordering only (no S, no classifier,
            no aliases, no block list); padding from KG statistics only.
Input schema = the guidance-bundle `parse` schema (kpct-parse-v1); see kpct_executor_readme.md.

CLI:
  python kpct_executor.py run --k k01_s0 --in parses.jsonl --out ranked.jsonl [--workers 8]
        input rows: {"quid": ..., "parse": {...}, "question": "..." (optional)}
  python kpct_executor.py selftest --k k01_s0 [--bundles DIR] [--workers 8]
"""
import os, sys, json, argparse, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('MTQ_CODE', os.path.join(HERE, 'struct'))
os.environ.setdefault('MTQ_ROOT', os.path.join(HERE, 'mtqdata'))
sys.path.insert(0, HERE)
import numpy as np
import torch  # noqa: F401  -- loads torch's bundled libgomp; lightgbm needs libgomp.so.1, absent in tkgqa-gpu:latest
import kpct_core as K

MODELS = os.environ.get('KPCT_MODELS', os.path.join(HERE, 'models'))
SUBSETS = ('k01_s0', 'k05_s0', 'k10_s0', 'k01_s1', 'k01_s2', 'k100', 'k0')
_EX = {}


class Executor:
    def __init__(self, k_subset, models=MODELS):
        assert k_subset in SUBSETS, k_subset
        self.k = k_subset
        d = os.path.join(models, k_subset)
        self.dir = d
        K.kg()
        self.FB = json.load(open(os.path.join(d, 'FB.json')))
        self.boosters = []
        if k_subset != 'k0':
            import lightgbm as lgb
            self.boosters = [lgb.Booster(model_file=os.path.join(d, f'S_seed{s}.txt')) for s in range(3)]
        self._parser = None
        self._linker = None

    # ---- optional: the subset's own structural parser (k<100: retrained on k%; k100: frozen v3)
    def parser(self):
        if self.k == 'k0':
            raise RuntimeError('k0 has no classifier')
        if self._parser is None:
            p = os.path.join(self.dir, 'parser.pt')
            self._parser = K.load_parser(p)
        return self._parser

    def linker(self):
        if self.k == 'k0':
            return K.make_linker()          # base linker: stemming + demonyms, no aliases / block list
        if self._linker is None:
            lk = json.load(open(os.path.join(self.dir, 'linker.json')))
            self._linker = K.make_linker(lk['alias_rules'], lk['ent_block'])
        return self._linker

    def parse_question(self, question, quid=None):
        """The subset's structural parse of a question (= the bundle parse for dev/test)."""
        rec = K.link_rec(self.linker(), quid, question)
        voc, net = self.parser()
        pr = K.parser_probs(voc, net, [rec])
        info = K.MC.parse_info(rec, {k: v[0] for k, v in pr.items()}, question)
        return K.parse_json(rec, info)

    def classifier_probs(self, rec, info):
        """replace the probabilities of the parse's choices by the subset classifier's probabilities
        (computed on the parse's own entity-masked text)."""
        voc, net = self.parser()
        pr = K.parser_probs(voc, net, [rec])
        pr = {k: v[0] for k, v in pr.items()}
        info = dict(info)
        info['rels'] = [(r, d, float(pr['rel'][r * 2 + d])) for r, d, _ in info['rels']]
        info['qt_p'] = float(pr['qtype'][K.QTYPES.index(info['qt'])])
        info['atype_p'] = float(pr['atype'][K.ATYPES.index(info['atype'])])
        info['level_p'] = float(pr['level'][K.LEVELS.index(info['level'])])
        return info

    def execute(self, parse_json, question=None, probs='given', top=50, details=20):
        """Execute one parse.
        returns {k_subset, valid, error, pool_size, n_operator_satisfied,
                 rule: [top-50 strings, RULE order, padded to >= 10 exactly as the prediction files],
                 s: {seed0, seed1, seed2: [top-50 strings, S_k order, ties by RULE order]} (None for k0),
                 s_mean: [top-50 by mean score of the 3 seeds] (None for k0),
                 candidates: [first `details` pool candidates in RULE order: answer, rule_rank, s_scores, operator_satisfied]}
        probs: 'given' (use the probabilities in the JSON; missing -> 1.0) or 'classifier' (the subset classifier's
        probabilities of the parse's choices; not for k0)."""
        out = dict(k_subset=self.k, valid=False, error=None, pool_size=0, n_operator_satisfied=0, rule=[], s=None,
                   s_mean=None, candidates=[])
        try:
            rec, info = K.rec_info_from_parse(parse_json, question)
        except (K.ParseError, KeyError, TypeError) as ex:
            out['error'] = f'{type(ex).__name__}: {ex}'
            return out
        if probs == 'classifier':
            info = self.classifier_probs(rec, info)
        anc = parse_json.get('anchor_mention')
        old = K.MC.anchor_index
        if anc is not None:
            K.MC.anchor_index = lambda ws, ments, qtext, _a=int(anc): (_a if 0 <= _a < len(ments) and len(ments) >= 2 else None)
        try:
            o = K.build_info(rec, info)
        finally:
            K.MC.anchor_index = old
        out['valid'] = True
        out['pool_size'] = len(o['names'])
        out['n_operator_satisfied'] = int(o['X'][:, K.MC.FIDX['sat_any']].sum()) if len(o['names']) else 0
        out['rule'] = K.make_ranked(o['names'], o, self.FB, k=top)
        if self.boosters:
            sc = K.predict_chunk(self.boosters, [o])
            out['s'] = {f'seed{b}': K.make_ranked([o['names'][j] for j in K.s_order(sc[b][0])], o, self.FB, k=top) for b in range(3)}
            mean = (sc[0][0] + sc[1][0] + sc[2][0]) / 3.0
            out['s_mean'] = K.make_ranked([o['names'][j] for j in K.s_order(mean)], o, self.FB, k=top)
        for j in range(min(details, len(o['names']))):
            out['candidates'].append(dict(answer=o['names'][j], rule_rank=j + 1,
                                          s_scores=[float(sc[b][0][j]) for b in range(3)] if self.boosters else None,
                                          operator_satisfied=bool(o['X'][j, K.MC.FIDX['sat_any']])))
        return out


def get(k_subset):
    if k_subset not in _EX:
        _EX[k_subset] = Executor(k_subset)
    return _EX[k_subset]


def execute(parse_json, k_subset, **kw):
    return get(k_subset).execute(parse_json, **kw)


# ------------------------------------------------------------------ helpers for LLM parsers
def _time_level(s):
    s = str(s)
    return {4: 'year', 7: 'month', 10: 'day'}.get(len(s))


def compact_to_parse(c, question, structural_parse=None):
    """Convert the compact LLM-parser JSON of the rule ({mentions:[{text, entity}], relation, answer_role, qtype,
    answer_type, granularity, time, anchor, order}) to the executor schema. If `structural_parse` (the bundle parse) is
    given, the compact parse becomes hypothesis rank 0 and the structural parse's remaining hypotheses follow
    (duplicates removed): relation list = [LLM relation] + structural relations; each LLM mention's alternatives =
    [LLM entity] + the alternatives of the structural mention whose span overlaps it; structural mentions that no LLM
    mention overlaps are dropped. Probabilities are taken from the structural parse where the same choice exists,
    else 1.0 (use probs='classifier' at execution to recompute them with the subset classifier)."""
    tmp = {'tokens': None, 'mentions': [], 'qtype': c['qtype'], 'answer_type': c['answer_type'], 'granularity': c['granularity']}
    p = {'schema': K.PARSE_SCHEMA}
    q2 = question
    t_ = K.L.parse_time(question)
    if t_ is not None:
        a, b = t_[2]; q2 = question[:a] + ' timex ' + question[b:]
    ws = K.L.toks(q2)
    p['tokens'] = ws
    if c.get('time'):
        lv = _time_level(c['time'])
        if lv is None:
            raise K.ParseError(f"bad time {c['time']}")
        p['time'] = {'level': lv, 'value': str(c['time'])}
    else:
        p['time'] = None
    used = set(); ments = []
    for m in c.get('mentions', []):
        span = K._locate(ws, m.get('text', ''), used) or K._locate(ws, m.get('entity', ''), used)
        if span is None:
            span = [len(ws) + len(ments)] * 2
        used.update(range(span[0], span[1] + 1))
        ments.append({'span': span, 'text': m.get('text', ''), 'alternatives': [{'entity': m['entity'], 'score': 1.0, 'alias': 0, 'full': 1}]})
    ments.sort(key=lambda m: m['span'][0])
    role = c.get('answer_role', 'object')
    d = 1 if role == 'object' else 0
    rels = [{'relation': c['relation'], 'direction': K.DIR_NAMES[d], 'prob': 1.0}]
    if structural_parse is not None:
        sp = structural_parse
        for r in sp['relations']:
            if (r['relation'], r['direction']) == (rels[0]['relation'], rels[0]['direction']):
                rels[0]['prob'] = r['prob']
            else:
                rels.append(dict(r))
        for m in ments:
            for sm in sp['mentions']:
                if not (sm['span'][1] < m['span'][0] or sm['span'][0] > m['span'][1]):
                    first = m['alternatives'][0]
                    for a in sm['alternatives']:
                        if K._norm_name(a['entity']) == K._norm_name(first['entity']):
                            first.update(score=a['score'], alias=a['alias'], full=a['full'])
                        elif len(m['alternatives']) < 5:
                            m['alternatives'].append(dict(a))
                    break
    p['mentions'] = ments
    p['relations'] = rels[:5]
    sp_lab = lambda key: (structural_parse or {}).get(key, {})
    for key, val in (('qtype', c['qtype']), ('answer_type', c['answer_type']), ('granularity', c['granularity'])):
        s_ = sp_lab(key)
        p[key] = {'label': val, 'prob': s_.get('prob', 1.0) if s_.get('label') == val else 1.0}
    order = c.get('order')
    p['operator_keywords'] = {'before_after': order if order in ('before', 'after') else None,
                              'first_last': order if order in ('first', 'last') else None,
                              'same_granularity': K.MC.same_gran_word(question)}
    if c.get('anchor'):
        an = K._norm_name(c['anchor'])
        idx = [i for i, m in enumerate(ments) if K._norm_name(m['alternatives'][0]['entity']) == an]
        if idx:
            p['anchor_mention'] = idx[0]
    return p


# ------------------------------------------------------------------ batch CLI
_W = {}


def _work(rows):
    ex = get(_W['k'])
    out = []
    for r in rows:
        res = ex.execute(r['parse'], question=r.get('question'), probs=_W['probs'])
        res['quid'] = r.get('quid')
        out.append(res)
    return out


def run_batch(k, rows, workers=8, probs='given'):
    import multiprocessing as mp
    _W.update(k=k, probs=probs)
    get(k)
    chunks = [rows[i:i + 50] for i in range(0, len(rows), 50)]
    if workers <= 1:
        return [x for c in chunks for x in _work(c)]
    with mp.get_context('fork').Pool(workers) as pool:
        return [x for r in pool.map(_work, chunks) for x in r]


def selftest(k, bundles, preds, workers=8):
    """Re-execute the parses of the subset's dev_sel2000 bundles; compare with the subset's dev_sel2000 RULE / S
    prediction files (full top-50 lists)."""
    bfile = os.path.join(bundles, k, 'dev_sel2000.jsonl')
    rows = [json.loads(l) for l in open(bfile)]
    res = run_batch(k, [{'quid': r['quid'], 'parse': r['parse']} for r in rows], workers)
    pred = {}
    arms = ['RULE'] + ([f'S_seed{s}' for s in range(3)] if k != 'k0' else [])
    for a in arms:
        pred[a] = {}
        for l in open(os.path.join(preds, k, f'dev_sel2000_{a}.jsonl')):
            x = json.loads(l); pred[a][x['quid']] = x['ranked']
    match = {a: 0 for a in arms}; match1 = {a: 0 for a in arms}
    for r in res:
        mine = {'RULE': r['rule'], **({f'S_seed{s}': r['s'][f'seed{s}'] for s in range(3)} if r['s'] else {})}
        for a in arms:
            match[a] += mine[a] == pred[a][r['quid']]
            match1[a] += mine[a][:1] == pred[a][r['quid']][:1]
    n = len(res)
    rep = dict(k_subset=k, n=n, invalid=sum(not r['valid'] for r in res),
               full_list_match={a: match[a] / n for a in arms}, top1_match={a: match1[a] / n for a in arms})
    rep['match_rate_all_arms'] = min(rep['full_list_match'].values())
    return rep


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run', 'selftest'])
    ap.add_argument('--k', required=True)
    ap.add_argument('--in', dest='inp')
    ap.add_argument('--out')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--probs', default='given', choices=['given', 'classifier'])
    ap.add_argument('--bundles', default=os.path.join(HERE, '..', 'bundles'))
    ap.add_argument('--preds', default=os.path.join(HERE, '..', 'preds'))
    a = ap.parse_args()
    t0 = time.time()
    if a.cmd == 'run':
        rows = [json.loads(l) for l in open(a.inp)]
        res = run_batch(a.k, rows, a.workers, a.probs)
        with open(a.out, 'w') as f:
            for r in res:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(json.dumps({'n': len(res), 'invalid': sum(not r['valid'] for r in res), 'secs': round(time.time() - t0, 1)}))
    else:
        print(json.dumps(selftest(a.k, a.bundles, a.preds, a.workers)))
