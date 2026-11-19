"""mtq_export.py -- result layout + provenance-fact export for the Structural QA track."""
import numpy as np
import multitq_eval as ME
import mtq_cand as MC
from mtq_align import tspan


def layout(summary):
    out = {}
    for k in (1, 10):
        b = summary[f'hits@{k}']
        out[f'hits@{k}'] = {
            'headline': dict(ME.headline_row(summary, k)),
            'qtype': {q: b['qtype'][q]['hits'] for q in MC.QTYPES if q in b['qtype']},
            'time_level_official': {t: b['time_level_official'][t]['hits'] for t in b['time_level_official']},
            'qtype_time_level_official': {t: b['qtype_time_level_official'][t]['hits'] for t in b['qtype_time_level_official']},
            'time_level_all': {t: b['time_level_all'][t]['hits'] for t in b['time_level_all']},
            'qtype_answer_type': {t: b['qtype_answer_type'][t]['hits'] for t in b['qtype_answer_type']},
        }
    out['n'] = summary['n_questions']
    out['n_missing_predictions'] = summary['n_missing_predictions']
    out['n_short_lists'] = summary['n_short_lists']
    return out


def provenance_facts(kg, meta, ranked_idx, names, rinfo, topics, rec, max_facts=20, per_cand=3, top_cands=12):
    """Top-20 facts for the reader: for S's top candidates in order, the facts linking the candidate's topic
    entity to it under its best relation hypothesis (time answers: pair facts in the date bucket), <=3 each,
    ordered by closeness to the parsed time constraint (or chronologically); then the generator's list."""
    info = meta['info']
    out, seen = [], set()
    tp = info.get('tp')
    lo = hi = None
    if tp is not None:
        lo, hi = tspan((tp[0], tp[1]))
    ments = rec['ments']
    ents = [m[2][0][0] for m in ments if m[2]]
    for j in ranked_idx[:top_cands]:
        r, d = rinfo[j]
        if info['atype'] == 'entity':
            t = topics[j]
            c = kg.e2i.get(names[j])
            if t is None or t < 0 or c is None or r < 0:
                continue
            fi, o, dd = MC.rel_slice(t, r, d)
            sel = fi[o == c]
            if not len(sel):
                continue
            dsel = kg.d[sel]
            if lo is not None:
                key = np.where(dsel < lo, lo - dsel, np.where(dsel > hi, dsel - hi, 0))
                sel = sel[np.argsort(key, kind='stable')]
            elif info['FL'] == 1 or info['qt'] == 'before_last':
                sel = sel[::-1]
        else:
            if len(ents) < 1 or r < 0:
                continue
            e1 = ents[0]
            f = kg.facts_of(e1)
            if len(ents) >= 2:
                e2 = ents[1]
                f = f[((kg.h[f] == e1) & (kg.t[f] == e2)) | ((kg.h[f] == e2) & (kg.t[f] == e1))]
            f = f[kg.r[f] == r]
            ks = MC.gran_key(kg.d[f], info['level'])
            bucket = names[j]
            sel = f[[MC.key_str(k_, info['level']) == bucket for k_ in ks]] if len(f) else f
        k_ = 0
        for fid in sel:
            fid = int(fid)
            if fid in seen:
                continue
            seen.add(fid); out.append(fid); k_ += 1
            if k_ >= per_cand or len(out) >= max_facts:
                break
        if len(out) >= max_facts:
            break
    for fid in meta['facts']:
        if len(out) >= max_facts:
            break
        if fid not in seen:
            seen.add(fid); out.append(fid)
    return out[:max_facts]
