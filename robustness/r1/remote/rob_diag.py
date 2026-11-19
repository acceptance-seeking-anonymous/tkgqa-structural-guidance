#!/usr/bin/env python3
"""rob_diag.py -- POST-HOC, gold-free diagnostic (after the single scoring pass): which structural parse fields change
between ORIG and each variant, and the top-20 bundle candidate answers of the ORIG / P1 / P3 rows (for a local recall
computation). Reads the rebuilt bundles only. Output: checks/diag_parse.json
"""
import json, sys, collections
R = sys.argv[1] if len(sys.argv) > 1 else '/kpct/robustness'
out = {}
for k in ('k01_s0', 'k05_s0', 'k100'):
    B = {}
    for l in open(f'{R}/bundles/{k}/variants.jsonl'):
        x = json.loads(l)
        B[int(x['quid'])] = x
    def fields(b):
        p = b['parse']
        return dict(entities=sorted(m['alternatives'][0]['entity'] for m in p['mentions']),
                    entities_any=sorted({a['entity'] for m in p['mentions'] for a in m['alternatives'][:3]}),
                    n_mentions=len(p['mentions']), time=(p['time'] or {}).get('value'), qtype=p['qtype']['label'],
                    answer_type=p['answer_type']['label'], granularity=p['granularity']['label'],
                    rel_top1=(p['relations'][0]['relation'], p['relations'][0]['direction']) if p['relations'] else None,
                    rel_top5=[(r['relation'], r['direction']) for r in p['relations'][:5]],
                    ba=p['operator_keywords']['before_after'], fl=p['operator_keywords']['first_last'],
                    same=p['operator_keywords']['same_granularity'], pool_empty=(b['pool_size'] == 0))
    quids = sorted({q // 10 for q in B})
    res = {}
    for vi, v in ((1, 'P1'), (2, 'P2'), (3, 'P3')):
        ch = collections.Counter(); n = 0
        for q in quids:
            if 10 * q + vi not in B:
                continue
            n += 1
            fo, fv = fields(B[10 * q]), fields(B[10 * q + vi])
            for key in fo:
                if key == 'pool_empty':
                    ch['pool_empty_variant'] += fv[key]; ch['pool_empty_orig'] += fo[key]
                elif key == 'rel_top5':
                    ch['rel_top1_in_orig_top5_lost'] += (fo['rel_top1'] is not None and fo['rel_top1'] not in fv['rel_top5'])
                else:
                    ch[key + '_changed'] += fo[key] != fv[key]
            ch['any_core_changed'] += any(fo[kk] != fv[kk] for kk in ('entities', 'time', 'qtype', 'answer_type', 'granularity', 'rel_top1', 'ba', 'fl', 'same'))
        res[v] = dict(n=n, **{kk: c / max(1, n) for kk, c in ch.items()})
    cands = {str(q): {v: [c['answer'] for c in B[10 * q + vi]['candidates']] for vi, v in ((0, 'ORIG'), (1, 'P1'), (3, 'P3')) if 10 * q + vi in B}
             for q in quids}
    out[k] = dict(change_rates=res, candidates=cands)
json.dump(out, open(f'{R}/checks/diag_parse.json', 'w'))
print(json.dumps({k: v['change_rates']['P3'] for k, v in out.items()}, indent=0)[:3000])
