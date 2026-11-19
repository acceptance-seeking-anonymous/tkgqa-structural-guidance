"""kpct_k100.py -- k = 100% guidance bundles from the FROZEN full-data structural system (no retraining).

dev_sel2000 / test: the frozen v3 linker + classifier + S_seed0-2 re-executed through the same code path as k<100
  (verified identical to the frozen prediction files; the frozen test files stay the registered RULE_100 / S_100).
train (reader_train60k): candidates, S scores and facts from the frozen reader exports (OOF classifier probabilities,
  S trained on the disjoint s_train100k); in_sample_parse = frozen v3 parser (in-sample) executed with RULE.
"""
import os, sys, json, pickle, time, gzip, collections
os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import kpct_core as K
import kpct_run as R

EXP = os.environ.get('KPCT_EXPORTS', os.path.join(HERE, '..', 'exp'))
NM = 'k100'


def frozen_linker():
    return K.make_linker(json.load(open(f'{K.CODE}/alias_rules.json')), json.load(open(f'{K.CODE}/ent_block.json')))


def w_train(idx):
    """rebuild the OOF pool of a reader_train60k question from the exported OOF parse; in-sample parse execution."""
    out = []
    for i in idx:
        g = R._G['tr']
        rec, info = g['oof_rec'][i], g['oof_info'][i]
        o = K.build_info(rec, info) if rec is not None else None
        rec2 = g['ins_rec'][i]
        info2 = K.MC.parse_info(rec2, R._probs_i(g['ins_probs'], i), g['qtext'][i])
        pj2 = K.parse_json(rec2, info2)
        o_full = K.build_info(rec2, info2)
        t1 = R.top1_parse(pj2)
        r_t, i_t = K.rec_info_from_parse(t1)
        o_top = K.build_info(r_t, i_t)
        res = dict(i=i, ins=dict(parse=pj2, top1_parse=t1, rule_top10_full=K.make_ranked(o_full['names'], o_full, g['FB'])[:10],
                                 rule_top10_top1=K.make_ranked(o_top['names'], o_top, g['FB'])[:10], pool_size_top1=len(o_top['names'])))
        cands = g['cands'].get(rec2['quid'] if rec is None else rec['quid'])
        if o is not None and cands is not None:
            pos = {n: j for j, n in enumerate(o['names'])}
            rows, rule_ok = [], 0
            for c in cands:      # (candidate, rank_RULE, S_seed0, f_sat_any)
                j = pos.get(c[0])
                rule_ok += (j == c[1])
                rows.append((c, j))
            res['rule_rank_consistent'] = rule_ok / max(1, len(rows))
            rows.sort(key=lambda x: (-x[0][2], x[0][1]))
            facts = g['facts'].get(rec['quid'], [])
            fpos = {f: k for k, f in enumerate(facts)}
            cl = []
            for c, j in rows[:20]:
                sup = K.cand_support(o, rec, j) if j is not None else []
                cl.append({'answer': c[0], 'rule_rank': int(c[1]) + 1, 's_score': float(c[2]), 's_rank': len(cl) + 1,
                           'operator_satisfied': bool(c[3]), 'support_fids': sup, 'support_fact_idx': [fpos[f] for f in sup if f in fpos]})
            res['candidates'] = cl
            top_j = rows[0][1] if rows and rows[0][1] is not None else 0
            res['operator_trace'] = K.operator_trace(o, rec, [top_j])
            res['pool_size'] = len(o['names']); res['n_operator_satisfied'] = int(o['X'][:, K.MC.FIDX['sat_any']].sum()) if len(o['names']) else 0
        else:
            res['candidates'] = []; res['pool_size'] = 0; res['n_operator_satisfied'] = 0
            res['operator_trace'] = 'no candidate retrieved (empty pool)' if o is not None else 'no exported OOF parse'
        out.append(res)
    return out


def oof_parse_from_export(x, qtext):
    """exported OOF parser output -> (rec, info); tokens recomputed with the fixed tokeniser; probs as exported."""
    kg = K.kg()
    q2 = qtext
    t_ = K.L.parse_time(qtext)
    if t_ is not None:
        a, b = t_[2]; q2 = qtext[:a] + ' timex ' + qtext[b:]
    ws = K.L.toks(q2)
    ments = [(m['tokens'][0], m['tokens'][1], [(kg.e2i[a[0]], float(a[1]), int(a[2]), 1) for a in m['alternatives']]) for m in x['mentions']]
    tp = (x['time']['level'], x['time']['value']) if x.get('time') else None
    rels = [(kg.r2i[r], K.DIR_CODES[d], float(p)) for r, d, p in x['relations_top5']]
    info = dict(qt=x['qtype_pred'], qt_p=float(x['qtype_p']), atype=x['answer_type_pred'], atype_p=float(x.get('atype_p', 1.0)),
                level=x['granularity_pred'], level_p=float(x.get('level_p', 1.0)), rels=rels,
                BA={'before': 0, 'after': 1, None: -1}[x['kw_before_after']], FL={'first': 0, 'last': 1, None: -1}[x['kw_first_last']],
                same=K.MC.same_gran_word(qtext), tp=tp)
    return dict(quid=x['quid'], ments=ments, ws=ws, tp=tp), info


def main():
    import multiprocessing as mp
    t0 = time.time()
    out = os.path.join(R.OUT, NM); os.makedirs(f'{out}/bundles', exist_ok=True); os.makedirs(f'{out}/preds', exist_ok=True)
    K.kg()
    FB = json.load(open(os.path.join(R.OUT, 'FB_k100.json')))
    LK = frozen_linker()
    voc, net = K.load_frozen_parser()
    boosters = [f'{K.CODE}/S_seed{s}.txt' for s in range(3)]
    dtext, dgold = R.dev_sel(); ttext = R.test_text()
    drec = R.link_all(LK, dtext); trec = R.link_all(LK, ttext)
    R._G['cfg'] = {'dev': dict(recs=drec, probs=K.parser_probs(voc, net, drec), qtext=[q['question'] for q in dtext], FB=FB, boosters=boosters),
                   'test': dict(recs=trec, probs=K.parser_probs(voc, net, trec), qtext=[q['question'] for q in ttext], FB=FB, boosters=boosters)}
    tseen, tk = R.template_sets(NM)
    frozen = {}
    for split, fsplit in (('dev_sel2000', 'dev'), ('test', 'test')):
        for a in ('RULE', 'S_seed0', 'S_seed1', 'S_seed2'):
            frozen[(split, a)] = {}
            for l in open(f'{EXP}/preds/{fsplit}_{a}.jsonl'):
                x = json.loads(l); frozen[(split, a)][x['quid']] = x['ranked']
    ident = {}
    devrows = []
    for kind, split in (('dev', 'dev_sel2000'), ('test', 'test')):
        n = len(R._G['cfg'][kind]['recs'])
        chunks = [[(kind, i) for i in range(a, min(n, a + 50))] for a in range(0, n, 50)]
        bf = open(f'{out}/bundles/{split}.jsonl', 'w')
        pf = {a: open(f'{out}/preds/{split}_{a}.jsonl' if split == 'dev_sel2000' else f'{out}/preds/regenerated_{split}_{a}.jsonl', 'w')
              for a in ('RULE', 'S_seed0', 'S_seed1', 'S_seed2', 'S_mean')}
        cnt = collections.Counter()
        with mp.get_context('fork').Pool(R.NPROC) as pool:
            for res in pool.imap(R.w_infer, chunks):
                for r in res:
                    quid = r['quid']
                    row = dict(quid=quid, question=R._G['cfg'][kind]['qtext'][r['i']], split=split, k_subset=NM,
                               template_unseen=bool(tk[split].get(quid) not in tseen),
                               producer='frozen full-data structural system (v3 linker + classifier, S_seed0-2); re-executed',
                               **r['bundle'])
                    bf.write(json.dumps(row, ensure_ascii=False) + '\n')
                    lists = {'RULE': r['rule'], 'S_seed0': r['S'][0], 'S_seed1': r['S'][1], 'S_seed2': r['S'][2], 'S_mean': r['S_mean']}
                    for a, lst in lists.items():
                        pf[a].write(json.dumps({'quid': quid, 'ranked': lst}) + '\n')
                        if a != 'S_mean':
                            f = frozen[(split, a)][quid]
                            cnt[(a, 'full')] += lst == f; cnt[(a, 'top1')] += lst[:1] == f[:1]
                    cnt['n'] += 1
                    if kind == 'dev':
                        g = dgold[quid]
                        devrows.append(dict(quid=quid, qtype=g['qtype'], unseen=row['template_unseen'], RULE1=K.hit1(g, r['rule']), RULE10=K.hit10(g, r['rule']),
                                            S1=[K.hit1(g, x) for x in r['S']], S10=[K.hit10(g, x) for x in r['S']], Smean1=K.hit1(g, r['S_mean']),
                                            rec20=int(any(K.hit1(g, [c['answer']]) for c in r['bundle']['candidates']))))
        bf.close(); [v.close() for v in pf.values()]
        ident[split] = {f'{a}_{m}': cnt[(a, m)] / cnt['n'] for a in ('RULE', 'S_seed0', 'S_seed1', 'S_seed2') for m in ('full', 'top1')}
        ident[split]['n'] = cnt['n']
        R.log(NM, split, ident[split], round(time.time() - t0), 's')
    R.pdump(devrows, f'{out}/dev_rows.pkl')
    summ = R.summarize_dev(devrows)
    summ['identity_with_frozen_prediction_files'] = ident
    # ---------------- train: reader_train60k from the frozen exports
    r60 = json.load(open(f'{K.CODE}/reader_train60k_quids.json'))
    keep = set(r60)
    trq = [q for q in K.load_questions('train') if q['quid'] in keep]
    assert len(trq) == 60001
    df_cols = ['quid', 'candidate', 'rank_RULE', 'S_seed0', 'f_sat_any', 'f_atype_p', 'f_level_p']
    import pandas as pd
    df = pd.read_parquet(f'{EXP}/exports/train60k_top50_candidates.parquet', columns=df_cols)
    cands = {q: list(zip(g.candidate, g.rank_RULE, g.S_seed0, g.f_sat_any)) for q, g in df.groupby('quid', sort=False)}
    pq = df.groupby('quid').first()[['f_atype_p', 'f_level_p']].to_dict('index')
    del df
    kg = K.kg()
    qd = {(kg.ent_names[kg.h[i]], kg.rel_names[kg.r[i]], kg.ent_names[kg.t[i]], kg.day_str[kg.d[i]]): i for i in range(kg.n - 1, -1, -1)}
    facts = {}
    for l in gzip.open(f'{EXP}/exports/train60k_top20_facts.jsonl.gz', 'rt'):
        x = json.loads(l)
        if x['quid'] in keep:
            facts[x['quid']] = [int(qd[tuple(f)]) for f in x['facts']]
    oofp = {}
    for l in gzip.open(f'{EXP}/exports/train_parser_outputs.jsonl.gz', 'rt'):
        x = json.loads(l)
        if x['quid'] in keep:
            oofp[x['quid']] = x
    byq = {q['quid']: q for q in trq}
    oof_rec, oof_info = [], []
    for q in trq:
        x = dict(oofp[q['quid']])
        if q['quid'] in pq:
            x['atype_p'] = pq[q['quid']]['f_atype_p']; x['level_p'] = pq[q['quid']]['f_level_p']
        r_, i_ = oof_parse_from_export(x, q['question'])
        oof_rec.append(r_); oof_info.append(i_)
    ins_rec = R.link_all(LK, [{'quid': q['quid'], 'question': q['question']} for q in trq])
    R._G['tr'] = dict(oof_rec=oof_rec, oof_info=oof_info, ins_rec=ins_rec, ins_probs=K.parser_probs(voc, net, ins_rec),
                      qtext=[q['question'] for q in trq], FB=FB, cands=cands, facts=facts)
    n = len(trq)
    chunks = [list(range(a, min(n, a + 100))) for a in range(0, n, 100)]
    ok = okf = 0; cons = []
    with open(f'{out}/bundles/train.jsonl', 'w') as bf, mp.get_context('fork').Pool(R.NPROC) as pool:
        for res in pool.imap(w_train, chunks):
            for r in res:
                q = trq[r['i']]
                ins = r['ins']
                t_hit = bool(K.hit1(q, ins['rule_top10_top1'])) and ins['pool_size_top1'] > 0
                f_hit = bool(K.hit1(q, ins['rule_top10_full']))
                ok += t_hit; okf += f_hit
                if 'rule_rank_consistent' in r:
                    cons.append(r['rule_rank_consistent'])
                pj = K.parse_json(oof_rec[r['i']], oof_info[r['i']])
                pj['note'] = 'frozen OOF parser output as exported (probabilities rounded to 4 decimals; answer-type / granularity probabilities from the exported S features)'
                row = dict(quid=q['quid'], question=q['question'], split='train', k_subset=NM, gold=q['answers'],
                           gold_meta={'answer_type': q['answer_type'], 'time_level': q['time_level'], 'qtype': q['qtype'], 'qlabel': q['qlabel']},
                           oof={'outer_fold': None, 'stack_trained_on': 'frozen: classifier OOF (train_oof_fold.npy), S on the disjoint s_train100k; linker aliases + block list in-sample (disclosed)'},
                           parse=pj, candidates=r['candidates'], facts=K.fact_rows(facts.get(q['quid'], [])),
                           operator_trace=r['operator_trace'], pool_size=r['pool_size'], n_operator_satisfied=r['n_operator_satisfied'],
                           in_sample_parse=dict(parse=ins['parse'], top1_parse=ins['top1_parse'], rule_top10_top1=ins['rule_top10_top1'],
                                                rule_top10_full=ins['rule_top10_full'], executes_to_gold=t_hit, executes_to_gold_full_parse=f_hit,
                                                note='frozen v3 parser (in-sample: trained on all train); executes_to_gold = top-1 hypothesis executed with operators + RULE order returns a gold answer at rank 1'))
                bf.write(json.dumps(row, ensure_ascii=False) + '\n')
    summ['train_in_sample'] = {'n': n, 'list': 'reader_train60k', 'executes_to_gold_top1_share': ok / n, 'executes_to_gold_full_parse_share': okf / n,
                               'export_rule_rank_consistency_mean': float(np.mean(cons)) if cons else None, 'n_with_exported_candidates': len(cons)}
    R.jdump(summ, f'{out}/dev_summary.json')
    R.log(NM, 'done', round(time.time() - t0), 's', json.dumps(summ['train_in_sample']))


if __name__ == '__main__':
    main()
