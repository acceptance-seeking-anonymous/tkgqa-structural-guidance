"""r2_pipeline.py -- local half of R2 rewrite generation: mechanical checks + LLM judge (R1 procedure, unchanged),
attempt-2 prompts, and the frozen R2 perturbed set. Driven by r2_build_perturbed_set.py; the judge is called through
robustness/llm_client.py.
Generation itself runs remotely (r2_gen.py, Qwen2.5-72B-Instruct-AWQ, greedy). Never touches answers or predictions.
"""
import collections, datetime, hashlib, json, sys
import numpy as np
import os
_H = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.abspath(os.path.join(_H, '..', '..', '..'))
sys.path.insert(0, os.path.join(_REPO, 'robustness', 'r1', 'local')); sys.path.insert(0, os.path.join(_REPO, 'src', 'struct')); sys.path.insert(0, _H)
import rob_common as RC
import mtq_lib as L

def load_gen(path):
    return {r['quid']: r for r in (json.loads(l) for l in open(path) if l.strip())}


def mech(original, text):
    js = RC.extract_json(text)
    para = js.get('paraphrase') if isinstance(js, dict) else None
    kept = js.get('kept_verbatim') if isinstance(js, dict) else None
    if not isinstance(para, str):
        return dict(paraphrase=None, kept=None, mech_ok=False, mech_reasons=['no_json'])
    ok, rs = RC.mechanical_check(original, para, kept, L.parse_time)
    return dict(paraphrase=para.strip(), kept=kept, mech_ok=ok, mech_reasons=rs)


def add_attempt(records, byq, gen, attempt, gen_model):
    for x, g in gen.items():
        a = dict(attempt=attempt, model=gen_model, raw=g['text'], finish_reason=g.get('finish_reason'), n_out=g.get('n_out'),
                 prompt_sha256=g.get('prompt_sha256'))
        a.update(mech(byq[x]['question'], g['text']))
        records[x]['attempts'].append(a)


def judge(records, byq, xs, att_idx, client):
    reqs = [{"prompt": RC.JUDGE_PROMPT.format(original=byq[x]['question'], paraphrase=records[x]['attempts'][att_idx]['paraphrase']),
             "max_tokens": RC.JUDGE_MAX_TOKENS} for x in xs]
    outs = []
    for s in range(0, len(reqs), 300):
        outs += client.complete(reqs[s:s + 300], max_concurrency=16, role='judge')
    for x, o in zip(xs, outs):
        js = RC.extract_json(o.get('text'))
        v = {k: (js.get(k) is True) for k in RC.JUDGE_KEYS} if isinstance(js, dict) else {k: False for k in RC.JUDGE_KEYS}
        records[x]['attempts'][att_idx].update(
            judge=dict(model=o.get('model'), raw=o.get('text'), usage=o.get('usage'), verdict=v, error=o.get('error'),
                       note=js.get('note') if isinstance(js, dict) else None, parsed=isinstance(js, dict)),
            accepted=all(v.values()))
    for x in records:
        a = records[x]['attempts'][att_idx] if len(records[x]['attempts']) > att_idx else None
        if a is not None and not a['mech_ok']:
            a['accepted'] = False


def attempt2_prompts(records, byq, path):
    rej = [x for x in sorted(records) if not records[x]['attempts'][0]['accepted']]
    with open(path, 'w') as f:
        for x in rej:
            a = records[x]['attempts'][0]
            prev = a['paraphrase'] if a['paraphrase'] else (a['raw'] or '')
            f.write(json.dumps(dict(quid=x, attempt=2, prompt=RC.GEN_PROMPT.format(question=byq[x]['question'])
                                    + RC.GEN_RETRY_SUFFIX.format(previous=prev)), ensure_ascii=False) + '\n')
    return rej


def jacc(a, b):
    A = set(RC._norm_cmp(a).split()); B = set(RC._norm_cmp(b).split())
    return len(A & B) / max(1, len(A | B))


def build_pset(records, byq, S2, P2, meta):
    """frozen R2 perturbed set: ORIG for the 2,000 sample questions + RW (first judge-accepted attempt) for the pool"""
    accepted = {}
    for x in P2:
        for a in records[x]['attempts']:
            if a.get('accepted'):
                accepted[x] = a; break
    variants = [dict(quid=x, variant='ORIG', vidx=0, squid=10 * x, question=byq[x]['question']) for x in S2]
    variants += [dict(quid=x, variant='RW', vidx=3, squid=10 * x + 3, question=accepted[x]['paraphrase']) for x in P2 if x in accepted]
    variants.sort(key=lambda r: r['squid'])
    vtxt = json.dumps([[r['squid'], r['question']] for r in variants], ensure_ascii=False)
    att = lambda x, i: records[x]['attempts'][i] if len(records[x]['attempts']) > i else None
    rej1 = [x for x in P2 if not att(x, 0)['accepted']]
    qts = sorted({byq[x]['qtype'] for x in P2})
    by_qt = {qt: dict(n=sum(byq[x]['qtype'] == qt for x in P2), accepted_attempt1=sum(byq[x]['qtype'] == qt and att(x, 0)['accepted'] for x in P2),
                      accepted_final=sum(byq[x]['qtype'] == qt and x in accepted for x in P2)) for qt in qts}
    for v in by_qt.values():
        v['rate_final'] = round(v['accepted_final'] / v['n'], 4)
    jr = lambda i: dict(collections.Counter(k for x in P2 if att(x, i) and att(x, i)['mech_ok'] for k, v in att(x, i)['judge']['verdict'].items() if not v))
    mr = lambda i: dict(collections.Counter(r.split(':')[0] for x in P2 if att(x, i) for r in att(x, i)['mech_reasons']))
    jm = sorted({(a.get('judge') or {}).get('model') for x in P2 for a in records[x]['attempts'] if a.get('judge')})
    pset = dict(name='multitq_robustness_R2_perturbed_set', **meta,
                created_utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                sample_quid_list_sha256=RC.list_sha(S2), pool_quid_list_sha256=RC.list_sha(P2),
                judge_models=jm, judge_label='API LLM judge (JUDGE_MODEL_ID), R1 JUDGE_PROMPT unchanged',
                counts=dict(ORIG=len(S2), pool=len(P2), attempt1_mech_ok=sum(att(x, 0)['mech_ok'] for x in P2), attempt1_accepted=len(P2) - len(rej1),
                            attempt2_n=len(rej1), attempt2_mech_ok=sum(bool(att(x, 1) and att(x, 1)['mech_ok']) for x in rej1),
                            attempt2_accepted=sum(bool(att(x, 1) and att(x, 1).get('accepted')) for x in rej1), RW_accepted=len(accepted),
                            rejected_final=len(P2) - len(accepted), low_n=len(accepted) < 200),
                acceptance_by_qtype=by_qt, judge_rejection_reasons_attempt1=jr(0), judge_rejection_reasons_attempt2=jr(1),
                mechanical_rejections_attempt1=mr(0), mechanical_rejections_attempt2=mr(1),
                judge_parse_failures=sum(1 for x in P2 for a in records[x]['attempts'] if a.get('judge') and not a['judge']['parsed']),
                rewrite_token_jaccard_vs_original=dict(mean=float(np.mean([jacc(byq[x]['question'], accepted[x]['paraphrase']) for x in accepted])),
                                                       median=float(np.median([jacc(byq[x]['question'], accepted[x]['paraphrase']) for x in accepted]))),
                variants_sha256=hashlib.sha256(vtxt.encode()).hexdigest(), n_variants=len(variants), variants=variants,
                records={str(x): records[x] for x in P2})
    return pset, accepted


def write_inputs(pset, d):
    rows = [{'quid': r['squid'], 'question': r['question']} for r in pset['variants']]
    with open(f'{d}/variants.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    json.dump(rows, open(f'{d}/variants_questions.json', 'w'), ensure_ascii=False)
    return len(rows)


def write_gold(pset, test_rows, path, test_sha):
    G = {q['quid']: q for q in test_rows}
    rq = sorted(r['quid'] for r in pset['variants'] if r['variant'] == 'RW')
    json.dump(dict(source='MultiTQ official test split questions/test.json', source_sha256=test_sha, n=len(rq),
                   note='judge-accepted R2 rewrites = PRIMARY set (amendment A1)',
                   questions={str(q): dict(answers=G[q]['answers'], answer_type=G[q]['answer_type'], qtype=G[q]['qtype']) for q in rq}),
              open(path, 'w'), indent=1)
    return len(rq)
