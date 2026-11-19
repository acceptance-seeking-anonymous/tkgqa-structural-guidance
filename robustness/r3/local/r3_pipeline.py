"""r3_pipeline.py -- local half of rule R3: Gemini rewrite generation for the R3 pool (two-attempt policy of R1/R2),
mechanical checks + LLM-judge pre-filter (R1 procedure, unchanged; functions copied verbatim from r2_pipeline.py), the
frozen R3 perturbed set, and the DEV / TRAINING paraphrase sets built from the remote Qwen outputs.
Library (called step by step; see scripts/reproduce_r3.sh). Never touches answers or predictions. The judge is called
through robustness/llm_client.py (JUDGE_MODEL_ID); the Gemini API key is read from GEMINI_API_KEY (llm_client.setting)
unless passed explicitly, and is never printed or written.
"""
import collections, concurrent.futures as cf, datetime, hashlib, json, os, sys, time
import numpy as np
import requests
_H = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.abspath(os.path.join(_H, '..', '..', '..'))
for _p in (os.path.join(_REPO, 'robustness'), os.path.join(_REPO, 'robustness', 'r1', 'local'),
           os.path.join(_REPO, 'robustness', 'r2', 'local'), os.path.join(_REPO, 'src', 'struct'), os.path.join(_H, '..')):
    sys.path.insert(0, _p)
import rob_common as RC
import r3_common as R3
import mtq_lib as L
import llm_client

# amendment A1 (registered before any R3 generation): gemini-2.5-flash is no longer available (HTTP 404);
# registered fallback gemini-2.5-pro, minimum thinkingBudget 128 (thinking cannot be disabled on 2.5 Pro);
# maxOutputTokens = 300 visible + 128 thinking = 428 (thinking counts against maxOutputTokens)
GEMINI_MODEL_A1 = 'gemini-2.5-pro'
GEN_CONFIG_A1 = dict(R3.GEMINI_GENERATION_CONFIG, maxOutputTokens=RC.GEN_MAX_TOKENS + 128, thinkingConfig={'thinkingBudget': 128})


def gemini_body_A1(prompt):
    b = R3.gemini_body(prompt)
    b['generationConfig'] = GEN_CONFIG_A1
    return b


def sha(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


# ------------------------------------------------------------------ Gemini (R3 generator)
def gemini_call(prompt, key, model=GEMINI_MODEL_A1, timeout=180, body_fn=gemini_body_A1):
    url = R3.GEMINI_ENDPOINT.format(model=model)
    body = body_fn(prompt)
    tr, last = 0, None
    for i in range(R3.GEMINI_TRANSPORT_RETRIES + 1):
        try:
            r = requests.post(url, params={'key': key}, json=body, timeout=timeout)
            if r.status_code == 200:
                txt, fin, mv, us, blk = R3.gemini_text(r.json())
                return dict(text=txt, finish_reason=fin, model_version=mv, usage=us, block_reason=blk, http_status=200,
                            transport_retries=tr, prompt_sha256=sha(prompt))
            last = f'HTTP {r.status_code}: {r.text[:300]}'
            if r.status_code not in (408, 429, 500, 502, 503, 504):
                break
        except Exception as e:
            last = repr(e)[:300]
        tr += 1
        time.sleep(min(60, 2 ** i * 2))
    return dict(text=None, finish_reason=None, model_version=None, usage=None, block_reason=None, http_status=None,
                transport_retries=tr, error=last, prompt_sha256=sha(prompt))


def gen_gemini(prompts, key=None, workers=8):
    """prompts: {quid: prompt} -> {quid: call record}; key defaults to GEMINI_API_KEY (llm_client.setting)"""
    key = key or llm_client.setting('GEMINI_API_KEY')
    if not key:
        raise RuntimeError('set GEMINI_API_KEY (see configs/llm.env.example)')
    out = {}
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(gemini_call, p, key): q for q, p in prompts.items()}
        for f in cf.as_completed(futs):
            out[futs[f]] = f.result()
    return out


# ------------------------------------------------------------------ mechanical checks + judge (= r2_pipeline.py)
def mech(original, text):
    js = RC.extract_json(text)
    para = js.get('paraphrase') if isinstance(js, dict) else None
    kept = js.get('kept_verbatim') if isinstance(js, dict) else None
    if not isinstance(para, str):
        return dict(paraphrase=None, kept=None, mech_ok=False, mech_reasons=['no_json'])
    ok, rs = RC.mechanical_check(original, para, kept, L.parse_time)
    return dict(paraphrase=para.strip(), kept=kept, mech_ok=ok, mech_reasons=rs)


def add_attempt_gemini(records, byq, gen, attempt):
    for x, g in gen.items():
        a = dict(attempt=attempt, model=GEMINI_MODEL_A1, raw=g['text'], finish_reason=g.get('finish_reason'),
                 model_version=g.get('model_version'), usage=g.get('usage'), block_reason=g.get('block_reason'),
                 http_status=g.get('http_status'), transport_retries=g.get('transport_retries'), error=g.get('error'),
                 prompt_sha256=g.get('prompt_sha256'))
        a.update(mech(byq[x]['question'], g['text']))
        records[x]['attempts'].append(a)


def judge_calls(pairs):
    """pairs: [(original, paraphrase)] -> list of judge records (R1 JUDGE_PROMPT, API LLM judge JUDGE_MODEL_ID, max_tokens 200)"""
    reqs = [{"prompt": RC.JUDGE_PROMPT.format(original=o, paraphrase=p), "max_tokens": RC.JUDGE_MAX_TOKENS} for o, p in pairs]
    outs = []
    for s in range(0, len(reqs), 300):
        outs += llm_client.complete(reqs[s:s + 300], max_concurrency=16, role='judge')
    res = []
    for o in outs:
        js = RC.extract_json(o.get('text'))
        v = {k: (js.get(k) is True) for k in RC.JUDGE_KEYS} if isinstance(js, dict) else {k: False for k in RC.JUDGE_KEYS}
        res.append(dict(model=o.get('model'), raw=o.get('text'), usage=o.get('usage'), verdict=v, error=o.get('error'),
                        note=js.get('note') if isinstance(js, dict) else None, parsed=isinstance(js, dict)))
    return res


def judge(records, byq, xs, att_idx):
    js = judge_calls([(byq[x]['question'], records[x]['attempts'][att_idx]['paraphrase']) for x in xs])
    for x, j in zip(xs, js):
        records[x]['attempts'][att_idx].update(judge=j, accepted=(j['error'] is None and all(j['verdict'].values())))
    for x in records:
        a = records[x]['attempts'][att_idx] if len(records[x]['attempts']) > att_idx else None
        if a is not None and not a['mech_ok']:
            a['accepted'] = False


def attempt2_prompts(records, byq):
    rej = [x for x in sorted(records) if not records[x]['attempts'][0]['accepted']]
    P = {}
    for x in rej:
        a = records[x]['attempts'][0]
        prev = a['paraphrase'] if a['paraphrase'] else (a['raw'] or '')
        P[x] = RC.GEN_PROMPT.format(question=byq[x]['question']) + RC.GEN_RETRY_SUFFIX.format(previous=prev)
    return P


def jacc(a, b):
    A = set(RC._norm_cmp(a).split()); B = set(RC._norm_cmp(b).split())
    return len(A & B) / max(1, len(A | B))


def build_r3_set(records, byq, S3, P3, meta):
    """frozen R3 perturbed set: ORIG for the 2,000 sample questions + RW3 (first judge-accepted attempt) for the pool"""
    accepted = {}
    for x in P3:
        for a in records[x]['attempts']:
            if a.get('accepted'):
                accepted[x] = a; break
    v = R3.VIDX
    variants = [dict(quid=x, variant='ORIG', vidx=v['ORIG'], squid=10 * x + v['ORIG'], question=byq[x]['question']) for x in S3]
    variants += [dict(quid=x, variant='RW3', vidx=v['RW3'], squid=10 * x + v['RW3'], question=accepted[x]['paraphrase']) for x in P3 if x in accepted]
    variants.sort(key=lambda r: r['squid'])
    vtxt = json.dumps([[r['squid'], r['question']] for r in variants], ensure_ascii=False)
    att = lambda x, i: records[x]['attempts'][i] if len(records[x]['attempts']) > i else None
    rej1 = [x for x in P3 if not att(x, 0)['accepted']]
    qts = sorted({byq[x]['qtype'] for x in P3})
    by_qt = {qt: dict(n=sum(byq[x]['qtype'] == qt for x in P3), accepted_attempt1=sum(byq[x]['qtype'] == qt and att(x, 0)['accepted'] for x in P3),
                      accepted_final=sum(byq[x]['qtype'] == qt and x in accepted for x in P3)) for qt in qts}
    for vv in by_qt.values():
        vv['rate_final'] = round(vv['accepted_final'] / vv['n'], 4)
    jr = lambda i: dict(collections.Counter(k for x in P3 if att(x, i) and att(x, i)['mech_ok'] for k, vv in att(x, i)['judge']['verdict'].items() if not vv))
    mr = lambda i: dict(collections.Counter(r.split(':')[0] for x in P3 if att(x, i) for r in att(x, i)['mech_reasons']))
    jm = sorted({(a.get('judge') or {}).get('model') for x in P3 for a in records[x]['attempts'] if a.get('judge')})
    mv = dict(collections.Counter(a.get('model_version') for x in P3 for a in records[x]['attempts']))
    fr = dict(collections.Counter(str(a.get('finish_reason')) for x in P3 for a in records[x]['attempts']))
    pset = dict(name='multitq_robustness_R3_perturbed_set', **meta,
                created_utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                sample_quid_list_sha256=RC.list_sha(S3), pool_quid_list_sha256=RC.list_sha(P3),
                generator_model=GEMINI_MODEL_A1, generator_config=GEN_CONFIG_A1, generator_model_versions=mv, generator_finish_reasons=fr,
                generator_errors=sum(1 for x in P3 for a in records[x]['attempts'] if a.get('error')),
                generator_blocked=sum(1 for x in P3 for a in records[x]['attempts'] if a.get('block_reason') or a.get('raw') is None),
                judge_models=jm, judge_label='API LLM judge (JUDGE_MODEL_ID), R1 JUDGE_PROMPT unchanged',
                counts=dict(ORIG=len(S3), pool=len(P3), attempt1_mech_ok=sum(att(x, 0)['mech_ok'] for x in P3), attempt1_accepted=len(P3) - len(rej1),
                            attempt2_n=len(rej1), attempt2_mech_ok=sum(bool(att(x, 1) and att(x, 1)['mech_ok']) for x in rej1),
                            attempt2_accepted=sum(bool(att(x, 1) and att(x, 1).get('accepted')) for x in rej1), RW3_accepted=len(accepted),
                            rejected_final=len(P3) - len(accepted), low_n=len(accepted) < 200),
                acceptance_by_qtype=by_qt, judge_rejection_reasons_attempt1=jr(0), judge_rejection_reasons_attempt2=jr(1),
                mechanical_rejections_attempt1=mr(0), mechanical_rejections_attempt2=mr(1),
                judge_parse_failures=sum(1 for x in P3 for a in records[x]['attempts'] if a.get('judge') and not a['judge']['parsed']),
                judge_errors=sum(1 for x in P3 for a in records[x]['attempts'] if a.get('judge') and a['judge']['error']),
                rewrite_token_jaccard_vs_original=dict(mean=float(np.mean([jacc(byq[x]['question'], accepted[x]['paraphrase']) for x in accepted])),
                                                       median=float(np.median([jacc(byq[x]['question'], accepted[x]['paraphrase']) for x in accepted]))),
                variants_sha256=hashlib.sha256(vtxt.encode()).hexdigest(), n_variants=len(variants), variants=variants,
                records={str(x): records[x] for x in P3})
    return pset, accepted


# ------------------------------------------------------------------ DEV / TRAINING sets from the remote Qwen outputs
def build_dev_set(dev_rows, gen1, gen2, judged, meta):
    """dev_rows: [{quid, qtype, question}] (dev_sel2000); gen1/gen2: {quid: qwen record}; judged: {quid: judge record}
    for the FIRST mechanically valid attempt. DEV_ACCEPTED = mech-ok and judge-accepted; rows with a missing judge
    verdict (judge unavailable) are flagged judge_missing and kept out of DEV_ACCEPTED but listed in DEV_MECH_OK."""
    byq = {int(r['quid']): r for r in dev_rows}
    v = R3.VIDX
    recs, variants, acc, mok = {}, [], [], []
    for q in sorted(byq):
        atts = [gen1[q]] + ([gen2[q]] if q in gen2 else [])
        first_ok = next((a for a in atts if a['mech_ok']), None)
        j = judged.get(q) if first_ok is not None else None
        accepted = bool(first_ok is not None and j is not None and j['error'] is None and all(j['verdict'].values()))
        recs[q] = dict(quid=q, qtype=byq[q]['qtype'], original=byq[q]['question'], attempts=atts,
                       chosen_attempt=first_ok['attempt'] if first_ok else None, judge=j, judge_missing=(first_ok is not None and j is None),
                       mech_ok=first_ok is not None, accepted=accepted)
        variants.append(dict(quid=q, variant='ORIG', vidx=v['ORIG'], squid=10 * q + v['ORIG'], question=byq[q]['question']))
        if first_ok is not None:
            mok.append(q)
            if accepted:
                acc.append(q)
                variants.append(dict(quid=q, variant='DEVPARA', vidx=v['DEVPARA'], squid=10 * q + v['DEVPARA'], question=first_ok['paraphrase']))
    variants.sort(key=lambda r: r['squid'])
    vtxt = json.dumps([[r['squid'], r['question']] for r in variants], ensure_ascii=False)
    qts = sorted({r['qtype'] for r in dev_rows})
    by_qt = {qt: dict(n=sum(recs[q]['qtype'] == qt for q in recs), mech_ok=sum(recs[q]['qtype'] == qt and recs[q]['mech_ok'] for q in recs),
                      accepted=sum(recs[q]['qtype'] == qt and recs[q]['accepted'] for q in recs)) for qt in qts}
    mr = lambda G: dict(collections.Counter(r.split(':')[0] for a in G.values() for r in a['mech_reasons']))
    jr = dict(collections.Counter(k for q in recs if recs[q]['judge'] for k, vv in recs[q]['judge']['verdict'].items() if not vv))
    dset = dict(name='multitq_dev_paraphrase_set', **meta, created_utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                dev_quid_list_sha256=RC.list_sha(list(byq)),
                counts=dict(dev=len(byq), attempt1_mech_ok=sum(gen1[q]['mech_ok'] for q in byq), attempt2_n=len(gen2),
                            attempt2_mech_ok=sum(a['mech_ok'] for a in gen2.values()), mech_ok_final=len(mok),
                            judged=sum(1 for q in recs if recs[q]['judge'] is not None), judge_missing=sum(recs[q]['judge_missing'] for q in recs),
                            judge_rejected=sum(1 for q in recs if recs[q]['judge'] is not None and not recs[q]['accepted']),
                            DEV_ACCEPTED=len(acc), finish_length=sum(a.get('finish_reason') == 'length' for G in (gen1, gen2) for a in G.values())),
                acceptance_by_qtype=by_qt, mechanical_rejections_attempt1=mr(gen1), mechanical_rejections_attempt2=mr(gen2),
                judge_rejection_reasons=jr, judge_models=sorted({recs[q]['judge']['model'] for q in recs if recs[q]['judge']}),
                rewrite_token_jaccard_vs_original=dict(mean=float(np.mean([jacc(recs[q]['original'], next(a for a in recs[q]['attempts'] if a['mech_ok'])['paraphrase']) for q in acc])) if acc else None),
                DEV_ACCEPTED_quid_list_sha256=RC.list_sha(acc), DEV_MECH_OK_quid_list_sha256=RC.list_sha(mok),
                variants_sha256=hashlib.sha256(vtxt.encode()).hexdigest(), n_variants=len(variants), variants=variants,
                records={str(q): recs[q] for q in sorted(recs)})
    return dset, acc


def build_train_set(order, gen, train_q_sha, meta):
    """one attempt per training question, mechanical checks only (no judge); failed rows kept and flagged"""
    rows = []
    for q in order:
        if q not in gen:
            continue
        g = gen[q]
        rows.append(dict(quid=q, paraphrase=g['paraphrase'], mech_ok=g['mech_ok'], mech_reasons=g['mech_reasons'], kept=g['kept'],
                         raw=g['text'], finish_reason=g['finish_reason'], n_out=g['n_out'], prompt_sha256=g['prompt_sha256']))
    covered = [r['quid'] for r in rows]
    prefix_ok = covered == [q for q in order[:len(covered)]]
    tset = dict(name='multitq_train5pct_paraphrase_set', **meta, created_utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                source='questions/train.json', source_sha256=train_q_sha, list='k05_s0 (n = 19,339), registered order r3_common.training_order',
                counts=dict(list_n=len(order), generated=len(rows), mech_ok=sum(r['mech_ok'] for r in rows), mech_failed=sum(not r['mech_ok'] for r in rows),
                            k01_s0_prefix_complete=len(rows) >= 3868, is_order_prefix=prefix_ok,
                            finish_length=sum(r['finish_reason'] == 'length' for r in rows)),
                mechanical_rejections=dict(collections.Counter(x.split(':')[0] for r in rows for x in r['mech_reasons'])),
                generated_quid_list_sha256=RC.list_sha(covered), mech_ok_quid_list_sha256=RC.list_sha([r['quid'] for r in rows if r['mech_ok']]),
                rows=rows)
    return tset
