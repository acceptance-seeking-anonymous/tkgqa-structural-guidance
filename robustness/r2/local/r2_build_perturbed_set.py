#!/usr/bin/env python3
"""r2_build_perturbed_set.py -- R2 perturbed set: fresh 2,000-question test sample disjoint from R1, 600-question pool,
rewrites generated remotely with an open-weights model (remote/r2_gen.py, Qwen2.5-72B-Instruct-AWQ, greedy), then the
unchanged R1 mechanical checks and R1 judge prompt (r2_pipeline.py), at most two attempts. Never reads answers except
to write the gold file of the accepted rewrites (scoring input).

  1. python r2_build_perturbed_set.py prompts --test-json test.json --r1-rule multitq_robustness_rule.json --outdir R2
       -> R2/attempt1_prompts.jsonl                       (run remote/gen_job.sh attempt1 -> attempt1_gen.jsonl + _stats.json)
  2. python r2_build_perturbed_set.py judge1 --test-json test.json --r1-rule ... --outdir R2
       -> R2/records_after_attempt1.json, R2/attempt2_prompts.jsonl   (run remote/gen_job.sh attempt2)
  3. python r2_build_perturbed_set.py build --test-json test.json --r1-rule ... --outdir R2 \
         [--rule multitq_robustness_rule_R2.json --amendments A1.json A2.json]
       -> R2/multitq_robustness_R2_perturbed_set.json, R2/variants.jsonl, R2/variants_questions.json, R2/R2_gold.json
Judge endpoint / model ID: robustness/llm_client.py (JUDGE_MODEL_ID).
"""
import argparse, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
for p in (HERE, os.path.join(REPO, 'robustness', 'r1', 'local'), os.path.join(REPO, 'src', 'struct'), os.path.join(REPO, 'robustness')):
    sys.path.insert(0, p)
import rob_common as RC          # noqa: E402
import r2_common as R2C          # noqa: E402
import r2_pipeline as PL         # noqa: E402
import llm_client                # noqa: E402

GEN_MODEL = 'Qwen/Qwen2.5-72B-Instruct-AWQ'


def fsha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest() if p and os.path.exists(p) else None


def sample(a):
    test_rows = json.load(open(a.test_json))
    R1 = json.load(open(a.r1_rule))
    samp, pool, _, _ = R2C.draw_sample_r2(test_rows, R1['sample']['quids'])
    S2 = sorted(x for v in samp.values() for x in v)
    P2 = sorted(x for v in pool.values() for x in v)
    return test_rows, {q['quid']: q for q in test_rows}, S2, P2


def cmd_prompts(a):
    _, byq, S2, P2 = sample(a)
    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, 'attempt1_prompts.jsonl'), 'w') as f:
        for x in P2:
            f.write(json.dumps(dict(quid=x, attempt=1, prompt=RC.GEN_PROMPT.format(question=byq[x]['question'])), ensure_ascii=False) + '\n')
    print(len(S2), len(P2))


def cmd_judge1(a):
    _, byq, S2, P2 = sample(a)
    gen1 = PL.load_gen(os.path.join(a.outdir, 'attempt1_gen.jsonl'))
    assert sorted(gen1) == sorted(P2), len(gen1)
    records = {x: {'quid': x, 'qtype': byq[x]['qtype'], 'original': byq[x]['question'], 'attempts': []} for x in P2}
    PL.add_attempt(records, byq, gen1, 1, GEN_MODEL)
    m_ok = [x for x in P2 if records[x]['attempts'][0]['mech_ok']]
    PL.judge(records, byq, m_ok, 0, llm_client)
    rej = PL.attempt2_prompts(records, byq, os.path.join(a.outdir, 'attempt2_prompts.jsonl'))
    json.dump({str(k): v for k, v in records.items()}, open(os.path.join(a.outdir, 'records_after_attempt1.json'), 'w'), ensure_ascii=False)
    print('attempt-1 accepted', len(P2) - len(rej), 'of', len(P2))


def cmd_build(a):
    test_rows, byq, S2, P2 = sample(a)
    records = {int(k): v for k, v in json.load(open(os.path.join(a.outdir, 'records_after_attempt1.json'))).items()}
    rej1 = [json.loads(l)['quid'] for l in open(os.path.join(a.outdir, 'attempt2_prompts.jsonl')) if l.strip()]
    gen2 = PL.load_gen(os.path.join(a.outdir, 'attempt2_gen.jsonl'))
    assert sorted(gen2) == sorted(rej1)
    PL.add_attempt(records, byq, gen2, 2, GEN_MODEL)
    m2 = [x for x in rej1 if records[x]['attempts'][1]['mech_ok']]
    PL.judge(records, byq, m2, 1, llm_client)
    st = {}
    for att in (1, 2):
        p = os.path.join(a.outdir, f'attempt{att}_gen_stats.json')
        st[att] = json.load(open(p)) if os.path.exists(p) else {}
        st[att].pop('example_rendered_prefix', None)
    rule = json.load(open(a.rule)) if a.rule else {}
    meta = dict(
        rule_sha256=rule.get('sha256'),
        amendments=[dict(file=os.path.basename(p), sha256=json.load(open(p)).get('sha256')) for p in (a.amendments or [])],
        generator_model=GEN_MODEL,
        generator_label='open-weights generator from a different family (Qwen2.5, 72B, 4-bit AWQ), vLLM greedy',
        generator_snapshot=st[1].get('snapshot'), generator_config_sha256=st[1].get('config_sha256'),
        vllm=st[1].get('vllm'), chat_template_sha256=st[1].get('chat_template_sha256'), decoding=st[1].get('decoding'),
        r2_gen_sha256=fsha(os.path.join(REPO, 'robustness', 'r2', 'remote', 'r2_gen.py')),
        rob_common_sha256=fsha(os.path.join(REPO, 'robustness', 'r1', 'local', 'rob_common.py')),
        mtq_lib_sha256=fsha(os.path.join(REPO, 'src', 'struct', 'mtq_lib.py')),
        prompt_files_sha256=dict(attempt1=fsha(os.path.join(a.outdir, 'attempt1_prompts.jsonl')),
                                 attempt2=fsha(os.path.join(a.outdir, 'attempt2_prompts.jsonl'))),
        gen_stats=dict(attempt1=st[1], attempt2=st[2]))
    pset, _ = PL.build_pset(records, byq, S2, P2, meta)
    out = os.path.join(a.outdir, 'multitq_robustness_R2_perturbed_set.json')
    json.dump(pset, open(out, 'w'), ensure_ascii=False, indent=1)
    n = PL.write_inputs(pset, a.outdir)
    g = PL.write_gold(pset, test_rows, os.path.join(a.outdir, 'R2_gold.json'), fsha(a.test_json))
    print(json.dumps(pset['counts']), n, g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prompts', 'judge1', 'build'])
    ap.add_argument('--test-json', required=True); ap.add_argument('--r1-rule', required=True)
    ap.add_argument('--outdir', default='R2'); ap.add_argument('--rule', default=None)
    ap.add_argument('--amendments', nargs='*', default=None)
    a = ap.parse_args()
    dict(prompts=cmd_prompts, judge1=cmd_judge1, build=cmd_build)[a.cmd](a)


if __name__ == '__main__':
    main()
