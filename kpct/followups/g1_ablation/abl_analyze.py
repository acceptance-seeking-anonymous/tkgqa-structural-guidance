#!/usr/bin/env python3
"""abl_analyze.py -- G_1 ablation: per-question dev metrics, prediction files and freeze record (CPU, dev answers only).

Reads the per-block generations written by abl_run.py, rebuilds the registered ranked lists ([answer] + S_k seed-0
list, deduplicated), scores them with the official multitq_eval.py on dev, and computes the registered per-question
metrics. Writes (job cwd): abl_perq.csv.gz, multitq_g1_ablation_predictions.tgz, multitq_g1_ablation_freeze.json,
multitq_g1_ablation_bundles_manifest.json.
"""
import argparse, csv, glob, gzip, hashlib, io, json, os, sys, tarfile, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gk_common as GC
import gk_ablation as AB


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def lsha(qs):
    return hashlib.sha256(json.dumps(sorted(int(q) for q in qs), separators=(',', ':')).encode('utf-8')).hexdigest()


def ranked_file(p, keep=None):
    out = {}
    for x in GC.read_jsonl(p):
        q = int(x['quid'])
        if keep is None or q in keep:
            rk = x['ranked']
            out[q] = json.loads(rk.replace("'", '"')) if isinstance(rk, str) else rk
    return out


def ans_in_groups(ans, groups):
    if GC.is_time(ans):
        return any(d.startswith(ans) for (h, r, t, ds) in groups for d in ds)
    return any(h == ans or t == ans for (h, r, t, ds) in groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/kpct/ablation_g1'); ap.add_argument('--K', default='/kpct')
    ap.add_argument('--dev_q', default='/data/questions/dev.json'); ap.add_argument('--eval', required=True)
    ap.add_argument('--out', default='.')
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(a.eval)))
    import multitq_eval as ME
    R, K = a.root, a.K
    prog = json.load(open(f'{R}/abl_run_progress.json'))
    Q = {q['quid']: q for q in json.load(open(a.dev_q))}
    sets = {}
    d = json.load(gzip.open(f'{R}/gen/sel_dev_sel2000.json.gz', 'rt'))
    sets['dev_sel2000'] = dict(rows=d['rows'], gen_files={'sel_dev_sel2000.json.gz': sha(f'{R}/gen/sel_dev_sel2000.json.gz')},
                               bundles=f'{K}/bundles/k01_s0/dev_sel2000.jsonl', s0=f'{K}/preds/k01_s0/dev_sel2000_S_seed0.jsonl',
                               gu=f'{K}/gu_retrieval/gu_dev_sel2000.jsonl.gz', stats=[d['stats']])
    mrows, mfiles, mstats = [], {}, []
    for p in sorted(glob.glob(f'{R}/gen/main_block_*.json.gz')):
        d = json.load(gzip.open(p, 'rt'))
        mrows += d['rows']; mfiles[os.path.basename(p)] = sha(p); mstats.append(d['stats'])
        assert d['adapter_sha256'] == '4d81d7a017e391af0c8118e55c9e707b53ca9b607cc397572832c7da54a3440a'
    if mrows:
        sets['D_main'] = dict(rows=mrows, gen_files=mfiles, bundles=f'{R}/bundles/dev_abl_k01_s0.jsonl',
                              s0=f'{R}/bundles/dev_abl_k01_s0_S_seed0.jsonl', gu=f'{R}/gu/gu_dev_abl.jsonl.gz', stats=mstats)
    tmp = os.path.join(a.out, 'pred_files'); os.makedirs(tmp, exist_ok=True)
    perq = []
    freeze = dict(name='multitq_g1_ablation_freeze', rule='multitq_g1_ablation_rule.json (sha256 e32304a1c1ed95098d3376d3aecb4183a6032d251714db4a704ae6367bebd0a8)',
                  created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  adapter=dict(path='$WORK_DIR/guided/adapters/G_k01_s0_ordS_s0/ep2', sha256='4d81d7a017e391af0c8118e55c9e707b53ca9b607cc397572832c7da54a3440a'),
                  run_progress=prog, sets={})
    for sname, S in sets.items():
        quids = sorted({r['quid'] for r in S['rows']})
        keep = set(quids)
        B = {}
        for r in GC.read_jsonl(S['bundles']):
            if int(r['quid']) in keep:
                b = GC.normalize_bundle(r); B[b['quid']] = b
        S0 = ranked_file(S['s0'], keep)
        GU = {}
        for x in GC.read_jsonl(S['gu']):
            if int(x['quid']) in keep:
                GU[int(x['quid'])] = [tuple(f) for f in x['facts']]
        assert set(B) == keep and set(S0) == keep
        variants = sorted({r['variant'] for r in S['rows']}, key=AB.VARIANTS.index)
        by_v = {v: {} for v in variants}
        for r in S['rows']:
            by_v[r['variant']][r['quid']] = r
        files = {}
        for v in variants:
            assert set(by_v[v]) == keep, (sname, v, len(by_v[v]))
            pp = os.path.join(tmp, f'pred_{v}_{sname}.jsonl'); rp = os.path.join(tmp, f'raw_{v}_{sname}.jsonl.gz')
            with open(pp, 'w') as fp, gzip.open(rp, 'wt') as fr:
                for q in quids:
                    r = by_v[v][q]
                    rk = GC.g_ranked(r['answer'], S0[q])
                    fp.write(json.dumps(dict(quid=q, ranked=rk)) + '\n')
                    fr.write(json.dumps({k: r[k] for k in ('quid', 'text', 'answer', 'valid', 'n_out', 'prompt_len', 'groups_dropped', 'groups_shown')}, ensure_ascii=False) + '\n')
                    b, qq = B[q], Q[q]
                    shown = AB.shown_groups(b, v, GU.get(q), r['groups_shown'])
                    cands = [c['candidate'] for c in b['candidates']]
                    perq.append(dict(set=sname, quid=q, qtype=qq['qtype'], answer_type=qq['answer_type'], variant=v,
                                     valid=int(r['valid']), hit1=int(ME.hit_at_k(qq, rk, 1)), hit10=int(ME.hit_at_k(qq, rk, 10)),
                                     in_cands=int(r['answer'] in cands), in_shown_facts=int(ans_in_groups(r['answer'], shown)),
                                     gold_in_shown_facts=int(AB.gold_in_groups(qq['answers'], shown)),
                                     gold_in_cands=int(any(GC.gold_hit(c, qq['answers']) for c in cands)),
                                     copy_s0=int(r['answer'] == S0[q][0]), s0_hit1=int(ME.hit_at_k(qq, S0[q], 1)),
                                     groups_shown=r['groups_shown'], groups_dropped=r['groups_dropped'], prompt_len=r['prompt_len'], n_out=r['n_out']))
            files[os.path.basename(pp)] = sha(pp); files[os.path.basename(rp)] = sha(rp)
        freeze['sets'][sname] = dict(n=len(quids), quid_list_sha256=lsha(quids), variants=variants, prediction_files=files,
                                     generation_files=S['gen_files'],
                                     inputs={k: dict(file=S[k], sha256=sha(S[k])) for k in ('bundles', 's0', 'gu')},
                                     generation_stats=S['stats'])
    pq = os.path.join(a.out, 'abl_perq.csv.gz')
    with gzip.open(pq, 'wt', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(perq[0].keys())); w.writeheader(); w.writerows(perq)
    tg = os.path.join(a.out, 'multitq_g1_ablation_predictions.tgz')
    with tarfile.open(tg, 'w:gz') as t:
        for fn in sorted(os.listdir(tmp)):
            t.add(os.path.join(tmp, fn), arcname=f'g1_ablation/{fn}')
    freeze['predictions_tgz_sha256'] = sha(tg); freeze['perq_sha256'] = sha(pq)
    freeze['code_sha256'] = {f: sha(os.path.join(os.path.dirname(os.path.abspath(__file__)), f))
                             for f in ('abl_analyze.py', 'abl_run.py', 'abl_build.py', 'gk_ablation.py', 'gk_common.py', 'gk_infer.py', 'gu_retrieve.py')}
    json.dump(freeze, open(os.path.join(a.out, 'multitq_g1_ablation_freeze.json'), 'w'), indent=1)
    bid = json.load(open(f'{R}/abl_build_identity.json'))
    gst = json.load(open(f'{R}/gu/stats_queries_gu_dev_abl.json'))
    man = dict(name='multitq_g1_ablation_bundles_manifest', created_utc=freeze['created_utc'],
               rebuilt_bundles=dict(bid['outputs']['bundles'], host_path='$WORK_DIR/ablation_g1/bundles/dev_abl_k01_s0.jsonl'),
               rebuilt_S_seed0=dict(bid['outputs']['S_seed0'], host_path='$WORK_DIR/ablation_g1/bundles/dev_abl_k01_s0_S_seed0.jsonl'),
               recipe='robustness/code/rob_build.py recipe (= kpct_run.stage_infer kind test) with the frozen k01_s0 executor models; question text only',
               builder_inputs=bid['inputs'], builder_errors=bid['builder_errors'],
               S2_identity_dev_sel2000=dict(field_identity=bid['field_identity'], rendered_user_turn_identity=bid['rendered_user_turn_identity'],
                                            pass_=bid['S2_pass'], note='parse identity < 1 only through ~1e-7 classifier probabilities (batch composition); probabilities are not rendered'),
               gu_retrieval=dict(file='$WORK_DIR/ablation_g1/gu/gu_dev_abl.jsonl.gz', sha256=sha(f'{R}/gu/gu_dev_abl.jsonl.gz'), stats=gst,
                                 S3_identity_dev_sel2000=prog['gates'].get('S3_gu_identity_dev_sel2000')),
               frozen_dev_sel2000=dict(bundles=sha(f'{K}/bundles/k01_s0/dev_sel2000.jsonl'), S_seed0=sha(f'{K}/preds/k01_s0/dev_sel2000_S_seed0.jsonl'),
                                       gu=sha(f'{K}/gu_retrieval/gu_dev_sel2000.jsonl.gz')),
               build_code_sha256=bid['code_sha256'])
    json.dump(man, open(os.path.join(a.out, 'multitq_g1_ablation_bundles_manifest.json'), 'w'), indent=1)
    print(json.dumps({s: dict(n=v['n'], variants=v['variants']) for s, v in freeze['sets'].items()}))


if __name__ == '__main__':
    main()
