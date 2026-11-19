#!/usr/bin/env python3
"""mbq_qa.py -- MultiQA (Chen et al., ACL 2023; czy1999/MultiTQ@94ad61c) at nested k% train subsets.

The official modules (MultiQA/qa_baselines.py, qa_datasets.py, utils.py, tcomplex.py) are imported
UNCHANGED. This driver replaces train_qa_model.py + utils.train only to (a) restrict the train split
to a registered quid list, (b) validate on dev_sel2000 instead of full dev, (c) extend the epoch
budget with dev early stopping, (d) seed the run, (e) keep the best state in memory and write
ranked top-10 predictions for dev_sel2000 and the answer-stripped test file, and (f) never compute
a test score. The optimisation step is the one of utils.train: Adam(lr) over qa_model.parameters(),
CrossEntropy on one randomly drawn gold answer id per question, batch 100, shuffle, num_workers 5.
Dev score each epoch = official utils.eval_multi Hits@1 (k=1); improvement = strictly greater
(official rule), so ties keep the earlier epoch.
One semantics-preserving speed-up: QA_Dataset_multi.timesToIds is memoised (pure function of
its argument and the fixed ts2id table).

Modes
  prep : build data/MultiTQ/questions/<k>/{train,dev,test}.json from qin/ + ner_*.jsonl.gz
  run  : --k <k01_s0|k05_s0|k10_s0|k100> --emb <off|full> --seed N --max_epochs M --patience P
         [--official_budget 10] [--smoke_epochs E]
"""
import argparse, contextlib, copy, gzip, hashlib, json, os, random, sys, time

W = os.path.abspath(os.getcwd())
MQ = os.path.join(W, 'MultiQA')
DUMMY_ENT = None


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def load_ner(names):
    out = {}
    for n in names:
        p = os.path.join(W, f'ner_{n}.jsonl.gz')
        if not os.path.exists(p):
            continue
        with gzip.open(p, 'rt', encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                out[r['quid']] = r
    return out


def prep(levels):
    ent0 = open(os.path.join(W, 'data/MultiTQ/kg/tkbc_processed_data/ent_id'), encoding='utf-8').readline().split('\t')[0]
    ner = load_ner(['dev_sel2000', 'test', 'train_k10', 'train_rest'])
    train = json.load(open(os.path.join(W, 'qin/train_all.json')))
    dev = json.load(open(os.path.join(W, 'qin/dev_sel2000.json')))
    test = json.load(open(os.path.join(W, 'qin/test_stripped.json')))

    def merge(q, stripped=False):
        r = dict(q)
        n = ner[q['quid']]
        r['time'], r['entities'], r['entity_positions'] = n['time'], n['entities'], n['entity_positions']
        if stripped:  # test: no gold reaches this job; placeholders satisfy the dataset constructor only
            r.update({'answers': [ent0], 'answer_type': 'entity', 'qtype': 'equal', 'qlabel': 'Single',
                      'time_level': 'day', '_stripped': True})
        return r
    dev_p = [merge(q) for q in dev]
    test_p = [merge(q, True) for q in test]
    info = {}
    for k in levels:
        if k == 'k100':
            members = None
        else:
            members = set(json.load(open(os.path.join(W, f'lists/{k}.json'))))
        tr = [merge(q) for q in train if (members is None or q['quid'] in members)]
        d = os.path.join(W, 'data/MultiTQ/questions', k)
        os.makedirs(d, exist_ok=True)
        json.dump(tr, open(os.path.join(d, 'train.json'), 'w'))
        json.dump(dev_p, open(os.path.join(d, 'dev.json'), 'w'))
        json.dump(test_p, open(os.path.join(d, 'test.json'), 'w'))
        info[k] = {'n_train': len(tr), 'n_dev': len(dev_p), 'n_test': len(test_p),
                   'no_entity_train': sum(1 for q in tr if not q['entity_positions'])}
    print(json.dumps(info), flush=True)
    json.dump(info, open(os.path.join(W, 'prep_info.json'), 'w'), indent=1)


def run(a):
    os.chdir(MQ); sys.path.insert(0, MQ)
    import numpy as np
    import torch
    import transformers
    import utils, qa_datasets
    from qa_baselines import QA_MultiQA
    from qa_datasets import QA_Dataset_MultiQA_muti
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))

    _orig, _cache = qa_datasets.QA_Dataset_multi.timesToIds, {}

    def _memo(self, times):
        key = tuple(times)
        if key not in _cache:
            _cache[key] = list(_orig(self, times))
        return list(_cache[key])
    qa_datasets.QA_Dataset_multi.timesToIds = _memo

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    tag = f'{a.k}_{a.emb}_seed{a.seed}'
    outdir = os.path.join(W, 'runs', tag); os.makedirs(outdir, exist_ok=True)
    logf = open(os.path.join(outdir, 'stdout.log'), 'w')
    args = argparse.Namespace(tkbc_model_file={'off': 'tcomplex.ckpt', 'full': 'tcomplex_fullkg.ckpt'}[a.emb],
                              dataset_name='MultiTQ', sub_dataset=a.k, tkg_file='full.txt', kg_dir='kg',
                              model='multiqa', lm_model='distilbert-base-uncased', load_from='', save_to=tag,
                              max_epochs=a.max_epochs, eval_k=1, valid_freq=1, batch_size=100, valid_batch_size=50,
                              frozen=1, lm_frozen=1, lr=2e-4, mode='train', eval_split='dev', extra_entities=False,
                              test='test', max_questions=50000)
    rec = {'tag': tag, 'k': a.k, 'emb': a.emb, 'seed': a.seed, 'args': vars(args), 'max_epochs': a.max_epochs,
           'patience': a.patience, 'official_budget': a.official_budget, 'torch': torch.__version__,
           'transformers': transformers.__version__, 'gpu': torch.cuda.get_device_name(0),
           'tkbc_sha256': sha(os.path.join(MQ, 'models/MultiTQ/kg_embeddings', args.tkbc_model_file)),
           'epochs': [], 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    t0 = time.time()
    with contextlib.redirect_stdout(logf), contextlib.redirect_stderr(logf):
        tkbc_model = utils.loadTkbcModel(f'models/MultiTQ/kg_embeddings/{args.tkbc_model_file}')
        dataset = QA_Dataset_MultiQA_muti(split='train', dataset_name='MultiTQ', args=args)
        valid_dataset = QA_Dataset_MultiQA_muti(split='dev', dataset_name='MultiTQ', args=args)
        test_dataset = QA_Dataset_MultiQA_muti(split='test', dataset_name='MultiTQ', args=args)
        qa_model = QA_MultiQA(tkbc_model, args).cuda()
    rec['n_train'], rec['n_dev'], rec['n_test'] = len(dataset), len(valid_dataset), len(test_dataset)
    assert all(q.get('_stripped') for q in test_dataset.data)
    rec['prep_s'] = time.time() - t0
    trainable = [n for n, p in qa_model.named_parameters() if p.requires_grad]
    rec['n_trainable_params'] = int(sum(p.numel() for p in qa_model.parameters() if p.requires_grad))

    def snapshot():
        sd = qa_model.state_dict()
        return {k: v.detach().clone().cpu() for k, v in sd.items()
                if not k.startswith(('lm_model.', 'tkbc_model.', 'entity_time_embedding.'))}  # frozen parts excluded

    optimizer = torch.optim.Adam(qa_model.parameters(), lr=args.lr)
    optimizer.zero_grad()
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=5,
                                         collate_fn=dataset._collate_fn)
    # fast dev Hits@1: same semantics as utils.eval_multi at k=1 (top-1 of torch.topk(scores, 10) per question,
    # official time truncation, any-gold match); checked against the official eval_multi at epoch 1 and at the end.
    dev_dl = torch.utils.data.DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False, num_workers=2,
                                         persistent_workers=True, collate_fn=valid_dataset._collate_fn)
    n_ent_ = len(valid_dataset.all_dicts['ent2id'])
    dev_gold_ = [(set(q['answers']), (len(q['answers'][0]) if q['answer_type'] == 'time' else 0)) for q in valid_dataset.data]

    def fast_dev_h1():
        qa_model.eval(); tops = []
        with torch.no_grad():
            for b in dev_dl:
                sc = qa_model.forward(b)
                _, ind = torch.topk(sc, 10, dim=1)
                tops += ind[:, 0].tolist()
        h = 0
        for a_id, (gold, L) in zip(tops, dev_gold_):
            p = valid_dataset.all_dicts['id2ent'][a_id] if a_id < n_ent_ else valid_dataset.all_dicts['id2ts'].get(a_id - n_ent_, '__no_ts__')
            if L in (4, 7):
                p = p[:L]
            h += int(p in gold)
        return h / len(dev_gold_)

    def official_dev_h1():
        with contextlib.redirect_stdout(logf), contextlib.redirect_stderr(logf):
            v, _ = utils.eval_multi(qa_model, valid_dataset, batch_size=args.valid_batch_size, split='dev', k=1)
        return v

    best, best_ep, best_sd = 0.0, None, None
    best10, best10_ep, best10_sd = 0.0, None, None
    since = 0
    n_ep = a.smoke_epochs if a.smoke_epochs else a.max_epochs
    for epoch in range(n_ep):
        te = time.time()
        qa_model.train(); eloss = 0.0; nb = 0
        for a_batch in loader:
            qa_model.zero_grad()
            answers_khot = a_batch[-1]
            scores = qa_model.forward(a_batch)
            loss = qa_model.loss(scores, answers_khot.cuda())
            loss.backward(); optimizer.step()
            eloss += loss.item(); nb += 1
        tt = time.time() - te
        dev_h1 = fast_dev_h1()
        row = {'epoch': epoch + 1, 'loss': eloss, 'steps': nb, 'train_s': round(tt, 2),
               'eval_s': round(time.time() - te - tt, 2), 'dev_h1': dev_h1}
        if epoch == 0:
            row['dev_h1_official_eval_multi'] = official_dev_h1()
        rec['epochs'].append(row)
        if dev_h1 > best:
            best, best_ep, best_sd, since = dev_h1, epoch + 1, snapshot(), 0
        else:
            since += 1
        if epoch + 1 <= a.official_budget and dev_h1 > best10:
            best10, best10_ep, best10_sd = dev_h1, epoch + 1, (best_sd if best_ep == epoch + 1 else snapshot())
        rec['best_dev_h1'], rec['best_epoch'] = best, best_ep
        rec['best10_dev_h1'], rec['best10_epoch'] = best10, best10_ep
        rec['elapsed_s'] = time.time() - t0
        json.dump(rec, open(os.path.join(outdir, 'run.json'), 'w'), indent=1)
        if a.patience and since >= a.patience and epoch + 1 >= a.official_budget:
            rec['early_stopped_at'] = epoch + 1
            break

    def predict(ds, path):
        qa_model.eval()
        dl = torch.utils.data.DataLoader(ds, batch_size=args.valid_batch_size, shuffle=False, num_workers=4,
                                         collate_fn=ds._collate_fn)
        out, i = [], 0
        n_ent = len(ds.all_dicts['ent2id'])
        with torch.no_grad():
            for b in dl:
                scores = qa_model.forward(b)
                for s in scores:
                    _, ind = torch.topk(s, 10)
                    ranked = []
                    for a_id in ind.tolist():
                        if a_id < n_ent:
                            ranked.append(ds.all_dicts['id2ent'][a_id])
                        else:
                            ranked.append(ds.all_dicts['id2ts'].get(a_id - n_ent, '__no_ts__'))
                    out.append({'quid': ds.data[i]['quid'], 'ranked': ranked}); i += 1
        assert i == len(ds.data)
        with open(path, 'w') as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        h = sha(path)
        with open(path, 'rb') as fi, gzip.open(path + '.gz', 'wb') as fo:
            fo.write(fi.read())
        return h

    rec['pred'] = {}
    for name, sd, ep in [('primary', best_sd, best_ep), ('budget10', best10_sd, best10_ep)]:
        if sd is None:
            continue
        if name == 'budget10' and ep == best_ep:  # same checkpoint as primary: copy instead of re-predicting
            import shutil
            for sp in ['dev_sel2000', 'test']:
                for ext in ['.jsonl', '.jsonl.gz']:
                    shutil.copyfile(os.path.join(outdir, f'{sp}_primary{ext}'), os.path.join(outdir, f'{sp}_budget10{ext}'))
            rec['pred'][name] = dict(rec['pred']['primary'], copied_from='primary')
            continue
        qa_model.load_state_dict(sd, strict=False)
        if name == 'primary':
            rec['primary_dev_h1_official_eval_multi'] = official_dev_h1()
        p_dev = os.path.join(outdir, f'dev_sel2000_{name}.jsonl'); p_test = os.path.join(outdir, f'test_{name}.jsonl')
        rec['pred'][name] = {'epoch': ep, 'dev_sha256': predict(valid_dataset, p_dev), 'test_sha256': predict(test_dataset, p_test)}
        if not a.smoke_epochs:
            torch.save(sd, os.path.join(outdir, f'state_{name}.pt'))
    # dev re-check with the official scorer semantics on the written file (dev labels only)
    dev_gold = {q['quid']: q for q in valid_dataset.data}
    for name in rec['pred']:
        if rec['pred'][name].get('copied_from'):
            rec['pred'][name]['dev_h1_from_file'] = rec['pred']['primary']['dev_h1_from_file']; continue
        hits = []
        for line in open(os.path.join(outdir, f'dev_sel2000_{name}.jsonl')):
            r = json.loads(line); q = dev_gold[r['quid']]; pred = r['ranked'][:1]
            if q['answer_type'] == 'time':
                L = len(q['answers'][0]); pred = [x[:L] for x in pred] if L in (4, 7) else pred
            hits.append(int(len(set(q['answers']) & set(pred)) > 0))
        rec['pred'][name]['dev_h1_from_file'] = sum(hits) / len(hits)
    rec['total_s'] = time.time() - t0
    rec['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    json.dump(rec, open(os.path.join(outdir, 'run.json'), 'w'), indent=1)
    os.system(f'chmod -R a+rX {outdir}')
    print(json.dumps({k: rec[k] for k in ['tag', 'best_dev_h1', 'best_epoch', 'best10_dev_h1', 'best10_epoch', 'total_s']}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prep', 'run'])
    ap.add_argument('--levels', default='k01_s0,k05_s0,k10_s0')
    ap.add_argument('--k'); ap.add_argument('--emb', default='off'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max_epochs', type=int, default=10); ap.add_argument('--patience', type=int, default=0)
    ap.add_argument('--official_budget', type=int, default=10); ap.add_argument('--smoke_epochs', type=int, default=0)
    a = ap.parse_args()
    if a.mode == 'prep':
        prep(a.levels.split(','))
    else:
        run(a)
