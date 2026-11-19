#!/usr/bin/env python3
"""mbq_tcx_full.py -- retrain the MultiQA TComplEx KG embedding on ALL kg/full.txt facts.

The shipped MultiQA checkpoint (models/MultiTQ/kg_embeddings/tcomplex.ckpt) was trained on
tkbc_processed_data/train.pickle (368,962 of 461,329 facts; random split). Our arms read every
fact of full.txt, so the graph-access-matched variant trains the same TComplEx (official
MultiQA/tkbc code: models.TComplEx, optimizers.TKBCOptimizer with the entity + time losses,
regularizers N3 / Lambda3) on train+valid+test pickles (= full.txt, 461,329 facts).
Hyper-parameters: rank and table sizes read from the shipped checkpoint; Adagrad lr 0.1,
batch 1024, emb_reg 1e-3, time_reg 1e-3 (train_complex_embedding.ipynb); 50 epochs
(learner.py default max_epochs); seed 0. No QA question or answer is read.
Diagnostic only (not used for selection): raw object-MRR on 20k sampled facts, for the shipped
checkpoint on its training facts and on its 92,367 held-out facts, and for the retrained one.
"""
import hashlib, json, os, pickle, sys, time
import numpy as np
import torch
from torch import optim

W = os.path.abspath(os.getcwd())
sys.path.insert(0, os.path.join(W, 'MultiQA', 'tkbc'))
import types  # noqa: E402
_stub = types.ModuleType('datasets'); _stub.TemporalDataset = object  # optimizers.py imports it only for a type hint
sys.modules['datasets'] = _stub
from models import TComplEx  # noqa: E402
from regularizers import N3, Lambda3  # noqa: E402
from optimizers import TKBCOptimizer  # noqa: E402

torch.manual_seed(0); np.random.seed(0)
P = os.path.join(W, 'data/MultiTQ/kg/tkbc_processed_data')
data = {s: pickle.load(open(os.path.join(P, s + '.pickle'), 'rb')).astype('int64') for s in ['train', 'valid', 'test']}
allf = np.vstack([data['train'], data['valid'], data['test']])
shipped = torch.load(os.path.join(W, 'MultiQA/models/MultiTQ/kg_embeddings/tcomplex.ckpt'), map_location='cpu')
n_ent, d2 = shipped['embeddings.0.weight'].shape
n_rel2 = shipped['embeddings.1.weight'].shape[0]
n_ts = shipped['embeddings.2.weight'].shape[0]
rank = d2 // 2
sizes = (n_ent, n_rel2, n_ent, n_ts)
log = {'sizes': sizes, 'rank': rank, 'n_facts': {k: int(v.shape[0]) for k, v in data.items()}, 'n_all': int(allf.shape[0])}
print(log, flush=True)


def recip(x):
    c = np.copy(x); tmp = np.copy(c[:, 0]); c[:, 0] = c[:, 2]; c[:, 2] = tmp; c[:, 1] += n_rel2 // 2
    return np.vstack((x, c))


@torch.no_grad()
def raw_mrr(model, facts, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(facts.shape[0], size=min(n, facts.shape[0]), replace=False)
    x = torch.from_numpy(facts[idx]).cuda()
    rr = []
    for b in range(0, x.shape[0], 1000):
        xb = x[b:b + 1000]
        sc = model.forward(xb)[0]
        tgt = sc.gather(1, xb[:, 2:3])
        rank_ = (sc > tgt).sum(1) + 1
        rr.append((1.0 / rank_.float()).cpu())
    return float(torch.cat(rr).mean())


m0 = TComplEx(sizes, rank, no_time_emb=False); m0.load_state_dict(shipped); m0 = m0.cuda().eval()
held = np.vstack([data['valid'], data['test']])
log['shipped_rawMRR_trainfacts'] = raw_mrr(m0, data['train'])
log['shipped_rawMRR_heldoutfacts'] = raw_mrr(m0, held)
del m0

model = TComplEx(sizes, rank, no_time_emb=False).cuda()
opt = optim.Adagrad(model.parameters(), lr=0.1)
emb_reg, time_reg = N3(1e-3), Lambda3(1e-3)
examples = torch.from_numpy(recip(allf))
t0 = time.time(); log['epochs'] = []
for ep in range(50):
    model.train()
    TKBCOptimizer(model, emb_reg, time_reg, opt, batch_size=1024, verbose=False).epoch(examples)
    if (ep + 1) % 10 == 0:
        model.eval()
        r = {'epoch': ep + 1, 'rawMRR_allfacts_sample': raw_mrr(model, allf), 'elapsed_s': time.time() - t0}
        log['epochs'].append(r); print(r, flush=True)
        json.dump(log, open(os.path.join(W, 'tcx_full_log.json'), 'w'), indent=1)
out = os.path.join(W, 'tcomplex_fullkg.ckpt')
torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out)
log['ckpt_sha256'] = hashlib.sha256(open(out, 'rb').read()).hexdigest()
log['train_s'] = time.time() - t0
json.dump(log, open(os.path.join(W, 'tcx_full_log.json'), 'w'), indent=1)
os.system(f'chmod a+r {out}')
print('done', log['ckpt_sha256'], flush=True)
