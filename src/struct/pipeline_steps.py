"""pipeline_steps.py -- transcription of the interactive steps of the full-data MultiTQ structural pipeline (Rules / Ranker).

Order of execution (CPU, 10 cores, env 'multitq': python 3.12, lightgbm 4.6, torch CPU):
  1. python run_alias.py                 -> alias_counts.pkl (train-only alias evidence, base linker)
     build_alias_rules()                 -> alias_rules.json  (count>=5, precision>=0.1, top-3 sectors per n-gram)
  2. python run_parse.py v1; build_block_list() -> ent_block.json ; python run_parse.py v3 (frozen; v2 discarded)
  3. train_parser()                      -> parser_clf_seed0.pt, parser_vocab.pkl, {dev,test}_parser_probs.npz
     train_oof()                         -> train_oof_parser_probs.npz (2-fold, seed 0; S / reader training features only)
  4. python run_cands.py dev dev_parser_probs.npz - 0 cands_dev.pkl
     python run_cands.py test test_parser_probs.npz - 0 cands_test.pkl
     python run_cands.py train train_oof_parser_probs.npz s_train100k_quids.json 200 cands_trainS.pkl
     python run_cands.py train train_oof_parser_probs.npz reader_train60k_quids.json 200 cands_trainR.pkl
  5. train_S()  (grid on dev at seed 0, then seeds 0/1/2)  -> S_seed{0,1,2}.txt, preds/*.jsonl, exports/*
The frozen artefacts (sha256 in multitq_structural_freeze.json) are authoritative; this file documents the code
that produced them (functions copied from the executed cells; float non-determinism of multithreaded
training may give bit-level differences on re-run).
"""
import os, json, pickle, collections, random, time
import numpy as np
import torch, torch.nn as nn
import lightgbm as lgb
import mtq_lib as L, mtq_clf as C, mtq_cand as MC, multitq_eval as ME

D = L.DATA


def build_alias_rules():
    ac = pickle.load(open('alias_counts.pkl', 'rb')); den, num = ac['den'], ac['num']
    byk = collections.defaultdict(list)
    for (k, S), cn in num.items():
        byk[k].append((cn, S))
    rules = {}
    for k, lst in byk.items():
        lst.sort(reverse=True)
        keep = [[S, round(cn / den[k], 4), cn] for cn, S in lst[:3] if cn >= 5 and cn / den[k] >= 0.1]
        if keep:
            rules[' '.join(k)] = keep
    json.dump(rules, open('alias_rules.json', 'w'))


def build_block_list(kg, Qtr, Ptr):
    cnt = collections.Counter(); good = collections.Counter()
    for q, r in zip(Qtr, Ptr):
        gold = {r['topic'], r['partner']}
        A = {kg.e2i.get(a) for a in q['answers']}
        for lo, hi, alts in r['ments']:
            e, eff, al, full = alts[0]
            k = (e, full, al); cnt[k] += 1; good[k] += (e in gold) or (e in A)
    bad = [[int(k[0]), int(k[1]), int(k[2])] for k, c in cnt.items() if c >= 10 and good[k] / c < 0.05]
    json.dump(bad, open('ent_block.json', 'w'))


def rel_targets(r):
    labs = []
    for rr, d in r['rel_labels']:
        labs += [rr * 2, rr * 2 + 1] if d == -1 else [rr * 2 + d]
    return labs


def train_clf(enc_tr, tr_rel, y_q, y_a, y_l, V, NR, tr_ids, seed=0, epochs=6):
    """Bag-of-n-gram multi-head classifier; 2 epochs uniform soft relation targets then 4 EM epochs."""
    torch.manual_seed(seed); np.random.seed(seed)
    net = C.Net(V, 96, {'rel': NR * 2, 'qtype': 6, 'atype': 2, 'level': 3})
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    tr_ids = np.array(tr_ids)
    sft = {i: (np.array(tr_rel[i]), np.full(len(tr_rel[i]), 1 / len(tr_rel[i]), dtype=np.float32)) for i in tr_ids if tr_rel[i]}
    for ep in range(epochs):
        if ep >= 2:
            pr = C.predict(net, [enc_tr[i] for i in tr_ids])['rel']
            for j, i in enumerate(tr_ids):
                if tr_rel[i]:
                    l = np.array(tr_rel[i]); p = pr[j, l] + 1e-6; sft[i] = (l, (p / p.sum()).astype(np.float32))
        net.train(); perm = tr_ids[np.random.permutation(len(tr_ids))]
        for s in range(0, len(perm), 512):
            ids = perm[s:s + 512]; idx, off = C.batchify(enc_tr, ids); o = net(idx, off)
            loss = sum(nn.functional.cross_entropy(o[k], torch.tensor(y[ids])) for k, y in (('qtype', y_q), ('atype', y_a), ('level', y_l)))
            rows = [(j, sft[i]) for j, i in enumerate(ids) if i in sft]
            lp = torch.log_softmax(o['rel'], -1)
            jj = np.concatenate([[j] * len(t[0]) for j, t in rows]); ll = np.concatenate([t[0] for j, t in rows])
            ww = np.concatenate([t[1] for j, t in rows])
            loss = loss - (lp[torch.tensor(jj), torch.tensor(ll)] * torch.tensor(ww)).sum() / len(rows)
            opt.zero_grad(); loss.backward(); opt.step()
    return net


def labels_for(q, names):
    A = set(q['answers'])
    if q['answer_type'] == 'time':
        Lg = len(q['answers'][0])
        if Lg in (4, 7):
            return np.array([n[:Lg] in A for n in names], np.int8)
    return np.array([n in A for n in names], np.int8)


S_BASE = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1], min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=20, num_threads=10, verbose=-1)
S_GRID = [(nl, lr) for nl in (31, 63, 127) for lr in (0.05, 0.1)]
S_SELECTED = dict(num_leaves=127, learning_rate=0.05, rounds=152)   # selected on dev (seed 0), see S_grid_dev.json


def make_ranked(ordered, meta, FB, k=50):
    """top-50 distinct strings; padded to >=10 with PRIOR then train-frequent answers of the predicted type."""
    out, seen = [], set()
    for n in ordered:
        if n not in seen:
            seen.add(n); out.append(n)
        if len(out) >= k:
            break
    if len(out) < 10:
        info = meta['info']; fb = FB['entity'] if info['atype'] == 'entity' else FB[info['level']]
        for n in list(meta['prior']) + fb:
            if n not in seen:
                seen.add(n); out.append(n)
            if len(out) >= 10:
                break
    return out
