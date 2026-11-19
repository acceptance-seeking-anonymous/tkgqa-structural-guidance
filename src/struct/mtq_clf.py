"""mtq_clf.py -- multi-head bag-of-n-gram classifier (fastText-style) over entity-masked question text.
Heads: relation-direction (soft multi-label from silver alignment, EM refinement), qtype, answer_type,
time_level, first/last, before/after.  Trained once on TRAIN with seed 0."""
import collections, numpy as np, torch, torch.nn as nn

QTYPES = ['equal', 'before_after', 'first_last', 'equal_multi', 'after_first', 'before_last']
ATYPES = ['entity', 'time']
LEVELS = ['day', 'month', 'year']


def masked_tokens(rec):
    ws = list(rec['ws'])
    for lo, hi, alts in rec['ments']:
        for j in range(lo, hi + 1):
            ws[j] = None
        ws[lo] = 'ENT'
    out = []
    for w in ws:
        if w is None:
            continue
        if w == 'timex':
            w = 'TIME_' + (rec['tp'][0] if rec['tp'] else 'x')
        out.append(w)
    return out


def ngrams(tokens):
    t = ['<s>'] + tokens + ['</s>']
    g = list(t)
    g += [t[i] + ' ' + t[i + 1] for i in range(len(t) - 1)]
    g += [t[i] + ' ' + t[i + 1] + ' ' + t[i + 2] for i in range(len(t) - 2)]
    return g


class Vocab:
    def __init__(self, docs, min_count=3):
        c = collections.Counter()
        for d in docs:
            c.update(set(d))
        self.itos = [k for k, v in c.items() if v >= min_count]
        self.stoi = {k: i + 1 for i, k in enumerate(self.itos)}  # 0 = pad

    def encode(self, d):
        return [self.stoi[x] for x in d if x in self.stoi] or [0]


class Net(nn.Module):
    def __init__(self, V, dim, heads):
        super().__init__()
        self.emb = nn.EmbeddingBag(V + 1, dim, mode='mean', padding_idx=0)
        self.drop = nn.Dropout(0.1)
        self.heads = nn.ModuleDict({k: nn.Linear(dim, n) for k, n in heads.items()})

    def forward(self, idx, off):
        h = self.drop(self.emb(idx, off))
        return {k: m(h) for k, m in self.heads.items()}


def batchify(enc, ids):
    seqs = [enc[i] for i in ids]
    off = np.cumsum([0] + [len(s) for s in seqs[:-1]])
    return torch.tensor(np.concatenate(seqs), dtype=torch.long), torch.tensor(off, dtype=torch.long)


def predict(net, enc, bs=4096):
    net.eval()
    outs = collections.defaultdict(list)
    with torch.no_grad():
        for s in range(0, len(enc), bs):
            ids = list(range(s, min(len(enc), s + bs)))
            idx, off = batchify(enc, ids)
            o = net(idx, off)
            for k, v in o.items():
                outs[k].append(torch.softmax(v, -1).numpy().astype(np.float32))
    return {k: np.concatenate(v) for k, v in outs.items()}
