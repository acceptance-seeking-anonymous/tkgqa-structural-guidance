"""cron_clf.py -- multi-head bag-of-word-n-gram classifier (fastText style, ported from mtq_clf.py) over
entity-masked question text.  Heads: relation (5), reasoning type (5), answer type (2), first/last adj (3),
before/after (3).  Trained on TRAIN only, seed 0."""
import collections, numpy as np, torch, torch.nn as nn

RELS = ['P54', 'P39', 'P166', 'P26', 'P108']
TYPES = ['simple_entity', 'simple_time', 'before_after', 'first_last', 'time_join']
ATYPES = ['entity', 'time']
ADJS = ['none', 'first', 'last']
BAS = ['none', 'before', 'after']
HEADS = {'rel': RELS, 'qtype': TYPES, 'atype': ATYPES, 'adj': ADJS, 'ba': BAS}


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
        self.itos = sorted(k for k, v in c.items() if v >= min_count)
        self.stoi = {k: i + 1 for i, k in enumerate(self.itos)}

    def encode(self, d):
        return [self.stoi[x] for x in d if x in self.stoi] or [0]


class Net(nn.Module):
    def __init__(self, V, dim=96):
        super().__init__()
        self.emb = nn.EmbeddingBag(V + 1, dim, mode='mean', padding_idx=0)
        self.drop = nn.Dropout(0.1)
        self.heads = nn.ModuleDict({k: nn.Linear(dim, len(v)) for k, v in HEADS.items()})

    def forward(self, idx, off):
        h = self.drop(self.emb(idx, off))
        return {k: m(h) for k, m in self.heads.items()}


def batchify(enc, ids):
    seqs = [enc[i] for i in ids]
    off = np.cumsum([0] + [len(s) for s in seqs[:-1]])
    return torch.tensor(np.concatenate(seqs), dtype=torch.long), torch.tensor(off, dtype=torch.long)


def labels_of(q):
    a = q['annotation']
    return dict(rel=RELS.index(next(iter(q['relations']))), qtype=TYPES.index(q['type']),
                atype=ATYPES.index(q['answer_type']), adj=ADJS.index(a.get('adj') or 'none'),
                ba=BAS.index(a.get('type') or 'none'))


def train(enc, Y, ids, V, seed=0, epochs=6, lr=2e-3, bs=512):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Net(V)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    ids = np.array(ids)
    for ep in range(epochs):
        net.train(); perm = ids[np.random.permutation(len(ids))]
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]; idx, off = batchify(enc, b); o = net(idx, off)
            loss = sum(nn.functional.cross_entropy(o[k], torch.tensor(Y[k][b])) for k in HEADS)
            opt.zero_grad(); loss.backward(); opt.step()
    return net


def predict(net, enc, bs=4096):
    net.eval(); outs = collections.defaultdict(list)
    with torch.no_grad():
        for s in range(0, len(enc), bs):
            ids = list(range(s, min(len(enc), s + bs)))
            idx, off = batchify(enc, ids); o = net(idx, off)
            for k, v in o.items():
                outs[k].append(torch.softmax(v, -1).numpy().astype(np.float32))
    return {k: np.concatenate(v) for k, v in outs.items()}
