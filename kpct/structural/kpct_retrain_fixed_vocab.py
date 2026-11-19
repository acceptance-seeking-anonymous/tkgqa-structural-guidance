"""kpct_retrain_fixed_vocab.py -- retraining driver for the k% structural stack with a fixed string-hash seed and the
preserved vocabulary index order.

Why: mtq_clf.Vocab builds its index order by iterating `set(doc)` (mtq_clf.py:35-41). The iteration order of a set of
strings depends on Python's per-process string-hash key. The original k% runs did not fix PYTHONHASHSEED, so a free
retraining run gets another vocabulary order, hence another embedding initialisation and other classifier weights.

What this driver does (kpct_run.py and kpct_core.py are NOT edited; they are imported and one class is replaced):
  1. It refuses to run unless PYTHONHASHSEED is set to a fixed integer (0..4294967295; default 0) before the
     interpreter starts (any fixed seed is accepted, see _require_hash_seed).
  2. It replaces kpct_core.VocabK. Every vocabulary built from training documents is first built exactly as before
     (mtq_clf.Vocab, min_count 3). Its token set is then looked up among the vocabularies of the frozen checkpoints
     (--preserved DIR: every *.pt below DIR that stores an 'itos' list).
       * token set equal to a frozen checkpoint's -> the frozen index order is imposed             (IMPOSED)
       * no frozen checkpoint with that token set  -> the order produced under the fixed hash seed is kept
                                                      (NOT_PRESERVED: repeatable for the same seed and Python version,
                                                       but not the order of the original run)
     The frozen checkpoints of a subset are parser_full_E6.pt, parser_full_E24.pt, fold0/parser.pt and fold1/parser.pt
     (the out/<subset>/ folder of the structural run). The 2-fold OOF classifiers inside stage fitA and the
     inner OOF classifiers inside stages fold0/fold1 were never saved, so their order cannot be imposed; they produce
     the classifier features of the S training rows.
  3. It appends one JSON row per vocabulary build to $KPCT_OUT/<subset>/vocab_order_log.jsonl.

usage (from the structural tree that holds kpct_run.py):
    PYTHONHASHSEED=0 python kpct_retrain_fixed_vocab.py all <subset> --preserved <frozen out/<subset> dir>
    PYTHONHASHSEED=0 python kpct_retrain_fixed_vocab.py <stage> <subset> --preserved <dir>
"""
import os, sys, json, glob, hashlib, inspect, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _require_hash_seed():
    """Return the fixed string-hash seed, or stop if string hashing is randomised.

    Any fixed PYTHONHASHSEED in 0..4294967295 is accepted; the documented default is 0. sys.flags.hash_randomization
    is not used for this test (it is 1 for every nonzero fixed seed). The seed must be in force from interpreter start: the string hash of a probe in this process is compared
    with the hash in a fresh interpreter started with the same seed.
    """
    v = os.environ.get('PYTHONHASHSEED')
    if v is None or not v.isdigit() or int(v) > 4294967295:
        raise SystemExit('kpct_retrain_fixed_vocab: set PYTHONHASHSEED to an integer in 0..4294967295 (default 0) '
                         'before starting Python; string-hash randomisation is active.')
    probe = 'kpct_retrain_fixed_vocab hash probe'
    fresh = subprocess.run([sys.executable, '-c', 'import sys; print(hash(sys.argv[1]))', probe],
                           env=dict(os.environ, PYTHONHASHSEED=v), capture_output=True, text=True, check=True)
    if int(fresh.stdout.strip()) != hash(probe):
        raise SystemExit('kpct_retrain_fixed_vocab: PYTHONHASHSEED=%s was not in force when this interpreter started; '
                         'set it before starting Python.' % v)
    return int(v)


def _set_key(itos):
    return hashlib.sha256('\n'.join(sorted(itos)).encode('utf-8')).hexdigest()


def load_preserved(pdir):
    import torch
    reg = {}
    for p in sorted(glob.glob(os.path.join(pdir, '**', '*.pt'), recursive=True)):
        d = torch.load(p, map_location='cpu', weights_only=False)
        if not isinstance(d, dict) or 'itos' not in d:
            continue
        itos = list(d['itos'])
        k = _set_key(itos)
        if k in reg and reg[k]['itos'] != itos:
            raise SystemExit(f'two frozen checkpoints with the same token set but different order: {reg[k]["src"]} {p}')
        reg.setdefault(k, {'itos': itos, 'src': []})['src'].append(os.path.relpath(p, pdir))
    if not reg:
        raise SystemExit(f'no frozen checkpoint with an itos list below {pdir}')
    return reg


def install(pdir, subset, out_root):
    import kpct_core as K
    reg = load_preserved(pdir)
    logp = os.path.join(out_root, subset, 'vocab_order_log.jsonl')
    os.makedirs(os.path.dirname(logp), exist_ok=True)
    Base = K.VocabK

    class VocabPreserved(Base):
        def __init__(self, docs=None, itos=None, min_count=3):
            if itos is not None:                      # loading a checkpoint: unchanged behaviour
                super().__init__(itos=itos)
                return
            natural = K.C.Vocab(docs, min_count=min_count).itos
            k = _set_key(natural)
            hit = reg.get(k)
            caller = [f.function for f in inspect.stack()[1:6]]
            row = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'pid': os.getpid(),
                   'stage': os.environ.get('KPCT_STAGE'), 'callers': caller, 'n_docs': len(docs), 'vocab_n': len(natural),
                   'token_set_sha256': k, 'hash_seed': os.environ.get('PYTHONHASHSEED')}
            if hit is not None:
                row.update(order='IMPOSED', frozen_sources=hit['src'], natural_order_equal_frozen=(natural == hit['itos']))
                super().__init__(itos=hit['itos'])
            else:
                row.update(order='NOT_PRESERVED')
                super().__init__(itos=natural)
            with open(logp, 'a') as f:
                f.write(json.dumps(row) + '\n')

    K.VocabK = VocabPreserved
    return reg


def main():
    seed = _require_hash_seed()
    a = sys.argv[1:]
    if '--preserved' not in a:
        raise SystemExit(__doc__)
    i = a.index('--preserved'); pdir = os.path.abspath(a[i + 1]); a = a[:i] + a[i + 2:]
    stage, nm = a[0], a[1]
    out_root = os.environ.get('KPCT_OUT', os.path.join(HERE, 'out'))
    if stage == 'all':
        import kpct_run as KR
        for st in KR.STAGES:
            KR.log('=== stage', st, nm, 'PYTHONHASHSEED', seed)
            env = dict(os.environ, KPCT_STAGE=st)
            r = subprocess.run([sys.executable, os.path.abspath(__file__), st, nm, '--preserved', pdir], env=env)
            if r.returncode != 0:
                raise SystemExit(f'stage {st} failed ({r.returncode})')
        return
    install(pdir, nm, out_root)
    import kpct_run as KR                                   # imports kpct_core (already patched)
    KR.run_stage(stage, nm)


if __name__ == '__main__':
    main()
