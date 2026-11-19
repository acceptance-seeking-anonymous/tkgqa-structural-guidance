#!/usr/bin/env python3
"""mbq_make_bundle.py -- question inputs of the matched MultiQA baseline job directory.

  python mbq_make_bundle.py --questions $DATA_DIR/questions --manifest <multitq_kpct_subsets.json> --out <job dir>

Writes (json.dump, default separators):
  qin/train_all.json       every MultiTQ train question (official order, all fields)
  qin/dev_sel2000.json     the dev questions of the registered dev_sel2000 list (official dev order, all fields)
  qin/test_stripped.json   every test question reduced to {quid, question}: no answer reaches the GPU job
  lists/<k>.json           quid list of each registered nested subset k01_s0, k05_s0, k10_s0 (sha256-checked)
The rest of the job directory (MultiQA code, data/MultiTQ/kg/tkbc_processed_data, the shipped TComplEx checkpoint)
comes from the upstream repository (see fetch_multiqa.sh).
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'kpct', 'reader'))
import gk_common as GC   # load_list: registered list recipe with sha256 check

LEVELS = ['k01_s0', 'k05_s0', 'k10_s0']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--questions', required=True, help='MultiTQ questions/ directory (train.json, dev.json, test.json)')
    ap.add_argument('--manifest', required=True, help='multitq_kpct_subsets.json of the k%% runs')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, 'qin'), exist_ok=True); os.makedirs(os.path.join(a.out, 'lists'), exist_ok=True)
    train = json.load(open(os.path.join(a.questions, 'train.json')))
    dev = json.load(open(os.path.join(a.questions, 'dev.json')))
    test = json.load(open(os.path.join(a.questions, 'test.json')))
    dsel = set(GC.load_list(a.manifest, 'dev_sel2000'))
    json.dump(train, open(os.path.join(a.out, 'qin', 'train_all.json'), 'w'))
    dv = [q for q in dev if int(q['quid']) in dsel]
    assert len(dv) == len(dsel) == 2000
    json.dump(dv, open(os.path.join(a.out, 'qin', 'dev_sel2000.json'), 'w'))
    json.dump([{'quid': q['quid'], 'question': q['question']} for q in test], open(os.path.join(a.out, 'qin', 'test_stripped.json'), 'w'))
    for k in LEVELS:
        json.dump(GC.load_list(a.manifest, k), open(os.path.join(a.out, 'lists', f'{k}.json'), 'w'))
    print(json.dumps({'train': len(train), 'dev_sel2000': len(dv), 'test': len(test),
                      **{k: len(json.load(open(os.path.join(a.out, 'lists', f'{k}.json')))) for k in LEVELS}}))


if __name__ == '__main__':
    main()
