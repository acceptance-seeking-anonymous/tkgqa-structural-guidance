"""Assemble the GPU-host staging tree: stage/kpct/{executor/, bundles/<subset>/, preds/<subset>/}."""
import os, sys, json, shutil, glob, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
OUT = os.path.join(HERE, 'out'); ST = os.path.join(HERE, 'stage', 'kpct')

def executor_base():
    ex = os.path.join(ST, 'executor'); os.makedirs(f'{ex}/struct', exist_ok=True); os.makedirs(f'{ex}/mtqdata/kg', exist_ok=True)
    for f in ('kpct_core.py', 'kpct_executor.py'):
        shutil.copy2(os.path.join(HERE, f), ex)
    for f in ('mtq_lib.py', 'mtq_align.py', 'mtq_clf.py', 'mtq_cand.py', 'mtq_export.py', 'multitq_eval.py'):
        shutil.copy2(os.path.join(HERE, 'struct', f), f'{ex}/struct/')
    for f in ('train.txt', 'valid.txt', 'test.txt'):
        shutil.copy2(os.path.join(HERE, 'mtqdata', 'kg', f), f'{ex}/mtqdata/kg/')
    return ex

def models_k(nm):
    import kpct_core as K
    ex = os.path.join(ST, 'executor'); md = f'{ex}/models/{nm}'; os.makedirs(md, exist_ok=True)
    if nm == 'k0':
        json.dump(K.build_FB_kg(), open(f'{md}/FB.json', 'w'))
        return md
    if nm == 'k100':
        for s in range(3):
            shutil.copy2(f'{K.CODE}/S_seed{s}.txt', md)
        voc, net = K.load_frozen_parser(); K.save_parser(f'{md}/parser.pt', voc, net)
        json.dump({'alias_rules': json.load(open(f'{K.CODE}/alias_rules.json')), 'ent_block': json.load(open(f'{K.CODE}/ent_block.json')),
                   'train_list': 'k100 (frozen full-train linker)'}, open(f'{md}/linker.json', 'w'))
        shutil.copy2(os.path.join(OUT, 'FB_k100.json'), f'{md}/FB.json')
        return md
    d = os.path.join(OUT, nm)
    E = json.load(open(f'{d}/selection_E.json'))['E_selected']
    for s in range(3):
        shutil.copy2(f'{d}/S_seed{s}.txt', md)
    shutil.copy2(f'{d}/parser_full_E{E}.pt', f'{md}/parser.pt')
    shutil.copy2(f'{d}/linker_full.json', f'{md}/linker.json')
    shutil.copy2(f'{d}/FB.json', f'{md}/FB.json')
    return md

def data_k(nm):
    d = os.path.join(OUT, nm)
    bd = os.path.join(ST, 'bundles', nm); pd_ = os.path.join(ST, 'preds', nm)
    os.makedirs(bd, exist_ok=True); os.makedirs(pd_, exist_ok=True)
    for f in glob.glob(f'{d}/bundles/*.jsonl'):
        dst = os.path.join(bd, os.path.basename(f))
        if os.path.exists(dst): os.remove(dst)
        os.link(f, dst)
    for f in glob.glob(f'{d}/preds/*.jsonl'):
        dst = os.path.join(pd_, os.path.basename(f))
        if os.path.exists(dst): os.remove(dst)
        os.link(f, dst)

if __name__ == '__main__':
    what = sys.argv[1:]
    executor_base()
    for nm in what:
        models_k(nm)
        if nm not in ('k0', 'k100'):
            data_k(nm)
    print('staged', what)
