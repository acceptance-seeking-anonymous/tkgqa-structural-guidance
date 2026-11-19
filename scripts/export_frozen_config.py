#!/usr/bin/env python3
"""export_frozen_config.py -- export the frozen hyperparameters and prompt texts from the code (single source of truth)
into configs/frozen_hyperparameters.json and configs/prompts/*.txt. The code reads its constants from the modules,
not from these files; rerun this script after any change and diff the outputs."""
import ast, json, os
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
SPEC = {
 'structural_ranker (kpct/structural/kpct_run.py)': ('kpct/structural/kpct_run.py', ['E_GRID', 'S_GRID', 'S_BASE', 'TRAIN_CAP']),
 'reader_dense_rag (kpct/reader/gk_common.py)': ('kpct/reader/gk_common.py', ['MODEL_8B', 'MODEL_70B', 'CHAT_DATE', 'MAX_LEN', 'MAX_NEW_G', 'LORA', 'OPT', 'E_MAX']),
 'parser (kpct/parser/pk_common.py)': ('kpct/parser/pk_common.py', ['MODEL_8B', 'MAX_LEN', 'MAX_NEW_P', 'LORA', 'OPT', 'E_MAX', 'TAUS', 'QTYPES', 'LEVELS', 'ATYPES', 'ROLES', 'ORDERS', 'KEYS']),
 'robustness_R1 (robustness/r1/local/rob_common.py)': ('robustness/r1/local/rob_common.py', ['SEED', 'N_SAMPLE', 'N_P3', 'PREFIX', 'GEN_MAX_TOKENS', 'JUDGE_MAX_TOKENS', 'MAX_ATTEMPTS', 'JUDGE_KEYS']),
 'robustness_R2 (robustness/r2/local/r2_common.py)': ('robustness/r2/local/r2_common.py', ['SEED_R2', 'N_SAMPLE', 'N_POOL', 'B_PRIMARY', 'BOOT_SEED_PRIMARY', 'ALPHA']),
}
PROMPTS = {'reader_G_head.txt': ('kpct/reader/gk_common.py', 'G_HEAD'), 'dense_rag_GU_head.txt': ('kpct/reader/gk_common.py', 'GU_HEAD'),
           'parser_P_head.txt': ('kpct/parser/pk_common.py', 'P_HEAD'), 'r1_generator_prompt.txt': ('robustness/r1/local/rob_common.py', 'GEN_PROMPT'),
           'r1_generator_retry_suffix.txt': ('robustness/r1/local/rob_common.py', 'GEN_RETRY_SUFFIX'), 'r1_r2_judge_prompt.txt': ('robustness/r1/local/rob_common.py', 'JUDGE_PROMPT')}


def consts(path):
    out = {}
    for node in ast.parse(open(os.path.join(REPO, path), encoding='utf-8').read()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                try:
                    out[node.targets[0].id] = eval(compile(ast.Expression(node.value), path, 'eval'), {'dict': dict, '__builtins__': {}}, {})
                except Exception:
                    pass
    return out


def main():
    cfg = {'_note': 'exported from the code by scripts/export_frozen_config.py; the code is authoritative',
           'paired_bootstrap': 'Poisson(1) weights shared by both arms; B = 2000, seed 0 (k% test pass, R1); B = 10,000, seed 20261029 (R2)'}
    for key, (path, names) in SPEC.items():
        c = consts(path)
        cfg[key] = {n: c[n] for n in names if n in c}
    os.makedirs(os.path.join(REPO, 'configs', 'prompts'), exist_ok=True)
    json.dump(cfg, open(os.path.join(REPO, 'configs', 'frozen_hyperparameters.json'), 'w'), indent=1, default=list)
    for fn, (path, name) in PROMPTS.items():
        open(os.path.join(REPO, 'configs', 'prompts', fn), 'w', encoding='utf-8').write(consts(path)[name])
    print('exported', len(cfg), 'sections,', len(PROMPTS), 'prompts')


if __name__ == '__main__':
    main()
