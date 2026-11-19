#!/bin/bash
# reproduce_kpct_1_subsets.sh -- nested qtype-stratified k% train lists (1/5/10%, two extra 1% draws), dev_sel2000,
# out-of-fold splits and template-key profile -> $SWORK/subsets/multitq_kpct_subsets.json (+ _subset_profile.json).
# List sha256 values are deterministic; the file sha256 changes with created_utc. CPU, about 1 minute.
# Needs the frozen full-data linker artefacts (alias_rules.json, ent_block.json) and train_oof_fold.npy in $PHASE1_STRUCT
# and the earlier 10,000-question dev list ($DEV_SEL10K_PATH; overlap report only).
set -euo pipefail
source "$(dirname "$0")/env_check.sh" DATA_DIR SWORK PHASE1_STRUCT DEV_SEL10K_PATH READER_TRAIN60K_PATH S_TRAIN100K_PATH
P=$SWORK/subsets_build; mkdir -p "$P/struct" "$P/kp" "$SWORK/subsets"
cp "$REPO"/kpct/subsets/*.py "$P/"; cp "$REPO"/src/struct/*.py "$REPO/src/multitq_eval.py" "$P/struct/"
cp "$PHASE1_STRUCT"/{alias_rules.json,ent_block.json,train_oof_fold.npy} "$P/struct/"
cd "$P"
python kpct_keys.py && mv -f keys_kpct.pkl kp/ 2>/dev/null || true
python - <<'PY'
import os
DEV_SEL10K_PATH = os.environ["DEV_SEL10K_PATH"]
exec(open("kpct_subsets.py").read())
exec(open("kpct_manifest.py").read())
PY
cp multitq_kpct_subsets.json multitq_kpct_subset_profile.json "$SWORK/subsets/"
