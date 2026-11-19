#!/bin/bash
# reproduce_kpct_0_setup.sh -- stage this repository's code into the GPU-host work tree ($WORK_DIR) in the layout the
# GPU-side scripts expect, and build the executor's private LightGBM library path. Run on the GPU host after stages
# 1-3 (scripts/reproduce_kpct_1_subsets.sh, _2_structural.sh) have produced $SWORK/stage/kpct/{executor,bundles,preds}.
#   $WORK_DIR/executor/          kpct_core.py, kpct_executor.py, struct/ (structural modules + scorer), mtqdata/kg, models/<k>/
#   $WORK_DIR/{bundles,preds}/   guidance bundles and Rules / Ranker prediction files per k (from $SWORK/stage/kpct)
#   $WORK_DIR/pylib/             lightgbm 4.6.0, numpy 2.5.4, scipy 1.18.1 (pip --target, used by CPU executor containers)
#   $WORK_DIR/guided/code/       Reader / Dense-RAG code      $WORK_DIR/parser/code/  Parser code      $WORK_DIR/k0/code/  k = 0 code
# Inputs not shipped: the MultiTQ release ($DATA_DIR), the full-data structural exports ($MTQ_DIR/phase1_structural)
# and the quid lists s_train100k_quids.json / reader_train60k_quids.json ($S_TRAIN100K_PATH, $READER_TRAIN60K_PATH).
set -euo pipefail
source "$(dirname "$0")/env_check.sh" DATA_DIR WORK_DIR SWORK S_TRAIN100K_PATH
K=$WORK_DIR; ST=$SWORK/stage/kpct; SUB=$SWORK/subsets/multitq_kpct_subsets.json
mkdir -p "$K"/{executor/struct,executor/mtqdata/kg,executor/models,guided/code,parser/code,k0/code/parser,bundles,preds}
# executor (frozen structural pipeline at every k)
cp "$REPO"/kpct/structural/{kpct_core.py,kpct_executor.py} "$K/executor/"
cp "$REPO"/src/struct/{mtq_align.py,mtq_cand.py,mtq_clf.py,mtq_export.py,mtq_lib.py} "$REPO/src/multitq_eval.py" "$K/executor/struct/"
cp "$DATA_DIR"/kg/{train,valid,test}.txt "$K/executor/mtqdata/kg/"
[ -d "$ST" ] && cp -r "$ST/executor/models/." "$K/executor/models/" && cp -r "$ST/bundles/." "$K/bundles/" && cp -r "$ST/preds/." "$K/preds/"
[ -d "$K/pylib" ] || docker run --rm -v "$K":/kpct "${GPU_IMAGE:-tkgqa-gpu:latest}" pip install -q --target /kpct/pylib lightgbm==4.6.0 numpy==2.5.4 scipy==1.18.1
# Reader (G_k) / Dense-RAG (GU_k)
cp "$REPO"/kpct/reader/* "$REPO/src/multitq_eval.py" "$SUB" "$S_TRAIN100K_PATH" "$K/guided/code/"
# Parser (P_k, P_0)
cp -r "$REPO"/kpct/parser/. "$K/parser/code/"; cp "$REPO/src/multitq_eval.py" "$SUB" "$K/parser/code/"
# k = 0 (P_0, G_0)
cp -r "$REPO"/kpct/k0/. "$K/k0/code/"; cp "$REPO/kpct/reader/gk_common.py" "$REPO/src/multitq_eval.py" "$SUB" "$K/k0/code/"
cp -r "$REPO"/kpct/parser/{p0_execute.py,p0_infer_70b.py,p0_prompts.py,p0_exemplars.json,p0_prompts,pk_common.py} "$K/k0/code/parser/"
echo "staged into $K"
