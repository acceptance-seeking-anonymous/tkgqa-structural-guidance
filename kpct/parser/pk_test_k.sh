#!/bin/bash
# STAGE B test pass for one FROZEN P_k arm (after pk_select.py and the freeze of adapter + tau by sha256; unscored):
#   pk_test_k.sh <llm_list> <executor_subset> <card> <checkpoint> <tau name: off|m030|m010|m003>
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set -eo pipefail
LST=$1; SUB=$2; CARD=$3; CK=$4; TN=$5
K=$WORK_DIR; P=$K/parser; C=$P/code; mkdir -p $P/raw/$LST $P/test/$LST $P/logs
bash $C/run_card_p.sh $CARD ${VLLM_IMAGE:-tkgqa-vllm:latest} infer_test_$LST $P "python3 /kpct/parser/code/pk_infer.py --split test --bundles /kpct/bundles/$SUB/test.jsonl --adapters $CK=/kpct/parser/adapters/P_${LST}_s0/$CK --out /kpct/parser/raw/$LST" > $P/logs/infer_test_$LST.log 2>&1
T=$(python3 -c "print({'off':'off','m030':'-0.30','m010':'-0.10','m003':'-0.03'}['$TN'])")
bash $C/run_cpu_p.sh exec_test_$LST $P 8 pylib "python3 /kpct/parser/code/pk_execute.py --k $SUB --split test --raw $CK=/kpct/parser/raw/$LST/raw_P_test_$CK.jsonl.gz --taus=$T --out /kpct/parser/test/$LST --workers 8" > $P/logs/exec_test_$LST.log 2>&1
sha256sum $P/test/$LST/pred_P_${SUB}_test_${CK}_tau${TN}_seed*.jsonl
