#!/bin/bash
# TRAINING-FREE PHASE: P_0 on BOTH cards (70B, vLLM TP2), then k0 execution on CPU.
#   p0_run.sh dev   -> 3 designs on dev_sel2000          p0_run.sh test <design>  -> the selected (frozen) design on test
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set -eo pipefail
SPLIT=$1; DES=${2:-d1,d2,d3}
K=$WORK_DIR; P=$K/parser; C=$P/code; mkdir -p $P/p0/raw $P/p0/exec $P/logs
if [ "$SPLIT" = dev ]; then Q=/data/questions/dev.json; SEL="--quids_list dev_sel2000"; S=dev_sel2000; else Q=/data/questions/test.json; SEL=""; S=test; fi
bash $C/run_card_p.sh 0,1 ${VLLM_IMAGE:-tkgqa-vllm:latest} p0_${S} $P "python3 /kpct/parser/code/p0_infer_70b.py --split $S --questions $Q $SEL --prompts /kpct/parser/code/p0_prompts --designs $DES --out /kpct/parser/p0/raw" > $P/logs/p0_infer_$S.log 2>&1
RAW=$(echo $DES | tr ',' '\n' | awk -v S=$S '{printf "%s%s=/kpct/parser/p0/raw/raw_P0_%s_%s.jsonl.gz", (NR>1?",":""), $1, S, $1}')
bash $C/run_cpu_p.sh p0exec_$S $P 8 pylib "python3 /kpct/parser/code/p0_execute.py --split $S --raw $RAW --questions $Q --out /kpct/parser/p0/exec --workers 8" > $P/logs/p0_exec_$S.log 2>&1
