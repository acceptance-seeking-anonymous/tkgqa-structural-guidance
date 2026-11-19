#!/bin/bash
# ps_dev.sh <ARM> <LLM list> <executor subset> <ckpt map name=raw_file_basename,...> <ckpts in training order>
# CPU only: executes every checkpoint x tau on dev_sel2000 with the FROZEN parser code / executor / S_k models
# (/kpct/parser/code, /kpct/executor, /kpct/preds), then pk_select.py; outputs under $WORK_DIR/parser_seeds
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
ARM=$1; LST=$2; SUB=$3; MAP=$4; CKS=$5
K=$WORK_DIR; PS=$K/parser_seeds; C=$K/parser/code
mkdir -p $PS/exec/$ARM $PS/select/$ARM $PS/logs
RAW=$(echo $MAP | tr ',' '\n' | awk -v A=$ARM '{split($0,x,"="); printf "%s%s=/kpct/parser_seeds/raw/%s/%s", (NR>1?",":""), x[1], A, x[2]}')
echo "$(date -u +%FT%TZ) $ARM execute $RAW"
bash $C/run_cpu_p.sh psx_$ARM $PS 8 pylib "python3 /kpct/parser/code/pk_execute.py --k $SUB --split dev_sel2000 --raw $RAW --preds /kpct/preds --out /kpct/parser_seeds/exec/$ARM --workers 8" > $PS/logs/exec_dev_$ARM.log 2>&1
echo "exec rc=$?"; tail -2 $PS/logs/exec_dev_$ARM.log
bash $C/run_cpu_p.sh pss_$ARM $PS 2 pylib "python3 /kpct/parser/code/pk_select.py --k $SUB --list $LST --exec_dir /kpct/parser_seeds/exec/$ARM --exec_stats /kpct/parser_seeds/exec/$ARM/exec_stats_P_${SUB}_dev_sel2000.json --ckpts $CKS --log /kpct/parser_seeds/select/$ARM/selection_log.json --out /kpct/parser_seeds/select/$ARM" > $PS/logs/select_$ARM.log 2>&1
echo "select rc=$?"; tail -1 $PS/logs/select_$ARM.log | cut -c1-600
