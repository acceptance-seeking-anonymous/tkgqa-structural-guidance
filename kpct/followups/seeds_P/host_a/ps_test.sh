#!/bin/bash
# ps_test.sh <ARM> <LLM list> <executor subset> <lora seed> <checkpoint> <tau name>  -- CPU only; frozen code / executor / S_k models
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
ARM=$1; LST=$2; SUB=$3; SEED=$4; CK=$5; TN=$6
K=$WORK_DIR; PS=$K/parser_seeds; C=$K/parser/code; mkdir -p $PS/test/$ARM $PS/logs
T=$(python3 -c "print({'off':'off','m030':'-0.30','m010':'-0.10','m003':'-0.03'}['$TN'])")
N=s${SEED}${CK}
bash $C/run_cpu_p.sh pst_$ARM $PS 8 pylib "python3 /kpct/parser/code/pk_execute.py --k $SUB --split test --raw $N=/kpct/parser_seeds/raw/$ARM/raw_P_test_$N.jsonl.gz --taus=$T --preds /kpct/preds --out /kpct/parser_seeds/test/$ARM --workers 8" > $PS/logs/exec_test_$ARM.log 2>&1
echo "exec rc=$?"; tail -2 $PS/logs/exec_test_$ARM.log | cut -c1-400
python3 $PS/code/ps_freeze.py --mode test --arm $ARM --list $LST --sub $SUB --seed $SEED --map $CK=$N
echo "freeze rc=$?"
