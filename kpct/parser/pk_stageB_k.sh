#!/bin/bash
# STAGE B (GPU) driver for one P_k arm:  pk_stageB_k.sh <llm_list> <executor_subset> <card>
#   1. LoRA training on the silver targets (one card, cuda:0 inside)            [GPU, usage_log]
#   2. vLLM decoding of dev_sel2000 with every saved checkpoint (one session)     [GPU, usage_log]
#   3. execution + fallback for every checkpoint x tau (CPU container, 8 workers) [CPU]
# Selection (pk_select.py) and the test pass (pk_test_k.sh) are separate steps run after this one.
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set -eo pipefail
LST=$1; SUB=$2; CARD=$3
K=$WORK_DIR; P=$K/parser; C=$P/code; mkdir -p $P/adapters $P/raw/$LST $P/exec/$LST $P/logs
SB=silver_${LST}; [ -f $P/silver/${SB}_train.jsonl ] || SB=silver_${LST}_from_${SUB}   # cap20k lists: silver_<cap>_from_<subset>
SIL=/kpct/parser/silver/${SB}_train.jsonl
bash $C/run_card_p.sh $CARD ${GPU_IMAGE:-tkgqa-gpu:latest} train_$LST $P "(pip install --break-system-packages -q --no-deps peft==0.21.2 || pip install --break-system-packages -q --no-deps peft) > /w/logs/pip_peft_$LST.log 2>&1; python3 /kpct/parser/code/pk_train.py --list $LST --silver $SIL --silver_manifest /kpct/parser/silver/${SB}_manifest.json --out /kpct/parser/adapters" > $P/logs/train_$LST.log 2>&1
AD=$(python3 -c "import json,sys; s=json.load(open('$P/adapters/P_${LST}_s0/train_stats.json')); print(','.join(f'{k}=/kpct/parser/adapters/P_${LST}_s0/{k}' for k in sorted(s['adapters'], key=lambda x: float(x[2:]))))")
python3 -c "import json; m=json.load(open('$C/multitq_kpct_subsets.json')); json.dump(m['dev_selection']['quids'], open('$P/dev_sel2000_quids.json','w'))"
bash $C/run_card_p.sh $CARD ${VLLM_IMAGE:-tkgqa-vllm:latest} infer_dev_$LST $P "python3 /kpct/parser/code/pk_infer.py --split dev_sel2000 --bundles /kpct/bundles/$SUB/dev_sel2000.jsonl --quids /kpct/parser/dev_sel2000_quids.json --adapters $AD --out /kpct/parser/raw/$LST" > $P/logs/infer_dev_$LST.log 2>&1
RAW=$(echo $AD | tr ',' '\n' | sed 's#=.*##' | awk -v L=$LST '{printf "%s%s=/kpct/parser/raw/%s/raw_P_dev_sel2000_%s.jsonl.gz", (NR>1?",":""), $1, L, $1}')
bash $C/run_cpu_p.sh exec_dev_$LST $P 8 pylib "python3 /kpct/parser/code/pk_execute.py --k $SUB --split dev_sel2000 --raw $RAW --out /kpct/parser/exec/$LST --workers 8" > $P/logs/exec_dev_$LST.log 2>&1
echo "stage B dev done for $LST"
