#!/bin/bash
# stageB_plan.sh -- Reader (G_k) / Dense-RAG (GU_k) track commands, in registered run order.
# Usage on the GPU host from $WORK_DIR/guided/code:   source stageB_plan.sh; <function> ...
# Card 0 is this track's card (card 1 = LLM-parser track). One process per card, no device_map.
# Before ANY dev_sel2000 score of G_k:  check_bundles <subset>  must exit 0 (schema, coverage, gold, STOP_OOF_LEAK).
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
: "${DATA_DIR:?set DATA_DIR (see configs/paths.env.example)}"
W=$WORK_DIR/guided/code
K=/kpct; C=/kpct/guided/code; M=$C/multitq_kpct_subsets.json
CARD=${CARD:-0}
S_PRED=${S_PRED:-/kpct/preds}              # S_k seed-0 prediction files: $S_PRED/<subset>/{dev_sel2000,test}_S_seed0.jsonl (structural track)
SLOG=/kpct/guided/multitq_kpct_guided_selection_log.jsonl
PIP="(pip install --break-system-packages -q --no-deps peft==0.21.2 || pip install --break-system-packages -q --no-deps peft) > /dev/null 2>&1"

check_bundles() {   # <subset>  (k01_s0 ... ; k100 uses the k100 bundles + s_train100k)
  docker run --rm -v $WORK_DIR:/kpct -v $DATA_DIR:/data:ro -w $C ${GPU_IMAGE:-tkgqa-gpu:latest} \
    python3 gk_check_bundles.py --bundles $(bdir $1) --subset $1 --manifest $M --data /data --s_train100k $C/s_train100k_quids.json \
    --out /kpct/guided/checks/check_$1.json
}
bdir() { echo /kpct/bundles/$1; }        # the structural track's bundles at EVERY k (incl. k100); converted exports = cross-check only
btrain() { local d=$(bdir $1); for e in .jsonl .jsonl.gz; do [ -f "$WORK_DIR${d#/kpct}/train$e" ] && { echo $d/train$e; return; }; done; echo $d/train.jsonl; }
train_G() {         # <subset> <list> <order S|random>   e.g. train_G k01_s0 k01_s0 S ; train_G k100 k100 S
  bash $W/run_card.sh $CARD ${GPU_IMAGE:-tkgqa-gpu:latest} train_G_$2_ord$3 $W "$PIP; python3 -u $C/gk_train.py --arm G --list $2 \
    --bundles $(btrain $1) --questions /data/questions/train.json --manifest $M --order $3 --out /kpct/guided/adapters \
    > /kpct/guided/logs/train_G_$2_ord$3.log 2>&1"
}
train_GU() {        # <list>
  bash $W/run_card.sh $CARD ${GPU_IMAGE:-tkgqa-gpu:latest} train_GU_$1 $W "$PIP; python3 -u $C/gk_train.py --arm GU --list $1 \
    --gu_facts /kpct/gu_retrieval/gu_train_union.jsonl.gz --questions /data/questions/train.json --manifest $M --out /kpct/guided/adapters \
    > /kpct/guided/logs/train_GU_$1.log 2>&1"
}
session_G() {       # <subset> <list> <runs name:order:dir,...> [controls]   one vLLM session: dev ckpts -> select -> freeze -> test
  bash $W/run_card.sh $CARD ${VLLM_IMAGE:-tkgqa-vllm:latest} sess_G_$2 $W "python3 -u $C/gk_run_arm.py --arm G --list $2 --runs $3 \
    --bundles_dir $(bdir $1) --dev_q /data/questions/dev.json --test_q /data/questions/test.json \
    --s_list_dev $S_PRED/$1/dev_sel2000_S_seed0.jsonl --s_list_test $S_PRED/$1/test_S_seed0.jsonl --controls=${4:-} \
    --manifest $M --eval $C/multitq_eval.py --kg /data/kg/full.txt --log $SLOG --out /kpct/guided/preds/G_$2 > /kpct/guided/logs/sess_G_$2.log 2>&1"
}
session_GU() {      # <list>
  bash $W/run_card.sh $CARD ${VLLM_IMAGE:-tkgqa-vllm:latest} sess_GU_$1 $W "python3 -u $C/gk_run_arm.py --arm GU --list $1 \
    --runs main:S:/kpct/guided/adapters/GU_$1_s0 --gu_dev /kpct/gu_retrieval/gu_dev_sel2000.jsonl.gz --gu_test /kpct/gu_retrieval/gu_test.jsonl.gz \
    --dev_q /data/questions/dev.json --test_q /data/questions/test.json \
    --manifest $M --eval $C/multitq_eval.py --kg /data/kg/full.txt --log $SLOG --out /kpct/guided/preds/GU_$1 > /kpct/guided/logs/sess_GU_$1.log 2>&1"
}
# ---------------------------------------------------------------- registered run order (this track's items)
# (2) k = 1 subset-seed-0 core:  G_1 order grid + G_1 + SHUF_G1/QONLY_G1 + GU_1
core_k1() {
  mkdir -p $WORK_DIR/guided/{logs,checks,preds,adapters}
  check_bundles k01_s0 || { echo "STOP_OOF_LEAK / schema check failed"; return 1; }
  train_G k01_s0 k01_s0 S && train_G k01_s0 k01_s0 random && \
  session_G k01_s0 k01_s0 ordS:S:/kpct/guided/adapters/G_k01_s0_ordS_s0,ordrandom:random:/kpct/guided/adapters/G_k01_s0_ordrandom_s0 SHUF,QONLY
  train_GU k01_s0 && session_GU k01_s0
}
# Later items use the order fixed in /kpct/guided/preds/G_k01_s0/order_decision_k01_s0.json (ORD below):
#   (4) k = 100:  check_bundles k100; train_G k100 k100 $ORD; session_G k100 k100 main:$ORD:/kpct/guided/adapters/G_k100_ord${ORD}_s0; train_GU k100; session_GU k100
#       (fallback 3 in force -> lists cap20k_k100 / cap20k_k10 instead of k100 / k10_s0)
#   (5) k = 10:   check_bundles k10_s0; train_G k10_s0 k10_s0 $ORD; session_G k10_s0 k10_s0 main:$ORD:...; train_GU k10_s0; session_GU k10_s0
#   (6) k = 5:    G_5 (as k = 10)                      (9) GU_5 (unless fallback 2)
#   (7) k = 1 subset seeds 1, 2: G only (unless fallback 6)   (8) G_0: 70B, NOT RUN under fallback 4
