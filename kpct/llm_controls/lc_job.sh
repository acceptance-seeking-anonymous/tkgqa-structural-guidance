#!/bin/bash
# lc_job.sh <item> [<item> ...] -- LLM-controls track (amendment LC1) on the GPU host, one card ($LC_CARD, default 1).
# Items (registered run order; cut point = GPLAIN_1 first, then GU_5):
#   setup gplain_train gplain_session gu5_train gu5_session rob_gplain rob_gu5 tokens
# Registered GPU-hour cap of the track: 5.5 GPU-h (usage-log track "llm_controls"); before every GPU item:
# spent + projected <= CAP, else NOT_RUN.
# Frozen code: $WORK_DIR/guided/code (sha256 recorded in setup, mounted read-only). Only $WORK_DIR/llm_controls is
# written. Resume-safe (gk_train / gk_run_arm skip completed work).
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
: "${MODEL_DIR:?set MODEL_DIR (see configs/paths.env.example)}"
J=$PWD; K=$WORK_DIR; LC=$K/llm_controls; CAP=5.5; L=$K/gpu_usage_kpct.tsv
mkdir -p $LC/code $LC/adapters $LC/preds $LC/logs $LC/robust $LC/tokens $J/out
for f in gk_plain.py gplain_train.py gplain_run_arm.py gplain_infer.py lc_tokens.py run_card_lc.sh lc_job.sh; do
  if [ -f $J/$f ]; then
    if [ -f $LC/code/$f ] && ! cmp -s $J/$f $LC/code/$f; then echo "CODE_CHANGED $f -- refusing to overwrite staged code" | tee -a $J/out/errors.txt; exit 2; fi
    cp -n $J/$f $LC/code/$f
  fi
done
RC="bash $LC/code/run_card_lc.sh"
# Memory-only options of amendment LC1-A1 (the reported runs used both: part of the card's memory was held by other
# processes): vLLM gpu_memory_utilization LC_GPU_MEM (default 0.40 as run; frozen Reader default 0.85) and gradient
# checkpointing in LoRA training (LC_GRAD_CKPT=1 as run; 0 = frozen recipe without it). Checkpointing changes memory
# and speed only; the memory fraction changes only the vLLM KV-cache size (batching), and greedy GPU decoding is not
# bitwise batch-invariant.
GMEM=${LC_GPU_MEM:-0.40}; GCK=""; [ "${LC_GRAD_CKPT:-1}" = 1 ] && GCK="--grad_ckpt"
C=/kpct/guided/code; M=$C/multitq_kpct_subsets.json; CL=/kpct/llm_controls/code
SLOG=/kpct/llm_controls/multitq_llm_controls_selection_log.jsonl
PIP="(pip install --break-system-packages -q --no-deps peft==0.21.2 || pip install --break-system-packages -q --no-deps peft) > /dev/null 2>&1"
spent() { awk -F'\t' '$1=="llm_controls"{s+=$7} END{printf "%.4f", s+0}' $L; }
gpu_cap_ok() { awk -v s=$(spent) -v p=$1 -v c=$CAP 'BEGIN{exit !(s+p<=c)}'; }
notrun() { echo "NOT_RUN $1 (GPU-hour cap: spent $(spent), projected $2, cap $CAP)" | tee -a $J/out/not_run.txt; }
sel_of() { python3 -c "import json; print(json.load(open('$1'))['selected'])"; }
AG=$LC/adapters/GPLAIN_k01_s0_s0; PG=$LC/preds/GPLAIN_k01_s0
AU=$LC/adapters/GU_k05_s0_s0;     PU=$LC/preds/GU_k05_s0

item() {
case $1 in
setup)
  { echo "utc $(date -u +%FT%TZ)"
    sha256sum $K/guided/code/{gk_common.py,gk_train.py,gk_infer.py,gk_select.py,gk_run_arm.py,multitq_eval.py,multitq_kpct_subsets.json} $LC/code/*
    ls -la $K/bundles/k01_s0/ $K/gu_retrieval/ $K/guided/adapters/G_k01_s0_ordS_s0/ $K/guided/adapters/GU_k01_s0_s0/
    sha256sum $K/bundles/k01_s0/* $K/gu_retrieval/gu_*.jsonl.gz
    for R in robustness robustness_r2; do ls -la $K/$R/bundles/k01_s0/ $K/$R/gu/ $K/$R/inputs/; sha256sum $K/$R/bundles/k01_s0/variants.jsonl $K/$R/gu/gu_variants.jsonl.gz $K/$R/inputs/variants_questions.json; done
    echo "spent_llm_controls $(spent)"; } > $J/out/setup_check.txt 2>&1 ;;
tokens)
  docker run --rm --name lc_tokens_$(date +%s) --cpus 12 -v $K:/kpct:ro -v $LC:/kpct/llm_controls -v $J/out:/workspace \
    -v $MODEL_DIR:/hf -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e OMP_NUM_THREADS=1 -e GK_CODE=/kpct/guided/code \
    -e TOKENIZERS_PARALLELISM=false ${GPU_IMAGE:-tkgqa-gpu:latest} python3 -u $CL/lc_tokens.py --manifest $M --workers 12 --out /workspace \
    > $J/out/tokens.log 2>&1; echo "tokens rc=$?" ;;
gplain_train)
  [ -f $AG/train_stats.json ] && { echo "gplain_train done"; return 0; }
  gpu_cap_ok 0.80 || { notrun GPLAIN_train 0.80; return 3; }
  BT=/kpct/bundles/k01_s0/train.jsonl; [ -f $K/bundles/k01_s0/train.jsonl ] || BT=/kpct/bundles/k01_s0/train.jsonl.gz   # = stageB_plan btrain
  $RC ${GPU_IMAGE:-tkgqa-gpu:latest} train_GPLAIN_k01_s0 5400 "$PIP; python3 -u $CL/gplain_train.py --arm G --list k01_s0 --name GPLAIN_k01_s0_s0 \
    --bundles $BT --questions /data/questions/train.json --manifest $M --order S --seed 0 $GCK \
    --out /kpct/llm_controls/adapters > /kpct/llm_controls/logs/train_GPLAIN_k01_s0.log 2>&1" ;;
gplain_session)
  [ -f $AG/train_stats.json ] || { echo "no GPLAIN adapter"; return 1; }
  gpu_cap_ok 0.95 || { notrun GPLAIN_session 0.95; return 3; }
  $RC ${VLLM_IMAGE:-tkgqa-vllm:latest} sess_GPLAIN_k01_s0 7200 "python3 -u $CL/gplain_run_arm.py --arm G --list k01_s0 \
    --runs plain:S:/kpct/llm_controls/adapters/GPLAIN_k01_s0_s0 --bundles_dir /kpct/bundles/k01_s0 \
    --dev_q /data/questions/dev.json --test_q /data/questions/test.json --manifest $M --eval $C/multitq_eval.py \
    --kg /data/kg/full.txt --log $SLOG --gpu_mem $GMEM --out /kpct/llm_controls/preds/GPLAIN_k01_s0 > /kpct/llm_controls/logs/sess_GPLAIN_k01_s0.log 2>&1" ;;
gu5_train)
  [ -f $AU/train_stats.json ] && { echo "gu5_train done"; return 0; }
  gpu_cap_ok 2.40 || { notrun GU_5_train 2.40; return 3; }
  $RC ${GPU_IMAGE:-tkgqa-gpu:latest} train_GU_k05_s0 10800 "$PIP; python3 -u $C/gk_train.py --arm GU --list k05_s0 \
    --gu_facts /kpct/gu_retrieval/gu_train_union.jsonl.gz --questions /data/questions/train.json --manifest $M $GCK \
    --out /kpct/llm_controls/adapters > /kpct/llm_controls/logs/train_GU_k05_s0.log 2>&1" ;;
gu5_session)
  [ -f $AU/train_stats.json ] || { echo "no GU_5 adapter"; return 1; }
  gpu_cap_ok 0.80 || { notrun GU_5_session 0.80; return 3; }
  $RC ${VLLM_IMAGE:-tkgqa-vllm:latest} sess_GU_k05_s0 7200 "python3 -u $C/gk_run_arm.py --arm GU --list k05_s0 \
    --runs main:S:/kpct/llm_controls/adapters/GU_k05_s0_s0 --gu_dev /kpct/gu_retrieval/gu_dev_sel2000.jsonl.gz \
    --gu_test /kpct/gu_retrieval/gu_test.jsonl.gz --dev_q /data/questions/dev.json --test_q /data/questions/test.json \
    --manifest $M --eval $C/multitq_eval.py --kg /data/kg/full.txt --log $SLOG --gpu_mem $GMEM --out /kpct/llm_controls/preds/GU_k05_s0 \
    > /kpct/llm_controls/logs/sess_GU_k05_s0.log 2>&1" ;;
rob_gplain)
  F=$PG/freeze_G_k01_s0.json; [ -f $F ] || { echo "GPLAIN not frozen"; return 1; }
  SEL=$(sel_of $PG/selection_G_k01_s0_plain.json)
  for R in robustness robustness_r2; do
    O=$LC/robust/$R/GPLAIN_k01_s0; [ -s $O/pred_G_variants_GPLAIN_k01_s0.jsonl ] && continue
    gpu_cap_ok 0.15 || { notrun rob_GPLAIN_$R 0.15; continue; }
    mkdir -p $O
    $RC ${VLLM_IMAGE:-tkgqa-vllm:latest} rob_GPLAIN_$R 2400 "python3 -u $CL/gplain_infer.py --arm G --split variants \
      --bundles /kpct/$R/bundles/k01_s0/variants.jsonl --adapters GPLAIN_k01_s0=/kpct/llm_controls/adapters/GPLAIN_k01_s0_s0/$SEL \
      --order S --kg /data/kg/full.txt --gpu_mem $GMEM --out /kpct/llm_controls/robust/$R/GPLAIN_k01_s0 > /kpct/llm_controls/logs/rob_GPLAIN_$R.log 2>&1"
    echo "rob GPLAIN $R rc=$?"
  done ;;
rob_gu5)
  F=$PU/freeze_GU_k05_s0.json; [ -f $F ] || { echo "GU_5 not frozen"; return 1; }
  SEL=$(sel_of $PU/selection_GU_k05_s0_main.json)
  for R in robustness robustness_r2; do
    O=$LC/robust/$R/GU_k05_s0; [ -s $O/pred_GU_variants_GU_k05_s0.jsonl ] && continue
    gpu_cap_ok 0.15 || { notrun rob_GU_5_$R 0.15; continue; }
    mkdir -p $O
    $RC ${VLLM_IMAGE:-tkgqa-vllm:latest} rob_GU5_$R 2400 "python3 -u $C/gk_infer.py --arm GU --split variants \
      --questions /kpct/$R/inputs/variants_questions.json --gu_facts /kpct/$R/gu/gu_variants.jsonl.gz \
      --gu_pad /kpct/llm_controls/adapters/GU_k05_s0_s0/gu_pad.json --adapters GU_k05_s0=/kpct/llm_controls/adapters/GU_k05_s0_s0/$SEL \
      --kg /data/kg/full.txt --gpu_mem $GMEM --out /kpct/llm_controls/robust/$R/GU_k05_s0 > /kpct/llm_controls/logs/rob_GU5_$R.log 2>&1"
    echo "rob GU5 $R rc=$?"
  done ;;
*) echo "unknown item $1"; return 1 ;;
esac
}
rc_all=0
for it in "$@"; do
  t0=$(date +%s); echo "== item $it start $(date -u +%FT%TZ) spent $(spent)"
  item $it; rc=$?
  echo "== item $it rc=$rc $(( $(date +%s) - t0 ))s $(date -u +%FT%TZ) spent $(spent)" | tee -a $J/out/items.log
  [ $rc -ne 0 ] && rc_all=$rc
  case $it in gplain_train|gu5_train) [ $rc -ne 0 ] && { echo "training $it failed -- stopping this job"; exit $rc; } ;; esac
done
exit $rc_all
