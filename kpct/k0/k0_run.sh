#!/bin/bash
# k0_run.sh <phase: p0 | g0 | dry> [jobdir]   -- k = 0 track driver on the GPU host (bare host; containers inside).
#   p0 : P_0 dev_sel2000 (3 designs, one TP2 session) -> k0 execution -> selection (dev_sel2000 only) -> STOP_DEV_FAILURE
#        check -> pre-test freeze -> P_0 test (selected design, TP2) -> k0 execution -> test freeze (unscored)
#        -> k0 guidance bundles for dev_sel2000 and test from the selected design's parses (CPU)
#   g0 : G_0 dev_sel2000 (3 designs, one TP2 session) -> predictions -> selection -> STOP_DEV_FAILURE check -> pre-test
#        freeze -> G_0 test (selected design, TP2) -> predictions -> test freeze (unscored)
#   dry: CPU plumbing run of both phases in a separate tree ($WORK_DIR/k0/dry/root/k0) on N = 5
#        dev_sel2000 questions with placeholder answers; the GPU steps are replaced by tokenizer-only prompt builds plus
#        stand-in generations (k0_dry.py). The dry "test" split is the same 5 dev questions (no test file is opened).
# Every step is resumable: it is skipped when its output exists (generation resumes from shards inside k0_infer_70b.py).
# Before each GPU step: both cards must be free (< 2000 MiB used; own stale k0_* containers are stopped, nothing else is
# touched) and usage-log GPU-h + the step's projected GPU-h must stay <= 52 (STOP_GPU_CAP, registered GPU-hour cap). Exit codes: 4 GPU busy,
# 5 STOP_DEV_FAILURE, 6 STOP_GPU_CAP, 7 STOP_TEST_INTEGRITY / missing freeze, 1 step failure.
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set -o pipefail
PH=$1; JD=${2:-$PWD}
H=$WORK_DIR; C=$H/k0/code; CC=/kpct/k0/code
if [ "$PH" = dry ]; then KH=$H/k0/dry/root/k0; KC=/kpct/k0/dry/root/k0; DRY=1; else KH=$H/k0; KC=/kpct/k0; DRY=0; fi
export KPCT_K0=$KC
CAP=52.0
declare -A PROJ=([P0_dev_sel2000]=0.85 [P0_test]=7.30 [G0_dev_sel2000]=0.95 [G0_test]=8.10)   # GPU-h (B1 stage-0 rates)
LOG=$KH/logs; mkdir -p $KH/p0/raw $KH/p0/exec $KH/g0/raw $KH/g0/pred $KH/bundles $KH/select $KH/freeze $LOG $JD/out
SLOG=$KC/multitq_kpct_k0_selection_log.jsonl
M=$CC/multitq_kpct_subsets.json; EV=$CC/multitq_eval.py; KG=/data/kg/full.txt
PM=$CC/parser/p0_prompts/p0_prompts_manifest.json; GM=$CC/g0_prompts/g0_prompts_manifest.json
if [ $DRY = 1 ]; then
  DEVQ=$KC/mock_questions.json; TESTQ=$KC/mock_questions.json; QDEV="--quids $KC/dry_quids.json"; QTEST="--quids $KC/dry_quids.json"
  SELMOCK="--mock_questions $KC/mock_questions.json"; FZDRY="--dry"
else
  DEVQ=/data/questions/dev.json; TESTQ=/data/questions/test.json; QDEV="--quids_list dev_sel2000"; QTEST=""; SELMOCK=""; FZDRY=""
fi
say() { echo "$(date -u +%FT%TZ) $*" | tee -a $LOG/k0_run_$PH.log; }
die() { say "STOP rc=$1: $2"; harvest; exit $1; }
cpu() { local tag=$1; shift; say "cpu $tag"; bash $C/run_cpu_k0.sh $tag $JD 8 ${GPU_IMAGE:-tkgqa-gpu:latest} "$*" >> $LOG/cpu_$tag.log 2>&1 || die 1 "cpu step $tag failed (log $LOG/cpu_$tag.log)"; }
vcpu() { local tag=$1; shift; say "vcpu $tag"; bash $C/run_cpu_k0.sh $tag $JD 8 ${VLLM_IMAGE:-tkgqa-vllm:latest} "$*" >> $LOG/cpu_$tag.log 2>&1 || die 1 "cpu step $tag failed (log $LOG/cpu_$tag.log)"; }
gpu_guard() {
  { echo "== $(date -u +%FT%TZ)"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; docker ps --format '{{.Names}} {{.Image}} {{.RunningFor}}'; } >> $LOG/gpu_state.log 2>&1
  local stale=$(docker ps --filter "name=^k0_" -q)
  [ -n "$stale" ] && { say "stopping own stale k0_ containers: $stale"; docker stop $stale >> $LOG/gpu_state.log 2>&1; sleep 5; }
  for i in 0 1; do
    local U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i | tr -d ' ')
    [ "$U" -lt 2000 ] || die 4 "card $i busy ($U MiB) -- no GPU job launched"
  done
}
gpu_cap() {
  local spent=$(awk -F'\t' 'NR>1 && $7 ~ /^[0-9.]+$/ {s+=$7} END {printf "%.4f", s}' $H/gpu_usage_kpct.tsv)
  say "usage log total $spent GPU-h + projected $1 (cap $CAP)"
  awk -v s=$spent -v p=$1 -v c=$CAP 'BEGIN{exit !(s+p<=c)}' || die 6 "STOP_GPU_CAP: $spent + $1 > $CAP"
}
done_raw() { local arm=$1 split=$2; shift 2; for d in "$@"; do [ -f $KH/${arm,,}/raw/raw_${arm}_${split}_$d.jsonl.gz.done ] || return 1; done; return 0; }
infer() {   # <arm> <split> <designs csv> <input args...>
  local arm=$1 split=$2 des=$3; shift 3
  done_raw $arm $split ${des//,/ } && { say "$arm $split $des raw complete"; return; }
  local cmd="python3 -u $CC/k0_infer_70b.py --arm $arm --split $split --designs $des --out $KC/${arm,,}/raw $*"
  if [ $DRY = 1 ]; then
    vcpu ${arm}_${split}_prompts "$cmd --dry_prompts"
    if [ $arm = P0 ]; then cpu ${arm}_${split}_fake "python3 $CC/k0_dry.py fake --arm P0 --split $split --designs $des --texts $KC/p0_standin_texts.json --out $KC/p0/raw"
    else cpu ${arm}_${split}_fake "python3 $CC/k0_dry.py fake --arm G0 --split $split --designs $des --bundles $KC/bundles/$split.jsonl --out $KC/g0/raw"; fi
  else
    gpu_guard; gpu_cap ${PROJ[${arm}_${split}]}
    say "gpu ${arm}_${split} ($des)"
    bash $C/run_card_k0.sh 0,1 ${VLLM_IMAGE:-tkgqa-vllm:latest} ${arm}_${split} $JD "$cmd" >> $LOG/gpu_${arm}_${split}.log 2>&1 || die 1 "gpu ${arm}_${split} failed"
  fi
  done_raw $arm $split ${des//,/ } || die 1 "$arm $split raw files missing after generation"
}
chosen() { python3 -c "import json;print(json.load(open('$KH/select/select_$1.json'))['selected'])"; }
fz_test_done() { python3 -c "import json,sys;sys.exit(0 if json.load(open('$KH/freeze/freeze_$1.json')).get('test_predictions') else 1)" 2>/dev/null; }
devfail() { python3 -c "import json,sys;sys.exit(1 if json.load(open('$KH/select/select_$1.json'))['DEV_FAILURE_trigger'] else 0)"; }
harvest() {
  cp $KH/select/*.json $KH/freeze/*.json $JD/out/ 2>/dev/null
  for f in $KH/p0/raw/infer_stats_*.json $KH/g0/raw/infer_stats_*.json $KH/p0/exec/exec_stats_*.json $KH/g0/pred/post_stats_*.json \
           $KH/bundles/bundle_stats_*.json $KH/p0/exec/pred_P0_dev_sel2000_*.jsonl $KH/g0/pred/pred_G0_dev_sel2000_*.jsonl; do
    [ -f $f ] && cp $f $JD/out/; done
  [ -f $KH/multitq_kpct_k0_selection_log.jsonl ] && cp $KH/multitq_kpct_k0_selection_log.jsonl $JD/out/
  for f in $KH/p0/exec/pred_P0_test_*.jsonl $KH/g0/pred/pred_G0_test_*.jsonl; do [ -f $f ] && gzip -c $f > $JD/out/$(basename $f).gz; done
  for f in $LOG/*.log; do tail -60 $f > $JD/out/tail__$(basename $f); done
  grep -P "^k0\t" $H/gpu_usage_kpct.tsv > $JD/out/usage_log_k0.tsv 2>/dev/null || true
  chmod -R a+rX $JD/out 2>/dev/null; true
}

say "== phase $PH start (dry=$DRY, outputs $KH)"
if [ $DRY = 1 ]; then
  [ -f $KH/mock_questions.json ] || cpu dry_prep "python3 $CC/k0_dry.py prep --n 5 --out $KC"
fi
# ----------------------------------------------------------------------------------------------------------- P_0
if [ $PH = p0 ] || [ $PH = dry ]; then
  infer P0 dev_sel2000 d1,d2,d3 --questions $DEVQ $QDEV
  RAWS=$(for d in d1 d2 d3; do printf "%s=%s," $d $KC/p0/raw/raw_P0_dev_sel2000_$d.jsonl.gz; done); RAWS=${RAWS%,}
  [ -f $KH/p0/exec/exec_stats_P0_dev_sel2000.json ] || cpu p0exec_dev "python3 $CC/parser/p0_execute.py --split dev_sel2000 --raw $RAWS --questions $DEVQ --out $KC/p0/exec --workers 8"
  PREDS=$(for d in d1 d2 d3; do printf "%s=%s," $d $KC/p0/exec/pred_P0_dev_sel2000_$d.jsonl; done); PREDS=${PREDS%,}
  [ -f $KH/select/select_P0.json ] || cpu p0select "python3 $CC/k0_select.py --arm P0 --cands $PREDS --stats $KC/p0/exec/exec_stats_P0_dev_sel2000.json --prompt_manifest $PM --dev $DEVQ --manifest $M --eval $EV --log $SLOG --out $KC/select/select_P0.json $SELMOCK"
  D=$(chosen P0); say "P_0 selected design $D"
  if [ $DRY = 0 ]; then devfail P0 || die 5 "STOP_DEV_FAILURE for P_0 (select_P0.json): one logged bug-fix iteration before any test pass"; fi
  [ -f $KH/freeze/freeze_P0.json ] || cpu p0freeze_pre "python3 $CC/k0_freeze.py --arm P0 --mode pre_test $FZDRY"
  [ -f $KH/freeze/freeze_P0.json ] || die 7 "no P_0 pre-test freeze"
  if ! fz_test_done P0; then
    infer P0 test $D --questions $TESTQ $QTEST
    [ -f $KH/p0/exec/pred_P0_test_$D.jsonl ] && [ -f $KH/p0/exec/exec_stats_P0_test.json ] || \
      cpu p0exec_test "python3 $CC/parser/p0_execute.py --split test --raw $D=$KC/p0/raw/raw_P0_test_$D.jsonl.gz --questions $TESTQ --out $KC/p0/exec --workers 8"
    cpu p0freeze_test "python3 $CC/k0_freeze.py --arm P0 --mode test $FZDRY" || die 7 "STOP_TEST_INTEGRITY P_0"
    fz_test_done P0 || die 7 "P_0 test freeze not written"
  fi
  for S in dev_sel2000 test; do
    Q=$DEVQ; [ $S = test ] && Q=$TESTQ
    [ -f $KH/bundles/bundle_stats_$S.json ] || cpu bundles_$S "python3 $CC/k0_bundles.py --split $S --exec_rows $KC/p0/exec/exec_P0_${S}_$D.jsonl.gz --pred $KC/p0/exec/pred_P0_${S}_$D.jsonl --design $D --questions $Q --out $KC/bundles/$S.jsonl --workers 8"
  done
  say "P_0 phase complete (design $D)"
fi
# ----------------------------------------------------------------------------------------------------------- G_0
if [ $PH = g0 ] || [ $PH = dry ]; then
  fz_test_done P0 || die 7 "P_0 is not frozen with test predictions"
  DP=$(chosen P0)
  for S in dev_sel2000 test; do [ -f $KH/bundles/bundle_stats_$S.json ] || die 7 "k0 bundles for $S missing"; done
  infer G0 dev_sel2000 g1,g2,g3 --bundles $KC/bundles/dev_sel2000.jsonl $QDEV
  RAWS=$(for g in g1 g2 g3; do printf "%s=%s," $g $KC/g0/raw/raw_G0_dev_sel2000_$g.jsonl.gz; done); RAWS=${RAWS%,}
  [ -f $KH/g0/pred/post_stats_G0_dev_sel2000.json ] || cpu g0post_dev "python3 $CC/k0_post_g0.py --split dev_sel2000 --raw $RAWS --p0_pred $KC/p0/exec/pred_P0_dev_sel2000_$DP.jsonl --kg $KG --out $KC/g0/pred"
  PREDS=$(for g in g1 g2 g3; do printf "%s=%s," $g $KC/g0/pred/pred_G0_dev_sel2000_$g.jsonl; done); PREDS=${PREDS%,}
  [ -f $KH/select/select_G0.json ] || cpu g0select "python3 $CC/k0_select.py --arm G0 --cands $PREDS --stats $KC/g0/pred/post_stats_G0_dev_sel2000.json --prompt_manifest $GM --dev $DEVQ --manifest $M --eval $EV --log $SLOG --out $KC/select/select_G0.json $SELMOCK"
  G=$(chosen G0); say "G_0 selected design $G"
  if [ $DRY = 0 ]; then devfail G0 || die 5 "STOP_DEV_FAILURE for G_0 (select_G0.json): one logged bug-fix iteration before any test pass"; fi
  [ -f $KH/freeze/freeze_G0.json ] || cpu g0freeze_pre "python3 $CC/k0_freeze.py --arm G0 --mode pre_test $FZDRY"
  [ -f $KH/freeze/freeze_G0.json ] || die 7 "no G_0 pre-test freeze"
  if ! fz_test_done G0; then
    infer G0 test $G --bundles $KC/bundles/test.jsonl $QTEST
    [ -f $KH/g0/pred/pred_G0_test_$G.jsonl ] && [ -f $KH/g0/pred/post_stats_G0_test.json ] || \
      cpu g0post_test "python3 $CC/k0_post_g0.py --split test --raw $G=$KC/g0/raw/raw_G0_test_$G.jsonl.gz --p0_pred $KC/p0/exec/pred_P0_test_$DP.jsonl --kg $KG --out $KC/g0/pred"
    cpu g0freeze_test "python3 $CC/k0_freeze.py --arm G0 --mode test $FZDRY" || die 7 "STOP_TEST_INTEGRITY G_0"
    fz_test_done G0 || die 7 "G_0 test freeze not written"
  fi
  say "G_0 phase complete (design $G)"
fi
harvest
say "== phase $PH done"
