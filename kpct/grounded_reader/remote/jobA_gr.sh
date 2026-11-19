#!/bin/bash
# jobA_gr.sh -- grounded_reader stage A (CPU only, no GPU, no gold): stage code, verify frozen inputs (exit 3 =
# STOP_TEST_INTEGRITY), build the Parser-grounded bundles for k05_s0, k100, k01_s0 on dev (+ I1b self-check), llm
# (R3 rewrites vidx 4 + dev paraphrases vidx 5/6), test, R1 P3 (vidx 3), R2 rewrites (vidx 3). Copies stats back.
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
J=$PWD; K=$WORK_DIR; GR=$K/grounded_reader
mkdir -p $GR/code $GR/logs $GR/checks $GR/build $GR/gen $GR/final
for f in gr_build.py gr_infer.py gr_prep.py gr_assemble.py gr_verify.py gr_dev_score.py gr_score.py run_card_gr.sh run_cpu_gr.sh jobA_gr.sh jobB_gr.sh jobC_gr.sh jobD_gr.sh expected_gr.json; do
  if [ -f $GR/code/$f ] && ! cmp -s $J/$f $GR/code/$f; then echo "CODE_CHANGED $f -- refusing to overwrite staged code"; exit 2; fi
  cp -n $J/$f $GR/code/$f
done
(cd $GR && sha256sum code/*) > $J/code_sha_A.txt
RC="bash $GR/code/run_cpu_gr.sh"
$RC verify 4 "python3 -u /kpct/grounded_reader/code/gr_verify.py /kpct/grounded_reader/code/expected_gr.json /kpct/grounded_reader/checks/verify_frozen.json" > $GR/logs/verify.log 2>&1
rc=$?; cp $GR/logs/verify.log $GR/checks/verify_frozen.json $J/ 2>/dev/null
[ $rc -eq 0 ] || { echo "verify rc=$rc (STOP_TEST_INTEGRITY)"; exit $rc; }
$RC lsdata 1 "ls -la /data/questions /kpct/robustness/P/k05_s0 /kpct/robustness_r2/P/k05_s0 /kpct/robustness/preds/k05_s0; head -c 300 /kpct/parser/dev_sel2000_quids.json" > $J/lsdata.txt 2>&1
declare -A CK=( [k01_s0]=ep3 [k05_s0]=ep1 [k100]=ep0.5 )
declare -A GT=( [k01_s0]=ordS_ep2 [k05_s0]=main_ep2 [k100]=main_ep1 )
B="python3 -u /kpct/grounded_reader/code/gr_build.py"
for k in k05_s0 k100 k01_s0; do
  ck=${CK[$k]}
  if [ $k = k100 ]; then SD=/mtq/phase1_structural/preds/dev_S_seed{b}.jsonl; ST=/mtq/phase1_structural/preds/test_S_seed{b}.jsonl
  else SD=/kpct/preds/$k/dev_sel2000_S_seed{b}.jsonl; ST=/kpct/preds/$k/test_S_seed{b}.jsonl; fi
  O=/kpct/grounded_reader/build/$k
  [ -s $GR/build/$k/test/stats.json ] || $RC b_test_$k 14 "$B --k $k --split test --raw /kpct/parser/raw/$k/raw_P_test_$ck.jsonl.gz --bundles /kpct/bundles/$k/test.jsonl --fallback_s $ST --frozen_p /kpct/parser/test/$k/pred_P_${k}_test_${ck}_tauoff_seed{b}.jsonl --out $O/test --workers 14" > $GR/logs/b_test_$k.log 2>&1 &
  [ -s $GR/build/$k/dev/stats.json ] || $RC b_dev_$k 4 "$B --k $k --split dev --selfcheck --raw /kpct/parser/raw/$k/raw_P_dev_sel2000_$ck.jsonl.gz --bundles /kpct/bundles/$k/dev_sel2000.jsonl --fallback_s $SD --frozen_p /kpct/parser/exec/$k/pred_P_${k}_dev_sel2000_${ck}_tauoff_seed{b}.jsonl --out $O/dev --workers 4" > $GR/logs/b_dev_$k.log 2>&1 &
  [ -s $GR/build/$k/llm/stats.json ] || $RC b_llm_$k 4 "$B --k $k --split llm --vidx 4,5,6 --raw /kpct/robustness_r3/llm/P/raw/raw_P_variants_P_$k.jsonl.gz --bundles /kpct/robustness_r3/llm/bundles/$k/variants.jsonl --fallback_s /kpct/robustness_r3/llm/preds/$k/variants_S_seed{b}.jsonl --frozen_p /kpct/robustness_r3/llm/P/$k/pred_P_${k}_variants_P_${k}_tauoff_seed{b}.jsonl --out $O/llm --workers 4" > $GR/logs/b_llm_$k.log 2>&1 &
  for R in r1 r2; do
    if [ $R = r1 ]; then RR=/kpct/robustness; else RR=/kpct/robustness_r2; fi
    [ -s $GR/build/$k/$R/stats.json ] || $RC b_${R}_$k 2 "$B --k $k --split $R --vidx 3 --raw $RR/P/raw/raw_P_variants_P_$k.jsonl.gz --bundles $RR/bundles/$k/variants.jsonl --fallback_s $RR/preds/$k/variants_S_seed{b}.jsonl --frozen_p $RR/P/$k/pred_P_${k}_variants_P_${k}_tauoff_seed{b}.jsonl --out $O/$R --workers 2" > $GR/logs/b_${R}_$k.log 2>&1 &
  done
  wait
  for s in test dev llm r1 r2; do
    [ -s $GR/build/$k/$s/stats.json ] && cp $GR/build/$k/$s/stats.json $J/stats_${k}_$s.json || { echo "build $k $s missing"; tail -30 $GR/logs/b_${s}_$k.log; }
  done
done
cp $GR/logs/*.log $J/ 2>/dev/null
ls $J/stats_*.json | wc -l
echo "jobA_gr done"
