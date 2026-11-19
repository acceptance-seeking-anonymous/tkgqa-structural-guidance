#!/bin/bash
# nm_job.sh <root: dev|test> <spent_host_b_gpu_h> <arms: comma list of G_5,P_5,P_100,G_100>
# Normalizer track (rule multitq_normalizer_rule.json): the UNCHANGED frozen pipeline on the normalized question texts
# that need a run (identity-reuse rule), plus 200 identity-control rows (original text, synthetic quid 10q+0).
#   1. code staging (refuses to overwrite changed staged code) + integrity: sha256 of the frozen adapters / code used
#      against robustness_r3 expected_sha_r3.json, and of the frozen R1 rob_build.py (exit 3 = STOP_TEST_INTEGRITY)
#   2. structural rebuild (frozen R1 rob_build.py, unchanged) for k05_s0 and k100 -> bundles, RULE, S_seed0-2; I1 of the
#      control rows vs the frozen dev / test files (RULE and S top-1 >= 0.995 required, else exit 7 = STOP_PIPELINE_DRIFT)
#   3. frozen LLM arms on the GPU host physical card 0, one container each (nm_run_card.sh), registered order and cut: G_5 -> P session (P_5 [+ P_100]) -> G_100; before each GPU item
#      spent(the second GPU host normalizer) + spent(the GPU host normalizer) + projection <= 4.5 else NOT_RUN (P_100 dropped from the session
#      first if only P_5 fits; G_100 is the first item cut, then P_100)
#   4. P execution (frozen pk_execute.py, tau off, CPU)
#   5. copies predictions / checks / logs into the job dir for harvest
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
S=$1; SPB=$2; ARMS=$3
J=$PWD; K=$WORK_DIR; NM=$K/normalizer; L=$K/gpu_usage_kpct.tsv; CAP=4.5; GM=${GPU_MEM:-0.40}; CARD=${NM_CARD:-0}
O=/kpct/normalizer/$S
mkdir -p $NM/code $NM/logs $NM/$S/inputs $NM/$S/checks
for f in nm_run_card.sh nm_run_cpu.sh nm_job.sh nm_verify.py; do
  if [ -f $NM/code/$f ] && ! cmp -s $J/$f $NM/code/$f; then echo "CODE_CHANGED $f -- refusing to overwrite staged code"; exit 2; fi
  cp -n $J/$f $NM/code/$f
done
cp -n $K/robustness_r3/code/rob_build.py $NM/code/rob_build.py
if [ -s $NM/$S/inputs/variants.jsonl ] && ! cmp -s $J/variants_$S.jsonl $NM/$S/inputs/variants.jsonl; then echo "INPUT_CHANGED variants_$S"; exit 2; fi
cp $J/variants_$S.jsonl $NM/$S/inputs/variants.jsonl
RC="bash $NM/code/nm_run_cpu.sh"; RCG="bash $NM/code/nm_run_card.sh $CARD"
$RC verify_$S 2 "python3 -u /kpct/normalizer/code/nm_verify.py /kpct/robustness_r3/r3/code/expected_sha.json,/kpct/robustness_r3/dev/code/expected_sha.json /kpct/normalizer/$S/checks/verify_frozen.json" > $NM/logs/verify_$S.log 2>&1
rc=$?; cp $NM/logs/verify_$S.log $J/; cp $NM/$S/checks/verify_frozen.json $J/${S}_verify_frozen.json 2>/dev/null
[ $rc -eq 0 ] || { echo "STOP_TEST_INTEGRITY rc=$rc"; exit 3; }
# ---------------------------------------------------------------- 2. structural rebuild
for k in k05_s0 k100; do
  [ -s $NM/$S/checks/I1_$k.json ] && continue
  if [ $S = test ]; then
    if [ $k = k100 ]; then FP=/mtq/phase1_structural/preds; else FP=/kpct/preds/$k; fi; FB=/kpct/bundles/$k/test.jsonl
  else
    FP=/kpct/robustness_r3/dev/frozen/$k; FB=/kpct/bundles/$k/dev_sel2000.jsonl
  fi
  $RC build_${S}_$k 12 "python3 -u /kpct/normalizer/code/rob_build.py --root /kpct/normalizer/$S --k $k --workers 12 --frozen_preds $FP --frozen_bundles $FB" > $NM/logs/build_${S}_$k.log 2>&1 &
done
wait
for k in k05_s0 k100; do
  [ -s $NM/$S/checks/I1_$k.json ] || { echo "build $S $k missing"; tail -20 $NM/logs/build_${S}_$k.log; exit 4; }
  cp $NM/$S/checks/I1_$k.json $J/${S}_I1_$k.json
done
python3 - $NM/$S/checks <<'EOF' || { echo "STOP_PIPELINE_DRIFT"; exit 7; }
import json, sys
bad = []
for k in ("k05_s0", "k100"):
    I = json.load(open(f"{sys.argv[1]}/I1_{k}.json"))
    for arm, v in I["arms"].items():
        print(k, arm, "n", I["n"], "top1", round(v["top1"], 4))
        if v["top1"] < 0.995:
            bad.append((k, arm, v["top1"]))
sys.exit(1 if bad else 0)
EOF
# ---------------------------------------------------------------- 3. LLM arms (GPU-hour cap checked)
mkdir -p $J/out $NM/$S/G $NM/$S/P/raw
spent() { awk -F'\t' -v b=$SPB '$1=="normalizer"{s+=$7} END{printf "%.4f", s+b}' $L; }
gpu_cap_ok() { awk -v s=$(spent) -v p=$1 -v c=$CAP 'BEGIN{exit !(s+p<=c)}'; }
notrun() { echo "NOT_RUN $1 (GPU-hour cap: spent $(spent), projected $2, cap $CAP)" | tee -a $J/out/not_run.txt; }
N=$(grep -c . $NM/$S/inputs/variants.jsonl)
PG=$(awk -v n=$N 'BEGIN{printf "%.4f", (70 + 0.075*n)/3600}')
PP1=$(awk -v n=$N 'BEGIN{printf "%.4f", (70 + 0.025*n)/3600}')
PP2=$(awk -v n=$N 'BEGIN{printf "%.4f", (70 + 0.050*n)/3600}')
echo "root $S n_rows $N projections G $PG P1 $PP1 P2 $PP2 spent $(spent)" | tee $J/out/projections.txt
declare -A AD=( [k05_s0]=/kpct/guided/adapters/G_k05_s0_ordS_s0/ep2 [k100]=/kpct/guided/adapters/G_k100_ordS_s0/ep1 )
declare -A LV=( [5]=k05_s0 [100]=k100 )
has() { echo ",$ARMS," | grep -q ",$1,"; }
run_G() { local k=$1
  [ -s $NM/$S/G/$k/pred_G_variants_G_$k.jsonl ] && return
  gpu_cap_ok $PG || { notrun G_$k $PG; return; }
  mkdir -p $NM/$S/G/$k
  $RCG ${VLLM_IMAGE:-tkgqa-vllm:latest} G_${S}_$k 3600 "python3 -u /kpct/guided/code/gk_infer.py --arm G --split variants --bundles $O/bundles/$k/variants.jsonl --adapters G_$k=${AD[$k]} --order S --s_list $O/preds/$k/variants_S_seed0.jsonl --kg /data/kg/full.txt --gpu_mem $GM --out $O/G/$k" > $NM/logs/G_${S}_$k.log 2>&1
  echo "G $k rc=$?"
}
has G_5 && run_G k05_s0
PA=""
if has P_5 && [ ! -s $NM/$S/P/raw/raw_P_variants_P_k05_s0.jsonl.gz ]; then
  if has P_100 && gpu_cap_ok $PP2; then PA="P_k05_s0=/kpct/parser/adapters/P_k05_s0_s0/ep1,P_k100=/kpct/parser/adapters/P_k100_s0/ep0.5"
  elif gpu_cap_ok $PP1; then PA="P_k05_s0=/kpct/parser/adapters/P_k05_s0_s0/ep1"; has P_100 && notrun P_100 $PP2
  else notrun P_5 $PP1; has P_100 && notrun P_100 $PP2; fi
elif has P_100 && [ ! -s $NM/$S/P/raw/raw_P_variants_P_k100.jsonl.gz ]; then
  if gpu_cap_ok $PP1; then PA="P_k100=/kpct/parser/adapters/P_k100_s0/ep0.5"; else notrun P_100 $PP1; fi
fi
if [ -n "$PA" ]; then
  $RCG ${VLLM_IMAGE:-tkgqa-vllm:latest} P_${S}_infer 3600 "python3 -u /kpct/parser/code/pk_infer.py --split variants --bundles $O/bundles/k05_s0/variants.jsonl --adapters $PA --gpu_mem $GM --out $O/P/raw" > $NM/logs/P_${S}_infer.log 2>&1
  echo "P infer rc=$?"
fi
has G_100 && run_G k100
# ---------------------------------------------------------------- 4. P execution (CPU)
for k in k05_s0 k100; do
  [ -s $NM/$S/P/raw/raw_P_variants_P_$k.jsonl.gz ] || continue
  [ -s $NM/$S/P/$k/pred_P_${k}_variants_P_${k}_tauoff_seed2.jsonl ] && continue
  mkdir -p $NM/$S/P/$k
  $RC P_exec_${S}_$k 12 "python3 -u /kpct/parser/code/pk_execute.py --k $k --split variants --raw P_$k=$O/P/raw/raw_P_variants_P_$k.jsonl.gz --bundles $O/bundles/$k/variants.jsonl --preds $O/preds --taus=off --out $O/P/$k --workers 12" > $NM/logs/P_exec_${S}_$k.log 2>&1
  echo "P exec $k rc=$?"
done
# ---------------------------------------------------------------- 5. harvest copies
cd $NM/$S && tar czf $J/out/nm_${S}_predictions.tgz inputs preds G P checks 2>/dev/null; cd $J
cp $NM/logs/*_${S}*.log $J/out/ 2>/dev/null
head -1 $L > $J/out/usage_log_normalizer.tsv; grep -P "^normalizer\t" $L >> $J/out/usage_log_normalizer.tsv
echo "spent_total $(spent)"
echo "nm_job162 $S done"
