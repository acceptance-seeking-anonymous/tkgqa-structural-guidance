#!/bin/bash
# jobA.sh -- robustness probe stage A (CPU only): stage code + inputs, verify frozen files (STOP_TEST_INTEGRITY),
# extract frozen ORIG rows, rebuild bundles / RULE / S lists for k01_s0, k05_s0, k100 on every variant, template keys.
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
: "${MTQ_DIR:?set MTQ_DIR (see configs/paths.env.example)}"
J=$PWD; K=$WORK_DIR; R=$K/robustness
mkdir -p $R/code $R/inputs $R/logs $R/checks
# expected_sha.json (frozen-file integrity list, STOP_TEST_INTEGRITY) is written by scripts/freeze_integrity_refs.sh
# right after the test pass ($WORK_DIR/integrity/expected_sha_R1.json); a self-made list only on explicit opt-in.
if [ ! -f $J/expected_sha.json ]; then
  if [ -f "$WORK_DIR/integrity/expected_sha_R1.json" ]; then cp "$WORK_DIR/integrity/expected_sha_R1.json" $J/expected_sha.json
  elif [ "${ALLOW_SELF_REFERENCE:-0}" = 1 ]; then
    echo "WARNING: expected_sha.json built from the current files (ALLOW_SELF_REFERENCE=1): the integrity stop is then vacuous" >&2
    python3 "${REPO:?set REPO to the repository root}/robustness/fill_expected_sha.py" "$REPO/robustness/r1/remote/expected_sha.template.json" $J/expected_sha.json --kpct $WORK_DIR --mtq $MTQ_DIR --strict
  else
    echo "ERROR: no integrity list. Run scripts/freeze_integrity_refs.sh after the test pass (or set ALLOW_SELF_REFERENCE=1)." >&2; exit 1
  fi
fi
cp $J/rob_build.py $J/rob_templates.py $J/rob_collect_frozen.py $J/expected_sha.json $J/run_card_r.sh $J/run_cpu_r.sh $J/jobB.sh $J/jobC.sh $J/jobD.sh $R/code/
cp $J/variants.jsonl $J/variants_questions.json $R/inputs/
(cd $R && sha256sum code/* inputs/*) > $J/code_inputs_sha.txt; cp $J/code_inputs_sha.txt $R/checks/
uptime > $J/host_state.txt; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >> $J/host_state.txt; docker ps --format '{{.Names}} {{.Image}} {{.RunningFor}}' >> $J/host_state.txt
bash $R/code/run_cpu_r.sh collect 4 "python3 -u /kpct/robustness/code/rob_collect_frozen.py /kpct/robustness" > $R/logs/collect.log 2>&1
rc=$?; cp $R/logs/collect.log $J/
if [ $rc -ne 0 ]; then echo "collect rc=$rc (STOP_TEST_INTEGRITY?)"; cp $R/checks/verify_frozen.json $J/ 2>/dev/null; exit $rc; fi
for k in k01_s0 k05_s0 k100; do
  if [ -s $R/checks/I1_$k.json ]; then continue; fi
  if [ $k = k100 ]; then FP=/mtq/phase1_structural/preds; else FP=/kpct/preds/$k; fi
  bash $R/code/run_cpu_r.sh build_$k 6 "python3 -u /kpct/robustness/code/rob_build.py --k $k --workers 6 --frozen_preds $FP --frozen_bundles /kpct/bundles/$k/test.jsonl" > $R/logs/build_$k.log 2>&1 &
done
wait
[ -s $R/checks/template_flags.json ] || bash $R/code/run_cpu_r.sh templates 12 "python3 -u /kpct/robustness/code/rob_templates.py /kpct/robustness 12" > $R/logs/templates.log 2>&1
cp $R/checks/I1_*.json $R/checks/template_check.json $R/checks/template_flags.json $R/checks/verify_frozen.json $R/checks/frozen_orig_rows.json $R/checks/frozen_bundle_fields_*.json $J/ 2>/dev/null
cp $R/logs/*.log $J/ 2>/dev/null
ls -la $R/bundles/* $R/preds/*
for k in k01_s0 k05_s0 k100; do [ -s $R/checks/I1_$k.json ] || { echo "build $k missing"; exit 4; }; done
[ -s $R/checks/template_flags.json ] || { echo "templates missing"; exit 5; }
echo "jobA done"
