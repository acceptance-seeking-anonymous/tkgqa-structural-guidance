#!/bin/bash
# jobA3.sh -- rule R3 stage A (CPU only). Two roots under $WORK_DIR/robustness_r3:
#   r3/  : R3 sample ORIG (2,000) + judge-accepted R3 rewrites (vidx 4)        frozen references = official TEST files
#   dev/ : dev_sel2000 ORIG (2,000) + judge-accepted dev paraphrases (vidx 5)   frozen references = DEV selection files
# For each root: verify frozen files by sha256 and extract the frozen ORIG rows (rob_collect_frozen.py, R1 code
# unchanged; exit 3 = STOP_TEST_INTEGRITY), rebuild bundles / RULE / S for k01_s0, k05_s0, k100 on every variant
# (rob_build.py, R1 code unchanged; I1 identity of the ORIG rows vs the frozen files), template keys (rob_templates.py).
# Then r3_combine.py builds the LLM-arm root llm/ from the rewrite rows of both roots.
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
J=$PWD; K=$WORK_DIR; R3=$K/robustness_r3
mkdir -p $R3/code $R3/logs $R3/checks
for f in rob_build.py rob_templates.py rob_collect_frozen.py r3_combine.py run_cpu_r3.sh run_card_r3.sh jobBC3.sh jobD3.sh; do
  if [ -f $R3/code/$f ] && ! cmp -s $J/$f $R3/code/$f; then echo "CODE_CHANGED $f -- refusing to overwrite staged code"; exit 2; fi
  cp -n $J/$f $R3/code/$f
done
for S in r3 dev; do
  mkdir -p $R3/$S/code $R3/$S/inputs $R3/$S/checks
  cp $R3/code/rob_build.py $R3/code/rob_templates.py $R3/code/rob_collect_frozen.py $R3/$S/code/
  cp $J/expected_sha_$S.json $R3/$S/code/expected_sha.json
  cp $J/variants_$S.jsonl $R3/$S/inputs/variants.jsonl
  python3 -c "import json; r=[json.loads(l) for l in open('$J/variants_$S.jsonl')]; json.dump([{'quid':x['quid'],'question':x['question']} for x in r], open('$R3/$S/inputs/variants_questions.json','w'), ensure_ascii=False)"
done
(cd $R3 && sha256sum code/* r3/code/* r3/inputs/* dev/code/* dev/inputs/*) > $J/code_inputs_sha_A.txt; cp $J/code_inputs_sha_A.txt $R3/checks/
RC="bash $R3/code/run_cpu_r3.sh"
for S in r3 dev; do
  $RC collect_$S 4 "python3 -u /kpct/robustness_r3/$S/code/rob_collect_frozen.py /kpct/robustness_r3/$S" > $R3/logs/collect_$S.log 2>&1
  rc=$?; cp $R3/logs/collect_$S.log $J/
  if [ $rc -ne 0 ]; then echo "collect $S rc=$rc (STOP_TEST_INTEGRITY?)"; cp $R3/$S/checks/verify_frozen.json $J/verify_frozen_$S.json 2>/dev/null; exit $rc; fi
done
# dev frozen references: rob_build reads <frozen_preds>/test_{RULE,S_seed0,S_seed1,S_seed2}.jsonl -> symlink dir to the dev files
$RC devlinks 1 "set -e; for k in k01_s0 k05_s0; do d=/kpct/robustness_r3/dev/frozen/\$k; mkdir -p \$d; for a in RULE S_seed0 S_seed1 S_seed2; do ln -sfn /kpct/preds/\$k/dev_sel2000_\$a.jsonl \$d/test_\$a.jsonl; done; done; \
  d=/kpct/robustness_r3/dev/frozen/k100; mkdir -p \$d; for a in RULE S_seed0 S_seed1 S_seed2; do ln -sfn /mtq/phase1_structural/preds/dev_\$a.jsonl \$d/test_\$a.jsonl; done; ls -l /kpct/robustness_r3/dev/frozen/*" > $R3/logs/devlinks.log 2>&1 || { echo "devlinks failed"; cat $R3/logs/devlinks.log; exit 6; }
for S in r3 dev; do
  for k in k01_s0 k05_s0 k100; do
    [ -s $R3/$S/checks/I1_$k.json ] && continue
    if [ $S = r3 ]; then
      if [ $k = k100 ]; then FP=/mtq/phase1_structural/preds; else FP=/kpct/preds/$k; fi; FB=/kpct/bundles/$k/test.jsonl
    else
      FP=/kpct/robustness_r3/dev/frozen/$k; FB=/kpct/bundles/$k/dev_sel2000.jsonl
    fi
    $RC build_${S}_$k 6 "python3 -u /kpct/robustness_r3/$S/code/rob_build.py --root /kpct/robustness_r3/$S --k $k --workers 6 --frozen_preds $FP --frozen_bundles $FB" > $R3/logs/build_${S}_$k.log 2>&1 &
  done
done
wait
for S in r3 dev; do
  [ -s $R3/$S/checks/template_flags.json ] || $RC templates_$S 12 "python3 -u /kpct/robustness_r3/$S/code/rob_templates.py /kpct/robustness_r3/$S 12" > $R3/logs/templates_$S.log 2>&1
  for k in k01_s0 k05_s0 k100; do [ -s $R3/$S/checks/I1_$k.json ] || { echo "build $S $k missing"; tail -20 $R3/logs/build_${S}_$k.log; exit 4; }; done
  for f in $R3/$S/checks/*.json; do cp $f $J/${S}_$(basename $f); done
done
$RC combine 2 "python3 -u /kpct/robustness_r3/code/r3_combine.py --roots /kpct/robustness_r3/r3,/kpct/robustness_r3/dev --out /kpct/robustness_r3/llm" > $R3/logs/combine.log 2>&1 || { echo "combine failed"; cat $R3/logs/combine.log; exit 5; }
cp $R3/llm/checks/combine.json $J/; cp $R3/logs/*.log $J/ 2>/dev/null
echo "jobA3 done"
