#!/bin/bash
# jobA3b.sh -- rule R3 amendment A2 (CPU only): root devj/ = dev_sel2000 ORIG + judge-REJECTED (mechanically valid) dev
# paraphrases (vidx 6) of 192 questions, so that the registered descriptive dev analysis on DEV_MECH_OK can be computed;
# same collect / rebuild / template steps as jobA3.sh, then r3_combine.py over r3/, dev/, devj/ with vidx 4, 5, 6.
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
J=$PWD; K=$WORK_DIR; R3=$K/robustness_r3; S=devj
mkdir -p $R3/$S/code $R3/$S/inputs $R3/$S/checks
cp $R3/code/rob_build.py $R3/code/rob_templates.py $R3/code/rob_collect_frozen.py $R3/$S/code/
cp $J/expected_sha_dev.json $R3/$S/code/expected_sha.json
cp $J/variants_devjrej.jsonl $R3/$S/inputs/variants.jsonl
python3 -c "import json; r=[json.loads(l) for l in open('$J/variants_devjrej.jsonl')]; json.dump([{'quid':x['quid'],'question':x['question']} for x in r], open('$R3/$S/inputs/variants_questions.json','w'), ensure_ascii=False)"
(cd $R3 && sha256sum $S/code/* $S/inputs/* code/r3_combine.py) > $J/code_inputs_sha_A3b.txt
RC="bash $R3/code/run_cpu_r3.sh"
$RC collect_$S 4 "python3 -u /kpct/robustness_r3/$S/code/rob_collect_frozen.py /kpct/robustness_r3/$S" > $R3/logs/collect_$S.log 2>&1 || { echo "collect $S failed"; exit 3; }
for k in k01_s0 k05_s0 k100; do
  [ -s $R3/$S/checks/I1_$k.json ] && continue
  $RC build_${S}_$k 6 "python3 -u /kpct/robustness_r3/$S/code/rob_build.py --root /kpct/robustness_r3/$S --k $k --workers 6 --frozen_preds /kpct/robustness_r3/dev/frozen/$k --frozen_bundles /kpct/bundles/$k/$( [ $k = k100 ] && echo dev_sel2000 || echo dev_sel2000 ).jsonl" > $R3/logs/build_${S}_$k.log 2>&1 &
done
wait
[ -s $R3/$S/checks/template_flags.json ] || $RC templates_$S 12 "python3 -u /kpct/robustness_r3/$S/code/rob_templates.py /kpct/robustness_r3/$S 12" > $R3/logs/templates_$S.log 2>&1
for k in k01_s0 k05_s0 k100; do [ -s $R3/$S/checks/I1_$k.json ] || { echo "build $S $k missing"; tail -20 $R3/logs/build_${S}_$k.log; exit 4; }; done
for f in $R3/$S/checks/*.json; do cp $f $J/${S}_$(basename $f); done
$RC combine 2 "python3 -u /kpct/robustness_r3/code/r3_combine.py --roots /kpct/robustness_r3/r3,/kpct/robustness_r3/dev,/kpct/robustness_r3/devj --vidx 4,5,6 --out /kpct/robustness_r3/llm" > $R3/logs/combine.log 2>&1 || { echo "combine failed"; cat $R3/logs/combine.log; exit 5; }
cp $R3/llm/checks/combine.json $J/; cp $R3/logs/*devj*.log $R3/logs/combine.log $J/ 2>/dev/null
echo "jobA3b done"
