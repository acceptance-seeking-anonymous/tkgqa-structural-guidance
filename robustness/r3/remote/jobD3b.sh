#!/bin/bash
# jobD3b.sh (= jobD3.sh plus the devj/ root of amendment A2) -- rule R3 freeze: sha256 of every probe output file (LLM-arm root llm/, structural roots r3/ and dev/, and
# the frozen ORIG rows) + predictions tarball. No scoring, no gold.
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
J=$PWD; K=$WORK_DIR; R3=$K/robustness_r3
cd $R3
find llm/G llm/GU llm/P llm/gu llm/GPLAIN llm/preds llm/bundles llm/inputs llm/checks r3/preds r3/bundles r3/checks r3/inputs dev/preds dev/bundles dev/checks dev/inputs devj/preds devj/bundles devj/checks devj/inputs \
  -type f \( -name '*.jsonl' -o -name '*.jsonl.gz' -o -name '*.json' \) 2>/dev/null | sort > $J/freeze_files.txt
xargs -a $J/freeze_files.txt sha256sum > $J/freeze_sha256.txt
cp $J/freeze_sha256.txt $R3/checks/freeze_sha256.txt
tar czf $J/multitq_robustness_R3_predictions.tgz llm/G llm/GU llm/P llm/gu llm/GPLAIN llm/preds llm/checks r3/preds r3/checks dev/preds dev/checks devj/preds devj/checks checks/freeze_sha256.txt
tar czf $R3/multitq_robustness_R3_bundles.tgz llm/bundles r3/bundles dev/bundles devj/bundles
sha256sum $J/multitq_robustness_R3_predictions.tgz $R3/multitq_robustness_R3_bundles.tgz > $J/tarball_sha256.txt
head -1 $K/gpu_usage_kpct.tsv > $J/usage_log_robustness_r3.tsv; grep -P "^robustness_r3\t" $K/gpu_usage_kpct.tsv >> $J/usage_log_robustness_r3.tsv
echo "jobD3 done"
