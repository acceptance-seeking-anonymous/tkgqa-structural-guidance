#!/bin/bash
# jobD.sh -- freeze: sha256 of every probe output file (predictions, raw generations, P exec records, GU retrievals,
# rebuilt bundles), packed predictions tarball for harvest. No scoring, no gold.
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
J=$PWD; K=$WORK_DIR; R=$K/robustness
cd $R
find preds G GU P gu bundles -type f \( -name '*.jsonl' -o -name '*.jsonl.gz' -o -name '*.json' \) | sort > $J/freeze_files.txt
xargs -a $J/freeze_files.txt sha256sum > $J/freeze_sha256.txt
cp $J/freeze_sha256.txt $R/checks/freeze_sha256.txt
tar czf $J/multitq_robustness_predictions.tgz preds G GU P gu checks/freeze_sha256.txt
tar czf $R/multitq_robustness_bundles.tgz bundles
sha256sum $J/multitq_robustness_predictions.tgz $R/multitq_robustness_bundles.tgz > $J/tarball_sha256.txt
ls -la $J/multitq_robustness_predictions.tgz $R/multitq_robustness_bundles.tgz
grep robustness $K/gpu_usage_kpct.tsv > $J/usage_log_robustness.tsv
echo "jobD done"
