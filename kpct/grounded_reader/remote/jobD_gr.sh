#!/bin/bash
# jobD_gr.sh -- grounded_reader single registered scoring pass (CPU; test gold read here for the first time for GR).
# freeze.json (written by local/gr_freeze.py before this job) is staged in the job cwd; gr_score.py verifies every listed sha256.
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
J=$PWD; K=$WORK_DIR; GR=$K/grounded_reader; RC="bash $GR/code/run_cpu_gr.sh"
mkdir -p $GR/score; cp $J/freeze.json $GR/score/freeze.json
$RC score 4 "python3 -u /kpct/grounded_reader/code/gr_score.py --freeze /kpct/grounded_reader/score/freeze.json --out /kpct/grounded_reader/score/gr" > $GR/logs/score.log 2>&1
echo "score rc=$?"; cp $GR/logs/score.log $GR/score/gr_* $J/
