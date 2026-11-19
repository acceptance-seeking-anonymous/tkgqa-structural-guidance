#!/bin/bash
# reproduce_robustness.sh <r1 | r2> -- wording-robustness probes on frozen arms (no retraining).
#  R1: rob_generate_r1.py (2,000-question sample; P1 prefix, P2 no '?', P3 API-LLM rewrites of a 600-question pool with
#      names / times protected, mechanical checks, LLM-judge pre-filter, <= 2 attempts) -> remote jobA-D (rebuild
#      bundles, frozen G / GU / P / Rules / Ranker on every variant, freeze) -> rob_score.py (single scoring pass;
#      paired Poisson bootstrap B = 2000, seed 0; Holm over the six reader / parser contrasts).
#  R2: disjoint 2,000-question sample; Qwen2.5-72B-Instruct-AWQ rewrites (remote/r2_gen.py) with the R1 checks and
#      judge prompt (local/r2_build_perturbed_set.py) -> remote jobA2 / jobBC2 / jobD2 -> kit/score_R2.py (B = 10,000,
#      seed 20261029, one-sided Holm over H1-H3).
#  Single-annotator human review (random 100 + targeted rows): r1/annotation/, r2/kit/ (kit builders, BOUND scorers).
# Generator / judge endpoint and model IDs: configs/llm.env.example (robustness/llm_client.py).
set -euo pipefail
source "$(dirname "$0")/env_check.sh" DATA_DIR OUT_DIR
P=${1:?r1 or r2}; O=$OUT_DIR/robustness_$P; mkdir -p "$O"; R=$REPO/robustness
if [ "$P" = r1 ]; then
  python3 "$R/r1/local/rob_generate_r1.py" --test-json "$DATA_DIR/questions/test.json" --rule "$REPO/protocol/multitq_robustness_rule.json" \
    --out "$O/multitq_robustness_perturbed_set.json"
  echo "next: write variants.jsonl / variants_questions.json from the perturbed set into a job dir with r1/remote/*, run jobA-D on the GPU host, then rob_score.py"
else
  python3 "$R/r2/local/r2_build_perturbed_set.py" prompts --test-json "$DATA_DIR/questions/test.json" --r1-rule "$REPO/protocol/multitq_robustness_rule.json" --outdir "$O"
  echo "next: remote/gen_job.sh attempt1 -> r2_build_perturbed_set.py judge1 -> gen_job.sh attempt2 -> r2_build_perturbed_set.py build -> jobA2 / jobBC2 / jobD2 -> kit/score_R2.py"
fi
