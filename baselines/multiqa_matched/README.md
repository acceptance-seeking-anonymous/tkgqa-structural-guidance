# Matched MultiQA baseline

MultiQA (Chen, Liao, Zhao, ACL 2023), the supervised baseline released with MultiTQ, trained on exactly the registered
nested 1 / 5 / 10 % train-question subsets used by the Rules, Ranker, Reader and Parser arms (and on the full train
split), selected on dev_sel2000 only and scored once on the official test split. Registered rule:
`protocol/multitq_matched_baseline_rule.json` (descriptive comparator; no pass/fail margin).

## Third-party code (not redistributed)

`fetch_multiqa.sh <job dir>` clones https://github.com/czy1999/MultiTQ, checks out commit
`94ad61c31235467cc077a99e3a5ee105ac1dd04b`, verifies the sha256 of the upstream files that are used, and lays out the job
directory exactly as the upstream repository (`MultiQA/`, `data/MultiTQ/kg/tkbc_processed_data/`, the shipped TComplEx
checkpoint from `MultiQA/models/Models.zip`). No upstream file is patched:

| upstream file | use |
|---|---|
| `MultiQA/qa_baselines.py`, `qa_datasets.py`, `utils.py`, `tcomplex.py` | imported unchanged by `mbq_qa.py` (model, datasets, official `eval_multi`) |
| `MultiQA/ner_task.py` | per-question logic ported into `mbq_ner.py` (flair tagger, fuzzy KG-name matching, time extraction) |
| `MultiQA/tkbc/` | imported by `mbq_tcx_full.py` (TComplEx, N3 / Lambda3 regularisers, TKBCOptimizer) |

## Our code

| file | role |
|---|---|
| `mbq_make_bundle.py` | question inputs of the job: all train questions, the dev_sel2000 questions, answer-stripped test questions, subset quid lists |
| `mbq_ner.py` | NER preprocessing (official logic; flair 0.15.1 instead of 0.11.3, which does not install on Python 3.12) |
| `mbq_tcx_full.py` | graph-access-matched variant: TComplEx retrained on all `kg/full.txt` facts (the shipped checkpoint saw the train split of the KG only) |
| `mbq_qa.py` | training driver around the official modules: registered train subset, dev_sel2000 validation, seeded runs, dev early stopping, top-10 predictions for dev_sel2000 and the answer-stripped test file; never computes a test score |
| `mbq_sched.py`, `plans/` | runs a plan (levels, embedding off/full, seeds, epochs, patience) with up to P concurrent processes on one GPU; `plan_C.json` (1%, 5%) and `plan_DE.json` (10% + 100%, the union of the registered `plan_D.json` / `plan_E.json`) were run; `plan_smoke.json` = dev-only feasibility |
| `jobA.sh`, `jobB.sh`, `jobQA.sh` | GPU-host wrappers (NER + TComplEx; optional dev-only smoke; training per plan) |
| `mbq_select_freeze.py` | dev_sel2000 selection of the embedding variant per level (ties -> shipped embedding), sha256 freeze of every test prediction file |
| `mbq_score.py` | single test pass: official Hits@1 / Hits@10 (src/multitq_eval.py), Poisson bootstrap CIs, paired contrasts with the k% arms |

Wrapper: `scripts/reproduce_matched_baseline.sh prepare | ner_tcx | smoke | train | freeze | score`.
`mbq_ner.py`, `mbq_qa.py`, `mbq_tcx_full.py`, `mbq_sched.py` and the plans are byte-identical to the files whose sha256 is
recorded in the rule; the GPU wrappers were rewritten for this release (paths and images from `configs/paths.env`).
GPU training is not bitwise deterministic; compare Hits@1 rather than file hashes.
