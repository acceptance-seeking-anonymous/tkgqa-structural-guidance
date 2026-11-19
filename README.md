# Structural guidance and LLM interfaces for temporal KGQA (MultiTQ, CronQuestions)

Code accompanying an anonymous submission. It covers the LLM-free structural pipeline (temporal parser + deterministic
temporal operators = **Rules**; LightGBM LambdaMART **Ranker** over 34 structural features), the nested training-fraction
(k%) protocol, the three LLM interfaces fitted on the same lists (**Reader**, **Parser**, **Dense-RAG** control), the
training-free k = 0 arms, the single test pass with paired bootstrap / Holm contrasts, the follow-up analyses, the two
wording-robustness probes with their single-annotator audit kits, the CronQuestions structural comparison, and the
research-track additions: a matched supervised baseline (MultiQA on the same subsets), structural ablations, LLM
controls, descriptive output analyses, and the rewording-robustness method tracks (a third rewrite probe R3,
Parser-first gating, a Parser-grounded Reader and a question normalizer).

**Code only.** No dataset, processed output, prediction, model weight / adapter, log or annotation sheet is
distributed. Every script reads its inputs from paths given by environment variables (`configs/paths.env.example`);
nothing outside the repository is referenced otherwise.

## Layout

```
README.md  LICENSE
environment/   environment.yml, requirements-*.txt, Dockerfile (training / CPU-executor image), Dockerfile.vllm
configs/       paths.env.example, llm.env.example, frozen_hyperparameters.json, prompts/, test_pass_analyze_config.template.json
src/           multitq_eval.py (official MultiTQ scorer port) and struct/ (full-data structural pipeline modules)
tests/         evaluator unit tests, research-track and rewording-robustness unit tests, data-free toy end-to-end test
kpct/          subsets/, structural/, reader/, parser/, k0/, test_pass/, followups/{g1_ablation,clean_silver_p1,seeds_G,seeds_P},
               llm_controls/, parser_first/, grounded_reader/, normalizer/
baselines/     multiqa_matched/ (matched MultiQA baseline: our drivers; the upstream code is fetched, not redistributed)
robustness/    llm_client.py, fill_expected_sha.py, r1/{local,remote,annotation}, r2/{local,remote,kit}, r3/{local,remote}
analysis/      type_breakdown.py, reader_mechanism.py, rewording_contrasts.py, tipping_point.py,
               structural_ablation/, rewrite_tables/, coverage/
cronquestions/ CronQuestions structural pipeline and evaluator
protocol/      anonymised copies of the registered decision rules and amendments
scripts/       reproduce_*.sh (stage wrappers), freeze_integrity_refs.sh, smoke_test.sh, export_frozen_config.py, env_check.sh
```

## Names used in the code

| Paper | Code | Notes |
|---|---|---|
| Rules | `RULE_k` | structural parser + operators, deterministic ordering (`src/struct/mtq_cand.py`) |
| Ranker | `S_k` (`S_seed0-2`) | LightGBM lambdarank, 34 features (`mtq_cand.FEATS`); hits averaged over 3 ranker seeds |
| Reader | `G_k` | guided LoRA answerer on the guidance bundle (`kpct/reader/gk_*.py`) |
| Dense-RAG | `GU_k` | BAAI/bge-large-en-v1.5 top-20 facts + LoRA (`gu_retrieve.py`, `gk_*.py --arm GU`) |
| Parser | `P_k` | LoRA JSON parser + symbolic execution and Ranker re-ranking (`kpct/parser/pk_*.py`) |
| k = 0 arms | `P_0`, `G_0` | Llama-3.1-70B-Instruct, few-shot parser / zero-shot reader (`kpct/k0/`) |
| 1% / 5% / 10% / Full | `k01_s0` / `k05_s0` / `k10_s0` / `k100` | extra 1% draws `k01_s1`, `k01_s2` |
| selection set | `dev_sel2000` | 2,000 dev questions used for all selection |
| unseen templates | `U_k` | test questions whose masked template key is absent from the structural training list |
| R1 / R2 | `robustness/r1`, `robustness/r2` | rewrite probes; R1 variant `P3` = rewrite, R2 variant `RW` |
| matched MultiQA baseline | `MBQ_<k>` (`MBQ_OFF_*`, `MBQ_FULL_*`) | shipped vs full-KG TComplEx embedding; dev-selected per level |
| Parser-alone ablation | `PK`, `PALONE`, `PREL`, `PENT` | frozen P_k; Parser hypothesis only; + relation / + entity alternatives only |
| plain-evidence Reader | `GPLAIN_1` | Reader_1 inputs rendered as plain alphabetical lists (`kpct/llm_controls/gk_plain.py`) |
| R3 | `robustness/r3` | third rewrite probe (gemini-2.5-pro rewrites, synthetic quid 10q+4); dev paraphrases `DEVPARA` (10q+5) |
| Parser-first variants | `V0`-`V3(t)` | gated choice between the `PALONE` and `PK` lists of the cached Parser generations (`kpct/parser_first/pf_lib.py`) |
| Parser-grounded Reader | `GR_k` (`GR-A_k`, `GR-K_k`) | frozen G_k on bundles built from the PALONE / PK parse (`kpct/grounded_reader/`) |
| question normalizer | `N` (`NR`, `NS`, `NP`, `NG`) | LoRA normalizer in front of the unchanged RULE / S / P / G arms (`kpct/normalizer/`) |

## Setup

1. Environments: `conda env create -f environment/environment.yml` (CPU stages); build the two GPU images
   `docker build -f environment/Dockerfile -t tkgqa-gpu:latest environment/` and
   `docker build -f environment/Dockerfile.vllm -t tkgqa-vllm:latest environment/`. peft 0.21.2 is installed inside the
   training container at run time (`--no-deps`), as in the experiments.
2. Data: MultiTQ (github.com/czy1999/MultiTQ, commit 94ad61c; unzip `data/Dataset.zip`, set `DATA_DIR` to the folder with
   `kg/` and `questions/`) and, for `cronquestions/`, the official CronKGQA data release (`CRON_DATA`). For
   `baselines/multiqa_matched/`, `fetch_multiqa.sh` clones the same commit and uses its MultiQA code, KG files and
   TComplEx checkpoint unchanged (third-party code is not redistributed here).
3. Models (Hugging Face cache in `MODEL_DIR`, mounted read-only and offline): meta-llama/Llama-3.1-8B-Instruct,
   meta-llama/Llama-3.1-70B-Instruct, BAAI/bge-large-en-v1.5, Qwen/Qwen2.5-72B-Instruct-AWQ (`robustness/r2/remote/dl_qwen.sh`).
4. Paths: `cp configs/paths.env.example configs/paths.env`, edit, `source configs/paths.env`.
5. API LLM roles (R1 generator, R1/R2 judge): `configs/llm.env.example` and `robustness/llm_client.py` (any endpoint with
   the `/v1/chat/completions` interface). Set `GEN_MODEL_ID` / `JUDGE_MODEL_ID` to the model IDs reported in the paper.
   The R3 generator (Google Gemini API, gemini-2.5-pro) reads its key from `GEMINI_API_KEY` (same file); the R3 judge
   is `JUDGE_MODEL_ID`.

Container mount convention used by all GPU-host scripts: `/kpct` = `$WORK_DIR`, `/data` = `$DATA_DIR` (read-only),
`/hf` = `$MODEL_DIR`, `/mtq` = `$MTQ_DIR`, `/w` = the job directory.

## Paper item -> code

| Paper item | Code | Wrapper |
|---|---|---|
| Full-data structural pipeline (linker, classifier, candidate generation, operators, Rules, Ranker features) | `src/struct/` (`pipeline_steps.py` lists the executed order) | -- |
| Nested k% lists, dev_sel2000, OOF splits, template keys | `kpct/subsets/` | `scripts/reproduce_kpct_1_subsets.sh` |
| Rules / Ranker retrained per k, guidance bundles, executor | `kpct/structural/` | `scripts/reproduce_kpct_2_structural.sh` |
| GPU-host staging | -- | `scripts/reproduce_kpct_0_setup.sh` |
| Dense-RAG retrieval (bge-large, top-20) | `kpct/reader/gu_retrieve.py`, `a2_driver.sh` | `scripts/reproduce_kpct_4_dense_rag_retrieval.sh` |
| Reader G_k, Dense-RAG GU_k, SHUF/QONLY controls, selection, freeze | `kpct/reader/` | `scripts/reproduce_kpct_5_reader.sh` |
| Parser P_k (silver targets, LoRA, execution, tau grid, selection, freeze) | `kpct/parser/` | `scripts/reproduce_kpct_6_parser.sh` |
| k = 0 70B arms P_0, G_0 | `kpct/k0/`, `kpct/parser/p0_*` | `scripts/reproduce_kpct_7_k0.sh` |
| Test pass: hash check, official scoring, paired Poisson bootstrap (B = 2000, seed 0), KP1-KP4, Holm over KP2-KP4 | `kpct/test_pass/` | `scripts/reproduce_kpct_8_test_pass.sh` |
| Per-question-type breakdown | `analysis/type_breakdown.py` | `scripts/reproduce_analysis.sh` |
| Reader_1 mechanism (copy / fix / loss vs Ranker seed 0) | `analysis/reader_mechanism.py`; `kpct/followups/g1_ablation/abl_part4.py` (gold coverage of shown facts) | `scripts/reproduce_analysis.sh` |
| G_1 input-intervention ablation (dev) | `kpct/followups/g1_ablation/` | `scripts/reproduce_followups.sh g1_ablation` |
| Clean-silver P_1 retrain (dev) | `kpct/followups/clean_silver_p1/` (+ `kpct/parser/pk_train.py --order_seed`) | `scripts/reproduce_followups.sh clean_silver_p1` |
| LoRA-seed replications G_1, P_1, P_5 | `kpct/followups/seeds_G/`, `kpct/followups/seeds_P/` (`host_b` = second GPU host) | `scripts/reproduce_followups.sh seeds_*` |
| R1 probe: rewrites with protected names / times, mechanical checks, LLM-judge pre-filter, frozen arms, scoring | `robustness/r1/local/{rob_common,rob_generate_r1,rob_score}.py`, `robustness/r1/remote/` | `scripts/reproduce_robustness.sh r1` |
| R2 probe: disjoint sample, Qwen2.5-72B-AWQ rewrites, H1-H3 (B = 10,000, seed 20261029, one-sided Holm) | `robustness/r2/` | `scripts/reproduce_robustness.sh r2` |
| Single-annotator targeted + random audit kits, BOUND scorers, re-scoring after review | `robustness/r1/annotation/`, `robustness/r2/kit/` | -- |
| Post-hoc parser-vs-reader rewording contrasts; R1 tipping point | `analysis/rewording_contrasts.py`, `analysis/tipping_point.py` | `scripts/reproduce_analysis.sh` |
| CronQuestions structural comparison | `cronquestions/` | `scripts/reproduce_cronquestions.sh` |
| Matched supervised baseline: MultiQA trained on the same nested k% subsets (shipped vs full-KG TComplEx, dev_sel2000 selection, single test pass) | `baselines/multiqa_matched/` | `scripts/reproduce_matched_baseline.sh` |
| Structural ablation (a): Parser alone vs Parser plus structural alternatives (CPU replay of the cached Parser generations; test, R1, R2) | `analysis/structural_ablation/sa_parser_replay.py`, `sa_score_A.py` | `scripts/reproduce_structural_ablation.sh A` |
| Structural ablation (b): Ranker feature-group retrain ablation (six groups, two composites, hash-seed noise floor) | `analysis/structural_ablation/sb_ablate.py`, `sa_score_B.py`; `kpct/structural/kpct_retrain_fixed_vocab.py` | `scripts/reproduce_structural_ablation.sh B` |
| LLM controls: plain-evidence Reader GPLAIN_1, Dense-RAG at 5% (GU_5); GPU training / selection / test and R1, R2 passes, then one CPU scoring pass (`lc_score.py` -> `lc_analyze.py`); rules `protocol/multitq_kpct_amendment_LC1*.json` | `kpct/llm_controls/` | `scripts/reproduce_llm_controls.sh`, `... score` |
| Absolute rewrite accuracy on R1 / R2 (A1) | `analysis/rewrite_tables/a1_rewrite.py` | `scripts/reproduce_output_analyses.sh A1` |
| Coverage-conditioned outcomes (A2): gold in displayed facts / candidates; Dense-RAG retrieved-fact and Parser top-20 coverage | `analysis/coverage/` | `scripts/reproduce_output_analyses.sh A2`, `A2_followup` |
| Rewording probe R3: 2,000 test questions disjoint from R1 / R2, 600-question rewrite pool (Gemini rewrites, mechanical checks + LLM judge), dev / training paraphrases (Qwen2.5-72B-AWQ), frozen arms on the rewrite rows, R3 secondary analysis, audit selection and BOUND check | `robustness/r3/` | `scripts/reproduce_r3.sh` |
| R3 single-annotator audit scoring (random-100 and targeted acceptance, BOUND for Parser-first, normalizer and Parser-grounded Reader) | `robustness/r3/r3_audit_score.py` | `python robustness/r3/r3_audit_score.py --help` |
| Parser-first gating (V0-V3 on the cached Parser generations; dev selection; test, R3, R1, R2) | `kpct/parser_first/` | `scripts/reproduce_parser_first.sh` |
| Parser-grounded Reader GR_k (frozen G_k on Parser-grounded bundles; dev selection; test, R3, R1, R2) | `kpct/grounded_reader/` | `scripts/reproduce_grounded_reader.sh` |
| Question normalizer N (LoRA rewrite into template wording before the unchanged arms; dev selection; test, R3, R1, R2) | `kpct/normalizer/` | `scripts/reproduce_normalizer.sh` |
| Registered rules and amendments | `protocol/` | -- |

Frozen hyperparameters (Ranker grid, LoRA recipe, decoding limits, tau grid, probe sizes and seeds) are defined in the
code; `configs/frozen_hyperparameters.json` and `configs/prompts/` are exports produced by `scripts/export_frozen_config.py`.
The P_0 prompt designs and exemplars are in `kpct/parser/p0_prompts/` and `kpct/parser/p0_exemplars.json`; the G_0
designs in `kpct/k0/g0_prompts.py`.

## Checks

`bash scripts/smoke_test.sh` byte-compiles every module, runs `tests/test_multitq_eval.py` (the data-dependent cases run
when `MULTITQ_DIR` points to the MultiTQ data) and `tests/toy_end_to_end.py` (20 synthetic questions: official Hits@1,
paired Poisson bootstrap, Holm). `robustness/r2/kit/tests_R2.py` re-checks the R2 scoring and audit-kit code against frozen R1 outputs (R1
predictions, review key / gold, freeze; not shipped). `tests/test_research_track.py` checks the research-track code on
synthetic data (ablation bootstrap = evaluator bootstrap, the Ranker feature-group partition, coverage tables, the
matched-baseline inputs and statistics). `tests/test_robustness_methods.py` checks the R3 training order, contrasts,
audit selection and BOUND check, the Parser-first routing and the normalizer selection rule.

## Notes on reproduction

- The stage wrappers in `scripts/` chain the commands that produced the reported results; the wrappers themselves were
  assembled for this release and have only been syntax-checked. GPU-host scripts expect two GPUs (one process per card;
  70B arms use tensor parallelism over both).
- Frozen structural execution is deterministic on CPU. Structural retraining is repeatable only with a fixed
  `PYTHONHASHSEED` (the classifier vocabulary order follows Python's string hashing); the wrappers set 0. LoRA training
  and vLLM greedy decoding are not bitwise deterministic: compare Hits@1 and top-1 agreement rather than file hashes.
- Freeze records, selection logs and integrity lists are written at run time. `src/multitq_eval.py` and
  `kpct/parser/p0_exemplars.json` are byte-identical to the registered files (sha256 `1f85bbdc...` and `5a3ea332...`,
  as recorded in `protocol/`), so the scorer-hash constants in the code and in the protocol agree. Checks that compared
  other code or rule files with hashes recorded at registration print a warning instead of stopping (the copies in
  this release differ byte-wise from the registered ones: paths replaced by variables, comments edited).
- Integrity references are frozen once: `scripts/freeze_integrity_refs.sh` (called at the end of
  `scripts/reproduce_kpct_8_test_pass.sh`) hashes the files listed in `robustness/*/remote/expected_sha.template.json`
  and `kpct/followups/seeds_P/host_a/ref_seed0.template.json` into read-only files under `$WORK_DIR/integrity/`. The
  R1 / R2 probe wrappers and the parser-seed scoring stop if these files are missing; `ALLOW_SELF_REFERENCE=1` builds
  them from the current files instead, which makes the check vacuous. `analysis/reader_mechanism.py` prints a note when
  G_1 Hits@1 differs from the frozen value (expected after re-training); `EXPECT_FROZEN_G1=1` makes this fatal.
- Research-track additions: the matched baseline and the LLM controls run on one GPU (`MBQ_CARD`, `LC_CARD`). The LLM
  controls were run with the two memory-only options of amendment LC1-A1 (part of the card's memory was held by other
  processes): gradient checkpointing in LoRA training (`LC_GRAD_CKPT=1`, the default) and vLLM
  `gpu_memory_utilization` 0.40 (`LC_GPU_MEM`, default 0.40; the frozen Reader recipe uses 0.85). `LC_GRAD_CKPT=0
  LC_GPU_MEM=0.85` restores the frozen recipe (training peaks at about 84 GB; vLLM reserves 85% of the card). Neither
  option changes the optimisation or the decoding rule, but greedy GPU decoding is not bitwise batch-invariant. Part B of
  the structural ablation retrains with `kpct/structural/kpct_retrain_fixed_vocab.py`, which requires a fixed
  `PYTHONHASHSEED` and imposes the vocabulary order of the structural run's saved checkpoints (`$SWORK/out/<k>`); the
  out-of-fold vocabularies follow the hash seed, hence the second seed (7) as a noise floor.
  `analysis/coverage/a2_gu_parser_followup.py` compares its inputs with the sha256 of the reported run's frozen files
  and only notes a mismatch unless `EXPECT_FROZEN_HASHES=1`.
- Rewording-robustness method tracks: the frozen LLM arms on rewrites and on normalized questions run with vLLM
  `gpu_memory_utilization` 0.40 as in the LLM controls (`GPU_MEM`); the Qwen paraphrase session sizes its KV cache from
  the free card memory at launch. The R3 sets, the normalizer item files, the dev selections and the grounded-Reader
  freeze were built by library calls (listed in the reproduce scripts). `expected_*.template.json` are filled with the
  hashes of your own frozen files (`reproduce_r3.sh integrity`, `reproduce_grounded_reader.sh integrity`); the
  normalizer trains and decodes on a second host with one 24 GB card (`WORK_DIR_B`).
- `src/struct/pipeline_steps.py` transcribes the executed order of the full-data pipeline; the top-level wrappers it
  names for full-data classifier, out-of-fold and Ranker training (`train_parser`, `train_oof`, `train_S`) are not
  included (their building blocks `train_clf`, `run_cands.py` and the LightGBM settings of `kpct/structural/kpct_run.py` are).
- Not shipped and needed for a full rerun: the MultiTQ / CronQuestions releases, the base models, the quid lists
  `s_train100k_quids.json` / `reader_train60k_quids.json` and the earlier 10,000-question dev list, and the outputs of
  the full-data structural training (linker artefacts, classifier, Ranker models, out-of-fold exports).

## License

MIT (see `LICENSE`).
