# R2 paraphrase audit ({N} rows): annotator instructions

Operational instructions for the annotator. This file is not paper text.

Review only `{SHEET}.xlsx` (or the `.csv`, same content). Rows are identified by `audit_id` (A001-A{NNN}).
The rows are rewrites of MultiTQ test questions written by a second generator. Each one passed the mechanical
checks and an LLM judge, with names and times protected. The rows are in a random order.

1. Single annotator: one author reviews every row alone. Do not discuss rows with anyone until the sheet is returned.
2. Do not open `key_do_not_open_before_review/`, any model output, the robustness reports, or earlier audits while
   reviewing.
3. For each row, read `original_question`, `gold_answers` (the official answers to the original question; several
   answers are separated by ` | `), `qtype`, and then `rewrite`. Fill the four columns below.
4. Fill every row. A blank or `unsure` row counts as not accepted. A row is accepted only if both
   `same_answer_set` and `same_constraint_time` are `Y`.
5. Save as `{SHEET}_<initials>.xlsx` in this folder and tell the analyst.

| column | values | meaning |
|---|---|---|
| `same_answer_set` | `Y`, `N`, `unsure` | `Y` if the rewrite asks for exactly the same set of correct answers as the original: the gold answers are all correct for the rewrite, and the rewrite would admit no other answers. Put `N` if the rewrite narrows or widens the type of answer (for example "who" becomes "which country" or "which person" while a gold answer is a person, an organisation or a group), or if it changes the relation, the direction (before/after, first/last) or an entity. Put `unsure` if you cannot decide. |
| `same_constraint_time` | `Y`, `N` | `Y` if the temporal constraint is unchanged: the same date or period, the same granularity (day, month, year), and the same anchoring (before / after / first / last / equal, and the same anchor event). Otherwise `N`. |
| `notes` | free text | A short reason for every `N` or `unsure`. |
| `reviewer_initials` | your initials | The same on every row. |

Judge each rewrite on its wording, under an ordinary reading. Spelling, word order and register changes are fine
if the meaning is unchanged. Do not use world knowledge about which answer a system would give.

Constructed calibration example (not from the sheet):

- Original: "Who negotiated with Country A after Person B?" Gold: `Ministry X | Person C`.
- Rewrite 1: "After Person B, who held negotiations with Country A?" -> `Y`, `Y`.
- Rewrite 2: "After Person B, which official negotiated with Country A?" -> `same_answer_set = N` ("official" excludes
  the ministry), `same_constraint_time = Y`. Note: "who -> official narrows the answer type".
- Rewrite 3: "Who negotiated with Country A before Person B?" -> `N`, `N`. Note: "after -> before".

Expected time: about {MINUTES} minutes ({N} rows at about 35 seconds each).

## Analyst only (after the sheet is returned)

```
python key_do_not_open_before_review/audit_score_R2.py --sheet {SHEET}_<initials>.xlsx \
    --selection key_do_not_open_before_review/audit_R2_selection_key.csv \
    --gold key_do_not_open_before_review/multitq_robustness_R2_gold.json \
    --pred-dir <extracted multitq_robustness_R2_predictions.tgz> --scorer key_do_not_open_before_review/multitq_eval.py \
    --freeze key_do_not_open_before_review/multitq_robustness_R2_freeze.json --out audit_R2_result.json
```

Registered analyses (rule R2, amendment A2): random-100 acceptance with Wilson 95% CI; targeted acceptance;
BOUND = H1-H3 on all judge-accepted rewrites minus the human-rejected rows (B = 10,000, seed 20261029, one-sided,
Holm), labelled BOUND_HOLDS / BOUND_FAILS; audited-accepted-only contrasts (descriptive). Paper wording:
"human-audited by one author: all rewrites on which the parser outperformed the reader plus a random 100".
Keep the completed sheet unchanged, because it is the audit record.
