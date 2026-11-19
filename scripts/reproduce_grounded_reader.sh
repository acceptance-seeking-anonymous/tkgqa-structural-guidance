#!/bin/bash
# reproduce_grounded_reader.sh <step> [card] [levels] -- Parser-grounded Reader (protocol/
# multitq_robustness_grounded_reader_rule.json): the frozen Reader G_k on bundles built from the CACHED Parser
# generations (PALONE = Parser hypothesis only; PK = the frozen P_k parse), k in {k05_s0, k100}; prompt-identical rows
# reuse the frozen G_k output. GPU host steps (code staged into $WORK_DIR/grounded_reader/code by stage A):
#  integrity  hash the frozen inputs listed in kpct/grounded_reader/remote/expected_gr.template.json (null entries are
#             recorded only) into $WORK_DIR/integrity/expected_gr.json
#  stageA     CPU: frozen-input check (exit 3 = STOP_TEST_INTEGRITY) and bundle building on dev, the R3 rewrite root,
#             test, R1 P3 and R2 rewrites (remote/jobA_gr.sh)
#  dev        one vLLM session on one card: dev + DEV_ACCEPTED generation, assembly, dev selection (remote/jobB_gr.sh)
#  final      one vLLM session for the dev-selected sources (selection.json in the job dir) + assembly (remote/jobC_gr.sh)
#  freeze     local: kpct/grounded_reader/local/gr_freeze.build_freeze(job_dir, selection, rule_id, rule_sha256,
#             selection_id, out) from the final_sha256 lists of step final -> freeze.json (copy it into the job dir)
#  score      CPU: the single scoring pass (remote/jobD_gr.sh -> gr_score.py; full test, R3 secondary, R1 / R2)
# JOBDIR (default ./gr_job) is the job directory. vLLM memory fraction: GPU_MEM (default 0.40, as run).
set -euo pipefail
source "$(dirname "$0")/env_check.sh" WORK_DIR
G=$REPO/kpct/grounded_reader; JD=${JOBDIR:-$PWD/gr_job}; mkdir -p "$JD"
case "${1:?step}" in
  integrity)
    source "$(dirname "$0")/env_check.sh" MTQ_DIR DATA_DIR
    I=$WORK_DIR/integrity; mkdir -p "$I"
    [ -f "$I/expected_gr.json" ] && { echo "kept existing $I/expected_gr.json"; exit 0; }
    python3 - "$G/remote/expected_gr.template.json" "$I/expected_gr.json" <<'EOF'
import hashlib, json, os, sys
T = json.load(open(sys.argv[1])); M = {'/kpct/': os.environ['WORK_DIR'], '/mtq/': os.environ['MTQ_DIR'], '/data/': os.environ['DATA_DIR']}
def host(p):
    for a, b in M.items():
        if p.startswith(a):
            return os.path.join(b, p[len(a):])
    return p
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()
out = {p: (sha(host(p)) if v == '' else None) for p, v in T.items()}
json.dump(out, open(sys.argv[2], 'w'), indent=1); print(len(out), 'entries,', sum(v is not None for v in out.values()), 'hashed')
EOF
    chmod a-w "$I/expected_gr.json" ;;
  stageA)
    cp "$G"/remote/*.py "$G"/remote/*.sh "$JD/"; cp "$WORK_DIR/integrity/expected_gr.json" "$JD/expected_gr.json"
    (cd "$JD" && bash jobA_gr.sh) ;;
  dev)    (cd "$JD" && bash jobB_gr.sh "${2:-1}" "${3:-k05_s0,k100}") ;;
  final)  [ -f "$JD/selection.json" ] || { echo "copy the registered selection (json {level: PALONE|PK|G_k}) to $JD/selection.json" >&2; exit 2; }
          (cd "$JD" && bash jobC_gr.sh "${2:-1}" "${3:-k05_s0,k100}") ;;
  freeze) echo "python3 -c 'import sys; sys.path.insert(0, \"$G/local\"); import gr_freeze; gr_freeze.build_freeze(\"$JD\", <selection>, <rule id>, <rule sha256>, <selection id>, \"$JD/freeze.json\")'" ;;
  score)  [ -f "$JD/freeze.json" ] || { echo "write $JD/freeze.json first (step freeze)" >&2; exit 2; }
          (cd "$JD" && bash jobD_gr.sh) ;;
  *) echo "unknown step $1" >&2; exit 2 ;;
esac
