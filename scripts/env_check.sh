#!/bin/bash
# env_check.sh VAR [VAR ...] -- source from the reproduce scripts: fail early when a required path variable is unset.
# All paths come from configs/paths.env (copy configs/paths.env.example); no script references a path outside the
# repository except through these variables.
for v in "$@"; do
  if [ -z "${!v:-}" ]; then echo "ERROR: set $v (see configs/paths.env.example)" >&2; exit 2; fi
done
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export REPO
