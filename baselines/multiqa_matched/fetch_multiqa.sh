#!/bin/bash
# fetch_multiqa.sh <job dir> -- third-party MultiQA code and data for the matched baseline (not redistributed here).
# Source: https://github.com/czy1999/MultiTQ at commit 94ad61c31235467cc077a99e3a5ee105ac1dd04b (MultiQA, Chen et al.,
# ACL 2023). No file of the upstream code is modified: mbq_qa.py imports qa_baselines.py, qa_datasets.py, utils.py and
# tcomplex.py as released, mbq_ner.py ports the logic of ner_task.py, mbq_tcx_full.py imports MultiQA/tkbc/.
# Job directory layout produced (the upstream repository layout):
#   <job>/MultiQA/                                         upstream MultiQA/ code
#   <job>/MultiQA/models/MultiTQ/kg_embeddings/tcomplex.ckpt  shipped TComplEx (from MultiQA/models/Models.zip)
#   <job>/data/MultiTQ/kg/tkbc_processed_data/{ent_id,rel_id,ts_id,train,valid,test.pickle}  (from data/Dataset.zip)
# Set MULTIQA_SRC to an existing checkout of that commit to skip the clone.
set -euo pipefail
JD=${1:?usage: fetch_multiqa.sh <job dir>}; mkdir -p "$JD"
COMMIT=94ad61c31235467cc077a99e3a5ee105ac1dd04b
SRC=${MULTIQA_SRC:-$JD/_multitq_src}
if [ ! -d "$SRC/.git" ] && [ ! -d "$SRC/MultiQA" ]; then
  git clone -q https://github.com/czy1999/MultiTQ.git "$SRC"
fi
[ -d "$SRC/.git" ] && git -C "$SRC" checkout -q "$COMMIT"
# the upstream files used unchanged (sha256 as recorded in protocol/multitq_matched_baseline_rule.json)
(cd "$SRC/MultiQA" && sha256sum -c - <<'EOF'
9e2dd86554e5392fc7c56b831bbb7c66ba709e7a43f4a30fa16618668a6566d0  qa_baselines.py
cbd74efec521d485ce9b44299ede56717dfacb02146dfe333c4ac360d23eea1f  qa_datasets.py
46c80d7d1a0ab0510e7245b3715556db91f97efb827f67e8fd8cbb4f0d796004  utils.py
4003e46c02b20d04c50d75f7e05f59a07a9e1dfb923d13e12ad63b4c29fb76b8  tcomplex.py
d915707acbc04a16a89df3901f5413639f7abac62e9924f8e1a6ff5cf82d06ba  ner_task.py
EOF
)
mkdir -p "$JD/MultiQA" "$JD/data/MultiTQ/kg/tkbc_processed_data" "$JD/MultiQA/models/MultiTQ/kg_embeddings" "$JD/MultiQA/models/MultiTQ/qa_models"
(cd "$SRC/MultiQA" && tar cf - --exclude='models/Models.zip' .) | (cd "$JD/MultiQA" && tar xf -)
T=$(mktemp -d)
unzip -q -o "$SRC/MultiQA/models/Models.zip" 'MultiTQ/kg_embeddings/tcomplex.ckpt' -d "$T"
cp "$T/MultiTQ/kg_embeddings/tcomplex.ckpt" "$JD/MultiQA/models/MultiTQ/kg_embeddings/"
for f in ent_id rel_id ts_id train.pickle valid.pickle test.pickle; do
  unzip -q -o "$SRC/data/Dataset.zip" "MultiTQ/kg/tkbc_processed_data/$f" -d "$T"
  cp "$T/MultiTQ/kg/tkbc_processed_data/$f" "$JD/data/MultiTQ/kg/tkbc_processed_data/"
done
rm -rf "$T"
echo "e10544da3dcdd88a0608c34a501e31b5b4f4e29eb7ab76c4d4d63f309682857c  $JD/MultiQA/models/MultiTQ/kg_embeddings/tcomplex.ckpt" | sha256sum -c -
