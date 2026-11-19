#!/bin/bash
# pf_replay_all.sh -- Parser-first track: CPU replay of CACHED Parser generations (no LLM inference) with the
# unchanged structural-ablation replay script, inside ${GPU_IMAGE:-tkgqa-gpu:latest} (no GPUs), kpct tree mounted read-only at /kpct.
cd /workspace
mkdir -p out logs
run() {  # tag k split raw bundles fbdir fbprefix frozen_pattern
  python3 -u sa_parser_replay.py --k "$2" --split "$3" --raw "$4" --bundles "$5" --fallback_dir "$6" \
     --fallback_prefix "$7" --frozen_pred "$8" --out "out/$1" --workers 4 > "logs/$1.log" 2>&1
  echo "$1 rc=$?" >> logs/rc.txt
}
P=/kpct/parser; L=/kpct/robustness_r3/llm
run dev_k01_s0 k01_s0 dev_sel2000 $P/raw/k01_s0/raw_P_dev_sel2000_ep3.jsonl.gz /kpct/bundles/k01_s0/dev_sel2000.jsonl /kpct/preds/k01_s0 dev_sel2000 "$P/exec/k01_s0/pred_P_k01_s0_dev_sel2000_ep3_tauoff_seed{b}.jsonl" &
run dev_k05_s0 k05_s0 dev_sel2000 $P/raw/k05_s0/raw_P_dev_sel2000_ep1.jsonl.gz /kpct/bundles/k05_s0/dev_sel2000.jsonl /kpct/preds/k05_s0 dev_sel2000 "$P/exec/k05_s0/pred_P_k05_s0_dev_sel2000_ep1_tauoff_seed{b}.jsonl" &
run dev_k10_s0 k10_s0 dev_sel2000 $P/raw/k10_s0/raw_P_dev_sel2000_ep0.5.jsonl.gz /kpct/bundles/k10_s0/dev_sel2000.jsonl /kpct/preds/k10_s0 dev_sel2000 "$P/exec/k10_s0/pred_P_k10_s0_dev_sel2000_ep0.5_tauoff_seed{b}.jsonl" &
run dev_k100 k100 dev_sel2000 $P/raw/k100/raw_P_dev_sel2000_ep0.5.jsonl.gz /kpct/bundles/k100/dev_sel2000.jsonl /kpct/preds/k100 dev_sel2000 "$P/exec/k100/pred_P_k100_dev_sel2000_ep0.5_tauoff_seed{b}.jsonl" &
for k in k01_s0 k05_s0 k100; do
  run var_$k $k variants $L/P/raw/raw_P_variants_P_$k.jsonl.gz $L/bundles/$k/variants.jsonl $L/preds/$k variants "$L/P/$k/pred_P_${k}_variants_P_${k}_tauoff_seed{b}.jsonl" &
done
wait
# input hashes
{ for k in k01_s0 k05_s0 k10_s0 k100; do sha256sum /kpct/bundles/$k/dev_sel2000.jsonl $P/raw/$k/raw_P_dev_sel2000_*.jsonl.gz /kpct/preds/$k/dev_sel2000_S_seed*.jsonl $P/exec/$k/pred_P_${k}_dev_sel2000_*_tauoff_seed*.jsonl; done;
  for k in k01_s0 k05_s0 k100; do sha256sum $L/bundles/$k/variants.jsonl $L/P/raw/raw_P_variants_P_$k.jsonl.gz $L/preds/$k/variants_S_seed*.jsonl $L/P/$k/pred_P_${k}_variants_P_${k}_tauoff_seed*.jsonl; done;
  sha256sum sa_parser_replay.py /kpct/parser/code/pk_common.py /kpct/executor/kpct_executor.py /kpct/executor/kpct_core.py; } > inputs_sha.txt
# keep only what the registered variants use: PK / PALONE fb lists + pools (sha256 of every written file stays in replay_stats)
find out -name 'pred_PREL_*' -delete; find out -name 'pred_PENT_*' -delete; find out -name 'pred_*_strict_*' -delete
find out -name 'pool_PREL_*' -delete; find out -name 'pool_PENT_*' -delete
cp out/*/replay_stats_*.json . 2>/dev/null
tar czf replay_out.tgz out logs inputs_sha.txt
chmod -R a+rwX /workspace
