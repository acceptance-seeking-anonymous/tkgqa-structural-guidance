#!/bin/bash
# nm_infer_job.sh <run name> <items file (in job dir)> <adapters: ep0.5,ep1,...> <spent_total_gpu_h> <projection_gpu_h>
# Normalizer decoding on the second GPU host RTX 4090 with vLLM 0.10.2 (image vllm/vllm-openai:v0.10.2), one engine session, the
# listed checkpoints of N_k05_s0_s0 one after another. GPU-hour cap: spent + projection <= 4.5 else NOT_RUN (exit 9).
# Usage_log row per container in $W/gpu_usage_normalizer_hostb.tsv (track normalizer).
: "${WORK_DIR_B:?set WORK_DIR_B (see configs/paths.env.example)}"
: "${MODEL_DIR:?set MODEL_DIR (see configs/paths.env.example)}"
set +e
RUN=$1; ITEMS=$2; CK=$3; SPENT=${4:-0}; PROJ=${5:-0.1}
J=$PWD; W=$WORK_DIR_B/normalizer; L=$W/gpu_usage_normalizer_hostb.tsv; CAP=4.5
mkdir -p $W/runs/$RUN $W/inputs $W/logs
for f in nm_common.py nm_infer.py; do
  if [ -f $W/code/$f ] && ! cmp -s $J/$f $W/code/$f; then echo "CODE_CHANGED $f"; exit 2; fi
done
if [ -f $W/inputs/$ITEMS ] && ! cmp -s $J/$ITEMS $W/inputs/$ITEMS; then echo "INPUT_CHANGED $ITEMS"; exit 2; fi
cp -n $J/$ITEMS $W/inputs/
python3 -c "import json; rows=[json.loads(l) for l in open('$W/inputs/$ITEMS')]; open('$W/runs/$RUN/items_text.jsonl','w').write(''.join(json.dumps(dict(iid=r['iid'], text=r['text']), ensure_ascii=False)+'\n' for r in rows))"
spent_b() { awk -F'\t' 'NR>1 && $1=="normalizer"{s+=$7} END{printf "%.4f", s+0}' $L; }
TOT=$(awk -v a=$SPENT -v b=$(spent_b) 'BEGIN{printf "%.4f", a+b}')
awk -v s=$TOT -v p=$PROJ -v c=$CAP 'BEGIN{exit !(s+p<=c)}' || { echo "NOT_RUN $RUN (spent $TOT projected $PROJ cap $CAP)"; exit 9; }
AD=""; for c in $(echo $CK | tr ',' ' '); do AD="$AD,N_$c=$W/adapters/N_k05_s0_s0/$c"; done; AD=${AD#,}
for i in $(seq 1 60); do n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .); [ "$n" = 0 ] && break; sleep 60; done
[ "$n" = 0 ] || { echo "4090 busy; not launching"; exit 75; }
CN=nmb_infer_${RUN}_$(date +%s); t0=$(date +%s)
( sleep 3600; docker stop $CN > /dev/null 2>&1 && echo "WATCHDOG stopped $CN" >&2 ) & WD=$!
docker run --rm --name $CN --gpus all --ipc=host --shm-size=8g --entrypoint python3 -v $W:$W -v $MODEL_DIR:/hf \
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e TOKENIZERS_PARALLELISM=false -w $W vllm/vllm-openai:v0.10.2 \
  $W/code/nm_infer.py --items $W/runs/$RUN/items_text.jsonl --adapters $AD --probe $W/adapters/N_k05_s0_s0/probe_prompt_ids.json --out $W/runs/$RUN > $W/logs/infer_$RUN.log 2>&1
rc=$?; t1=$(date +%s); kill $WD 2>/dev/null
printf "normalizer\thost_b\t1\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -u -d @$t0 +%FT%TZ)" "$(date -u -d @$t1 +%FT%TZ)" "$((t1-t0))" \
  "$(awk -v s=$((t1-t0)) 'BEGIN{printf "%.4f", s/3600}')" "$rc" "infer_$RUN" "$CN" >> $L
echo "infer rc=$rc"
docker run --rm -v $W:$W ${GPU_IMAGE:-tkgqa-gpu:latest} bash -c "chmod -R a+rX $W/runs/$RUN" > /dev/null 2>&1
mkdir -p $J/out; cp $W/runs/$RUN/raw_*.jsonl.gz $W/runs/$RUN/infer_stats.json $J/out/ 2>/dev/null; cp $W/logs/infer_$RUN.log $J/out/
cp $L $J/out/usage_log_normalizer_hostb.tsv
echo "spent_b $(spent_b)"
exit $rc
