#!/bin/bash
# append the second-host parser_seeds usage-log rows to the shared k% GPU usage log on the GPU host (flock-serialised, idempotent by container name)
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
K=$WORK_DIR; L=$K/gpu_usage_kpct.tsv; PS=$K/parser_seeds
mkdir -p $PS/usage_log; cp gpu_usage_pseeds.tsv $PS/usage_log/gpu_usage_pseeds_hostb.tsv
(
  flock 9
  awk -F'\t' 'NR>1' gpu_usage_pseeds.tsv | while IFS=$'\t' read tr host cards nc st en sec gh rc tag cn; do
    grep -q "$cn" $L || printf "parser_seeds_hostb\thostb:%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$cards" "$nc" "$st" "$en" "$sec" "$gh" "$rc" "$tag" "$cn" >> $L
  done
) 9>$L.lock
grep -P "^parser_seeds_hostb\t" $L > out_rows.tsv; awk -F'\t' '{s+=$7} END{printf "parser_seeds_hostb rows %d gpu_h %.4f\n", NR, s}' out_rows.tsv
awk -F'\t' 'NR>1{s+=$7} END{printf "shared usage log total gpu_h %.4f\n", s}' $L
df -h $WORK_DIR | tail -1
