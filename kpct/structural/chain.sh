#!/bin/bash
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES OMP_NUM_THREADS=1 KPCT_NPROC=8
while [ ! -f out/k05_s0/dev_summary.json ]; do sleep 10; done
for s in k10_s0 k01_s1 k01_s2; do
  mkdir -p out/$s
  python kpct_run.py all $s > out/$s/run.log 2>&1 || echo "FAILED $s" >> out/chain.log
  echo "done $s $(date)" >> out/chain.log
done
mkdir -p out/k100
python kpct_k100.py > out/k100/run.log 2>&1 || echo "FAILED k100" >> out/chain.log
echo "done k100 $(date)" >> out/chain.log
