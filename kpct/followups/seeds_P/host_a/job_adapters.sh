#!/bin/bash
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set -e
K=$WORK_DIR; PS=$K/parser_seeds
tar xf seed_adapters.tar -C $PS/
python3 - <<'PY'
import json, hashlib, os
K = os.path.expanduser("$WORK_DIR"); PS = K + "/parser_seeds"
for arm, n, c in [("P_k05_s0_lora1","P_k05_s0_s1","ep1"),("P_k05_s0_lora2","P_k05_s0_s2","ep2"),("P_k01_s0_lora1","P_k01_s0_s1","ep1"),("P_k01_s0_lora2","P_k01_s0_s2","ep2")]:
    fz = json.load(open(f"{PS}/freeze/freeze_{arm}.json"))
    h = hashlib.sha256(open(f"{PS}/adapters/{n}/{c}/adapter_model.safetensors", "rb").read()).hexdigest()
    print(arm, c, h == fz["adapter"]["sha256"], h[:16])
PY
chmod -R a+rX $PS/adapters; du -sh $PS/adapters
