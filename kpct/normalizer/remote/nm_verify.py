#!/usr/bin/env python3
"""nm_verify.py <expected_sha.json[,expected_sha.json...]> <out.json> -- normalizer track integrity check: sha256 of every
frozen adapter, frozen code file and frozen prediction file listed in the robustness_r3 freezes (expected_sha_r3.json,
expected_sha_dev.json), plus the frozen R1 rob_build.py staged for this track. Exit 3 on any mismatch
(STOP_TEST_INTEGRITY). Reads no prediction content and no gold."""
import hashlib, json, os, sys
ROB_BUILD = ("/kpct/normalizer/code/rob_build.py", "0400d972eb1f33ee0c5501e6ee05c58eab3d83e6c0b81a245bbb23b211a19edf")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


out = dict(files={}, bad=[])
for ef in sys.argv[1].split(","):
    E = json.load(open(ef))
    for grp in ("adapters", "code", "pred_files"):
        for p, s in E.get(grp, {}).items():
            if p in out["files"]:
                continue
            got = sha(p) if os.path.exists(p) else None
            out["files"][p] = dict(group=grp, expected=s, got=got, ok=got == s)
            if got != s:
                out["bad"].append(p)
p, s = ROB_BUILD
got = sha(p) if os.path.exists(p) else None
out["files"][p] = dict(group="code", expected=s, got=got, ok=got == s)
if got != s:
    out["bad"].append(p)
out["n_checked"] = len(out["files"]); out["n_bad"] = len(out["bad"])
os.makedirs(os.path.dirname(sys.argv[2]), exist_ok=True)
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps(dict(n_checked=out["n_checked"], n_bad=out["n_bad"], bad=out["bad"][:5])), flush=True)
sys.exit(3 if out["bad"] else 0)
