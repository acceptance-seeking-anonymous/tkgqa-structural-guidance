
import json, os, sys, hashlib, datetime
def build_freeze(job_dir, selection, rule_vid, rule_sha, sel_vid, out):
    files = {}
    for fn, pre in (("final_sha256.txt", "/kpct/grounded_reader/"), ("final_gen_sha256.txt", "/kpct/grounded_reader/")):
        for l in open(os.path.join(job_dir, fn)):
            if l.strip():
                s, p = l.split(None, 1); files[pre + p.strip()] = s
    fz = dict(name="grounded_reader_freeze", rule_version_id=rule_vid, rule_sha256=rule_sha, selection_record_version_id=sel_vid,
              selection=selection, label="reuse of the official MultiTQ test split (+ R3 / R1 P3 / R2 rewrites); frozen BEFORE the single scoring pass; scored=False",
              frozen_utc=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), scored=False, gold_read_for_GR_test=False, files=files)
    s = json.dumps(fz, indent=1, sort_keys=True)
    fz["sha256_of_body"] = hashlib.sha256(s.encode()).hexdigest()
    json.dump(fz, open(out, "w"), indent=1)
    return fz
