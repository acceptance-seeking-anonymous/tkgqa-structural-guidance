import json, sys
sel = set(json.load(open(sys.argv[3])))
fz = {}
for l in open(sys.argv[1]):
    x = json.loads(l); fz[int(x["quid"])] = x["parse"]
rb = {}
for l in open(sys.argv[2]):
    x = json.loads(l)
    if int(x["quid"]) in sel: rb[int(x["quid"])] = x["parse"]
def walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b): out["key_diff"].append(path); return
        for k in a: walk(a[k], b[k], path + "/" + k, out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b): out["len_diff"].append(path); return
        for i, (u, v) in enumerate(zip(a, b)): walk(u, v, path + "[]", out)
    elif isinstance(a, float) or isinstance(b, float):
        try:
            d = abs(float(a) - float(b))
        except Exception:
            out["type_diff"].append(path); return
        if d > 0:
            out["n_float_diff"] += 1; out["max_abs"] = max(out["max_abs"], d)
            out["paths"][path] = max(out["paths"].get(path, 0.0), d)
    elif a != b:
        out["nonfloat_diff"].append(path)
out = dict(n=0, n_rows_identical=0, n_float_diff=0, max_abs=0.0, paths={}, key_diff=[], len_diff=[], type_diff=[], nonfloat_diff=[])
for q in sorted(sel):
    o2 = dict(n=0, n_rows_identical=0, n_float_diff=0, max_abs=0.0, paths={}, key_diff=[], len_diff=[], type_diff=[], nonfloat_diff=[])
    walk(fz[q], rb[q], "parse", o2)
    out["n"] += 1; out["n_rows_identical"] += int(fz[q] == rb[q])
    out["n_float_diff"] += o2["n_float_diff"]; out["max_abs"] = max(out["max_abs"], o2["max_abs"])
    for k, v in o2["paths"].items(): out["paths"][k] = max(out["paths"].get(k, 0.0), v)
    for k in ("key_diff", "len_diff", "type_diff", "nonfloat_diff"): out[k] += [f"{q}:{p}" for p in o2[k]]
for k in ("key_diff", "len_diff", "type_diff", "nonfloat_diff"): out[k + "_n"] = len(out[k]); out[k] = out[k][:20]
json.dump(out, open("abl_parse_diff.json", "w"), indent=1); print(json.dumps({k: v for k, v in out.items() if k != "paths"}), out["paths"])
