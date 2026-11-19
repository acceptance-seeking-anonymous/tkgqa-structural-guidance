import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
keys = ["n", "list", "list_sha256", "prompt_ids_sha256_first100", "target_in_source", "n_truncated", "len_mean", "lora", "opt", "max_len", "epochs", "versions", "prompt_sha256", "train_tok", "total_steps", "order", "arm"]
diff = [k for k in keys if a.get(k) != b.get(k)]
out = dict(seed0=sys.argv[1], new=sys.argv[2], compared=keys, differences=diff, STOP_RECIPE=bool(diff), new_seed=b.get("seed"))
json.dump(out, open(sys.argv[3], "w"), indent=1); print(json.dumps(out)); sys.exit(1 if diff else 0)
